"""Tests of tycho.agent.runner, prompts, nightly and weekly with a scripted fake client (no network)."""
from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
import yaml

from tycho.agent.nightly import run_nightly
from tycho.agent.prompts import LANGUAGE_INSTRUCTIONS, NIGHTLY_SYSTEM, WEEKLY_SYSTEM, system_prompt, task_text
from tycho.agent.runner import (FALLBACK_BETA, MAX_TOKENS, RUNNER_AUTHOR, AgentRunner, AgentUnavailable, Pricing,
                                Usage, observation_window)
from tycho.agent.tools import iso_week, local_date
from tycho.agent.weekly import run_weekly
from tycho.config import Paths
from tycho.survey.catalog import Catalog

SKY_V0 = Path(__file__).resolve().parents[1] / "configs" / "sky_v0.yaml"
SYSTEM = "You are a test agent."
DAY = 86_400.0


# ---------------------------------------------------------------------------------------------- the fake client
class FakeMessages:
    """``client.beta.messages`` replaying a script of responses (or exceptions) and recording every request."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))  # the runner keeps appending to its message list
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, script: list) -> None:
        self.beta = SimpleNamespace(messages=FakeMessages(script))

    @property
    def requests(self) -> list[dict]:
        return self.beta.messages.requests


def text(t: str):
    return SimpleNamespace(type="text", text=t)


def thinking():
    return SimpleNamespace(type="thinking", thinking="", signature="sig-123")


def tool_use(tid: str, name: str, tool_input: dict):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=tool_input)


def response(*blocks, stop: str = "end_turn", usage: tuple | None = (100, 50, 0, 0), model: str = "claude-opus-5-5",
             stop_details=None):
    u = None if usage is None else SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1],
                                                   cache_read_input_tokens=usage[2],
                                                   cache_creation_input_tokens=usage[3])
    return SimpleNamespace(id="msg_1", type="message", role="assistant", content=list(blocks), stop_reason=stop,
                           usage=u, model=model, stop_details=stop_details)


def api_error(cls, status: int, message: str = "boom"):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    if cls is anthropic.APIConnectionError:
        return cls(request=request)
    return cls(message, response=httpx2.Response(status, request=request), body=None)


class EchoTools:
    """A minimal tool box: "echo" returns its input, "boom" raises."""

    specs = [{"name": "echo", "description": "Echo.", "strict": True,
              "input_schema": {"type": "object", "properties": {"x": {"type": "integer", "description": "x"}},
                               "required": ["x"], "additionalProperties": False}}]

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def dispatch(self, name, input):  # noqa: A002
        self.calls.append((name, input))
        if name == "boom":
            raise RuntimeError("kaput")
        return json.dumps({"echo": input}), False


@pytest.fixture
def catalog(tmp_path):
    cat = Catalog(tmp_path / "data" / "catalog.sqlite")
    yield cat
    cat.close()


def runs(cat: Catalog) -> list[dict]:
    return cat.query("SELECT * FROM agent_runs ORDER BY run_id")


def tool_calls(cat: Catalog) -> list[dict]:
    return cat.query("SELECT * FROM agent_tool_calls ORDER BY id")


def keys_anywhere(obj) -> set[str]:
    if isinstance(obj, dict):
        return set(obj).union(*(keys_anywhere(v) for v in obj.values()))
    if isinstance(obj, (list, tuple)):
        return set().union(*(keys_anywhere(v) for v in obj)) if obj else set()
    return set()


# ---------------------------------------------------------------------------------------------- the loop
def test_tool_use_then_end_turn_with_usage_audit_and_history(catalog):
    first = response(thinking(), text("checking"), tool_use("t1", "echo", {"x": 1}), tool_use("t2", "echo", {"x": 2}),
                     stop="tool_use", usage=(1000, 200, 0, 500))
    client = FakeClient([first, response(text("done"), usage=(50, 100, 1000, None))])
    tools = EchoTools()
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", tools)
    assert result.status == "completed" and result.stop_reason == "end_turn" and result.final_text == "done"
    assert (result.n_api_calls, result.n_tool_calls) == (2, 2) and result.error is None
    assert tools.calls == [("echo", {"x": 1}), ("echo", {"x": 2})]
    first_req, second_req = client.requests
    assert first_req["messages"] == [{"role": "user", "content": "go"}]
    user, assistant, results = second_req["messages"]
    assert assistant["role"] == "assistant" and assistant["content"] == first.content  # full content, thinking too
    assert results == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": '{"echo": {"x": 1}}', "is_error": False},
        {"type": "tool_result", "tool_use_id": "t2", "content": '{"echo": {"x": 2}}', "is_error": False}]}
    assert result.messages[-1]["content"][0].text == "done"
    u = result.usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (1050, 300, 1000, 500)
    expected = (1050 * 4 + 300 * 20 + 1000 * 0.2 + 500 * 5) / 1e6
    assert result.est_cost_usd == pytest.approx(expected)
    (row,) = runs(catalog)
    assert row["run_id"] == result.run_id and row["kind"] == "nightly" and row["model"] == "claude-opus-5-5"
    assert (row["status"], row["stop_reason"], row["error"]) == ("completed", "end_turn", None)
    assert (row["input_tokens"], row["output_tokens"], row["cache_read_tokens"], row["cache_write_tokens"]) == \
        (1050, 300, 1000, 500)
    assert row["est_cost_usd"] == pytest.approx(expected) and row["n_tool_calls"] == 2
    assert row["finished_at"] is not None
    audit = tool_calls(catalog)
    assert [(a["run_id"], a["tool"], a["input_json"], a["is_error"]) for a in audit] == \
        [(result.run_id, "echo", '{"x": 1}', 0), (result.run_id, "echo", '{"x": 2}', 0)]
    assert audit[0]["output_preview"] == '{"echo": {"x": 1}}'


def test_requests_carry_effort_fallbacks_and_caching_but_no_thinking_or_tool_choice(catalog):
    client = FakeClient([response(tool_use("t1", "echo", {"x": 1}), stop="tool_use"), response(text("ok"))])
    AgentRunner(catalog, client, effort="high").run("weekly", SYSTEM, "go", EchoTools())
    for req in client.requests:
        assert req["model"] == "claude-opus-5-5" and req["max_tokens"] == MAX_TOKENS == 16000
        assert req["system"] == [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
        assert req["tools"] == EchoTools.specs
        assert req["output_config"] == {"effort": "high"}
        assert req["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
        assert req["fallbacks"] == "default" and req["cache_control"] == {"type": "ephemeral"}
        assert not {"thinking", "tool_choice", "budget_tokens", "temperature", "extra_body"} & keys_anywhere(
            {k: v for k, v in req.items() if k != "messages"})
    a, b = client.requests
    assert json.dumps([a["system"], a["tools"]]) == json.dumps([b["system"], b["tools"]])  # cacheable prefix
    plain = FakeClient([response(text("ok"))])
    AgentRunner(catalog, plain, fallbacks=False, auto_cache=False).run("nightly", SYSTEM, "go", EchoTools())
    assert not {"betas", "fallbacks", "cache_control"} & set(plain.requests[0])
    with pytest.raises(ValueError):
        AgentRunner(catalog, plain, effort="extreme")


def test_the_real_sdk_accepts_the_requests_and_replays_thinking_blocks(catalog):
    """End to end through anthropic's client with a mock HTTP transport (no network): the SDK accepts every keyword,
    sends the fallback beta header, and real response blocks (thinking included) go back verbatim."""
    bodies, headers = [], []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        headers.append(request.headers)
        if len(bodies) == 1:
            content = [{"type": "thinking", "thinking": "", "signature": "sig-1"},
                       {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"x": 7}}]
            stop, usage = "tool_use", {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": None,
                                       "cache_creation_input_tokens": 50}
        else:
            content, stop, usage = [{"type": "text", "text": "done"}], "end_turn", {"input_tokens": 10,
                                                                                    "output_tokens": 5}
        return httpx2.Response(200, json={"id": f"msg_{len(bodies)}", "type": "message", "role": "assistant",
                                          "model": "claude-opus-5-5", "content": content, "stop_reason": stop,
                                          "stop_sequence": None, "usage": usage})

    client = anthropic.Anthropic(api_key="test-key", base_url="https://api.anthropic.com", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == "completed" and result.final_text == "done"
    assert (result.usage.input_tokens, result.usage.cache_write_tokens, result.usage.cache_read_tokens) == (110, 50, 0)
    assert all(FALLBACK_BETA in h.get("anthropic-beta", "") for h in headers)
    first, second = bodies
    assert first["fallbacks"] == "default" and first["output_config"] == {"effort": "medium"}
    assert first["tools"][0]["strict"] is True and "thinking" not in first and "tool_choice" not in first
    assert second["messages"][1] == {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"x": 7}}]}
    assert second["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"echo": {"x": 7}}', "is_error": False}]}
    assert second["system"] == first["system"] and second["tools"] == first["tools"]


def test_pause_turn_resends_the_paused_content(catalog):
    paused = response(text("partial"), stop="pause_turn")
    client = FakeClient([paused, response(text("finished"))])
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == "completed" and result.n_api_calls == 2
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"] and second[1]["content"] == paused.content


def test_refusal_stops_without_retry_and_records_details(catalog):
    details = SimpleNamespace(type="refusal", category="cyber", explanation="looks like exploit work",
                              fallback_credit_token="secret-token", recommended_model=None)
    client = FakeClient([response(text("I can't"), stop="refusal", stop_details=details), response(text("never"))])
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == "refused" and len(client.requests) == 1
    assert result.stop_details == {"type": "refusal", "category": "cyber", "explanation": "looks like exploit work",
                                   "recommended_model": None}
    (row,) = runs(catalog)
    assert row["status"] == "refused" and row["stop_reason"] == "refusal"
    assert "cyber" in row["error"] and "secret-token" not in row["error"]


@pytest.mark.parametrize("stop, status", [("max_tokens", "max_tokens"), ("model_context_window_exceeded", "stopped")])
def test_max_tokens_and_unexpected_stops_end_the_run(catalog, stop, status):
    client = FakeClient([response(text("cut"), stop=stop), response(text("never"))])
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == status and len(client.requests) == 1 and result.error
    assert runs(catalog)[0]["status"] == status and runs(catalog)[0]["stop_reason"] == stop


def test_iteration_cap_executes_the_last_tool_calls(catalog):
    script = [response(tool_use(f"t{i}", "echo", {"x": i}), stop="tool_use") for i in range(4)]
    client = FakeClient(script)
    result = AgentRunner(catalog, client, max_iterations=3).run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == "iteration_cap" and len(client.requests) == 3 and result.n_tool_calls == 3
    assert "max_iterations" in result.error and runs(catalog)[0]["status"] == "iteration_cap"
    assert len(tool_calls(catalog)) == 3


def test_cost_cap_stops_after_the_response_that_reaches_it(catalog):
    expensive = response(tool_use("t1", "echo", {"x": 1}), stop="tool_use", usage=(10_000, 0, 0, 0))  # $0.04
    client = FakeClient([expensive, response(text("never"))])
    tools = EchoTools()
    result = AgentRunner(catalog, client, max_cost_usd=0.03).run("nightly", SYSTEM, "go", tools)
    assert result.status == "cost_cap" and len(client.requests) == 1
    assert tools.calls == [("echo", {"x": 1})] and result.est_cost_usd == pytest.approx(0.04)
    (row,) = runs(catalog)
    assert row["status"] == "cost_cap" and row["est_cost_usd"] == pytest.approx(0.04) and "cap" in row["error"]


def test_wrap_up_notice_is_appended_once_near_the_caps(catalog):
    script = [response(tool_use(f"t{i}", "echo", {"x": i}), stop="tool_use") for i in range(3)]
    client = FakeClient(script + [response(text("bye"))])
    result = AgentRunner(catalog, client, max_iterations=4, wrap_up_calls=2).run("nightly", SYSTEM, "go",
                                                                                 EchoTools())
    assert result.status == "completed" and len(client.requests) == 4

    def last_user_blocks(request):
        return [b["type"] for b in request["messages"][-1]["content"]]

    assert last_user_blocks(client.requests[1]) == ["tool_result"]  # 3 calls left
    assert last_user_blocks(client.requests[2]) == ["tool_result", "text"]  # 2 calls left: the notice
    notice = client.requests[2]["messages"][-1]["content"][-1]["text"]
    assert notice.startswith("[runner]") and "2 API call(s) left" in notice and "Finish now" in notice
    assert last_user_blocks(client.requests[3]) == ["tool_result"]  # only once
    assert client.requests[3]["messages"][:-2] == client.requests[2]["messages"]  # history is append-only
    pricey = FakeClient([response(tool_use("t", "echo", {"x": 1}), stop="tool_use", usage=(500_000, 0, 0, 0)),
                         response(text("done"))])  # $2.00 = 80 % of the cap
    assert AgentRunner(catalog, pricey, max_cost_usd=2.5).run("nightly", SYSTEM, "go", EchoTools()).status == \
        "completed"
    assert last_user_blocks(pricey.requests[1]) == ["tool_result", "text"]


def test_tool_exceptions_become_error_results_and_the_run_continues(catalog):
    client = FakeClient([response(tool_use("t1", "boom", {}), tool_use("t2", "echo", {"x": 5}), stop="tool_use"),
                         response(text("recovered"))])
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == "completed"
    boom, echo = client.requests[1]["messages"][2]["content"]
    assert boom["tool_use_id"] == "t1" and boom["is_error"] is True
    assert "internal error in tool boom: RuntimeError: kaput" in json.loads(boom["content"])["error"]
    assert echo["is_error"] is False
    assert [a["is_error"] for a in tool_calls(catalog)] == [1, 0]


def test_usage_accounting_tolerates_missing_fields(catalog):
    client = FakeClient([response(text("a"), stop="pause_turn", usage=None),
                         response(text("b"), usage=(7, 3, None, None))])
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    u = result.usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (7, 3, 0, 0)
    usage = Usage()
    usage.add({"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": 3})
    assert Pricing().cost(usage) == pytest.approx((1 * 4 + 2 * 20 + 3 * 0.2) / 1e6)


def test_fallback_served_responses_are_reported(catalog):
    client = FakeClient([response(text("served elsewhere"), model="claude-opus-5")])
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == "completed" and result.served_by == ["claude-opus-5"]


def test_tool_use_stop_without_blocks_is_an_error(catalog):
    result = AgentRunner(catalog, FakeClient([response(text("?"), stop="tool_use")])).run("nightly", SYSTEM, "go",
                                                                                         EchoTools())
    assert result.status == "error" and "without any tool_use block" in result.error


@pytest.mark.parametrize("exc, needle", [
    (api_error(anthropic.RateLimitError, 429), "rate limited (429)"),
    (api_error(anthropic.InternalServerError, 500), "server error (500)"),
    (api_error(anthropic.APIStatusError, 529, "overloaded"), "server error (529)"),
    (api_error(anthropic.BadRequestError, 400, "bad field"), "rejected (400): bad field"),
    (api_error(anthropic.APIConnectionError, 0), "cannot reach the Claude API"),
])
def test_api_failures_end_the_run_with_status_error(catalog, exc, needle):
    client = FakeClient([response(tool_use("t1", "echo", {"x": 1}), stop="tool_use"), exc])
    result = AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == "error" and needle in result.error and result.n_api_calls == 1
    (row,) = runs(catalog)
    assert row["status"] == "error" and needle in row["error"] and row["n_tool_calls"] == 1


@pytest.mark.parametrize("cls, status", [(anthropic.AuthenticationError, 401), (anthropic.PermissionDeniedError, 403)])
def test_auth_failures_raise_agent_unavailable(catalog, cls, status):
    client = FakeClient([api_error(cls, status, "invalid x-api-key")])
    with pytest.raises(AgentUnavailable, match=str(status)):
        AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    (row,) = runs(catalog)
    assert row["status"] == "unavailable" and str(status) in row["error"] and row["finished_at"] is not None


def test_other_exceptions_are_recorded_and_raised(catalog):
    client = FakeClient([response(tool_use("t1", "echo", {"x": 1}), stop="tool_use"), KeyError("bug")])
    with pytest.raises(KeyError):
        AgentRunner(catalog, client).run("nightly", SYSTEM, "go", EchoTools())
    (row,) = runs(catalog)
    assert row["status"] == "error" and "KeyError" in row["error"] and row["n_tool_calls"] == 1


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    """An environment in which the SDK finds no credentials (and no profile on disk)."""
    import os
    for key in list(os.environ):
        if key.startswith("ANTHROPIC_"):
            monkeypatch.delenv(key)
    empty = tmp_path / "empty-home"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(empty / ".config"))
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(empty / "anthropic"))


def test_missing_credentials_raise_agent_unavailable_before_any_request(catalog, no_credentials):
    with pytest.raises(AgentUnavailable, match="ANTHROPIC_API_KEY"):
        AgentRunner(catalog).run("nightly", SYSTEM, "go", EchoTools())
    (row,) = runs(catalog)
    assert row["status"] == "unavailable" and row["input_tokens"] == 0


def test_dry_run_writes_the_first_request_without_a_client(catalog, no_credentials):
    runner = AgentRunner(catalog, dry_run=True)
    result = runner.run("nightly", SYSTEM, "go", EchoTools())
    assert result.status == "dry_run" and result.run_id is None and runs(catalog) == []
    path = catalog.root / "reports" / "agent" / "dryrun-nightly.json"
    assert result.dry_run_path == path
    request = json.loads(path.read_text())["request"]
    assert request == runner.build_request(SYSTEM, EchoTools.specs, [{"role": "user", "content": "go"}])
    assert runner._client is None


def test_constructor_validation(catalog):
    for kwargs in ({"max_iterations": 0}, {"max_cost_usd": 0.0}, {"max_tokens": 0}, {"model": ""},
                   {"wrap_up_calls": -1}, {"wrap_up_cost_frac": 0.0}, {"wrap_up_cost_frac": 1.5}):
        with pytest.raises(ValueError):
            AgentRunner(catalog, **kwargs)
    with pytest.raises(ValueError):
        AgentRunner(catalog, FakeClient([])).run("../evil", SYSTEM, "go", EchoTools())


# ---------------------------------------------------------------------------------------------- prompts
def test_system_prompts_are_stable_and_language_configurable():
    for kind, base in (("nightly", NIGHTLY_SYSTEM), ("weekly", WEEKLY_SYSTEM)):
        english = system_prompt(kind)
        assert english == system_prompt(kind, "en") and english.startswith(base)
        for lang, sentence in LANGUAGE_INSTRUCTIONS.items():
            prompt = system_prompt(kind, lang)
            assert prompt == system_prompt(kind, lang)
            assert prompt.startswith(base) and prompt.rstrip().endswith(sentence)
        assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}", base)  # no dates or times: cacheable
        for phrase in ("noise floor", "positive controls", "transients", "3 seeds", "replay", "Kepler",
                       "falsif"):
            assert phrase in base, (kind, phrase)
    assert "write_survey_log exactly once" in NIGHTLY_SYSTEM and "write_report" not in NIGHTLY_SYSTEM
    assert "write_report exactly once" in WEEKLY_SYSTEM and "fit_power_law" in WEEKLY_SYSTEM
    for duty in ("Quality control", "Classification review", "unknown queue", "Proposals", "hypothesis",
                 "boundar", "replication", "stale workers", "disk"):
        assert duty in NIGHTLY_SYSTEM, duty
    assert "中文" in system_prompt("nightly", "zh") and "日本語" in system_prompt("weekly", "ja")
    with pytest.raises(ValueError):
        system_prompt("nightly", "fr")
    with pytest.raises(ValueError):
        system_prompt("hourly")
    assert "write_survey_log" in task_text("nightly") and "write_report" in task_text("weekly")


# ---------------------------------------------------------------------------------------------- nightly / weekly
@pytest.fixture
def root(tmp_path):
    root = tmp_path / "survey"
    (root / "configs").mkdir(parents=True)
    shutil.copy(SKY_V0, root / "configs" / "sky_v0.yaml")
    return root


def write_config(root: Path, **agent) -> None:
    (root / "tycho.yaml").write_text(yaml.safe_dump({"agent": agent}))


def test_run_nightly_dry_run_needs_no_credentials(root, no_credentials):
    write_config(root, language="zh")
    path = run_nightly(root, dry_run=True)
    assert path == Paths(root).reports / "agent" / "dryrun-nightly.json"
    request = json.loads(path.read_text())["request"]
    assert request["system"][0]["text"] == system_prompt("nightly", "zh")
    assert [t["name"] for t in request["tools"]][-1] == "write_survey_log"
    assert "write_report" not in [t["name"] for t in request["tools"]]
    assert request["output_config"] == {"effort": "medium"} and request["fallbacks"] == "default"
    assert "thinking" not in request and "tool_choice" not in request
    content = request["messages"][0]["content"]
    assert content.startswith(task_text("nightly"))
    packet = json.loads(content.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
    assert packet["run"] == "nightly" and packet["window"]["hours"] == 24.0
    assert packet["budget"] == {"max_api_calls": 40, "max_cost_usd": 3.0}
    assert packet["overview"]["sky_version"] == "sky-v0" and "qc_flags" in packet["overview"]
    with Catalog(Paths(root).db) as cat:
        assert runs(cat) == []
    weekly = json.loads(run_weekly(root, dry_run=True).read_text())["request"]
    assert weekly["output_config"] == {"effort": "high"} and weekly["tools"][-1]["name"] == "write_report"


def test_disabled_agent_refuses_real_runs_but_allows_dry_runs(root):
    write_config(root, enabled=False)
    with pytest.raises(AgentUnavailable, match="disabled"):
        run_nightly(root, client=FakeClient([]))
    assert run_nightly(root, dry_run=True).exists()


def test_run_nightly_writes_the_agents_log_and_a_markdown_copy(root):
    log = {"title": "Night 1", "markdown": "## QC\nAll quiet."}
    client = FakeClient([
        response(tool_use("t1", "survey_overview", {}), stop="tool_use"),
        response(thinking(), tool_use("t2", "write_survey_log", log), stop="tool_use"),
        response(text("Logged.")),
    ])
    path = run_nightly(root, client=client)
    with Catalog(Paths(root).db) as cat:
        assert path == Paths(root).reports / "nightly" / f"{local_date(cat.now())}.md"
        (entry,) = cat.query("SELECT * FROM survey_log")
        (run,) = runs(cat)
        audit = tool_calls(cat)
    assert (entry["author"], entry["kind"], entry["title"]) == ("agent", "nightly", "Night 1")
    assert run["status"] == "completed" and run["n_tool_calls"] == 2 and run["kind"] == "nightly"
    assert [a["tool"] for a in audit] == ["survey_overview", "write_survey_log"]
    overview = json.loads(client.requests[1]["messages"][2]["content"][0]["content"])
    assert overview["sky_version"] == "sky-v0" and "caps" in overview
    copy_text = path.read_text()
    assert copy_text.startswith("# Night 1\n\n## QC\nAll quiet.") and f"Agent run {run['run_id']}" in copy_text
    second = run_nightly(root, client=FakeClient([response(tool_use("t", "write_survey_log", log), stop="tool_use"),
                                                  response(text("ok"))]))
    assert second.name == f"{path.stem}-2.md"


def test_run_nightly_writes_a_fallback_entry_when_the_agent_does_not(root):
    path = run_nightly(root, client=FakeClient([response(text("I looked around and ran out of ideas."))]))
    with Catalog(Paths(root).db) as cat:
        (entry,) = cat.query("SELECT * FROM survey_log")
    assert entry["author"] == RUNNER_AUTHOR and entry["kind"] == "nightly"
    assert "without writing its survey log" in entry["markdown"]
    assert "I looked around and ran out of ideas." in entry["markdown"]
    assert "I looked around" in path.read_text()


def test_run_weekly_returns_the_report_with_a_run_footer(root):
    report = {"title": "Week in review", "markdown": "## Summary\nNothing yet."}
    client = FakeClient([response(tool_use("t1", "write_report", report), stop="tool_use"), response(text("done"))])
    path = run_weekly(root, client=client)
    with Catalog(Paths(root).db) as cat:
        assert path == Paths(root).reports / "weekly" / f"{iso_week(cat.now())}.md"
        (entry,) = cat.query("SELECT * FROM survey_log")
        (run,) = runs(cat)
    assert (entry["author"], entry["kind"], entry["title"]) == ("agent", "weekly", "Week in review")
    body = path.read_text()
    assert body.startswith("# Week in review\n\n## Summary\nNothing yet.\n") and f"Agent run {run['run_id']}" in body
    assert client.requests[0]["output_config"] == {"effort": "high"}
    fallback = run_weekly(root, client=FakeClient([response(text("no report today"))]))
    assert fallback.name == f"{iso_week(run['started_at'])}-2.md" and "no report today" in fallback.read_text()


def test_observation_window_starts_at_the_previous_completed_run(tmp_path):
    clock = SimpleNamespace(t=1_790_000_000.0)
    with Catalog(tmp_path / "data" / "catalog.sqlite", clock=lambda: clock.t) as cat:
        window = observation_window(cat, "nightly")
        assert window.start == clock.t - DAY and "no previous completed" in window.basis
        done = cat.start_agent_run("nightly", "m")
        cat.finish_agent_run(done, status="completed")
        clock.t += 3600.0
        failed = cat.start_agent_run("nightly", "m")
        cat.finish_agent_run(failed, status="error")
        clock.t += 20 * 3600.0
        assert observation_window(cat, "nightly").start == clock.t - 21 * 3600.0
        assert observation_window(cat, "weekly").start == clock.t - 7 * DAY
        clock.t += 30 * DAY
        capped = observation_window(cat, "nightly")
        assert capped.start == clock.t - 7 * DAY and "maximum look-back" in capped.basis
