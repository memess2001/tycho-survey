"""The survey agent's conversation loop (SPEC §7): one Claude conversation over the survey tools, with hard caps,
usage accounting and a complete audit trail in the catalog.

Why a manual loop rather than the SDK's tool runner
    The loop enforces caps that are checked after every response (``max_iterations`` API calls, an estimated cost of
    ``max_cost_usd``), records token usage and cost in ``agent_runs`` after every response (so a run killed midway still
    shows what it spent), writes an ``agent_tool_calls`` row for every tool call, and runs deterministically against a
    scripted fake client in the tests.

Requests (Claude Opus 5.5 through ``client.beta.messages.create``)
    ``model`` (default ``claude-opus-5-5``), ``max_tokens`` 16000 (non-streaming), ``system`` as one text block with an
    ephemeral ``cache_control`` breakpoint, ``tools`` (strict schemas; the list never changes within a run),
    ``output_config={"effort": ...}`` ("medium" nightly, "high" weekly; this model's default effort is medium, so it is
    always sent explicitly), a top-level ``cache_control`` (automatic caching of the growing conversation, the second
    breakpoint), and, unless disabled, server-side refusal fallbacks: ``betas=["server-side-fallback-2026-07-01"]`` with
    ``fallbacks="default"``. The model always thinks, so no ``thinking`` parameter is sent (``disabled`` or a token
    budget would be rejected); forced ``tool_choice`` is rejected by this model, so tool choice stays automatic and the
    system prompt tells the model which tools to call.

    The system prompt and the tool list are byte-identical across the requests of a run, and the history is append
    only: every response's full ``content`` (thinking blocks included) is appended unchanged, and the tool results of
    one assistant turn go back in a single user message. This keeps the prompt cache valid and satisfies the API's
    check that replayed thinking blocks belong to an unedited conversation.

Stop reasons
    ``tool_use``: every tool_use block is executed (``ToolBox.dispatch``; an exception becomes an error result) and all
    results are returned in one user message. When the run nears a cap (at most ``wrap_up_calls`` API calls left, or
    an estimated cost of ``wrap_up_cost_frac`` of the cap), a single notice asking the model to finish is appended to
    that message as a text block after the results (append-only, as above). ``pause_turn``: the paused response is
    resent as is. ``end_turn``: done. ``refusal``: the run stops without retry (the server-side fallback, when enabled,
    has already been tried) and the stop details are recorded. ``max_tokens``: the run stops. Any other stop reason
    stops the run.

Run statuses (``agent_runs.status``)
    running (while in progress), completed, refused, max_tokens, iteration_cap, cost_cap, stopped (an unexpected stop
    reason), error (an API failure after the SDK's own retries, or an exception) and unavailable (no usable credentials;
    :class:`AgentUnavailable` is raised). A cap is checked after the tool calls of a response have been executed, so
    work the model has already paid for (for example the final ``write_survey_log``) is not lost; a run can therefore
    exceed ``max_cost_usd`` by at most the cost of one response.

Cost
    est. cost = (input · in + output · out + cache reads · read + cache writes · write) / 10⁶ with the per-million-token
    prices of :class:`Pricing` (Claude Opus 5.5: $4 / $20 / $0.20 / $5). Missing or null usage fields count as 0.
    Responses served by a fallback model are priced at the configured model's rates; their model names are reported in
    ``AgentRunResult.served_by``.

Errors
    The SDK retries 408/409/429/5xx and connection errors itself (:data:`SDK_MAX_RETRIES`); what still fails is caught
    as typed SDK exceptions, most specific first: authentication (401) and permission (403) errors raise
    :class:`AgentUnavailable`; rate limits (429), server errors (≥ 500), other API errors and connection errors end the
    run with status "error" and are recorded. A client is created lazily with ``anthropic.Anthropic()``, which resolves
    ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login`` profile; without any of them the run raises
    :class:`AgentUnavailable` before making a request.

Sessions
    :func:`run_session` is the driver shared by :func:`tycho.agent.nightly.run_nightly` and
    :func:`tycho.agent.weekly.run_weekly`: config, sky, catalog, observation window, tools, runner, then the
    kind-specific context packet and finalisation.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from tycho.agent.prompts import system_prompt
from tycho.agent.tools import SurveyTools, ToolLimits, iso_utc, write_text_atomic
from tycho.config import Paths, load_config
from tycho.survey.catalog import Catalog
from tycho.survey.sky import Sky

try:
    import anthropic
except ImportError:  # pragma: no cover - the SDK is the optional "agent" extra; dry runs work without it
    anthropic = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16_000
"""Per-response output cap (thinking included); suits non-streaming requests."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"
"""Beta flag of the ``fallbacks="default"`` form of server-side refusal fallbacks."""

EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

SDK_MAX_RETRIES = 4
"""Retries of the SDK's own backoff for 408/409/429/5xx and connection errors (its default is 2; an unattended run
can afford to wait out a short overload)."""

WRAP_UP_CALLS = 3
WRAP_UP_COST_FRAC = 0.8
"""Defaults of when the runner asks the model to finish: at most this many API calls left, or this fraction of the
cost cap spent."""

RUNNER_AUTHOR = "agent-runner"
"""``survey_log.author`` of entries the driver writes when the model did not write its log or report."""

CREDENTIALS_HELP = ("Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) in the environment of the job that runs the agent "
                    "(for launchd: the plist's EnvironmentVariables), or run `ant auth login` as that user.")

WINDOWS: dict[str, tuple[float, float]] = {"nightly": (86_400.0, 7 * 86_400.0), "weekly": (7 * 86_400.0, 31 * 86_400.0)}
"""Observation window per run kind: (default length without a previous completed run, longest look-back)."""

_KIND = re.compile(r"^[a-z][a-z0-9_-]*$")


class AgentUnavailable(RuntimeError):
    """The agent cannot run: no usable Claude API credentials, the SDK is missing, or the agent is disabled."""


class _ApiFailure(Exception):
    """An API call failed after the SDK's retries (the run ends with status "error")."""


# ---------------------------------------------------------------------------------------------- usage and cost
@dataclass(frozen=True)
class Pricing:
    """Prices in USD per million tokens."""

    input: float = 4.0
    output: float = 20.0
    cache_read: float = 0.20
    cache_write: float = 5.0

    def cost(self, usage: "Usage") -> float:
        return (usage.input_tokens * self.input + usage.output_tokens * self.output
                + usage.cache_read_tokens * self.cache_read + usage.cache_write_tokens * self.cache_write) / 1e6


OPUS_5_5_PRICING = Pricing()
"""Claude Opus 5.5: $4 input, $20 output, $0.20 cache read, $5 cache write (5-minute TTL) per million tokens."""


@dataclass
class Usage:
    """Token totals of a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage: Any) -> None:
        """Add one response's ``usage`` (an SDK object or mapping; None and missing/null fields count as 0)."""
        if usage is None:
            return
        self.input_tokens += _count(usage, "input_tokens")
        self.output_tokens += _count(usage, "output_tokens")
        self.cache_read_tokens += _count(usage, "cache_read_input_tokens")
        self.cache_write_tokens += _count(usage, "cache_creation_input_tokens")


def _count(usage: Any, name: str) -> int:
    value = _field(usage, name)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _field(obj: Any, name: str) -> Any:
    """Attribute ``name`` of an SDK object, or key ``name`` of a mapping (None if absent)."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


# ---------------------------------------------------------------------------------------------- results
@dataclass
class AgentRunResult:
    """Outcome of one agent run.

    Attributes:
        kind, model: the run kind and the requested model.
        status: see the module docstring ("dry_run" for a dry run).
        run_id: the ``agent_runs`` row (None for a dry run).
        stop_reason: stop reason of the last response; stop_details: the refusal details (category, explanation).
        final_text: the text of the most recent response that contained text.
        n_api_calls, n_tool_calls, usage, est_cost_usd: what the run consumed.
        error: why the run ended early (None when completed).
        served_by: models other than the requested one that served responses (server-side fallback).
        messages: the conversation (user and assistant turns, as sent).
        dry_run_path: the file a dry run wrote.
    """

    kind: str
    model: str
    status: str
    run_id: int | None = None
    stop_reason: str | None = None
    stop_details: dict[str, Any] | None = None
    final_text: str = ""
    n_api_calls: int = 0
    n_tool_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    est_cost_usd: float = 0.0
    error: str | None = None
    served_by: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    dry_run_path: Path | None = None


class ToolBox(Protocol):
    """What the runner needs from a tool set (implemented by :class:`tycho.agent.tools.SurveyTools`)."""

    @property
    def specs(self) -> list[dict[str, Any]]: ...

    def dispatch(self, name: str, input: Mapping[str, Any]) -> tuple[str, bool]: ...  # noqa: A002


# ---------------------------------------------------------------------------------------------- the runner
class AgentRunner:
    """Runs one agent conversation with caps and auditing (see the module docstring).

    Args:
        catalog: where runs (``agent_runs``) and tool calls (``agent_tool_calls``) are recorded.
        client: an ``anthropic.Anthropic``-like client (default: created lazily on the first real run).
        model: model id.
        effort: ``output_config.effort`` (one of :data:`EFFORT_LEVELS`).
        max_iterations: maximum number of API calls per run.
        max_cost_usd: estimated-cost cap per run.
        fallbacks: enable server-side refusal fallbacks (``fallbacks="default"`` with its beta flag).
        max_tokens: per-response output cap.
        pricing: prices for the cost estimate.
        auto_cache: send the top-level ``cache_control`` (automatic caching of the conversation).
        wrap_up_calls, wrap_up_cost_frac: when to append the wrap-up notice (see the module docstring).
        dry_run: write the would-be first request to ``<dry_run_dir>/dryrun-<kind>.json`` instead of calling the API.
        dry_run_dir: default ``<survey root>/reports/agent``.
    """

    def __init__(self, catalog: Catalog, client: Any = None, model: str = DEFAULT_MODEL, effort: str = "medium",
                 max_iterations: int = 40, max_cost_usd: float = 3.0, fallbacks: bool = True, *,
                 max_tokens: int = MAX_TOKENS, pricing: Pricing = OPUS_5_5_PRICING, auto_cache: bool = True,
                 wrap_up_calls: int = WRAP_UP_CALLS, wrap_up_cost_frac: float = WRAP_UP_COST_FRAC,
                 dry_run: bool = False, dry_run_dir: Path | str | None = None) -> None:
        if effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {effort!r}")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError(f"max_iterations must be a positive integer, got {max_iterations!r}")
        if not max_cost_usd > 0:
            raise ValueError(f"max_cost_usd must be positive, got {max_cost_usd!r}")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}")
        if not model:
            raise ValueError("model must be a non-empty model id")
        if isinstance(wrap_up_calls, bool) or not isinstance(wrap_up_calls, int) or wrap_up_calls < 0:
            raise ValueError(f"wrap_up_calls must be a non-negative integer, got {wrap_up_calls!r}")
        if not 0.0 < wrap_up_cost_frac <= 1.0:
            raise ValueError(f"wrap_up_cost_frac must be in (0, 1], got {wrap_up_cost_frac!r}")
        self.catalog = catalog
        self.model = model
        self.effort = effort
        self.max_iterations = max_iterations
        self.max_cost_usd = float(max_cost_usd)
        self.fallbacks = bool(fallbacks)
        self.max_tokens = max_tokens
        self.pricing = pricing
        self.auto_cache = bool(auto_cache)
        self.wrap_up_calls = wrap_up_calls
        self.wrap_up_cost_frac = float(wrap_up_cost_frac)
        self.dry_run = bool(dry_run)
        self.dry_run_dir = Path(dry_run_dir) if dry_run_dir is not None else Paths(catalog.root).reports / "agent"
        self._client = client

    @classmethod
    def from_config(cls, catalog: Catalog, cfg: Mapping[str, Any], kind: str, *, client: Any = None,
                    dry_run: bool = False) -> "AgentRunner":
        """A runner configured by the ``agent`` section of a survey config (effort ``<kind>_effort``)."""
        agent = cfg.get("agent") or {}
        return cls(catalog, client, model=str(agent.get("model", DEFAULT_MODEL)),
                   effort=str(agent.get(f"{kind}_effort", "medium")),
                   max_iterations=int(agent.get("max_iterations", 40)),
                   max_cost_usd=float(agent.get("max_cost_usd", 3.0)),
                   fallbacks=bool(agent.get("fallbacks", True)), dry_run=dry_run)

    def build_request(self, system: str, tool_specs: Sequence[Mapping[str, Any]],
                      messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The keyword arguments of one ``client.beta.messages.create`` call."""
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": [dict(t) for t in tool_specs],
            "messages": list(messages),
            "output_config": {"effort": self.effort},
        }
        if self.auto_cache:
            request["cache_control"] = {"type": "ephemeral"}
        if self.fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def run(self, kind: str, system: str, user_content: str | list[dict[str, Any]], tools: ToolBox) -> AgentRunResult:
        """Run one conversation: ``system`` prompt, the first user message ``user_content``, the tools of ``tools``.

        Returns the result (also recorded in ``agent_runs``). Raises :class:`AgentUnavailable` when no usable
        credentials exist; any other exception is recorded as status "error" and re-raised.
        """
        if not _KIND.match(kind):
            raise ValueError(f"invalid run kind {kind!r}")
        specs = list(tools.specs)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
        if self.dry_run:
            return self._dry_run(kind, system, specs, messages)
        result = AgentRunResult(kind=kind, model=self.model, status="running", messages=messages)
        result.run_id = self.catalog.start_agent_run(kind, self.model)
        try:
            client = self._get_client()
            self._loop(client, result, system, specs, tools)
        except AgentUnavailable as exc:
            result.status, result.error = "unavailable", str(exc)
            self._record(result, final=True)
            raise
        except BaseException as exc:
            result.status, result.error = "error", f"{type(exc).__name__}: {exc}"
            self._record(result, final=True)
            raise
        self._record(result, final=True)
        logger.info("agent run %s (%s) %s after %d API calls and %d tool calls; est. cost $%.3f", result.run_id, kind,
                    result.status, result.n_api_calls, result.n_tool_calls, result.est_cost_usd)
        return result

    # ------------------------------------------------------------------ the loop
    def _loop(self, client: Any, result: AgentRunResult, system: str, specs: list[dict[str, Any]],
              tools: ToolBox) -> None:
        messages = result.messages
        warned = False
        for _ in range(self.max_iterations):
            try:
                response = self._call(client, self.build_request(system, specs, messages))
            except _ApiFailure as exc:
                result.status, result.error = "error", str(exc)
                logger.error("agent run %s: %s", result.run_id, exc)
                return
            result.n_api_calls += 1
            result.usage.add(_field(response, "usage"))
            result.est_cost_usd = self.pricing.cost(result.usage)
            served = _field(response, "model")
            if served and served != self.model and served not in result.served_by:
                result.served_by.append(served)
                logger.warning("agent run %s: a response was served by %s instead of %s (server-side fallback)",
                               result.run_id, served, self.model)
            content = list(_field(response, "content") or [])
            messages.append({"role": "assistant", "content": content})
            stop = _field(response, "stop_reason")
            result.stop_reason = stop
            text = _text_of(content)
            if text:
                result.final_text = text
            self._record(result, final=False)
            if stop == "end_turn":
                result.status = "completed"
                return
            if stop == "refusal":
                result.status = "refused"
                result.stop_details = _stop_details(response)
                details = result.stop_details or {}
                result.error = (f"the model declined (category {details.get('category')}): "
                                f"{details.get('explanation') or 'no explanation given'}")
                logger.warning("agent run %s: %s", result.run_id, result.error)
                return
            if stop == "max_tokens":
                result.status = "max_tokens"
                result.error = f"a response reached max_tokens ({self.max_tokens}) and was cut off"
                return
            if stop == "tool_use":
                tool_results: list[dict[str, Any]] = self._execute_tools(result, content, tools)
                if not tool_results:
                    result.status, result.error = "error", "stop_reason tool_use without any tool_use block"
                    return
                if not warned and self._near_cap(result):
                    warned = True
                    tool_results.append({"type": "text", "text": self._wrap_up_notice(result)})
                messages.append({"role": "user", "content": tool_results})
            elif stop != "pause_turn":
                result.status, result.error = "stopped", f"unexpected stop_reason {stop!r}"
                return
            if result.est_cost_usd >= self.max_cost_usd:
                result.status = "cost_cap"
                result.error = (f"estimated cost ${result.est_cost_usd:.2f} reached the cap of "
                                f"${self.max_cost_usd:.2f}")
                logger.warning("agent run %s: %s", result.run_id, result.error)
                return
        result.status = "iteration_cap"
        result.error = f"stopped after {self.max_iterations} API calls (max_iterations) before the model finished"
        logger.warning("agent run %s: %s", result.run_id, result.error)

    def _near_cap(self, result: AgentRunResult) -> bool:
        calls_left = self.max_iterations - result.n_api_calls
        return 0 < calls_left <= self.wrap_up_calls or result.est_cost_usd >= self.wrap_up_cost_frac * self.max_cost_usd

    def _wrap_up_notice(self, result: AgentRunResult) -> str:
        return (f"[runner] This run is close to its limits: {self.max_iterations - result.n_api_calls} API call(s) "
                f"left and an estimated cost of ${result.est_cost_usd:.2f} of ${self.max_cost_usd:.2f}. Finish now: "
                f"make your final call (the survey log or the report) in your next response.")

    def _call(self, client: Any, request: dict[str, Any]) -> Any:
        """One API call; failures that survive the SDK's retries are mapped most-specific first."""
        try:
            return client.beta.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise AgentUnavailable(f"the Claude API rejected the credentials (401: {exc.message}). "
                                   f"{CREDENTIALS_HELP}") from exc
        except anthropic.PermissionDeniedError as exc:
            raise AgentUnavailable(f"the Claude API denied access (403: {exc.message}); check that these credentials "
                                   f"may use model {self.model}.") from exc
        except anthropic.RateLimitError as exc:
            raise _ApiFailure(f"rate limited (429) after the SDK's retries; try again later ({exc.message})") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _ApiFailure(f"Claude API server error ({exc.status_code}) after the SDK's retries; try again "
                                  f"later ({exc.message})") from exc
            raise _ApiFailure(f"Claude API request rejected ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise _ApiFailure(f"cannot reach the Claude API (network error or timeout after the SDK's retries): "
                              f"{exc}") from exc

    def _execute_tools(self, result: AgentRunResult, content: Sequence[Any], tools: ToolBox) -> list[dict[str, Any]]:
        """Run every tool_use block of one response; returns their tool_result blocks (in order) and audits each."""
        out = []
        for block in content:
            if _field(block, "type") != "tool_use":
                continue
            name, tool_input = _field(block, "name"), _field(block, "input")
            try:
                text, is_error = tools.dispatch(name, tool_input)
            except Exception as exc:  # a broken tool must not end the run: the model gets an error result
                logger.exception("agent run %s: tool %s raised", result.run_id, name)
                text = json.dumps({"error": f"internal error in tool {name}: {type(exc).__name__}: {exc}"})
                is_error = True
            result.n_tool_calls += 1
            self.catalog.log_tool_call(result.run_id, str(name), _audit_json(tool_input), text, is_error)
            out.append({"type": "tool_result", "tool_use_id": _field(block, "id"), "content": text,
                        "is_error": bool(is_error)})
        return out

    def _record(self, result: AgentRunResult, *, final: bool) -> None:
        """Write the run's usage and state to ``agent_runs`` (``final=False``: still running, no finish time)."""
        fields: dict[str, Any] = {
            "input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens,
            "cache_read_tokens": result.usage.cache_read_tokens, "cache_write_tokens": result.usage.cache_write_tokens,
            "est_cost_usd": result.est_cost_usd, "n_tool_calls": result.n_tool_calls,
            "stop_reason": result.stop_reason, "status": result.status if final else "running",
        }
        if final:
            fields["error"] = result.error
        else:
            fields["finished_at"] = None
        self.catalog.finish_agent_run(result.run_id, **fields)

    def _dry_run(self, kind: str, system: str, specs: list[dict[str, Any]],
                 messages: list[dict[str, Any]]) -> AgentRunResult:
        request = self.build_request(system, specs, messages)
        path = self.dry_run_dir / f"dryrun-{kind}.json"
        payload = {"method": "client.beta.messages.create",
                   "note": "the first request a real run would send (dry run: nothing was sent)",
                   "request": request}
        write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        logger.info("dry run: wrote the first %s request to %s", kind, path)
        return AgentRunResult(kind=kind, model=self.model, status="dry_run", messages=messages, dry_run_path=path)

    def _get_client(self) -> Any:
        """The client (created on first use). The SDK is required even with an injected client: its exception
        classes classify API failures."""
        if anthropic is None:
            raise AgentUnavailable("the 'anthropic' package is not installed (pip install 'tycho-survey[agent]')")
        if self._client is None:
            try:
                client = anthropic.Anthropic(max_retries=SDK_MAX_RETRIES)
            except Exception as exc:  # any failure to resolve credentials or configuration makes the agent unavailable
                raise AgentUnavailable(f"cannot create the Claude API client ({type(exc).__name__}: {exc}). "
                                       f"{CREDENTIALS_HELP}") from exc
            if not (client.api_key or client.auth_token or getattr(client, "credentials", None)):
                raise AgentUnavailable(f"no Claude API credentials found. {CREDENTIALS_HELP}")
            self._client = client
        return self._client


def _text_of(content: Sequence[Any]) -> str:
    return "\n\n".join(str(_field(b, "text")) for b in content if _field(b, "type") == "text" and _field(b, "text"))


def _stop_details(response: Any) -> dict[str, Any] | None:
    """The refusal details worth keeping (type, category, explanation, recommended model); None if absent."""
    details = _field(response, "stop_details")
    if details is None:
        return None
    return {k: _field(details, k) for k in ("type", "category", "explanation", "recommended_model")}


def _audit_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------------- sessions
@dataclass(frozen=True)
class Window:
    """The observation window of a run: its start (epoch seconds) and how it was chosen."""

    start: float
    basis: str


def observation_window(catalog: Catalog, kind: str) -> Window:
    """Since the start of the previous completed run of ``kind`` (so nothing finished since then is missed, even
    when later runs failed), looking back at most the kind's maximum; the default length without such a run."""
    default_s, max_s = WINDOWS[kind]
    now = catalog.now()
    rows = catalog.query("SELECT started_at FROM agent_runs WHERE kind = ? AND status = 'completed' "
                         "ORDER BY started_at DESC LIMIT 1", (kind,))
    if not rows:
        return Window(now - default_s, f"no previous completed {kind} run: the last {default_s / 3600:g} h")
    start = float(rows[0]["started_at"])
    if start < now - max_s:
        return Window(now - max_s, f"the previous completed {kind} run is older than the maximum look-back: "
                                   f"the last {max_s / 3600:g} h")
    return Window(start, f"since the previous completed {kind} run started ({iso_utc(start)})")


def run_session(kind: str, root: Path | str, *, dry_run: bool, client: Any,
                context: Callable[[SurveyTools, Window, dict[str, Any]], str],
                finalize: Callable[[Catalog, SurveyTools, AgentRunResult], Path]) -> Path:
    """One agent run of ``kind`` on the survey at ``root``; returns the file written by ``finalize`` (or, for a dry
    run, the dry-run request file).

    ``context(tools, window, budget)`` builds the first user message (``budget``: the run's ``max_api_calls`` and
    ``max_cost_usd``); ``finalize(catalog, tools, result)`` makes sure the run left its log or report and writes the
    markdown copy. Raises :class:`AgentUnavailable` when the agent is disabled
    (``agent.enabled: false``; dry runs are still allowed) or has no usable credentials.
    """
    paths = Paths(root)
    cfg = load_config(paths.root)
    agent_cfg = cfg.get("agent") or {}
    if not dry_run and not agent_cfg.get("enabled", True):
        raise AgentUnavailable("the agent is disabled (agent.enabled: false in tycho.yaml)")
    system = system_prompt(kind, str(agent_cfg.get("language", "en")))
    sky = Sky.from_yaml(paths.root / cfg["survey"]["sky"])
    with Catalog(paths.db, root=paths.root) as catalog:
        window = observation_window(catalog, kind)
        tools = SurveyTools(catalog, sky, paths, ToolLimits.from_config(cfg), kind=kind, config=cfg,
                            window_start=window.start)
        runner = AgentRunner.from_config(catalog, cfg, kind, client=client, dry_run=dry_run)
        budget = {"max_api_calls": runner.max_iterations, "max_cost_usd": runner.max_cost_usd}
        result = runner.run(kind, system, context(tools, window, budget), tools)
        if result.dry_run_path is not None:
            return result.dry_run_path
        return finalize(catalog, tools, result)


def describe_run(result: AgentRunResult) -> str:
    """One-line record of a run: status, stop reason, calls, tokens and estimated cost."""
    u = result.usage
    text = (f"Agent run {result.run_id} ({result.kind}, {result.model}): status {result.status}, stop reason "
            f"{result.stop_reason}, {result.n_api_calls} API calls, {result.n_tool_calls} tool calls, "
            f"{u.input_tokens} input and {u.output_tokens} output tokens (cache read {u.cache_read_tokens}, cache "
            f"write {u.cache_write_tokens}), estimated cost ${result.est_cost_usd:.2f}.")
    if result.served_by:
        text += f" Some responses were served by {', '.join(result.served_by)} (server-side fallback)."
    return text


def missing_output_markdown(result: AgentRunResult, tools: SurveyTools, what: str) -> str:
    """Markdown recording that a run ended without writing its ``what`` (log or report), for the human."""
    lines = [f"The {result.kind} agent run ended without writing its {what}; this entry was written by the runner.",
             "", describe_run(result)]
    if result.error:
        lines += ["", f"Reason: {result.error}"]
    calls = ", ".join(f"{name} x{n}" for name, n in sorted(tools.call_counts.items())) or "none"
    lines += ["", f"Tool calls: {calls}.", "", "## Last text from the agent", "", result.final_text or "(none)"]
    return "\n".join(lines)
