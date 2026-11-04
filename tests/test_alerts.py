"""Tests of tycho.survey.alerts: the notification log, channels (urllib and osascript mocked), rate limit and digest,
persisted state, and that channel failures never raise."""
from __future__ import annotations

import json
import subprocess
import urllib.error

import pytest

import tycho.survey.alerts as alerts
from tycho.survey.alerts import Notifier, transient_notification


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def getcode(self) -> int:
        return self.status


class Posts(list):
    """Requests captured by the fake ``urlopen``; ``control.error`` / ``control.status`` script its answer."""

    control: object


@pytest.fixture
def posts(monkeypatch):
    """Record every urllib request instead of sending it."""
    calls = Posts()

    def fake_urlopen(request, timeout=None):
        calls.append({"url": request.full_url, "body": json.loads(request.data.decode("utf-8")),
                      "headers": dict(request.header_items()), "timeout": timeout})
        if fake_urlopen.error is not None:
            raise fake_urlopen.error
        return FakeResponse(fake_urlopen.status)

    fake_urlopen.error = None
    fake_urlopen.status = 200
    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)
    calls.control = fake_urlopen
    return calls


def records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


CHANNELS = {"ntfy": {"enabled": True, "server": "https://ntfy.example/", "topic": "tycho-secret-topic"},
            "webhook": {"enabled": True, "url": "https://hooks.example/T000"}, "macos": {"enabled": False}}


def test_without_channels_every_message_is_logged_only(tmp_path, posts):
    log = tmp_path / "logs" / "notifications.log"
    n = Notifier({"max_per_hour": 1}, log)
    assert n.channels == [] and n.min_severity == 2
    assert n.send("first", "a message", "high", ("tycho", "x")) is True
    assert n.send("second", "another", "urgent") is True  # no push channel: no rate limit, nothing held
    n.record("below threshold", "only logged")
    recs = records(log)
    assert [r["title"] for r in recs] == ["first", "second", "below threshold"]
    assert all(r["outcome"] == "logged" for r in recs) and recs[0]["priority"] == "high"
    assert recs[0]["tags"] == ["tycho", "x"] and recs[0]["time"].endswith("Z")
    assert posts == [] and n.held == 0


def test_ntfy_and_webhook_payloads(tmp_path, posts):
    n = Notifier({"max_per_hour": 10, "channels": CHANNELS}, tmp_path / "n.log")
    assert n.channels == ["ntfy", "webhook"]
    assert n.send("Tycho transient: pure synergy (severity 3)", "message body ✓", "high", ("tycho", "pure_synergy"))
    ntfy, hook = posts
    assert ntfy["url"] == "https://ntfy.example/"  # JSON publishing to the server root
    assert ntfy["body"] == {"topic": "tycho-secret-topic", "title": "Tycho transient: pure synergy (severity 3)",
                            "message": "message body ✓", "priority": 4, "tags": ["tycho", "pure_synergy"]}
    assert hook["url"] == "https://hooks.example/T000"
    assert hook["body"] == {"text": "*Tycho transient: pure synergy (severity 3)*\nmessage body ✓"}
    assert ntfy["timeout"] == alerts.HTTP_TIMEOUT_S
    assert ntfy["headers"]["Content-type"] == "application/json"
    rec = records(tmp_path / "n.log")[0]
    assert rec["outcome"] == "pushed" and rec["channels"] == {"ntfy": "ok", "webhook": "ok"}


def test_enabled_channels_without_target_are_disabled(tmp_path, posts):
    n = Notifier({"channels": {"ntfy": {"enabled": True, "topic": ""}, "webhook": {"enabled": True, "url": ""}}},
                 tmp_path / "n.log")
    assert n.channels == []


def test_long_messages_are_truncated(tmp_path, posts):
    n = Notifier({"channels": {"webhook": CHANNELS["webhook"]}}, tmp_path / "n.log")
    n.send("t", "x" * 10_000)
    text = posts[0]["body"]["text"]
    assert len(text) <= alerts.MAX_MESSAGE_CHARS + 10 and text.endswith("…")
    assert len(records(tmp_path / "n.log")[0]["message"]) == 10_000  # the log keeps everything


def test_rate_limit_holds_messages_and_sends_one_digest(tmp_path, posts):
    clock = FakeClock()
    log = tmp_path / "n.log"
    n = Notifier({"max_per_hour": 2, "channels": {"ntfy": CHANNELS["ntfy"]}}, log, clock=clock)
    assert n.send("one", "m1") and n.send("two", "m2")
    clock.t += 60
    assert n.send("three", "m3", "high") is True  # held, not lost
    assert n.send("four", "m4") is True
    assert len(posts) == 2 and n.held == 2
    assert [r["outcome"] for r in records(log)] == ["pushed", "pushed", "held", "held"]
    assert n.flush_digest() is False  # the window is still full
    clock.t += alerts.WINDOW_S - 59  # the first two sends leave the rolling hour
    assert n.flush_digest() is True
    digest = posts[-1]["body"]
    assert digest["title"].startswith("Tycho digest: 2 notifications")
    assert "three" in digest["message"] and "four" in digest["message"]
    assert digest["priority"] == alerts.PRIORITIES["high"] and digest["tags"] == ["tycho", "digest"]
    assert n.held == 0 and n.flush_digest() is False  # nothing left to send
    assert len(posts) == 3


def test_rate_limit_and_digest_survive_a_restart(tmp_path, posts):
    clock = FakeClock()
    log = tmp_path / "n.log"
    cfg = {"max_per_hour": 1, "channels": {"webhook": CHANNELS["webhook"]}}
    first = Notifier(cfg, log, clock=clock)
    assert first.send("a", "m") and first.send("b", "m")
    assert first.held == 1
    second = Notifier(cfg, log, clock=clock)  # e.g. the daemon restarted by launchd
    assert second.held == 1
    assert second.send("c", "m") and len(posts) == 1  # still rate limited: held
    assert second.held == 2
    clock.t += alerts.WINDOW_S + 1
    assert second.flush_digest() and "b" in posts[-1]["body"]["text"] and "c" in posts[-1]["body"]["text"]


def test_unreadable_state_is_ignored(tmp_path, posts):
    log = tmp_path / "n.log"
    log.with_name(log.name + ".state.json").write_text("{not json", encoding="utf-8")
    n = Notifier({"max_per_hour": 1, "channels": {"webhook": CHANNELS["webhook"]}}, log)
    assert n.held == 0 and n.send("x", "y")


@pytest.mark.parametrize("error", [
    urllib.error.URLError("no route to host"),
    urllib.error.HTTPError("https://ntfy.example/", 503, "Service Unavailable", {}, None),
    TimeoutError("timed out"),
    ValueError("unknown url type"),
])
def test_channel_failures_never_raise(tmp_path, posts, error):
    posts.control.error = error
    n = Notifier({"channels": CHANNELS}, tmp_path / "n.log")
    assert n.send("title", "message") is False  # every channel failed
    rec = records(tmp_path / "n.log")[0]
    assert rec["outcome"] == "failed"
    assert all(v.startswith("error:") for v in rec["channels"].values())
    assert n.flush_digest() is False


def test_http_error_status_and_partial_failure(tmp_path, posts):
    posts.control.status = 500
    n = Notifier({"channels": {"ntfy": CHANNELS["ntfy"]}}, tmp_path / "n.log")
    assert n.send("t", "m") is False
    assert records(tmp_path / "n.log")[0]["channels"]["ntfy"] == "error: HTTP 500"


def test_one_working_channel_is_enough(tmp_path, posts, monkeypatch):
    monkeypatch.setattr(alerts.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    posts.control.error = urllib.error.URLError("down")
    n = Notifier({"channels": CHANNELS | {"macos": {"enabled": True}}}, tmp_path / "n.log", platform="darwin")
    assert n.channels == ["ntfy", "webhook", "macos"]
    assert n.send("t", "m") is True
    assert records(tmp_path / "n.log")[0]["channels"]["macos"] == "ok"


def test_macos_channel_only_on_darwin_and_never_raises(tmp_path, monkeypatch):
    assert Notifier({"channels": {"macos": {"enabled": True}}}, tmp_path / "a.log", platform="linux").channels == []
    calls = []

    def missing_osascript(args, **kwargs):
        calls.append(args)
        raise FileNotFoundError("osascript")

    monkeypatch.setattr(alerts.subprocess, "run", missing_osascript)
    n = Notifier({"channels": {"macos": {"enabled": True}}}, tmp_path / "b.log", platform="darwin")
    assert n.send('a "quoted" title', "it's a message") is False
    assert calls[0][0] == "osascript" and calls[0][-2:] == ['a "quoted" title', "it's a message"]  # argv, no quoting


def test_unwritable_log_never_raises(tmp_path, posts):
    log_dir = tmp_path / "is_a_directory"
    log_dir.mkdir()
    n = Notifier({"channels": {"webhook": CHANNELS["webhook"]}}, log_dir)  # the "log file" is a directory
    assert n.send("t", "m") is True  # pushed even though logging failed
    n.record("t", "m")


def test_log_rotation(tmp_path, posts, monkeypatch):
    monkeypatch.setattr(alerts, "LOG_MAX_BYTES", 300)
    log = tmp_path / "n.log"
    n = Notifier({}, log)
    for i in range(40):
        n.send(f"message {i}", "x" * 50)
    rotated = sorted(p.name for p in tmp_path.glob("n.log.*") if not p.name.endswith("state.json"))
    assert rotated == ["n.log.1", "n.log.2", "n.log.3"]  # LOG_BACKUPS old files, no more
    assert log.stat().st_size < 600 and records(log)[-1]["title"] == "message 39"


def test_transient_notification_text():
    row = {"transient_id": 7, "object_id": "abc123", "kind": "late_activation", "severity": 3,
           "message": "late activation of {x1, yhat}", "task": "sparse_parity", "region": "ctrl", "seed": 2}
    title, message, priority, tags = transient_notification(row)
    assert title == "Tycho transient: late activation (severity 3)"
    assert message == "late activation of {x1, yhat}\nobject abc123, task sparse_parity, region ctrl, seed 2"
    assert priority == "high" and tags == ("tycho", "late_activation")
    assert transient_notification({**row, "severity": 1})[2] == "low"
