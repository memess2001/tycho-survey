"""Notifications (SPEC §6.4): push transients and health alerts to the researcher, rate limited, never raising.

:class:`Notifier` appends every message to a log file (JSON lines, one record per message) and pushes it to the
enabled channels of the ``notify`` config section:

* ``ntfy`` (``{server, topic}``): a JSON publish to the server root (``{"topic", "title", "message", "priority",
  "tags"}``), which avoids HTTP-header encoding limits for non-ASCII titles. Subscribe to the topic in the ntfy app;
  use a long random topic name, since anyone who knows it can read it.
* ``webhook`` (``{url}``): a Slack-compatible ``{"text": "*title*\\nmessage"}`` POST.
* ``macos``: a Notification Center banner via ``osascript`` (only on macOS; ignored elsewhere). Title and message
  are passed as script arguments, so no quoting is involved.

HTTP uses ``urllib`` with a :data:`HTTP_TIMEOUT_S` timeout. Every failure (network, HTTP status, bad URL, missing
``osascript``, an unwritable log) is logged and reported in the return value; :meth:`Notifier.send` never raises.

Rate limit and digest
    At most ``max_per_hour`` pushes in any rolling hour (``None`` or ≤ 0: unlimited). A message over the limit is
    logged and held for a digest; :meth:`Notifier.flush_digest` (called by the daemon every loop) pushes one digest
    message listing the held titles as soon as the window has room again. The digest counts as one push. The send
    times and the held messages are saved next to the log (``<log>.state.json``), so a restarting daemon cannot
    exceed the limit or lose a pending digest.

Severity
    :meth:`Notifier.send` takes a priority (``min``, ``low``, ``default``, ``high``, ``urgent``); the daemon decides
    what is pushed: transients with severity ≥ ``notify.min_severity`` are sent (priority :data:`SEVERITY_PRIORITY`),
    weaker ones only logged (:meth:`Notifier.record`).
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import tycho

logger = logging.getLogger(__name__)

__all__ = ["HTTP_TIMEOUT_S", "PRIORITIES", "SEVERITY_PRIORITY", "Notifier", "transient_notification"]

PRIORITIES: dict[str, int] = {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5}
"""Accepted priorities and their ntfy levels (an unknown priority is sent as "default")."""

SEVERITY_PRIORITY: dict[int, str] = {1: "low", 2: "default", 3: "high"}
"""Push priority of a transient of each severity (``tycho.analysis.events.TRANSIENT_SEVERITY``)."""

HTTP_TIMEOUT_S = 10.0
"""Timeout of one HTTP push."""

OSASCRIPT_TIMEOUT_S = 10.0
"""Timeout of one macOS notification."""

WINDOW_S = 3600.0
"""Length of the rate-limit window (``max_per_hour``)."""

MAX_MESSAGE_CHARS = 3500
"""Pushed messages are truncated to this length (ntfy turns messages above 4 KB into attachments)."""

DIGEST_MAX_TITLES = 20
"""Titles listed in a digest; further held messages are counted only."""

MAX_HELD = 500
"""Held messages kept for the digest (the oldest are dropped beyond this, but still counted)."""

_USER_AGENT = f"tycho-survey/{tycho.__version__}"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def transient_notification(row: Mapping[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    """(title, message, priority, tags) of a transient row of ``Catalog.pending_notifications``."""
    kind, severity = str(row.get("kind")), int(row.get("severity") or 1)
    title = f"Tycho transient: {kind.replace('_', ' ')} (severity {severity})"
    where = [f"object {row.get('object_id')}"]
    for key in ("task", "region"):
        if row.get(key) is not None:
            where.append(f"{key} {row[key]}")
    if row.get("seed") is not None:
        where.append(f"seed {row['seed']}")
    message = f"{row.get('message')}\n{', '.join(where)}"
    return title, message, SEVERITY_PRIORITY.get(severity, "default"), ("tycho", kind)


class Notifier:
    """Rate-limited notifications (see the module docstring).

    Args:
        cfg: the ``notify`` config section (``max_per_hour``, ``channels``; ``min_severity`` is exposed as an
            attribute for the caller).
        log_path: the notification log (JSON lines); its directory is created.
        clock: time source (seconds since the epoch), injectable for tests.
        platform: ``sys.platform`` value deciding whether the macOS channel is usable.
    """

    def __init__(self, cfg: Mapping[str, Any] | None, log_path: Path | str, *,
                 clock: Callable[[], float] = time.time, platform: str = sys.platform) -> None:
        cfg = dict(cfg or {})
        limit = cfg.get("max_per_hour")
        self.max_per_hour: int | None = int(limit) if limit is not None and int(limit) > 0 else None
        self.min_severity = int(cfg.get("min_severity", 2))
        channels = cfg.get("channels") or {}
        ntfy = channels.get("ntfy") or {}
        webhook = channels.get("webhook") or {}
        macos = channels.get("macos") or {}
        self.ntfy: dict[str, str] | None = None
        if ntfy.get("enabled") and ntfy.get("topic"):
            self.ntfy = {"server": str(ntfy.get("server") or "https://ntfy.sh").rstrip("/"),
                         "topic": str(ntfy["topic"])}
        elif ntfy.get("enabled"):
            logger.warning("notify.channels.ntfy is enabled but has no topic; ntfy disabled")
        self.webhook_url: str | None = None
        if webhook.get("enabled") and webhook.get("url"):
            self.webhook_url = str(webhook["url"])
        elif webhook.get("enabled"):
            logger.warning("notify.channels.webhook is enabled but has no url; webhook disabled")
        self.macos = bool(macos.get("enabled")) and platform == "darwin"
        self.log_path = Path(log_path)
        self.state_path = self.log_path.with_name(self.log_path.name + ".state.json")
        self._clock = clock
        self._lock = threading.Lock()
        self._sent: list[float] = []
        self._held: list[dict[str, Any]] = []
        self._n_held = 0
        self._load_state()

    # ------------------------------------------------------------------ public API
    @property
    def channels(self) -> list[str]:
        """Names of the enabled push channels."""
        return [name for name, on in (("ntfy", self.ntfy is not None), ("webhook", self.webhook_url is not None),
                                      ("macos", self.macos)) if on]

    @property
    def held(self) -> int:
        """Number of messages waiting for the digest."""
        return self._n_held

    def send(self, title: str, message: str, priority: str = "default", tags: Sequence[str] = ()) -> bool:
        """Log a message and push it (unless rate limited: then it is held for the digest).

        Returns:
            False only when every enabled channel failed; True when it was pushed by at least one channel, held for
            the digest, or there is no push channel (logged only). Never raises.
        """
        try:
            with self._lock:
                return self._send(str(title), str(message), _priority(priority), tuple(str(t) for t in tags))
        except Exception:  # last line of defence: a notifier must never take its caller down
            logger.exception("notifier: unexpected error while sending %r", title)
            return False

    def record(self, title: str, message: str, priority: str = "low", tags: Sequence[str] = ()) -> None:
        """Log a message without pushing it (e.g. a transient below ``min_severity``). Never raises."""
        try:
            with self._lock:
                self._log(self._clock(), str(title), str(message), _priority(priority), tuple(str(t) for t in tags),
                          "logged", {})
        except Exception:
            logger.exception("notifier: unexpected error while recording %r", title)

    def flush_digest(self) -> bool:
        """Push the digest of held messages if there are any and the rate limit allows one push.

        Returns True if a digest went out (to at least one channel). Never raises.
        """
        try:
            with self._lock:
                return self._flush_digest()
        except Exception:
            logger.exception("notifier: unexpected error while sending the digest")
            return False

    # ------------------------------------------------------------------ internals
    def _send(self, title: str, message: str, priority: str, tags: tuple[str, ...]) -> bool:
        now = self._clock()
        if not self.channels:
            self._log(now, title, message, priority, tags, "logged", {})
            return True
        if not self._has_room(now):
            self._held.append({"time": now, "title": title, "priority": priority})
            self._n_held += 1
            del self._held[:-MAX_HELD]
            self._save_state()
            self._log(now, title, message, priority, tags, "held", {})
            logger.info("notifier: rate limit (%s/h) reached; %r held for the digest", self.max_per_hour, title)
            return True
        results = self._push(title, message, priority, tags)
        ok = any(r == "ok" for r in results.values())
        if ok:
            self._sent.append(now)
            self._save_state()
        self._log(now, title, message, priority, tags, "pushed" if ok else "failed", results)
        if not ok:
            logger.warning("notifier: every channel failed for %r: %s", title, results)
        return ok

    def _flush_digest(self) -> bool:
        if not self._n_held or not self.channels:
            return False
        now = self._clock()
        if not self._has_room(now):
            return False
        n, held = self._n_held, list(self._held)
        lines = [f"- {_iso(h['time'])} [{h['priority']}] {h['title']}" for h in held[-DIGEST_MAX_TITLES:]]
        if n > len(lines):
            lines.insert(0, f"(the {n - len(lines)} earlier ones are not listed)")
        title = f"Tycho digest: {n} notification{'s' if n != 1 else ''} held back by the rate limit"
        priority = max((h["priority"] for h in held), key=lambda p: PRIORITIES[p], default="default")
        results = self._push(title, "\n".join(lines), priority, ("tycho", "digest"))
        ok = any(r == "ok" for r in results.values())
        if ok:
            self._sent.append(now)
            self._held, self._n_held = [], 0
            self._save_state()
        self._log(now, title, "\n".join(lines), priority, ("tycho", "digest"), "pushed" if ok else "failed", results)
        return ok

    def _has_room(self, now: float) -> bool:
        self._sent = [t for t in self._sent if now - t < WINDOW_S]
        return self.max_per_hour is None or len(self._sent) < self.max_per_hour

    def _push(self, title: str, message: str, priority: str, tags: tuple[str, ...]) -> dict[str, str]:
        """Deliver to every enabled channel; channel name -> "ok" or "error: ..."."""
        text = message if len(message) <= MAX_MESSAGE_CHARS else message[:MAX_MESSAGE_CHARS - 1] + "…"
        results: dict[str, str] = {}
        if self.ntfy is not None:
            body = {"topic": self.ntfy["topic"], "title": title, "message": text,
                    "priority": PRIORITIES[priority], "tags": list(tags)}
            results["ntfy"] = _post_json(self.ntfy["server"] + "/", body)
        if self.webhook_url is not None:
            results["webhook"] = _post_json(self.webhook_url, {"text": f"*{title}*\n{text}"})
        if self.macos:
            results["macos"] = _macos_notification(title, text)
        return results

    def _log(self, now: float, title: str, message: str, priority: str, tags: tuple[str, ...], outcome: str,
             channels: Mapping[str, str]) -> None:
        record = {"time": _iso(now), "ts": now, "outcome": outcome, "priority": priority, "title": title,
                  "message": message, "tags": list(tags), "channels": dict(channels)}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("notifier: cannot append to %s: %s", self.log_path, exc)

    def _load_state(self) -> None:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            sent = [float(t) for t in state.get("sent", []) if math.isfinite(float(t))]
            held = [{"time": float(h["time"]), "title": str(h["title"]), "priority": _priority(h.get("priority"))}
                    for h in state.get("held", [])]
            n_held = max(int(state.get("n_held", len(held))), len(held))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("notifier: ignoring unreadable state %s (%s)", self.state_path, exc)
            return
        self._sent, self._held, self._n_held = sent, held, n_held

    def _save_state(self) -> None:
        state = {"sent": self._sent, "held": self._held, "n_held": self._n_held}
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(state), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError as exc:
            logger.error("notifier: cannot save its state to %s: %s", self.state_path, exc)


def _priority(priority: Any) -> str:
    p = str(priority or "default").lower()
    return p if p in PRIORITIES else "default"


def _post_json(url: str, payload: Mapping[str, Any]) -> str:
    """POST JSON; "ok" for a 2xx answer, else "error: ..." (never raises)."""
    try:
        request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
                                         method="POST")
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            status = int(getattr(response, "status", None) or response.getcode())
        return "ok" if 200 <= status < 300 else f"error: HTTP {status}"
    except urllib.error.HTTPError as exc:
        return f"error: HTTP {exc.code} {exc.reason}"
    except urllib.error.URLError as exc:
        return f"error: {exc.reason}"
    except Exception as exc:  # timeouts, bad URLs, TLS, anything else: never raise
        return f"error: {type(exc).__name__}: {exc}"


_OSASCRIPT = ("on run argv", "display notification (item 2 of argv) with title (item 1 of argv)", "end run")


def _macos_notification(title: str, message: str) -> str:
    """Notification Center banner through osascript; "ok" or "error: ..." (never raises)."""
    args = ["osascript"]
    for line in _OSASCRIPT:
        args += ["-e", line]
    try:
        out = subprocess.run([*args, title, message], capture_output=True, text=True, timeout=OSASCRIPT_TIMEOUT_S,
                             check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error: {type(exc).__name__}: {exc}"
    return "ok" if out.returncode == 0 else f"error: osascript exit {out.returncode}: {out.stderr.strip()[:200]}"

