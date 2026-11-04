"""The nightly agent run (SPEC §7): quality control, classification review, proposals and the survey log.

``run_nightly(root)`` builds the context packet (the observation window since the previous completed nightly run,
what finished in it, the QC overview and the caps left), runs the agent with the nightly tool set at the
``agent.nightly_effort`` level, makes sure a survey-log entry exists (if the model did not call ``write_survey_log``,
the runner writes an entry recording why, with the agent's last text) and writes a markdown copy of the entry to
``reports/nightly/<local date>.md`` (``-2``, ``-3``, ... when a date already has one). With ``dry_run=True`` it only
writes the would-be first request to ``reports/agent/dryrun-nightly.json``; no credentials are needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tycho.agent.prompts import task_text
from tycho.agent.runner import (RUNNER_AUTHOR, AgentRunResult, Window, describe_run, missing_output_markdown,
                                run_session)
from tycho.agent.tools import SurveyTools, fit_json, iso_utc, local_date, local_time, unique_path, write_text_atomic
from tycho.survey.catalog import Catalog

CONTEXT_MAX_CHARS = 40_000
"""Size budget of the context packet's JSON (shrunk like an over-long tool output)."""


def run_nightly(root: Path | str, *, dry_run: bool = False, client: Any = None) -> Path:
    """Run the nightly agent on the survey at ``root``; returns the markdown copy of the night's survey-log entry
    (or, with ``dry_run``, the dry-run request file). Raises ``AgentUnavailable`` without usable credentials or when
    ``agent.enabled`` is false (dry runs excepted)."""
    return run_session("nightly", root, dry_run=dry_run, client=client, context=nightly_context,
                       finalize=_finalize)


def nightly_context(tools: SurveyTools, window: Window, budget: Mapping[str, Any]) -> str:
    """The first user message of a nightly run: the fixed task text and the context packet (JSON); ``budget`` holds
    the run's ``max_api_calls`` and ``max_cost_usd``."""
    now = tools.now()
    packet = {
        "run": "nightly",
        "now_utc": iso_utc(now),
        "now_local": local_time(now),
        "window": {"start_utc": iso_utc(window.start), "hours": round((now - window.start) / 3600.0, 2),
                   "basis": window.basis},
        "budget": dict(budget),
        "in_window": tools.period_summary(window.start),
        "overview": tools.overview(),
    }
    return f"{task_text('nightly')}\n\n```json\n{fit_json(packet, CONTEXT_MAX_CHARS)}\n```"


def _finalize(catalog: Catalog, tools: SurveyTools, result: AgentRunResult) -> Path:
    entry_id = tools.log_entry_id
    if entry_id is None:
        entry_id = catalog.add_log(RUNNER_AUTHOR, "nightly",
                                   f"Nightly run {local_date(tools.now())}: no survey log from the agent",
                                   missing_output_markdown(result, tools, "survey log"))
    entry = catalog.query("SELECT entry_id, author, title, markdown FROM survey_log WHERE entry_id = ?",
                          (entry_id,))[0]
    path = unique_path(tools.paths.reports / "nightly", local_date(tools.now()))
    write_text_atomic(path, f"# {entry['title']}\n\n{entry['markdown']}\n\n---\n\n"
                            f"*Survey-log entry {entry['entry_id']} by {entry['author']}. {describe_run(result)}*\n")
    return path
