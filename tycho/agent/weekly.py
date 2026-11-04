"""The weekly agent run (SPEC §7): the weekly report with candidate empirical laws, calibration status and
recommendations.

``run_weekly(root)`` builds the context packet (the reporting period since the previous completed weekly run, at most
31 days and 7 days by default, what happened in it, the QC overview), runs the agent with the weekly tool set (read
only, plus ``fit_power_law`` and ``write_report``) at the ``agent.weekly_effort`` level, and returns the report file
``reports/weekly/<ISO year>-W<ISO week>.md``. The tool ``write_report`` writes that file and a survey-log entry; the
runner appends a footer recording the run. If the model did not write the report, the runner writes one recording
why, with the agent's last text, both as a file and as a survey-log entry. With ``dry_run=True`` it only writes the
would-be first request to ``reports/agent/dryrun-weekly.json``; no credentials are needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tycho.agent.prompts import task_text
from tycho.agent.runner import (RUNNER_AUTHOR, AgentRunResult, Window, describe_run, missing_output_markdown,
                                run_session)
from tycho.agent.tools import (SurveyTools, fit_json, iso_utc, iso_week, local_time, unique_path,
                               write_text_atomic)
from tycho.survey.catalog import Catalog

CONTEXT_MAX_CHARS = 40_000
"""Size budget of the context packet's JSON (shrunk like an over-long tool output)."""


def run_weekly(root: Path | str, *, dry_run: bool = False, client: Any = None) -> Path:
    """Run the weekly agent on the survey at ``root``; returns the report file (or, with ``dry_run``, the dry-run
    request file). Raises ``AgentUnavailable`` without usable credentials or when ``agent.enabled`` is false (dry
    runs excepted)."""
    return run_session("weekly", root, dry_run=dry_run, client=client, context=weekly_context, finalize=_finalize)


def weekly_context(tools: SurveyTools, window: Window, budget: Mapping[str, Any]) -> str:
    """The first user message of a weekly run: the fixed task text and the context packet (JSON); ``budget`` holds
    the run's ``max_api_calls`` and ``max_cost_usd``."""
    now = tools.now()
    packet = {
        "run": "weekly",
        "now_utc": iso_utc(now),
        "now_local": local_time(now),
        "iso_week": iso_week(now),
        "period": {"start_utc": iso_utc(window.start), "days": round((now - window.start) / 86_400.0, 2),
                   "basis": window.basis},
        "budget": dict(budget),
        "in_period": tools.period_summary(window.start),
        "overview": tools.overview(),
    }
    return f"{task_text('weekly')}\n\n```json\n{fit_json(packet, CONTEXT_MAX_CHARS)}\n```"


def _finalize(catalog: Catalog, tools: SurveyTools, result: AgentRunResult) -> Path:
    footer = f"\n---\n\n*{describe_run(result)}*\n"
    if tools.report_path is not None:
        path = tools.report_path
        write_text_atomic(path, path.read_text(encoding="utf-8") + footer)
        return path
    week = iso_week(tools.now())
    title = f"Weekly report {week}: not written by the agent"
    markdown = missing_output_markdown(result, tools, "report")
    path = unique_path(tools.paths.reports / "weekly", week)
    write_text_atomic(path, f"# {title}\n\n{markdown}\n{footer}")
    catalog.add_log(RUNNER_AUTHOR, "weekly", title, markdown)
    return path
