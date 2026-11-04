"""The survey agent (SPEC §7): a Claude-based observer that runs nightly (quality control, classification review,
follow-up requests, proposals of the next configurations, the survey log) and weekly (the weekly report with candidate
empirical laws), acting only through the bounded, audited tools of :mod:`tycho.agent.tools`.

Modules: :mod:`~tycho.agent.prompts` (system prompts), :mod:`~tycho.agent.tools` (tools and their JSON schemas),
:mod:`~tycho.agent.runner` (the capped, audited conversation loop), :mod:`~tycho.agent.nightly` and
:mod:`~tycho.agent.weekly` (the two entry points).
"""
from tycho.agent.nightly import run_nightly
from tycho.agent.runner import AgentRunner, AgentRunResult, AgentUnavailable, Pricing, Usage
from tycho.agent.tools import TOOL_SPECS, SurveyTools, ToolLimits, tool_specs
from tycho.agent.weekly import run_weekly

__all__ = ["AgentRunResult", "AgentRunner", "AgentUnavailable", "Pricing", "SurveyTools", "TOOL_SPECS", "ToolLimits",
           "Usage", "run_nightly", "run_weekly", "tool_specs"]
