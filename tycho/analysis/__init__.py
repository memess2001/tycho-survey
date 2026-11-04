"""Tycho subpackage."""
# The functions summarize() and classify() are deliberately not re-exported: they share their modules' names, and
# binding them here would shadow the submodules (``import tycho.analysis.summarize as m`` would return the function and
# ``tycho.analysis.summarize.summarize`` would fail). Import them from tycho.analysis.summarize / .classify.
from tycho.analysis.classify import LABELS, Classification
from tycho.analysis.events import RIEvent, Transient, detect_events, find_transients
from tycho.analysis.ricurve import RICurve
from tycho.analysis.summarize import find_drops

__all__ = ["LABELS", "Classification", "RICurve", "RIEvent", "Transient", "detect_events", "find_drops",
           "find_transients"]
