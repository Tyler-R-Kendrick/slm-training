"""Calculated arity and adaptive precision analyzer (CAP0-02)."""

from __future__ import annotations

from slm_training.dsl.analysis.arity.analyzer import ArityAnalyzer
from slm_training.dsl.analysis.arity.report import ArityReport, ContinuationSummary
from slm_training.dsl.analysis.arity.types import (
    AnalysisBounds,
    StateAtom,
    StateSignature,
    SupportOracle,
    SupportQuery,
    SupportResult,
    SupportVerdict,
)

__all__ = [
    "AnalysisBounds",
    "ArityAnalyzer",
    "ArityReport",
    "ContinuationSummary",
    "StateAtom",
    "StateSignature",
    "SupportOracle",
    "SupportQuery",
    "SupportResult",
    "SupportVerdict",
]
