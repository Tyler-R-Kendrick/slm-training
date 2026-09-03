"""Code-quality measurement: module size, complexity, and package structure.

Enforced by ``python -m scripts.verify_code_quality``; contract documented in
``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from slm_training.quality.baseline import RatchetOutcome, compare
from slm_training.quality.complexity import RULES, RuffUnavailableError, Violation
from slm_training.quality.gate import (
    MAX_COMPONENT_LINES,
    QualityReport,
    analyse,
    evaluate,
    merged_dimensions,
)
from slm_training.quality.imports import ModuleInfo, parse_modules, resolve_edges
from slm_training.quality.martin import ComponentMetrics, cycles
from slm_training.quality.report import render
from slm_training.quality.sources import MAX_MODULE_LINES, SourceFile, discover

__all__ = [
    "MAX_COMPONENT_LINES",
    "MAX_MODULE_LINES",
    "RULES",
    "ComponentMetrics",
    "ModuleInfo",
    "QualityReport",
    "RatchetOutcome",
    "RuffUnavailableError",
    "SourceFile",
    "Violation",
    "analyse",
    "compare",
    "cycles",
    "discover",
    "evaluate",
    "merged_dimensions",
    "parse_modules",
    "render",
    "resolve_edges",
]
