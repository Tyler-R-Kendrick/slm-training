"""Deterministic invalid-program operators with verified clean repairs."""

from slm_training.data.corrupt.oracle import (
    CorruptionCase,
    CorruptionNotApplicable,
    CorruptionOperator,
    OperatorFamily,
    ScopedCorruption,
    build_corruption,
    build_scoped_corruptions,
    generate_corruptions,
)

__all__ = [
    "CorruptionCase",
    "CorruptionNotApplicable",
    "CorruptionOperator",
    "OperatorFamily",
    "ScopedCorruption",
    "build_corruption",
    "build_scoped_corruptions",
    "generate_corruptions",
]
