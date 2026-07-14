"""Deterministic invalid-program operators with verified clean repairs."""

from slm_training.data.corrupt.oracle import (
    CorruptionCase,
    CorruptionNotApplicable,
    CorruptionOperator,
    OperatorFamily,
    build_corruption,
    generate_corruptions,
)

__all__ = [
    "CorruptionCase",
    "CorruptionNotApplicable",
    "CorruptionOperator",
    "OperatorFamily",
    "build_corruption",
    "generate_corruptions",
]
