"""Canonical program roots, generation, and derivative projection."""

from slm_training.data.progspec.generate import (
    CoverageTracker,
    GenerationResult,
    TypedProgramGenerator,
)
from slm_training.data.progspec.schema import ProgramSpec, emit_record

__all__ = [
    "CoverageTracker",
    "GenerationResult",
    "ProgramSpec",
    "TypedProgramGenerator",
    "emit_record",
]
