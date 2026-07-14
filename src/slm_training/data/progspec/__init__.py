"""Canonical program roots, generation, and derivative projection."""

from slm_training.data.progspec.generate import (
    CoverageCell,
    CoverageTracker,
    GenerationResult,
    GeneratorConfig,
    ProgramGenerator,
    generate_program_specs,
)
from slm_training.data.progspec.schema import ProgramSpec, emit_record

__all__ = [
    "CoverageCell",
    "CoverageTracker",
    "GenerationResult",
    "GeneratorConfig",
    "ProgramGenerator",
    "ProgramSpec",
    "emit_record",
    "generate_program_specs",
]
