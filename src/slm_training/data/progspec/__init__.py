"""Canonical program roots, generation, and derivative projection."""

from typing import Any

from slm_training.data.progspec.schema import ProgramSpec, emit_record

_GENERATOR_EXPORTS = {
    "CoverageCell",
    "CoverageTracker",
    "GenerationResult",
    "GeneratorConfig",
    "ProgramGenerator",
    "generate_program_specs",
}


def __getattr__(name: str) -> Any:
    """Load generator exports without creating a language-contract import cycle."""
    if name not in _GENERATOR_EXPORTS:
        raise AttributeError(name)
    from slm_training.data.progspec import generate

    return getattr(generate, name)


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
