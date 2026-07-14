"""ProgramSpec: the canonical program as the root dataset object + lineage."""

from __future__ import annotations

from slm_training.data.progspec.schema import (
    TASKS,
    TASK_APPLY_PATCH,
    TASK_CANONICALIZE,
    TASK_COMPLETE,
    TASK_GENERATE,
    TASK_INPAINT,
    TASK_NOOP,
    TASK_PATCH,
    TASK_REPAIR_MINIMAL,
    TASK_SEMANTIC_PATCH,
    TASK_VISUAL_GENERATE,
    TASK_VISUAL_PATCH,
    TIER_BRONZE,
    TIER_GOLD,
    TIER_QUARANTINE,
    TIER_SILVER,
    ProgramSpec,
    assign_split_groups,
    resolve_split_group_id,
    structural_family_id,
)

__all__ = [
    "ProgramSpec",
    "assign_split_groups",
    "resolve_split_group_id",
    "structural_family_id",
    "TASKS",
    "TASK_GENERATE",
    "TASK_REPAIR_MINIMAL",
    "TASK_CANONICALIZE",
    "TASK_COMPLETE",
    "TASK_INPAINT",
    "TASK_PATCH",
    "TASK_APPLY_PATCH",
    "TASK_VISUAL_GENERATE",
    "TASK_VISUAL_PATCH",
    "TASK_SEMANTIC_PATCH",
    "TASK_NOOP",
    "TIER_GOLD",
    "TIER_SILVER",
    "TIER_BRONZE",
    "TIER_QUARANTINE",
]
