"""Canonical evaluation policies shared by every model-build entrypoint."""

from __future__ import annotations

from slm_training.levers import (
    CHECKPOINT_DECLARED_POLICY,
    DEFAULT_EVALUATION_POLICY,
    STRICT_COMPILER_TREE_POLICY,
    STRICT_COMPILER_TREE_POLICY_ID,
    STRICT_EVALUATION_POLICY,
)


EVALUATION_POLICIES: dict[str, dict[str, object]] = {
    # Retain checkpoint architecture/conditioning for diagnostic comparisons,
    # but never inherit unsafe completion settings from checkpoint metadata.
    CHECKPOINT_DECLARED_POLICY: STRICT_EVALUATION_POLICY,
    STRICT_COMPILER_TREE_POLICY_ID: STRICT_COMPILER_TREE_POLICY,
}


def apply_evaluation_policy(config: object) -> None:
    """Normalize a model-build config to one complete evaluation policy."""
    policy_id = str(
        getattr(config, "evaluation_policy", DEFAULT_EVALUATION_POLICY)
        or DEFAULT_EVALUATION_POLICY
    )
    if getattr(config, "compiler_decode_mode", "off") == "tree":
        policy_id = STRICT_COMPILER_TREE_POLICY_ID
        setattr(config, "evaluation_policy", policy_id)
    try:
        policy = EVALUATION_POLICIES[policy_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown evaluation_policy {policy_id!r}; "
            f"expected one of {sorted(EVALUATION_POLICIES)}"
        ) from exc
    for field, value in policy.items():
        setattr(config, field, value)


def apply_strict_compiler_tree_policy(config: object) -> None:
    """Apply the canonical honest compiler-tree policy to a model config."""
    setattr(config, "evaluation_policy", STRICT_COMPILER_TREE_POLICY_ID)
    apply_evaluation_policy(config)


__all__ = [
    "CHECKPOINT_DECLARED_POLICY",
    "EVALUATION_POLICIES",
    "STRICT_COMPILER_TREE_POLICY",
    "STRICT_COMPILER_TREE_POLICY_ID",
    "STRICT_EVALUATION_POLICY",
    "apply_evaluation_policy",
    "apply_strict_compiler_tree_policy",
]
