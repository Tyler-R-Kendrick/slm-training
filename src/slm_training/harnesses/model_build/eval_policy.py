"""Canonical evaluation policies shared by every model-build entrypoint."""

from __future__ import annotations

from typing import Any


CHECKPOINT_DECLARED_POLICY = "checkpoint_declared"
STRICT_COMPILER_TREE_POLICY_ID = "strict_compiler_tree"

# Fields shared by exact choice completion and strict compiler-tree decoding.
# Keep this bundle atomic: partial copies previously produced incomparable runs
# with honest mode enabled but slot-contract constrained decoding disabled.
STRICT_EVALUATION_POLICY: dict[str, Any] = {
    "grammar_constrained": True,
    "grammar_ltr_primary": True,
    "grammar_finalize_validate": True,
    "schema_in_context": True,
    "slot_contract_in_context": True,
    "slot_contract_constrained_decode": True,
    "honest_slot_contract": True,
    "design_md_in_context": False,
    "allow_unconstrained_fallback": False,
}
STRICT_COMPILER_TREE_POLICY: dict[str, Any] = {
    **STRICT_EVALUATION_POLICY,
    "output_tokenizer": "lexer",
    "compiler_decode_mode": "tree",
}
EVALUATION_POLICIES: dict[str, dict[str, Any]] = {
    CHECKPOINT_DECLARED_POLICY: {},
    STRICT_COMPILER_TREE_POLICY_ID: STRICT_COMPILER_TREE_POLICY,
}


def apply_evaluation_policy(config: object) -> None:
    """Normalize a model-build config to one complete evaluation policy."""
    policy_id = str(
        getattr(config, "evaluation_policy", CHECKPOINT_DECLARED_POLICY)
        or CHECKPOINT_DECLARED_POLICY
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
