"""Canonical evaluation policies shared by matrices and trace collectors."""

from __future__ import annotations

from typing import Any


STRICT_COMPILER_TREE_POLICY: dict[str, Any] = {
    "output_tokenizer": "lexer",
    "grammar_ltr_primary": True,
    "grammar_finalize_validate": True,
    "compiler_decode_mode": "tree",
    "schema_in_context": True,
    "slot_contract_in_context": True,
    "slot_contract_constrained_decode": True,
    "honest_slot_contract": True,
    "design_md_in_context": False,
    "allow_unconstrained_fallback": False,
}


def apply_strict_compiler_tree_policy(config: object) -> None:
    """Apply the canonical honest compiler-tree policy to a model config."""
    for field, value in STRICT_COMPILER_TREE_POLICY.items():
        setattr(config, field, value)


__all__ = ["STRICT_COMPILER_TREE_POLICY", "apply_strict_compiler_tree_policy"]
