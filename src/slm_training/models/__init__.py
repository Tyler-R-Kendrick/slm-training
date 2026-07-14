"""Neural models for OpenUI TwoTower training."""

from __future__ import annotations

from typing import Any

from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text


def __getattr__(name: str) -> Any:
    """Load the PyTorch training model only when explicitly requested."""
    if name in {"TwoTowerConfig", "TwoTowerModel"}:
        from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

        return {"TwoTowerConfig": TwoTowerConfig, "TwoTowerModel": TwoTowerModel}[name]
    if name in {"CausalLMOpenUIConfig", "CausalLMOpenUIPlugin"}:
        from slm_training.models.causal_lm_openui import (
            CausalLMOpenUIConfig,
            CausalLMOpenUIPlugin,
        )

        return {
            "CausalLMOpenUIConfig": CausalLMOpenUIConfig,
            "CausalLMOpenUIPlugin": CausalLMOpenUIPlugin,
        }[name]
    raise AttributeError(name)

__all__ = [
    "DSLNativeTokenizer",
    "CausalLMOpenUIConfig",
    "CausalLMOpenUIPlugin",
    "OpenUITokenizer",
    "SymbolTable",
    "TwoTowerConfig",
    "TwoTowerModel",
    "tokenize_text",
]
