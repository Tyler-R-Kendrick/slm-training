"""Toy layout DSL — proves GrammarBackend is not OpenUI-specific."""

from __future__ import annotations

from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR


class ToyLayoutBackend(LarkFileBackend):
    """Minimal alternate DSL for grammar-based training smoke tests."""

    def __init__(self) -> None:
        super().__init__(
            dsl_id="toy-layout",
            grammar_path=GRAMMARS_DIR / "toy_layout.lark",
            description="Minimal toy layout DSL (grammar-training smoke test)",
            root_name="root",
            call_as_component=True,
            prop_order={
                "row": ["children", "gap"],
                "col": ["children", "gap"],
                "text": ["text", "size"],
                "button": ["label", "action"],
                "stack": ["children", "direction"],
            },
            structural_extras=frozenset(
                {"root", "row", "col", "text", "button", "stack", "(", ")", "[", "]", ",", "=", '"'}
            ),
        )
