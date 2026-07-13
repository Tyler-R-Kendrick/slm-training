"""OpenUI via Lark grammar file — in-process AST without Node."""

from __future__ import annotations

from slm_training.dsl.openui_tokens import STRUCTURAL_TOKENS
from slm_training.grammar_backends.lark_backend import LarkFileBackend
from slm_training.grammar_backends.types import GRAMMARS_DIR


class OpenUILarkBackend(LarkFileBackend):
    """Parse OpenUI Lang with ``grammars/openui.lark`` into ElementNode-like ASTs."""

    def __init__(self) -> None:
        super().__init__(
            dsl_id="openui-lark",
            grammar_path=GRAMMARS_DIR / "openui.lark",
            description="OpenUI Lang subset via Lark (in-process AST)",
            root_name="root",
            call_as_component=True,
            prop_order_path=GRAMMARS_DIR / "openui_prop_order.json",
            structural_extras=STRUCTURAL_TOKENS,
        )
