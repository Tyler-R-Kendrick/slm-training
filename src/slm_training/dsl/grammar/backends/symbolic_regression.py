"""Symbolic-regression expression DSL backend (SRP-001 / SLM-441).

Minimal generic-Lark grammar for a single ``root`` expression over declared
input variables and a small allow-listed function set (see
``symbolic_regression_pack.ALLOWED_OPERATORS``). This is a placeholder:
SRP-002 (SLM-453) owns the typed symbolic-expression IR/grammar/codec that
supersedes it.
"""

from __future__ import annotations

from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR


class SymbolicRegressionBackend(LarkFileBackend):
    """``arith_sketch``-style statement/expr grammar plus function calls."""

    def __init__(self) -> None:
        super().__init__(
            dsl_id="symbolic-regression",
            grammar_path=GRAMMARS_DIR / "symbolic_regression.lark",
            description=(
                "Minimal arithmetic-plus-allow-listed-function grammar for a "
                "single `root` symbolic-regression expression."
            ),
            root_name="root",
            call_as_component=False,
            structural_extras=frozenset({"=", "(", ")", "+", "-", "*", "/", ","}),
        )
