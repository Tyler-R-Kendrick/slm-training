"""Arithmetic sketch DSL backend (G4 / SLM-36)."""

from __future__ import annotations

from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR


class ArithSketchBackend(LarkFileBackend):
    """Straight-line arithmetic reasoning traces with a required `root`
    answer binding; semantics live in the pack's deterministic evaluator."""

    def __init__(self) -> None:
        super().__init__(
            dsl_id="arith-sketch",
            grammar_path=GRAMMARS_DIR / "arith_sketch.lark",
            description="Straight-line arithmetic sketch DSL (G4 reasoning traces)",
            root_name="root",
            call_as_component=False,
            structural_extras=frozenset({"=", "(", ")", "+", "-", "*", "/"}),
        )
