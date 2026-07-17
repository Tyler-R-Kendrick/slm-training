"""OpenUI wiring for the enumerative support oracle (VSS0-04).

Adapts the deterministic compiler forest and the lang-core validity check to the
problem-independent :class:`~slm_training.dsl.solver.support.EnumerativeSupportOracle`.

* :class:`OpenUIForestExpander` advances a token prefix through
  :func:`build_completion_forest`, projecting each next decision with the VSS0-03
  :func:`completion_forest_state` adapter. A chosen ``eos`` path terminates the
  program; a ``coverage == "none"`` forest is bottom; a ``partial``/``none``
  child is reported ``INCOMPLETE`` so the oracle keeps it ``UNKNOWN`` (it can
  never be exhaustively covered).
* :class:`OpenUIWellFormedVerifier` runs the deterministic lang-core parse/schema
  check. A genuine ``ParseError`` is a hard ``REJECT``; a missing bridge, timeout,
  or other runtime fault is ``UNAVAILABLE`` (→ ``UNKNOWN``), **never**
  ``UNSUPPORTED`` — the timeout-vs-UNSAT distinction the contract requires.

This module is Torch-free and is not invoked by decode by default.
"""

from __future__ import annotations

from typing import Any

from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.grammar.fastpath.token_map import decode_prefix
from slm_training.dsl.solver.adapters import completion_forest_state
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleId,
    SolverBounds,
)
from slm_training.dsl.solver.support import (
    ExpandStatus,
    ExpandStep,
    VerifyOutcome,
    VerifyStatus,
)

WELL_FORMED_PROFILE = "openui/lang-core-validate/well-formed@0.2.x"


class OpenUIWellFormedVerifier:
    """Deterministic lang-core well-formedness verifier (G0-G2 surface).

    ``profile`` is the recorded verifier identity. This checks structural
    validity only (``reward_label`` is ``well_formed_not_behavioral``); it is not
    a behavioral or ship verdict.
    """

    def __init__(self, *, profile: str = WELL_FORMED_PROFILE) -> None:
        self._profile = profile

    @property
    def profile(self) -> str:
        return self._profile

    def verify(self, program: str) -> VerifyOutcome:
        from slm_training.dsl import lang_core

        if not lang_core.bridge_available():
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail="bridge_unavailable")
        try:
            lang_core.validate(program)
        except lang_core.ParseError:
            return VerifyOutcome(VerifyStatus.REJECT, detail="parse_error")
        except RuntimeError as exc:  # bridge/timeout — capability, not a rejection
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail=type(exc).__name__)
        return VerifyOutcome(VerifyStatus.ACCEPT)


class OpenUIForestExpander:
    """Bounded deterministic expander over the OpenUI choice/compiler forest."""

    def __init__(
        self,
        tokenizer: Any,
        prefix_ids: tuple[int, ...] | list[int],
        *,
        pack_id: str,
        constraint_version: str,
        bounds: SolverBounds,
        max_path_tokens: int = 8,
    ) -> None:
        self._tok = tokenizer
        self._pack_id = pack_id
        self._cv = constraint_version
        self._bounds = bounds
        self._mpt = max(1, int(max_path_tokens))
        self._eos = int(tokenizer.eos_id)
        self._prefix_by_fp: dict[str, tuple[int, ...]] = {}
        self._root = self._project(tuple(int(t) for t in prefix_ids))

    def _project(self, prefix: tuple[int, ...]) -> FiniteDomainState:
        forest = build_completion_forest(self._tok, list(prefix), max_path_tokens=self._mpt)
        state = completion_forest_state(
            prefix_ids=prefix,
            forest=forest,
            pack_id=self._pack_id,
            constraint_version=self._cv,
            bounds=self._bounds,
        )
        self._prefix_by_fp[state.fingerprint] = prefix
        return state

    # --- ProblemExpander protocol ---------------------------------------- #
    @property
    def problem_id(self) -> str:
        return self._root.problem_id

    @property
    def pack_id(self) -> str:
        return self._pack_id

    @property
    def constraint_version(self) -> str:
        return self._cv

    @property
    def bounds(self) -> SolverBounds:
        return self._bounds

    def root_state(self) -> FiniteDomainState:
        return self._root

    def successor(
        self, state: FiniteDomainState, hole_id: HoleId, value: DomainValue
    ) -> ExpandStep:
        prefix = self._prefix_by_fp.get(state.fingerprint)
        if prefix is None:
            # The oracle only expands states this expander created; a miss means
            # an out-of-band state we cannot faithfully advance -> UNKNOWN.
            return ExpandStep(
                ExpandStatus.INCOMPLETE, coverage="none", detail="unknown_state"
            )
        payload = value.payload
        token_ids = tuple(int(tok) for tok in payload.get("token_ids", ()))
        kind = str(payload.get("kind", ""))
        if kind == "eos" or (len(token_ids) == 1 and token_ids[0] == self._eos):
            program = decode_prefix(self._tok, list(prefix))
            return ExpandStep(
                ExpandStatus.TERMINAL,
                program=program,
                coverage="complete",
                detail=f"prefix_len={len(prefix)}",
            )
        new_prefix = prefix + token_ids
        forest = build_completion_forest(
            self._tok, list(new_prefix), max_path_tokens=self._mpt
        )
        if forest.coverage == "none":
            return ExpandStep(ExpandStatus.DEAD, coverage="none", detail="illegal_prefix")
        if not forest.paths:
            return ExpandStep(
                ExpandStatus.INCOMPLETE,
                coverage=forest.coverage,
                detail="no_enumerated_actions",
            )
        child = completion_forest_state(
            prefix_ids=new_prefix,
            forest=forest,
            pack_id=self._pack_id,
            constraint_version=self._cv,
            bounds=self._bounds,
        )
        self._prefix_by_fp[child.fingerprint] = new_prefix
        return ExpandStep(
            ExpandStatus.CONTINUE, next_state=child, coverage=forest.coverage
        )
