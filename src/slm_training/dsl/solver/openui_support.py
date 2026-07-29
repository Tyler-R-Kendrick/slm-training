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

from typing import Any, TypeAlias

from slm_training.dsl.grammar.fastpath.completion_kernel import CompletionSession
from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest
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

# Coverage-"none" marker for a rejected advance (identical to a fresh build's
# dead-prefix forest: no paths, no terminals).
_DEAD_FOREST = CompletionForest((), "none")

StateId: TypeAlias = int
EdgeId: TypeAlias = int


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
    """Bounded deterministic expander over the OpenUI choice/compiler forest.

    Holds one request-local packed
    :class:`~slm_training.dsl.grammar.fastpath.completion_kernel.CompletionSession`:
    ``_project``/``successor`` reuse interned states (advance along the chosen
    path's token ids, then project the target state's outgoing edges) instead
    of building a fresh forest per node with no shared state.  Certificates
    remain payload-based (kind, token ids, versions, digests) — interned
    state ids never leave the expander.
    """

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
        self._session = CompletionSession(tokenizer, max_path_tokens=self._mpt)
        self._prefix_by_fp: dict[str, tuple[int, ...]] = {}
        self._sid_by_fp: dict[str, int] = {}
        self._edge_ids: dict[tuple[StateId, HoleId, DomainValue], EdgeId] = {}
        self._successors: dict[tuple[StateId, EdgeId], ExpandStep] = {}
        self._root = self._project(tuple(int(t) for t in prefix_ids), None)

    def _project(
        self, prefix: tuple[int, ...], state_id: int | None
    ) -> FiniteDomainState:
        sid = self._session.seed(prefix) if state_id is None else state_id
        forest = self._session.outgoing(sid)
        state = completion_forest_state(
            prefix_ids=prefix,
            forest=forest,
            pack_id=self._pack_id,
            constraint_version=self._cv,
            bounds=self._bounds,
        )
        self._prefix_by_fp[state.fingerprint] = prefix
        self._sid_by_fp[state.fingerprint] = sid
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
        sid = self._sid_by_fp.get(state.fingerprint)
        if prefix is None or sid is None:
            # The oracle only expands states this expander created; a miss means
            # an out-of-band state we cannot faithfully advance -> UNKNOWN.
            return ExpandStep(
                ExpandStatus.INCOMPLETE, coverage="none", detail="unknown_state"
            )
        edge_key = (sid, hole_id, value)
        edge_id = self._edge_ids.get(edge_key)
        if edge_id is None:
            edge_id = len(self._edge_ids)
            self._edge_ids[edge_key] = edge_id
        cache_key = (sid, edge_id)
        cached = self._successors.get(cache_key)
        if cached is not None:
            from slm_training.models.decode_stats import get_active_stats

            stats = get_active_stats()
            if stats is not None:
                stats.solver_successor_cache_hits += 1
            return cached
        from slm_training.models.decode_stats import get_active_stats

        stats = get_active_stats()
        if stats is not None:
            stats.solver_successor_cache_misses += 1
            stats.solver_successor_expansions += 1

        def _done(step: ExpandStep) -> ExpandStep:
            self._successors[cache_key] = step
            return step

        payload = value.payload
        token_ids = tuple(int(tok) for tok in payload.get("token_ids", ()))
        kind = str(payload.get("kind", ""))
        if kind == "eos" or (len(token_ids) == 1 and token_ids[0] == self._eos):
            program = decode_prefix(self._tok, list(prefix))
            return _done(
                ExpandStep(
                    ExpandStatus.TERMINAL,
                    program=program,
                    coverage="complete",
                    detail=f"prefix_len={len(prefix)}",
                )
            )
        new_prefix = prefix + token_ids
        child_sid = self._session.advance_path(sid, token_ids)
        forest = (
            self._session.outgoing(child_sid)
            if child_sid is not None
            else _DEAD_FOREST
        )
        if forest.coverage == "none":
            return _done(
                ExpandStep(
                    ExpandStatus.DEAD,
                    coverage="none",
                    detail="illegal_prefix",
                )
            )
        if not forest.paths:
            return _done(
                ExpandStep(
                    ExpandStatus.INCOMPLETE,
                    coverage=forest.coverage,
                    detail="no_enumerated_actions",
                )
            )
        child = completion_forest_state(
            prefix_ids=new_prefix,
            forest=forest,
            pack_id=self._pack_id,
            constraint_version=self._cv,
            bounds=self._bounds,
        )
        self._prefix_by_fp[child.fingerprint] = new_prefix
        self._sid_by_fp[child.fingerprint] = child_sid
        return _done(
            ExpandStep(
                ExpandStatus.CONTINUE,
                next_state=child,
                coverage=forest.coverage,
            )
        )
