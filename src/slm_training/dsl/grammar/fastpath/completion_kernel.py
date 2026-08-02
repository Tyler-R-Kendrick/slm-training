"""Request-local packed completion session (V2 kernel).

One :class:`CompletionSession` per completion-domain request.  It interns
(parser-state, semantic-state) pairs so convergent suffix states are expanded
once, shares the outgoing-edge computation (the existing
``_build_openui_completion_forest_impl`` authority), and path-compresses
forced singleton chains.

Equivalence contract (this change never widens authority):

* ``outgoing`` returns exactly the forest the scan-based reference build
  returns for the same prefix — the semantic-state views substituted inside
  the impl are parity-proven in ``tests/test_dsl/test_semantic_state.py``.
* ``terminal_witness`` returns a typed proof using the reference builder's
  first-path order and per-candidate search allowance.  Each query uses a
  bounded LRU only for states encountered during that query.  Incomplete
  coverage and exhausted search authority are UNKNOWN, never silently
  collapsed into UNSUPPORTED.
* Certificates and payloads stay token-based; interned state ids are
  request-local and never escape the session.

Nothing here is keyed globally by prefixes; every cache dies with the
request.  ``TimeoutError`` from the cooperative decode deadline propagates —
it is never caught.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from slm_training.dsl.grammar.fastpath import semantic_state as _ss
from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    _build_openui_completion_forest_direct,
    _official_schema,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import (
    decode_prefix,
    token_surface_piece,
)
from slm_training.models.decode_stats import check_decode_deadline

__all__ = [
    "CompletionSession",
    "WitnessResult",
    "WitnessStatus",
]


class WitnessStatus(str, Enum):
    """Authority status for a bounded terminal-reachability query."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WitnessResult:
    """Typed bounded reachability result.

    ``witness`` is populated only for :attr:`WitnessStatus.SUPPORTED`.
    """

    status: WitnessStatus
    witness: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.status is WitnessStatus.SUPPORTED and not self.witness:
            raise ValueError("SUPPORTED requires a non-empty witness")
        if self.status is not WitnessStatus.SUPPORTED and self.witness:
            raise ValueError(f"{self.status.value} cannot carry a witness")

    @property
    def supported(self) -> bool:
        return self.status is WitnessStatus.SUPPORTED


@dataclass
class _StateRecord:
    """One interned state: a positioned engine fork plus its semantic state."""

    engine: OpenUIIncrementalEngine
    semantic: Any
    prefix_ids: tuple[int, ...] | None
    parent_state_id: int | None = None
    token_id: int | None = None


class CompletionSession:
    """Interned, request-local completion graph over one authority context.

    The authority context (tokenizer identity, slot contract, content floor,
    path bound, runtime symbols, schema flags, explain) is fixed at
    construction — states from different contexts never share a session, so
    no cache key needs to repeat it.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        slot_contract: tuple[str, ...] | list[str] = (),
        min_content: int = 0,
        max_path_tokens: int = 8,
        runtime_symbols: tuple[Any, ...] = (),
        enforce_schema_component_types: bool = False,
        reserved_content_slots: int = 0,
        reserve_unresolved_content: bool = False,
        explain: bool = False,
        node_budget: int = 16,
    ) -> None:
        # Every hard-authority input is owned by this request-local object;
        # no state or proof cache crosses contexts.
        self._tokenizer = tokenizer
        self._slot_contract = tuple(slot_contract or ())
        self._min_content = int(min_content)
        self._max_path_tokens = int(max_path_tokens)
        self._enforce_schema_component_types = bool(enforce_schema_component_types)
        self._reserved_content_slots = int(reserved_content_slots)
        self._reserve_unresolved_content = bool(reserve_unresolved_content)
        self._explain = bool(explain)
        self._node_budget = max(1, int(node_budget))
        self._schema = _official_schema()
        self._semantic_arena = _ss.SemanticArena()
        self._runtime_symbols = tuple(runtime_symbols)

        self._states: list[_StateRecord] = []
        self._intern: dict[tuple[Any, Any], int] = {}
        self._transitions: dict[tuple[int, int], int | None] = {}
        self._outgoing: dict[int, CompletionForest] = {}
        # Exact top-level witness queries recur across sibling completion
        # domains.  Cache only a fully returned verdict for the identical
        # (interned state, room) request.  Nested subproblems remain
        # query-local so the historical per-candidate node allowance and
        # first-path ordering are unchanged; UNKNOWN and interrupted queries
        # are never persisted.
        self._witness_roots: dict[tuple[int, int], WitnessResult] = {}
        self._forced: dict[tuple[int, int], tuple[tuple[int, ...], int, str]] = {}
        self._counters: dict[str, int] = {
            "session_starts": 1,
            "state_intern_hits": 0,
            "state_intern_misses": 0,
            "unique_states": 0,
            "edges_built": 0,
            "transition_cache_hits": 0,
            "transition_cache_misses": 0,
            "domain_cache_hits": 0,
            "domain_cache_misses": 0,
            "reachability_cache_hits": 0,
            "reachability_cache_misses": 0,
            "witness_states_expanded": 0,
            "forced_closure_hits": 0,
            "forced_closure_tokens": 0,
            "direct_terminal_feeds": 0,
            "full_sync_fallbacks": 0,
            "full_prefix_lex_bytes": 0,
            "parser_forks": 0,
            "candidate_engine_allocations": 0,
            "scope_reference_scans_avoided": 0,
        }

    # --- introspection ----------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Session instrumentation snapshot (request-local counters)."""
        return dict(self._counters)

    def close(self) -> None:
        """Release request-owned parser, semantic, and proof state eagerly."""
        self._states.clear()
        self._intern.clear()
        self._transitions.clear()
        self._outgoing.clear()
        self._witness_roots.clear()
        self._forced.clear()
        self._semantic_arena.clear()

    def __del__(self) -> None:
        # Completion-domain requests are short-lived.  Eagerly sever Lark's
        # parser/value-stack object graph rather than waiting for cyclic GC.
        try:
            self.close()
        except Exception:
            pass

    @property
    def unique_states(self) -> int:
        return self._counters["unique_states"]

    # --- state interning ---------------------------------------------------

    def _fork(self, engine: OpenUIIncrementalEngine) -> OpenUIIncrementalEngine:
        self._counters["parser_forks"] += 1
        return engine.copy_control()

    def _intern_state(
        self,
        engine: OpenUIIncrementalEngine,
        semantic: Any,
        prefix_ids: tuple[int, ...] | None,
        *,
        parent_state_id: int | None = None,
        token_id: int | None = None,
    ) -> int:
        key = (engine.parser_state_key(), semantic)
        state_id = self._intern.get(key)
        if state_id is not None:
            self._counters["state_intern_hits"] += 1
            return state_id
        if prefix_ids is None and (parent_state_id is None or token_id is None):
            raise ValueError("a deferred prefix requires its parent state and token")
        state_id = len(self._states)
        self._intern[key] = state_id
        self._states.append(
            _StateRecord(
                engine,
                semantic,
                prefix_ids,
                parent_state_id=parent_state_id,
                token_id=token_id,
            )
        )
        self._counters["state_intern_misses"] += 1
        self._counters["unique_states"] += 1
        return state_id

    def seed(
        self,
        prefix_ids: tuple[int, ...] | list[int],
        *,
        state: Any | None = None,
        engine: Any | None = None,
    ) -> int:
        """Intern the session's initial state for ``prefix_ids``.

        Reuses ``state.engine`` (a ``GrammarDecodeState``) or an explicit
        pre-positioned ``engine`` via fork; only a session with neither
        constructs a fresh ``OpenUIIncrementalEngine`` (counted as a
        candidate-engine allocation).
        """
        ids = tuple(int(token_id) for token_id in prefix_ids)
        bos_id = int(self._tokenizer.bos_id)
        forbidden = {
            int(self._tokenizer.pad_id),
            int(self._tokenizer.mask_id),
            int(self._tokenizer.eos_id),
            int(self._tokenizer.unk_id),
        }
        for index, token_id in enumerate(ids):
            if token_id in forbidden:
                raise ValueError(
                    f"special token {token_id} is not legal in a completion prefix"
                )
            if token_id == bos_id and index != 0:
                raise ValueError("BOS is legal only at prefix position zero")
        semantic = _ss.initial_state(
            self._tokenizer, arena=self._semantic_arena
        )
        for token_id in ids:
            semantic = _ss.advance(
                semantic,
                token_id,
                self._tokenizer,
                schema=self._schema,
                arena=self._semantic_arena,
            )
        holder_engine = getattr(state, "engine", None) if state is not None else None
        if isinstance(holder_engine, OpenUIIncrementalEngine):
            positioned = self._fork(holder_engine)
        elif isinstance(engine, OpenUIIncrementalEngine):
            positioned = self._fork(engine)
        else:
            self._counters["candidate_engine_allocations"] += 1
            positioned = OpenUIIncrementalEngine()
        if state is not None and callable(getattr(state, "sync_ids", None)):
            text = state.sync_ids(self._tokenizer, list(ids))
        else:
            text = decode_prefix(self._tokenizer, list(ids))
        if not positioned.set_prefix(text):
            raise ValueError("completion prefix is rejected by the grammar")
        if not getattr(positioned, "_control_only", False):
            positioned = positioned.copy_control()
        self._counters["full_prefix_lex_bytes"] += len(text)
        return self._intern_state(positioned, semantic, ids)

    def prefix_ids_of(self, state_id: int) -> tuple[int, ...]:
        return self._materialize_prefix(int(state_id))

    def _materialize_prefix(self, state_id: int) -> tuple[int, ...]:
        """Materialize one persistent prefix only when an authority scan needs it."""
        record = self._states[int(state_id)]
        if record.prefix_ids is not None:
            return record.prefix_ids
        suffix: list[int] = []
        cursor = int(state_id)
        while True:
            ancestor = self._states[cursor]
            if ancestor.prefix_ids is not None:
                prefix = ancestor.prefix_ids + tuple(reversed(suffix))
                record.prefix_ids = prefix
                return prefix
            if ancestor.parent_state_id is None or ancestor.token_id is None:
                raise RuntimeError("deferred completion prefix has no concrete ancestor")
            suffix.append(int(ancestor.token_id))
            cursor = int(ancestor.parent_state_id)

    def semantic_of(self, state_id: int) -> Any:
        return self._states[int(state_id)].semantic

    def restore(self, state_id: int) -> int:
        """Validate and return an immutable interned state id for rollback."""
        sid = int(state_id)
        if sid < 0 or sid >= len(self._states):
            raise IndexError(f"completion state id out of range: {sid}")
        return sid

    def accepts_eos(self, state_id: int) -> bool:
        """Whether the exact constrained domain admits EOS at ``state_id``."""
        eos_id = int(self._tokenizer.eos_id)
        return any(
            path.kind == "eos" and tuple(path.token_ids) == (eos_id,)
            for path in self.outgoing(int(state_id)).paths
        )

    # --- transitions --------------------------------------------------------

    def advance(self, state_id: int, token_id: int) -> int | None:
        """Advance one token from ``state_id``; interned, memoized, fail closed.

        Returns the child state id, or ``None`` when the grammar rejects the
        token (the pre-feed engine state was restored exactly).
        """
        key = (int(state_id), int(token_id))
        cached = self._transitions.get(key, _MISSING)
        if cached is not _MISSING:
            self._counters["transition_cache_hits"] += 1
            return cached
        self._counters["transition_cache_misses"] += 1
        record = self._states[int(state_id)]
        tid = int(token_id)
        if tid in {
            int(self._tokenizer.pad_id),
            int(self._tokenizer.mask_id),
            int(self._tokenizer.bos_id),
            int(self._tokenizer.eos_id),
            int(self._tokenizer.unk_id),
        }:
            # EOS is a terminal certificate, not a parser transition.  Query
            # it with accepts_eos(); every other special is always illegal.
            self._transitions[key] = None
            return None
        kind_of = getattr(self._tokenizer, "kind_of", None)
        raw_kind = kind_of(tid) if callable(kind_of) else None
        kind = str(getattr(raw_kind, "value", ""))
        if kind == "macro":
            expand = getattr(self._tokenizer, "expand_macros", None)
            expanded = list(expand([tid])) if callable(expand) else []
            if not expanded or expanded == [tid]:
                self._transitions[key] = None
                return None
        child = self._fork(record.engine)
        # ``child`` is already an isolated control fork and is discarded when
        # the feed rejects.  Avoid cloning it a second time solely to restore
        # a branch that cannot be observed again.
        outcome = child.feed_token_id(
            self._tokenizer,
            tid,
            _restore_on_reject=False,
            _token_kind=raw_kind,
        )
        if outcome is None:
            # Ambiguous junction — canonical text route.
            self._counters["full_sync_fallbacks"] += 1
            piece = token_surface_piece(self._tokenizer, tid)
            self._counters["full_prefix_lex_bytes"] += len(piece)
            outcome = child.advance_checked(piece)
        elif outcome:
            self._counters["direct_terminal_feeds"] += 1
        if not outcome:
            self._transitions[key] = None
            return None
        semantic = _ss.advance(
            record.semantic,
            tid,
            self._tokenizer,
            schema=self._schema,
            arena=self._semantic_arena,
            _token_kind=kind,
        )
        child_id = self._intern_state(
            child,
            semantic,
            None,
            parent_state_id=int(state_id),
            token_id=tid,
        )
        self._transitions[key] = child_id
        return child_id

    def advance_path(
        self, state_id: int, token_ids: tuple[int, ...] | list[int]
    ) -> int | None:
        """Fold ``token_ids`` through :meth:`advance`; ``None`` on rejection."""
        current: int | None = int(state_id)
        for token_id in token_ids:
            if current is None:
                return None
            current = self.advance(current, int(token_id))
        return current

    # --- outgoing edges ------------------------------------------------------

    def outgoing(self, state_id: int, *, state: Any | None = None) -> CompletionForest:
        """The exact completion forest at ``state_id`` (shared per session).

        ``state`` (a ``GrammarDecodeState``) is honored only for the seeded
        initial build so its ``sync_ids`` side effect matches the reference
        authority path byte-for-byte; such builds are never cached (same rule
        as the memoizing front).
        """
        check_decode_deadline()
        state_id = int(state_id)
        if state is None:
            cached = self._outgoing.get(state_id)
            if cached is not None:
                self._counters["domain_cache_hits"] += 1
                return cached
        self._counters["domain_cache_misses"] += 1
        record = self._states[state_id]
        engine = None if state is not None else self._fork(record.engine)
        scan_counter: dict[str, int] = {"avoided": 0}
        forest = _build_openui_completion_forest_direct(
            self._tokenizer,
            list(self._materialize_prefix(state_id)),
            state=state,
            engine=engine,
            slot_contract=list(self._slot_contract),
            max_path_tokens=self._max_path_tokens,
            min_content=self._min_content,
            enforce_schema_component_types=self._enforce_schema_component_types,
            reserved_content_slots=self._reserved_content_slots,
            reserve_unresolved_content=self._reserve_unresolved_content,
            explain=self._explain,
            semantic_state=record.semantic,
            scan_counter=scan_counter,
        )
        check_decode_deadline()
        self._counters["edges_built"] += 1
        self._counters["scope_reference_scans_avoided"] += scan_counter["avoided"]
        if state is None:
            self._outgoing[state_id] = forest
        return forest

    # --- bounded terminal-witness DP -----------------------------------------

    def terminal_witness(self, state_id: int, room: int) -> WitnessResult:
        """Return V1's first-in-path-order witness with typed uncertainty.

        Each top-level candidate receives the historical 16-node allowance
        and a 64-entry query-local LRU. This preserves the reference's
        behavior-visible expansion order. Results affected by partial
        authority or budget depletion are UNKNOWN and never enter the
        session-wide memo.
        """
        sid = int(state_id)
        rm = max(0, int(room))
        check_decode_deadline()
        root_key = (sid, rm)
        root_cached = self._witness_roots.get(root_key)
        if root_cached is not None:
            self._counters["reachability_cache_hits"] += 1
            return root_cached

        nodes_left = self._node_budget
        query_cache: OrderedDict[tuple[int, int], WitnessResult] = OrderedDict()

        def _eval(current: int, remaining: int) -> WitnessResult:
            nonlocal nodes_left
            check_decode_deadline()
            key = (int(current), int(remaining))
            cached = query_cache.get(key)
            if cached is not None:
                query_cache.move_to_end(key)
                self._counters["reachability_cache_hits"] += 1
                return cached
            if remaining <= 0:
                result = WitnessResult(WitnessStatus.UNSUPPORTED)
            elif nodes_left <= 0:
                result = WitnessResult(WitnessStatus.UNKNOWN)
            else:
                nodes_left -= 1
                self._counters["reachability_cache_misses"] += 1
                self._counters["witness_states_expanded"] += 1
                forest = self.outgoing(current)
                saw_unknown = forest.coverage != "complete"
                result = WitnessResult(WitnessStatus.UNSUPPORTED)
                for path in forest.paths:
                    tokens = tuple(int(token_id) for token_id in path.token_ids)
                    if not tokens or len(tokens) > remaining:
                        continue
                    if path.kind == "eos":
                        result = WitnessResult(WitnessStatus.SUPPORTED, tokens)
                        break
                    child = self.advance_path(current, tokens)
                    if child is None:
                        continue
                    suffix = _eval(child, remaining - len(tokens))
                    if suffix.status is WitnessStatus.SUPPORTED:
                        result = WitnessResult(
                            WitnessStatus.SUPPORTED, tokens + suffix.witness
                        )
                        break
                    saw_unknown |= suffix.status is WitnessStatus.UNKNOWN
                else:
                    if saw_unknown:
                        result = WitnessResult(WitnessStatus.UNKNOWN)
            query_cache[key] = result
            query_cache.move_to_end(key)
            if len(query_cache) > 64:
                query_cache.popitem(last=False)
            return result

        result = _eval(sid, rm)
        if result.status is not WitnessStatus.UNKNOWN:
            self._witness_roots[root_key] = result
        return result

    # --- forced singleton closure ---------------------------------------------

    def forced_closure(
        self, state_id: int, room: int
    ) -> tuple[tuple[int, ...], int, str]:
        """Maximal exact singleton chain from ``state_id``.

        Follows only complete-coverage single-path forests; stops at a branch,
        an EOS path, a literal boundary, incomplete authority, or budget.
        Returns ``(token_ids, terminal_state_id, coverage)``.
        """
        check_decode_deadline()
        key = (int(state_id), int(room))
        cached = self._forced.get(key)
        if cached is not None:
            self._counters["forced_closure_hits"] += 1
            return cached
        tokens: list[int] = []
        current = int(state_id)
        coverage = "complete"
        while len(tokens) < int(room):
            check_decode_deadline()
            forest = self.outgoing(current)
            coverage = forest.coverage
            if coverage != "complete" or len(forest.paths) != 1:
                break
            path = forest.paths[0]
            if path.kind == "eos" or not path.token_ids:
                break
            remaining = int(room) - len(tokens)
            if len(path.token_ids) > remaining:
                break
            if any(
                str(self._tokenizer.id_to_token.get(int(token_id), ""))
                in {"LIT_STR", "LIT_NUM", "LIT_END"}
                for token_id in path.token_ids
            ):
                break  # literal boundary: never compress across opaque content
            nxt = self.advance_path(current, path.token_ids)
            if nxt is None:
                break
            tokens.extend(int(token_id) for token_id in path.token_ids)
            current = nxt
        result = (tuple(tokens), current, coverage)
        check_decode_deadline()
        self._forced[key] = result
        self._counters["forced_closure_tokens"] += len(tokens)
        return result


_MISSING = object()
