"""Request-local packed completion session (V2 kernel).

One :class:`CompletionSession` per completion-domain request.  It interns
(parser-state, semantic-state) pairs so convergent suffix states are expanded
once, shares the outgoing-edge computation (the existing
``_build_openui_completion_forest_impl`` authority) and ONE bounded
terminal-witness DP across every candidate and every position of the
request, and path-compresses forced singleton chains.

Equivalence contract (this change never widens authority):

* ``outgoing`` returns exactly the forest the scan-based reference build
  returns for the same prefix — the semantic-state views substituted inside
  the impl are parity-proven in ``tests/test_dsl/test_semantic_state.py``.
* ``terminal_witness`` keeps V1's per-candidate expansion discipline: a fresh
  ``node_budget`` (16, matching ``dsl/pack.py``'s historical ``nodes_left``)
  per top-level query, free repeats of an already-charged (state, room) pair
  within one query (V1's ``lru_cache`` behavior), and first-witness-in-path-
  order selection.  The session-wide memo stores only *budget-independent*
  results — a ``None`` produced under a depleted node budget is recomputed
  for a fresh query, never cached (the memo-poisoning fix).
* Certificates and payloads stay token-based; interned state ids are
  request-local and never escape the session.

Nothing here is keyed globally by prefixes; every cache dies with the
request.  ``TimeoutError`` from the cooperative decode deadline propagates —
it is never caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slm_training.dsl.grammar.fastpath import semantic_state as _ss
from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    _build_openui_completion_forest_impl,
    _official_schema,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import (
    decode_prefix,
    token_surface_piece,
)
from slm_training.models.decode_stats import check_decode_deadline

__all__ = ["CompletionSession"]


@dataclass
class _StateRecord:
    """One interned state: a positioned engine fork plus its semantic state."""

    engine: OpenUIIncrementalEngine
    semantic: Any
    prefix_ids: tuple[int, ...]


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
        # ``runtime_symbols`` is authority context only: it participates in
        # the pack-level request fingerprint (``dsl/pack.py``
        # ``_openui_domain_fingerprint``) and is intentionally NOT stored on
        # the session — nothing in the kernel reads it, and a dead stored
        # copy is how authority context and behavior drift apart.  Sessions
        # are request-local and constructed per request with the symbols in
        # scope, so the context is fixed by construction.
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

        self._states: list[_StateRecord] = []
        self._intern: dict[tuple[Any, Any], int] = {}
        self._transitions: dict[tuple[int, int], int | None] = {}
        self._outgoing: dict[int, CompletionForest] = {}
        # (state, room) -> budget-costing keys (in evaluation order) of a
        # proven-UNSUPPORTED exploration.  Only fully explored, complete-
        # authority None verdicts are stored (see ``terminal_witness``);
        # UNKNOWN (incomplete authority) and budget-truncated verdicts stay
        # query-local, and witness results are kept query-local.
        self._reach: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        self._witness: dict[
            tuple[int, int], tuple[tuple[int, ...], tuple[tuple[int, int], ...]]
        ] = {}
        self._forced: dict[tuple[int, int], tuple[tuple[int, ...], int, str]] = {}
        self._results: dict[tuple[int, int, str], Any] = {}
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
            "witness_cache_hits": 0,
            "witness_cache_misses": 0,
            "witness_states_expanded": 0,
            "forced_closure_hits": 0,
            "forced_closure_tokens": 0,
            "direct_terminal_feeds": 0,
            "full_sync_fallbacks": 0,
            "full_prefix_lex_bytes": 0,
            "parser_forks": 0,
            "candidate_engine_allocations": 0,
            "scope_reference_scans_avoided": 0,
            "result_cache_hits": 0,
            "result_cache_misses": 0,
        }

    # --- introspection ----------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Session instrumentation snapshot (request-local counters)."""
        return dict(self._counters)

    @property
    def unique_states(self) -> int:
        return self._counters["unique_states"]

    # --- state interning ---------------------------------------------------

    def _fork(self, engine: OpenUIIncrementalEngine) -> OpenUIIncrementalEngine:
        self._counters["parser_forks"] += 1
        return engine.copy()

    def _intern_state(
        self,
        engine: OpenUIIncrementalEngine,
        semantic: Any,
        prefix_ids: tuple[int, ...],
    ) -> int:
        key = (engine.parser_state_key(), semantic)
        state_id = self._intern.get(key)
        if state_id is not None:
            self._counters["state_intern_hits"] += 1
            return state_id
        state_id = len(self._states)
        self._intern[key] = state_id
        self._states.append(_StateRecord(engine, semantic, prefix_ids))
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
        semantic = _ss.initial_state(self._tokenizer)
        for token_id in ids:
            semantic = _ss.advance(semantic, token_id, self._tokenizer, schema=self._schema)
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
        in_sync = bool(
            state is not None
            and callable(getattr(state, "engine_in_sync", None))
            and state.engine_in_sync(list(ids), text)
        )
        if not in_sync:
            positioned.set_prefix(text)
            self._counters["full_prefix_lex_bytes"] += len(text)
        return self._intern_state(positioned, semantic, ids)

    def prefix_ids_of(self, state_id: int) -> tuple[int, ...]:
        return self._states[int(state_id)].prefix_ids

    def semantic_of(self, state_id: int) -> Any:
        return self._states[int(state_id)].semantic

    def restore_decode_state(
        self,
        state_id: int,
        state: Any,
        *,
        context: str,
        pool: dict[tuple[str, tuple[int, ...]], tuple[object, int]] | None,
    ) -> None:
        """Restore a decoder row from an interned state without prefix replay."""
        record = self._states[int(state_id)]
        state.engine = self._fork(record.engine)
        state.prefix_ids = list(record.prefix_ids)
        state.prefix_text = decode_prefix(self._tokenizer, list(record.prefix_ids))
        state.engine_ids_len = len(record.prefix_ids)
        state.completion_session = self
        state.completion_session_context = str(context)
        state.completion_state_id = int(state_id)
        state.completion_stats_snapshot = self.stats()
        state.completion_session_pool = pool
        state.clear_position_memo()

    def cached_result(self, state_id: int, room: int, fingerprint: str) -> Any | None:
        result = self._results.get((int(state_id), int(room), str(fingerprint)))
        self._counters[
            "result_cache_hits" if result is not None else "result_cache_misses"
        ] += 1
        return result

    def cache_result(
        self, state_id: int, room: int, fingerprint: str, result: Any
    ) -> Any:
        self._results[(int(state_id), int(room), str(fingerprint))] = result
        return result

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
        child = self._fork(record.engine)
        outcome = child.feed_token_id(self._tokenizer, int(token_id))
        if outcome is None:
            # Ambiguous junction — canonical text route.
            self._counters["full_sync_fallbacks"] += 1
            piece = token_surface_piece(self._tokenizer, int(token_id))
            self._counters["full_prefix_lex_bytes"] += len(piece)
            outcome = child.advance_checked(piece)
        elif outcome:
            self._counters["direct_terminal_feeds"] += 1
        if not outcome:
            self._transitions[key] = None
            return None
        semantic = _ss.advance(
            record.semantic, int(token_id), self._tokenizer, schema=self._schema
        )
        child_id = self._intern_state(
            child, semantic, record.prefix_ids + (int(token_id),)
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
        forest = _build_openui_completion_forest_impl(
            self._tokenizer,
            list(record.prefix_ids),
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
        self._counters["edges_built"] += 1
        self._counters["scope_reference_scans_avoided"] += scan_counter["avoided"]
        if state is None:
            self._outgoing[state_id] = forest
        return forest

    # --- bounded terminal-witness DP -----------------------------------------

    def terminal_witness_proof(
        self, state_id: int, room: int
    ) -> tuple[tuple[int, ...] | None, bool]:
        """Return a witness and whether a missing witness was fully proved.

        One bounded proof shared across all candidates/positions of the
        request.  Per-query discipline is exactly V1's (``dsl/pack.py``'s
        reference ``_tail_from``): a fresh node budget, and a per-query
        LRU memo (capacity 64, like the reference's ``lru_cache``) whose
        evictions are behavior-visible — an evicted key is re-evaluated and
        re-charged.  The session-wide memo stores only *proven-UNSUPPORTED*
        entries: a ``None`` whose exploration was complete — every expanded
        node had ``coverage == "complete"`` and no expansion was truncated by
        budget.  Such a verdict is exact for its ``room`` and for every
        *smaller* room (reachability is monotone in budget), which is why
        replaying it can never change a verdict — but it is NOT exact for a
        *larger* room, so the memo is keyed by the exact ``(state, room)``
        pair and must never be widened to a ``≤ room`` key.  Each entry
        carries the budget-costing keys its exploration evaluated, in
        evaluation order, so a hit charges exactly what V1's fresh evaluation
        would have charged and refreshes the query LRU in the same order.  A
        ``None`` produced under a depleted budget or under incomplete
        authority (UNKNOWN, never UNSUPPORTED — verified-scope-solver law)
        is never stored, and witness results stay query-local because a
        cached witness replayed under a smaller fresh budget could prove
        more than V1.
        """
        nodes_left = self._node_budget
        # Per-query LRU, capacity 64 — mirrors the reference's
        # ``@lru_cache(maxsize=64)`` including eviction re-charges.
        query_cache: dict[tuple[int, int], tuple[tuple[int, ...] | None, bool]] = {}
        _QUERY_CACHE_CAP = 64

        def _qget(
            key: tuple[int, int]
        ) -> tuple[tuple[int, ...] | None, bool] | None:
            value = query_cache.get(key)
            if value is not None:
                query_cache[key] = query_cache.pop(key)  # refresh recency
            return value

        def _qput(
            key: tuple[int, int], value: tuple[tuple[int, ...] | None, bool]
        ) -> None:
            if key in query_cache:
                query_cache.pop(key)
            query_cache[key] = value
            while len(query_cache) > _QUERY_CACHE_CAP:
                query_cache.pop(next(iter(query_cache)))

        def _eval(
            sid: int, rm: int
        ) -> tuple[tuple[int, ...] | None, bool, tuple[tuple[int, int], ...]]:
            """Return (witness|None, proven, budget-costing keys in eval order)."""
            nonlocal nodes_left
            check_decode_deadline()
            key = (sid, rm)
            hit = _qget(key)
            if hit is not None:
                return hit[0], hit[1], ()
            if rm <= 0:
                # Budget-independent verdict, zero node cost in V1 (the
                # reference checks ``room`` before decrementing).
                _qput(key, (None, True))
                return None, True, ()
            if nodes_left <= 0:
                # Depleted-query artifact: never enters the session memo.
                _qput(key, (None, False))
                return None, False, ()
            shared_witness = self._witness.get(key)
            if shared_witness is not None:
                self._counters["witness_cache_hits"] += 1
                witness, taxed_keys = shared_witness
                charged: list[tuple[int, int]] = []
                for taxed_key in taxed_keys:
                    if _qget(taxed_key) is not None:
                        continue
                    if nodes_left <= 0:
                        _qput(key, (None, False))
                        return None, False, tuple(charged)
                    nodes_left -= 1
                    _qput(taxed_key, (None, True))
                    charged.append(taxed_key)
                _qput(key, (witness, True))
                return witness, True, tuple(charged)
            self._counters["witness_cache_misses"] += 1
            shared = self._reach.get(key)
            if shared is not None:
                self._counters["reachability_cache_hits"] += 1
                charged = []
                for taxed_key in shared:
                    if _qget(taxed_key) is not None:
                        continue
                    if nodes_left <= 0:
                        # V1 spent the fresh budget before completing this
                        # exact replay: query-local UNKNOWN.
                        _qput(key, (None, False))
                        return None, False, tuple(charged)
                    nodes_left -= 1
                    _qput(taxed_key, (None, True))
                    charged.append(taxed_key)
                return None, True, tuple(charged)
            self._counters["reachability_cache_misses"] += 1
            self._counters["witness_states_expanded"] += 1
            nodes_left -= 1
            forest = self.outgoing(sid)
            if forest.coverage != "complete":
                # UNKNOWN, not UNSUPPORTED: partial authority proves nothing
                # either way (verified-scope-solver law — unknown never
                # removes).  The verdict is query-local only; it must NEVER
                # enter the session memo, or a transiently incomplete
                # authority becomes a session-permanent, cross-candidate
                # false UNSUPPORTED.  A later query with repaired authority
                # re-expands this node.
                _qput(key, (None, False))
                return None, False, ()
            proven = True
            taxed: list[tuple[int, int]] = [key]
            for path in forest.paths:
                tokens = tuple(int(token_id) for token_id in path.token_ids)
                if not tokens:
                    raise ValueError("completion graph edges must consume a token")
                if len(tokens) > rm:
                    continue
                if path.kind == "eos":
                    _qput(key, (tokens, True))
                    stored = tuple(taxed)
                    self._witness[key] = (tokens, stored)
                    return tokens, True, stored
                child = self.advance_path(sid, tokens)
                if child is None:
                    continue
                witness, child_proven, child_taxed = _eval(child, rm - len(tokens))
                if witness is not None:
                    result = tokens + witness
                    _qput(key, (result, True))
                    stored = tuple([*taxed, *child_taxed])
                    self._witness[key] = (result, stored)
                    return result, True, stored
                proven = proven and child_proven
                taxed.extend(child_taxed)
            _qput(key, (None, proven))
            if proven:
                stored = tuple(taxed)
                self._reach[key] = stored
                return None, True, stored
            return None, False, ()

        witness, proven, _ = _eval(int(state_id), int(room))
        return witness, proven

    def terminal_witness(self, state_id: int, room: int) -> tuple[int, ...] | None:
        """First-in-path-order terminal witness within ``room`` tokens, or None."""
        return self.terminal_witness_proof(state_id, room)[0]

    # --- forced singleton closure ---------------------------------------------

    def forced_closure(
        self, state_id: int, room: int
    ) -> tuple[tuple[int, ...], int, str]:
        """Maximal exact singleton chain from ``state_id``.

        Follows only complete-coverage single-path forests; stops at a branch,
        an EOS path, a literal boundary, incomplete authority, or budget.
        Returns ``(token_ids, terminal_state_id, coverage)``.
        """
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
            if len(tokens) + len(path.token_ids) > int(room):
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
            remaining = int(room) - len(tokens) - len(path.token_ids)
            if self.terminal_witness(nxt, remaining) is None:
                break
            tokens.extend(int(token_id) for token_id in path.token_ids)
            current = nxt
        result = (tuple(tokens), current, coverage)
        self._forced[key] = result
        self._counters["forced_closure_tokens"] += len(tokens)
        return result


_MISSING = object()
