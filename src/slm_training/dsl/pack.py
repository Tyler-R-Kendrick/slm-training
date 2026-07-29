"""F1 (SLM-34): the DSL-pack contract.

A **DSL pack** bundles everything the training stack needs to treat a language
as a first-class target, mirroring the pack-contract table in
``docs/design/design-patterns-dsl.md`` field-for-field:

==================  ========================================================
Pack slot           Field on :class:`DslPack`
==================  ========================================================
grammar             ``backend`` (:class:`GrammarBackend`: parse / serialize /
                    schema / structural tokens / stream checks)
validity oracle     ``oracle`` (record- or source-level verdict; for OpenUI
                    the G0–G12 gate stack in ``data/verify/stack.py``)
typed-AST generator ``corpus_generator`` (factory returning a seeded
                    coverage-guided generator)
canonicalizer       ``canonicalize`` (confluent codec round-trip, NOT an
                    e-graph — see ``dsl/canonicalize.py``)
scope rules         ``scope_extractor`` (AST-derived scope slices)
placeholder policy  ``placeholder_policy`` (:class:`PlaceholderPolicy`)
==================  ========================================================

Plus honesty metadata required by F3/F4:

* ``reward_label`` — what the oracle actually measures (e.g.
  ``"well_formed_not_behavioral"``): rewards derived from the oracle must be
  labeled with this string, never presented as behavioral correctness.
* Slots are :class:`Protocol`-typed callables/objects, not concrete Lark
  paths, so the F4 ontology variant (grammar → graph-walk constraint,
  oracle → ontology reasoner) can fill the same slots.

Packs may be **partial**: a slot a language genuinely does not provide yet is
``None``, and :meth:`DslPack.require` fails closed with a message naming the
pack and the missing slot. ``toy-layout`` is the shipped partial example.

The registry here does not duplicate the grammar-backend registry
(``dsl/grammar/backends``): the ``backend`` slot references it, and pack
resolution follows the same ``SLM_GRAMMAR_DSL`` / ``active_dsl()`` convention.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, fields, replace
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    runtime_checkable,
)

from slm_training.dsl.grammar.backends import GrammarBackend, get_backend
from slm_training.dsl.placeholders import (
    CONTENT_PROPS,
    PLACEHOLDER_RE,
    extract_placeholders,
    is_placeholder,
)

if TYPE_CHECKING:
    from slm_training.dsl.grammar_capabilities import GrammarCapabilityAuthorityV1


class PackSlotUnavailable(NotImplementedError):
    """A pack was asked for a slot it does not (yet) provide."""


@runtime_checkable
class Canonicalizer(Protocol):
    def __call__(self, source: str) -> str: ...


@runtime_checkable
class ValidityOracle(Protocol):
    """Record- or source-level validity verdict (see ``reward_label``)."""

    def __call__(self, record: Any, /) -> Any: ...


@runtime_checkable
class OperatorLibrary(Protocol):
    registry_fingerprint: str


@dataclass(frozen=True)
class PlaceholderPolicy:
    """How user content is kept out of the program surface.

    ``slot_contract`` maps a source (plus optional declared inventory) to the
    ordered placeholder inventory used by codec slot pointers.
    """

    placeholder_re: re.Pattern[str]
    content_props: frozenset[str]
    slot_contract: Callable[..., tuple[str, ...]]
    is_placeholder: Callable[[str], bool] = is_placeholder
    extract: Callable[[str], list[str]] = extract_placeholders


@dataclass(frozen=True)
class DslPack:
    """One language's full pack. ``None`` slots are honest gaps — use
    :meth:`require` to fail closed with a clear message."""

    pack_id: str
    backend: GrammarBackend
    placeholder_policy: PlaceholderPolicy
    reward_label: str
    # Tree-edit inventory is pack-owned. Empty tuples mean the pack does not
    # support that variant; consumers must fail closed instead of borrowing
    # OpenUI's alphabet.
    leaf_components: tuple[str, ...] = ()
    container_components: tuple[str, ...] = ()
    component_property_domains: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    statement_templates: tuple[tuple[str, str], ...] = ()
    canonicalize: Canonicalizer | None = None
    oracle: ValidityOracle | None = None
    corpus_generator: Callable[..., Any] | None = None
    scope_extractor: Callable[..., list[Any]] | None = None
    prop_order: Callable[[], Mapping[str, Sequence[str]]] | None = None
    incremental_engine: Callable[[], Any] | None = None
    capsule_problem_builder: Callable[..., Any] | None = None
    capsule_summary_extractor: Callable[..., Any] | None = None
    capsule_materializer: Callable[..., Any] | None = None
    capsule_local_oracle: Callable[..., Any] | None = None
    capsule_global_oracle: Callable[..., Any] | None = None
    opaque_region_extractor: Callable[..., Any] | None = None
    fragment_parser: Callable[..., Any] | None = None
    region_splicer: Callable[..., Any] | None = None
    operator_library: OperatorLibrary | None = None
    grammar_capability_authority: GrammarCapabilityAuthorityV1 | None = None
    # Request-independent completion authority. Scope/semantic overlays remain
    # live and pack-owned; loaders must certify this artifact before use.
    completion_artifact: str | None = None

    def filled_slots(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self) if getattr(self, f.name) is not None)

    def require(self, slot: str) -> Any:
        """Return the named slot, failing closed when the pack omits it."""
        if slot not in {f.name for f in fields(self)}:
            raise AttributeError(f"unknown pack slot {slot!r}")
        value = getattr(self, slot)
        if value is None:
            raise PackSlotUnavailable(
                f"DSL pack {self.pack_id!r} does not provide slot {slot!r}; "
                f"filled slots: {sorted(self.filled_slots())}"
            )
        return value


_PACKS: dict[str, DslPack] = {}
_BUILTINS_LOADED: bool = False
# Backend ids that resolve to the openui pack (same language, different parser).
_ALIASES = {
    "openui-lark": "openui",
    "openui-langcore": "openui",
    "lark-openui": "openui",
    "default": "openui",
    "auto": "openui",
}


def register_pack(pack: DslPack) -> DslPack:
    _PACKS[pack.pack_id] = pack
    return pack


def list_packs() -> list[str]:
    _ensure_builtin_packs()
    return sorted(_PACKS)


def _active_dsl() -> str:
    try:
        from slm_training.models.grammar import active_dsl

        return active_dsl()
    except Exception:  # noqa: BLE001 - torch-free contexts fall back to env
        return os.getenv("SLM_GRAMMAR_DSL") or "openui"


def get_pack(dsl: str | None = None) -> DslPack:
    """Resolve a pack by id, following ``SLM_GRAMMAR_DSL`` when ``dsl`` is None."""
    _ensure_builtin_packs()
    key = (dsl or _active_dsl()).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _PACKS:
        raise KeyError(f"unknown DSL pack {dsl!r}; known={sorted(_PACKS)}")
    return _PACKS[key]


# --------------------------------------------------------------------------
# Builtin packs. Slot bodies import lazily to keep dsl/ importable without
# torch and to avoid data/ <-> dsl/ import cycles.
# --------------------------------------------------------------------------


def _openui_canonicalize(source: str) -> str:
    from slm_training.dsl.canonicalize import canonicalize

    return canonicalize(source, dsl="openui")


def _openui_oracle(record: Any, context: Any = None) -> Any:
    """G0-G12 gate-stack verdict. Accepts an ExampleRecord or raw source."""
    from slm_training.data.verify import verify_record
    from slm_training.dsl.schema import ExampleRecord

    if isinstance(record, str):
        record = ExampleRecord(
            id="pack-oracle",
            prompt="pack oracle probe",
            openui=record,
            placeholders=extract_placeholders(record),
            split="train",
            source="fixture",
        )
    return verify_record(record, context)


def _openui_generator(config: Any = None, *, seed: int = 0) -> Any:
    from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator

    return ProgramGenerator(config or GeneratorConfig(), seed=seed)


def _openui_scope_extractor(source: str, **kwargs: Any) -> list[Any]:
    from slm_training.data.scope_extract import extract_scope_slices

    return extract_scope_slices(source, dsl="openui", **kwargs)


def _openui_fragment_parser(
    source: str, kind: str, grammar_category: str | None = None
) -> str:
    """Validate a typed fragment behind the OpenUI pack boundary."""
    from slm_training.data.scope_extract import TERMINAL_CATEGORIES
    from slm_training.dsl.parser import validate_output

    category = (
        TERMINAL_CATEGORIES.get(str(grammar_category), str(grammar_category).lower())
        if grammar_category is not None
        else None
    )
    return validate_output(source, kind, category)  # type: ignore[arg-type]


def _openui_prop_order() -> Mapping[str, Sequence[str]]:
    from slm_training.dsl.production_codec import _prop_order

    return _prop_order("openui")


def _openui_engine() -> Any:
    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine

    return OpenUIIncrementalEngine()


def _openui_completion_frontier(prefix: str) -> frozenset[str]:
    engine = _openui_engine()
    if not engine.set_prefix(prefix):
        return frozenset()
    return engine.next_terminals()


def _openui_completion_domain(request: Any) -> Any:
    """Build OpenUI's scoped finite domain and prove each action reaches EOS.

    This is deliberately pack-owned: component schema, binder scope, literal
    framing, and the OpenUI static language contract do not leak into the
    grammar-agnostic decoder.

    The kind-ids (lexer-native) branch runs through one request-local packed
    :class:`~slm_training.dsl.grammar.fastpath.completion_kernel.CompletionSession`:
    interned states, shared outgoing edges, and bounded terminal-witness
    queries, with exactly the reference's external contract and per-candidate
    expansion discipline. ``_openui_completion_domain_reference`` also
    remains the compatibility implementation for tokenizers without native
    kind ids.
    """
    from slm_training.models.decode_stats import check_decode_deadline

    check_decode_deadline()
    if not callable(getattr(request.tokenizer, "kind_ids", None)):
        # Word-tokenizer exports (including ONNX) do not carry the lexer-native
        # binder/component inventory — the witness-projection branch is
        # unchanged and shared with the reference.
        return _openui_completion_domain_reference(request)
    from slm_training.dsl.grammar.fastpath.completion_kernel import (
        CompletionSession,
        WitnessStatus,
    )
    from slm_training.dsl.grammar_capabilities import (
        CompletionDomainCandidateV1,
        CompletionDomainV1,
    )

    prefix = tuple(int(token_id) for token_id in request.prefix_ids)
    budget = int(request.remaining_tokens) if request.remaining_tokens is not None else 64
    fingerprint = _openui_domain_fingerprint(request, prefix, budget)
    # Budget and prefix vary as a row advances; every other input is hard
    # authority and must match before a request-local session may be reused.
    authority_key = (
        _openui_tokenizer_authority_fingerprint(request.tokenizer),
        tuple(
            tuple(str(token) for token in expansion)
            for expansion in getattr(request.tokenizer, "macro_expansions", ())
        ),
        tuple(request.slot_contract),
        int(request.min_content),
        int(request.max_path_tokens),
        hashlib.sha256(
            json.dumps(
                tuple(request.runtime_symbols),
                default=str,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        bool(request.explain),
    )
    row_state = request.state
    session = (
        getattr(row_state, "completion_session", None)
        if getattr(row_state, "completion_authority_key", None) == authority_key
        else None
    )
    if not isinstance(session, CompletionSession):
        session = CompletionSession(
            request.tokenizer,
            slot_contract=tuple(request.slot_contract),
            min_content=request.min_content,
            max_path_tokens=request.max_path_tokens,
            runtime_symbols=tuple(request.runtime_symbols),
            explain=request.explain,
        )
    seed_id = None
    if row_state is not None and session is getattr(
        row_state, "completion_session", None
    ):
        seed_id = getattr(row_state, "completion_prefix_states", {}).get(prefix)
        current_id = getattr(row_state, "completion_state_id", None)
        if current_id is not None:
            try:
                if session.prefix_ids_of(int(current_id)) == prefix:
                    seed_id = int(current_id)
            except (IndexError, KeyError, TypeError, ValueError):
                seed_id = None
    if seed_id is None:
        seed_id = session.seed(
            prefix,
            state=(
                row_state
                if session is not getattr(row_state, "completion_session", None)
                else None
            ),
        )
    if row_state is not None and callable(
        getattr(row_state, "bind_completion_session", None)
    ):
        row_state.bind_completion_session(
            session, authority_key, int(seed_id), prefix
        )
    batch_cache = getattr(row_state, "completion_batch_cache", None)
    batch_key = (authority_key, prefix, budget)
    row_cache = getattr(row_state, "completion_domain_cache", None)
    row_cache_key = ("packed_completion_domain", *batch_key)
    if row_cache is not None:
        cached_domain = row_cache.get(row_cache_key)
        if cached_domain is not None:
            check_decode_deadline()
            return cached_domain
    if batch_cache is not None:
        cached_domain = batch_cache.domains.get(batch_key)
        from slm_training.models.decode_stats import get_active_stats

        stats = get_active_stats()
        if cached_domain is not None:
            if stats is not None:
                stats.completion_shared_domain_hits += 1
            check_decode_deadline()
            return cached_domain
        if stats is not None:
            stats.completion_shared_domain_misses += 1

    def _finish(domain: Any) -> Any:
        check_decode_deadline()
        if row_state is not None and callable(
            getattr(row_state, "_collect_completion_stats", None)
        ):
            row_state._collect_completion_stats()
        if batch_cache is not None:
            batch_cache.domains[batch_key] = domain
        if row_cache is not None:
            row_cache[row_cache_key] = domain
        if row_state is None:
            session.close()
        return domain

    initial = session.outgoing(int(seed_id))
    if initial.coverage != "complete":
        return _finish(
            CompletionDomainV1(
                status="incomplete",
                scope_fingerprint=fingerprint,
                terminals=initial.terminals,
                reason="openui_completion_forest_incomplete",
            )
        )

    candidates: list[Any] = []
    unsupported = False
    for path in initial.paths:
        tokens = tuple(int(token_id) for token_id in path.token_ids)
        if not tokens or len(tokens) > budget:
            unsupported = True
            continue
        if path.kind == "eos":
            witness = tokens
        else:
            child = session.advance_path(seed_id, tokens)
            if child is None:
                unsupported = True
                continue
            proof = session.terminal_witness(child, budget - len(tokens))
            if proof.status is WitnessStatus.UNKNOWN:
                # Preserve V1's behavior-visible per-candidate expansion
                # discipline: an unproven candidate is omitted while proven
                # siblings remain available. UNKNOWN itself is never cached
                # as a negative kernel fact.
                unsupported = True
                continue
            if proof.status is WitnessStatus.UNSUPPORTED:
                unsupported = True
                continue
            suffix = tuple(int(token_id) for token_id in proof.witness)
            witness = tokens + suffix
            if not suffix:
                unsupported = True
                continue
        candidates.append(
            CompletionDomainCandidateV1(
                token_ids=tokens,
                kind=path.kind,
                terminal_witness=witness,
            )
        )
    if not candidates:
        return _finish(
            CompletionDomainV1(
                status="incomplete",
                scope_fingerprint=fingerprint,
                terminals=initial.terminals,
                reason="terminal_witness_unavailable",
            )
        )
    return _finish(
        CompletionDomainV1(
            status="complete",
            candidates=tuple(candidates),
            scope_fingerprint=fingerprint,
            terminals=initial.terminals,
            reason="witness_pruned" if unsupported else "",
        )
    )


def _openui_domain_fingerprint(request: Any, prefix: tuple[int, ...], budget: int) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "prefix": prefix,
                "runtime_symbols": request.runtime_symbols,
                "slot_contract": request.slot_contract,
                "tokenizer_version": int(getattr(request.tokenizer, "version", 0)),
                "macro_expansions": getattr(
                    request.tokenizer, "macro_expansions", ()
                ),
                "remaining_tokens": budget,
                "max_path_tokens": request.max_path_tokens,
                "min_content": request.min_content,
            },
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _openui_tokenizer_authority_fingerprint(tokenizer: Any) -> str:
    """Content identity for completion authority shared across batch rows."""
    tokenizer_type = type(tokenizer)
    return hashlib.sha256(
        json.dumps(
            {
                "type": f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}",
                "version": int(getattr(tokenizer, "version", 0)),
                "pad_id": int(getattr(tokenizer, "pad_id", -1)),
                "bos_id": int(getattr(tokenizer, "bos_id", -1)),
                "eos_id": int(getattr(tokenizer, "eos_id", -1)),
                "mask_id": int(getattr(tokenizer, "mask_id", -1)),
                "unk_id": int(getattr(tokenizer, "unk_id", -1)),
                "id_to_token": sorted(
                    (int(token_id), str(token))
                    for token_id, token in getattr(tokenizer, "id_to_token", {}).items()
                ),
                "id_to_kind": sorted(
                    (int(token_id), str(kind))
                    for token_id, kind in getattr(tokenizer, "id_to_kind", {}).items()
                ),
                "sym_slots": int(getattr(tokenizer, "sym_slots", 0)),
                "bind_slots": int(getattr(tokenizer, "bind_slots", 0)),
                "state_slots": int(getattr(tokenizer, "state_slots", 0)),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _openui_completion_domain_reference(request: Any) -> Any:
    """Pre-kernel reference and non-native-tokenizer compatibility path.

    Native OpenUI production uses the packed kernel. Tokenizers without
    lexer-native kind ids retain this fail-closed witness implementation;
    tests and benchmarks also call it directly for byte-for-byte parity.
    """
    from functools import lru_cache

    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        _build_openui_completion_forest,
    )
    from slm_training.dsl.grammar_capabilities import (
        CompletionDomainCandidateV1,
        CompletionDomainV1,
    )

    prefix = tuple(int(token_id) for token_id in request.prefix_ids)
    budget = int(request.remaining_tokens) if request.remaining_tokens is not None else 64
    fingerprint = _openui_domain_fingerprint(request, prefix, budget)
    if not callable(getattr(request.tokenizer, "kind_ids", None)):
        # Word-tokenizer exports (including ONNX) do not carry the lexer-native
        # binder/component inventory.  Project declared, grammar-validated pack
        # witnesses into that tokenizer instead of falling back to broad DFA
        # terminals or decoder-side component-name policy.
        source_prefix = prefix
        bos_id = getattr(request.tokenizer, "bos_id", None)
        if source_prefix and source_prefix[0] == bos_id:
            source_prefix = source_prefix[1:]
        significant_prefix = tuple(
            token_id
            for token_id in source_prefix
            if not request.tokenizer.decode([token_id]).isspace()
        )
        candidates_by_token: dict[int, Any] = {}
        for item in _openui_witness_candidates():
            if not significant_prefix and not item.source.lstrip().startswith("root"):
                continue
            encoded = tuple(
                int(token_id)
                for token_id in request.tokenizer.encode(item.source, add_special=False)
            )
            consumed = 0
            if significant_prefix:
                seen: list[int] = []
                for index, token_id in enumerate(encoded):
                    if not request.tokenizer.decode([token_id]).isspace():
                        seen.append(token_id)
                    if tuple(seen) == significant_prefix:
                        consumed = index + 1
                        break
                    if len(seen) >= len(significant_prefix):
                        break
                if tuple(seen) != significant_prefix:
                    continue
            tail = encoded[consumed:] + (int(request.tokenizer.eos_id),)
            if not tail or len(tail) > budget:
                continue
            # Whitespace is semantically ignorable for this grammar.  Project
            # both it and the next significant witness token so a proven
            # structural force (for example `=` after `root`) is not rejected
            # merely because a canonical witness happened to include a space.
            for offset, token_id in enumerate(tail):
                candidates_by_token.setdefault(
                    token_id,
                    CompletionDomainCandidateV1(
                        token_ids=(token_id,),
                        kind="witness_projection",
                        terminal_witness=tail[offset:],
                    ),
                )
                piece = request.tokenizer.decode([token_id])
                if not piece.isspace():
                    break
        if not candidates_by_token:
            return CompletionDomainV1(
                status="incomplete",
                scope_fingerprint=fingerprint,
                reason="tokenizer_projection_has_no_witness",
            )
        return CompletionDomainV1(
            status="complete",
            candidates=tuple(candidates_by_token.values()),
            scope_fingerprint=fingerprint,
        )
    initial = _build_openui_completion_forest(
        request.tokenizer,
        list(prefix),
        state=request.state,
        slot_contract=list(request.slot_contract),
        max_path_tokens=request.max_path_tokens,
        min_content=request.min_content,
        explain=request.explain,
    )
    if initial.coverage != "complete":
        return CompletionDomainV1(
            status="incomplete",
            scope_fingerprint=fingerprint,
            terminals=initial.terminals,
            reason="openui_completion_forest_incomplete",
        )

    # A completion domain is exact only when every exposed action has an actual
    # terminal continuation inside this request's remaining decode budget.
    # The search is over the pack's own finite forest, never model logits.
    #
    # Structural reuse, shared across every ``_tail_from`` proof below:
    # - ``_tail_forests`` memoizes the *stateless* forest for a node.  The
    #   build is a pure function of (node prefix, slot contract, path bounds,
    #   schema identity) — all fixed for this request by
    #   ``_official_schema()`` — so a cached forest is identical to a rebuild.
    #   Only stateless tail builds are memoized; the stateful initial build
    #   above is a different authority path and is never mixed into this cache.
    # - ``_tail_engines`` holds one engine per node, forked from its parent
    #   node (``OpenUIIncrementalEngine.copy``) so the child build feeds only
    #   the delta lexemes instead of re-lexing+re-feeding the whole prefix.
    #   Every build still calls ``set_prefix`` on the full decoded node text,
    #   and the engine falls back to a full sync whenever the fed lexemes do
    #   not extend cleanly, so each node's forest is identical to the
    #   from-scratch build.
    _tail_forests: dict[tuple[int, ...], Any] = {}
    _tail_engines: dict[tuple[int, ...], Any] = {}

    def _tail_forest(current: tuple[int, ...]) -> Any:
        forest = _tail_forests.get(current)
        if forest is None:
            forest = _build_openui_completion_forest(
                request.tokenizer,
                list(current),
                engine=_tail_engines.get(current),
                slot_contract=list(request.slot_contract),
                max_path_tokens=request.max_path_tokens,
                min_content=request.min_content,
            )
            _tail_forests[current] = forest
        return forest

    def _tail_from(
        start: tuple[int, ...], remaining: int
    ) -> tuple[int, ...] | None:
        # This is a decode-time proof, not an unbounded solver campaign.
        # Exhaustion is incomplete and refuses the position; it never becomes a
        # model-vocabulary fallback.  Each candidate receives the same bounded
        # chance to establish a witness, so an early complex branch cannot
        # starve a later simple one.
        nodes_left = 16
        _tail_engines.setdefault(tuple(start), _openui_engine())

        @lru_cache(maxsize=64)
        def _tail(current: tuple[int, ...], room: int) -> tuple[int, ...] | None:
            nonlocal nodes_left
            if room <= 0 or nodes_left <= 0:
                return None
            nodes_left -= 1
            forest = _tail_forest(current)
            if forest.coverage != "complete":
                return None
            for path in forest.paths:
                tokens = tuple(int(token_id) for token_id in path.token_ids)
                if not tokens or len(tokens) > room:
                    continue
                if path.kind == "eos":
                    return tokens
                child = current + tokens
                if child not in _tail_engines:
                    parent_engine = _tail_engines.get(current)
                    _tail_engines[child] = (
                        parent_engine.copy()
                        if parent_engine is not None
                        else _openui_engine()
                    )
                suffix = _tail(child, room - len(tokens))
                if suffix is not None:
                    return tokens + suffix
            return None

        return _tail(start, remaining)

    candidates: list[Any] = []
    unwitnessed = False
    for path in initial.paths:
        tokens = tuple(int(token_id) for token_id in path.token_ids)
        if not tokens or len(tokens) > budget:
            unwitnessed = True
            continue
        witness = (
            tokens
            if path.kind == "eos"
            else tokens + (_tail_from(prefix + tokens, budget - len(tokens)) or ())
        )
        if not witness or (path.kind != "eos" and len(witness) == len(tokens)):
            unwitnessed = True
            continue
        candidates.append(
            CompletionDomainCandidateV1(
                token_ids=tokens,
                kind=path.kind,
                terminal_witness=witness,
            )
        )
    if not candidates:
        return CompletionDomainV1(
            status="incomplete",
            scope_fingerprint=fingerprint,
            terminals=initial.terminals,
            reason="terminal_witness_unavailable",
        )
    return CompletionDomainV1(
        status="complete",
        candidates=tuple(candidates),
        scope_fingerprint=fingerprint,
        terminals=initial.terminals,
        reason="witness_pruned" if unwitnessed else "",
    )


def _openui_witness_candidates() -> tuple[Any, ...]:
    from slm_training.data.contract import RuntimeSymbol
    from slm_training.dsl.grammar_capabilities import GrammarWitnessCandidateV1

    base = 'root = TextContent(":w.text")'
    two_statements = 'item = TextContent(":w.text")\nroot = Stack([item], "column")'
    sources = [
        base,
        f"{base}\n",
        f"\n\n{base}",
        "root = Separator()",
        two_statements,
        'hero = TextContent(":smoke.hero.title")\nroot = Stack([hero], "column")',
        f"{two_statements}\n",
        '$s = true ? false : true\nroot = TextContent(":w.text")',
        *(
            f'$s = true {operator} false\nroot = TextContent(":w.text")'
            for operator in (
                "||",
                "&&",
                "==",
                "!=",
                ">=",
                "<=",
                ">",
                "<",
                "+",
                "-",
                "*",
                "/",
                "%",
            )
        ),
        '$s = !true\nroot = TextContent(":w.text")',
        '$s = -true\nroot = TextContent(":w.text")',
        '$s = {text: true}.text\nroot = TextContent(":w.text")',
        '$s = {Stack: true}.Stack\nroot = TextContent(":w.text")',
        '$s = [true][false]\nroot = TextContent(":w.text")',
        '$s = []\nroot = TextContent(":w.text")',
        '$s = [true, false]\nroot = TextContent(":w.text")',
        '$s = {}\nroot = TextContent(":w.text")',
        (
            '$s = {text: true, Stack: false, ":w.key": null}\n'
            'root = TextContent(":w.text")'
        ),
        '$s = "column"\nroot = TextContent(":w.text")',
        '$s = true\nroot = TextContent(":w.text")',
        '$s = null\nroot = TextContent(":w.text")',
        '$s = $s\nroot = TextContent(":w.text")',
        ('item = TextContent(":w.text")\n$s = item\nroot = Stack([item], "column")'),
        '$s = (true)\nroot = TextContent(":w.text")',
        'root = Stack([], "column", true)',
    ]
    candidates = []
    for source in sources:
        symbols = [
            RuntimeSymbol(surface=surface, role="external_entity")
            for surface in sorted(set(extract_placeholders(source)))
        ]
        symbols.extend(
            RuntimeSymbol(surface=surface, role="state")
            for surface in sorted(set(re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", source)))
        )
        candidates.append(
            GrammarWitnessCandidateV1(
                source=source,
                runtime_symbols=tuple(symbols),
            )
        )
    return tuple(candidates)


def _openui_grammar_capability_authority() -> GrammarCapabilityAuthorityV1:
    from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
    from slm_training.dsl.grammar_capabilities import (
        lark_authority,
        production_id,
    )

    backend = get_backend("openui")
    authority = lark_authority(
        grammar_path=GRAMMARS_DIR / "openui.lark",
        start_symbols=("start",),
        canonical_serialize=_openui_canonicalize,
        static_validate=backend.validate,
        scope_policy=_openui_scope_extractor,
        completion_frontier=_openui_completion_frontier,
        completion_domain=_openui_completion_domain,
        witness_candidates=_openui_witness_candidates,
    )
    unsupported: dict[str, str] = {}
    for production in authority.productions or ():
        lhs = str(production.lhs)
        rhs = tuple((symbol.kind, str(symbol.name)) for symbol in production.rhs)
        reason = None
        if lhs == "start" and (not rhs or rhs == (("nonterminal", "__start_star_0"),)):
            reason = "STATIC_SEMANTICS_REQUIRES_ROOT"
        elif lhs == "__start_star_0" and rhs[:1] == (
            ("nonterminal", "__start_star_0"),
        ):
            reason = "LEXER_COLLAPSES_REPEATED_NEWLINES"
        elif lhs == "primary" and rhs == (("terminal", "NUMBER"),):
            reason = "SYMBOLIC_SURFACE_FORBIDS_OPEN_NUMBER"
        elif lhs == "call_name" and rhs == (("terminal", "BUILTIN"),):
            reason = "SYMBOLIC_SURFACE_HAS_NO_DECLARED_BUILTIN"
        if reason is not None:
            unsupported[production_id(production)] = reason
    return replace(authority, unsupported_alternatives=unsupported)


def _openui_slot_contract(
    source: str, *, declared: Iterable[str] | None = None
) -> tuple[str, ...]:
    from slm_training.data.contract import canonical_slot_contract

    return canonical_slot_contract(source, declared=declared)


def _openui_opaque_region_extractor(source: str) -> tuple[Any, ...]:
    """Classify user-facing content placeholders as opaque CONTENT_VALUE regions."""
    import hashlib

    from slm_training.dsl.opaque_regions import (
        OpaqueRegion,
        OpaqueRegionKind,
        OpaqueRegionSummary,
    )
    from slm_training.dsl.placeholders import CONTENT_PROPS, extract_placeholders

    regions: list[Any] = []
    for placeholder in extract_placeholders(source):
        prop = placeholder.lstrip(":").split(".")[-1]
        if prop in CONTENT_PROPS:
            digest = hashlib.sha256(placeholder.encode("utf-8")).hexdigest()[:16]
            regions.append(
                OpaqueRegion(
                    region_id=f"openui:content:{placeholder}",
                    kind=OpaqueRegionKind.CONTENT_VALUE,
                    placeholder=placeholder,
                    source_digest=digest,
                    summary=OpaqueRegionSummary(),
                )
            )
    return tuple(regions)


def _toy_layout_scope_extractor(source: str, **kwargs: Any) -> list[Any]:
    from slm_training.data.scope_extract import extract_scope_slices

    return extract_scope_slices(source, dsl="toy-layout", **kwargs)


def _toy_layout_prop_order() -> Mapping[str, Sequence[str]]:
    backend = get_backend("toy-layout")
    order = getattr(backend, "prop_order", None)
    if not callable(order):  # pragma: no cover - LarkFileBackend always has it
        raise PackSlotUnavailable("toy-layout backend exposes no prop order")
    return order()


def _toy_layout_engine() -> Any:
    from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine

    return OpenUIIncrementalEngine(GRAMMARS_DIR / "toy_layout.lark")


def _ensure_builtin_packs() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    shared_policy = PlaceholderPolicy(
        placeholder_re=PLACEHOLDER_RE,
        content_props=CONTENT_PROPS,
        slot_contract=_openui_slot_contract,
    )
    register_pack(
        DslPack(
            pack_id="openui",
            backend=get_backend("openui"),
            placeholder_policy=shared_policy,
            # F3 honesty: the G0-G12 stack proves well-formedness (grammar,
            # schema, references, canonical idempotence) — NOT behavior.
            # Runtime/behavior gates only fire when evidence is supplied.
            reward_label="well_formed_not_behavioral",
            leaf_components=("TextContent", "Button", "Image", "TextInput"),
            container_components=("Stack", "Card", "Form"),
            component_property_domains={
                component: {"rest": (', "column"', "")}
                for component in ("Stack", "Card", "Form")
            },
            statement_templates=(
                ("Query", '"tool", {arg: $x}, {default: []}, 15'),
                ("Mutation", '"tool", {arg: $x}'),
            ),
            canonicalize=_openui_canonicalize,
            oracle=_openui_oracle,
            corpus_generator=_openui_generator,
            scope_extractor=_openui_scope_extractor,
            prop_order=_openui_prop_order,
            incremental_engine=_openui_engine,
            opaque_region_extractor=_openui_opaque_region_extractor,
            fragment_parser=_openui_fragment_parser,
            grammar_capability_authority=_openui_grammar_capability_authority(),
            completion_artifact=("resources/decode/openui_completion_v1.safetensors"),
        )
    )
    # Partial pack: toy-layout genuinely fills grammar, scope rules,
    # placeholder policy, prop order, and the incremental decode engine.
    # It has no canonicalizer, oracle, or typed-AST generator yet — those
    # slots fail closed via DslPack.require.
    register_pack(
        DslPack(
            pack_id="toy-layout",
            backend=get_backend("toy-layout"),
            placeholder_policy=shared_policy,
            reward_label="parse_only",
            leaf_components=("text", "button"),
            container_components=("row", "col"),
            scope_extractor=_toy_layout_scope_extractor,
            prop_order=_toy_layout_prop_order,
            incremental_engine=_toy_layout_engine,
        )
    )
    # F2 (SLM-43): GraphQL — graphql-js oracle, the schema IS the symbol table.
    # Registered even when the Node bridge is absent; backend.available()
    # reports the bridge state and oracle/canonicalize fail at call time, not
    # registration (mirrors toy-layout's partial-pack pattern).
    try:
        from slm_training.dsl.graphql_pack import build_graphql_pack

        register_pack(build_graphql_pack())
    except Exception:  # noqa: BLE001 - graphql pack is optional
        pass
    _BUILTINS_LOADED = True


__all__ = [
    "Canonicalizer",
    "DslPack",
    "PackSlotUnavailable",
    "OperatorLibrary",
    "PlaceholderPolicy",
    "ValidityOracle",
    "get_pack",
    "list_packs",
    "register_pack",
]
