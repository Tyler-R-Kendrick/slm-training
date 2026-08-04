"""Certified static control + Γ leaf filters (claim family A / E1–E4).

Production domains are obtained only after the completion artifact checker
accepts the request-independent projection. Scope and semantic (Γ) constraints
remain request-local leaf filters that may only **tighten** the static domain.
Lark remains the live parser authority. ``StaticLalrAdapter`` is an executable
projection of the certified control arrays, but it is import-only: it becomes
usable evidence only after ``require_certified_static_lalr`` proves lockstep
accepts-set equality against the live ``InteractiveParser`` over the canonical
corpus, and no decode path consumes it. Certificate payload is never rebranded
as a DPDA executor.

See ``docs/design/adr-constrained-diffusion-topology-split.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from slm_training.dsl.grammar.fastpath.completion_artifact import (
    StaticLalrAdapter,
    load_checked_completion_artifact,
)
from slm_training.dsl.grammar.fastpath.completion_kernel import CompletionSession
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import dsl_direct_terminal_map
from slm_training.dsl.grammar_capabilities import (
    CompletionDomainRequestV1,
    CompletionDomainV1,
)

__all__ = [
    "DomainParityReport",
    "DomainSnapshot",
    "ForcedMacroEdge",
    "STATIC_LALR_CORPUS",
    "StaticLalrCertificate",
    "assert_gamma_only_tightens",
    "candidate_multiset",
    "certify_static_lalr_adapter",
    "compare_domain_parity",
    "dynamic_forced_closure",
    "e9_warm_hard_prefix_classification",
    "extract_static_forced_macro",
    "production_domain",
    "production_domain_snapshot",
    "require_certified_static_lalr",
    "require_certified_static_projection",
    "static_forced_macro_edge",
    "static_lalr_adapter",
    "walk_static_forced_macro",
]


# E9: durable label for the warm hard-prefix ~89.8× row. Never "AOT cold".
E9_WARM_HARD_PREFIX_LABEL = "request_local_memo_reuse"
E9_WARM_HARD_PREFIX_CLAIM_CLASS = "fixture_or_scratch"
E9_WARM_HARD_PREFIX_ROW_ID = "warm_card_open_comma"
HARD_PREFIX_SURFACE = "root = Card([b1,"


@dataclass(frozen=True)
class DomainSnapshot:
    """Multiset + coverage view used for E1 parity (timing-free)."""

    status: str
    coverage: str
    reason: str
    candidates: tuple[tuple[int, ...], ...]
    candidate_kinds: tuple[str, ...]
    terminals: tuple[str, ...]
    authority: str
    gamma_applied: bool
    gamma_only_tightens: bool
    certified_projection: bool
    unknown_collapsed_to_unsupported: bool = False

    def multiset_digest(self) -> str:
        payload = "|".join(
            f"{kind}:{','.join(str(t) for t in tokens)}"
            for kind, tokens in sorted(
                zip(self.candidate_kinds, self.candidates, strict=True)
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DomainParityReport:
    """E1 pass/fail without mixing in latency claims."""

    equal: bool
    production: DomainSnapshot
    oracle: DomainSnapshot
    multiset_match: bool
    coverage_match: bool
    status_match: bool
    no_unknown_collapse: bool
    gamma_only_tightens: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ForcedMacroEdge:
    """AOT/static forced singleton chain certified against dynamic closure."""

    token_ids: tuple[int, ...]
    terminal_state_id: int
    coverage: str
    matched_dynamic: bool
    stopped_on: str  # branch | eos | literal | incomplete | budget | empty


def e9_warm_hard_prefix_classification() -> dict[str, str]:
    """Canonical labels for the warm hard-prefix speedup (E9)."""

    return {
        "row_id": E9_WARM_HARD_PREFIX_ROW_ID,
        "label": E9_WARM_HARD_PREFIX_LABEL,
        "claim_class": E9_WARM_HARD_PREFIX_CLAIM_CLASS,
        "not": "aot_cold",
        "prefix": HARD_PREFIX_SURFACE,
    }


def require_certified_static_projection(
    tokenizer: Any,
    *,
    engine: OpenUIIncrementalEngine | None = None,
) -> dict[str, Any]:
    """Fail closed unless the checked artifact projection matches live authority.

    Returns the certified ``direct_map``. Raises ``ValueError`` on any checker
    rejection — production never silently widens to an unchecked map.
    """

    eng = engine if engine is not None else OpenUIIncrementalEngine()
    checked = load_checked_completion_artifact(
        tokenizer, eng._parser, grammar_path=eng.grammar_path
    )
    live = dsl_direct_terminal_map(tokenizer, eng._parser.terminals)
    if live is None or checked.direct_map != live:
        raise ValueError("certified static projection does not match live authority")
    return checked.direct_map


# Mirrors the canonical E1 corpus exercised by
# ``tests/test_dsl/test_static_control_domain.py`` (complete programs, partial
# programs, and the hard prefix).
STATIC_LALR_CORPUS: tuple[str, ...] = (
    'root = TextContent(":slot_0")\n',
    "root = Card([])\n",
    'root = Card([b1, b2])\nb1 = TextContent(":slot_0")\nb2 = TextContent(":slot_1")\n',
    "root = Card([b1",
    HARD_PREFIX_SURFACE,
    "root =",
    "root",
    "",
)


@dataclass(frozen=True)
class StaticLalrCertificate:
    """Lockstep evidence that the adapter tracks the live Lark parser exactly."""

    equal: bool
    programs: int
    steps: int
    states_visited: int
    mismatches: tuple[str, ...]


def static_lalr_adapter(
    tokenizer: Any = None,
    *,
    engine: OpenUIIncrementalEngine | None = None,
) -> StaticLalrAdapter:
    """Return the adapter carried by the checked artifact (fail-closed)."""

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    eng = engine if engine is not None else OpenUIIncrementalEngine()
    tok = tokenizer if tokenizer is not None else DSLNativeTokenizer.build()
    checked = load_checked_completion_artifact(
        tok, eng._parser, grammar_path=eng.grammar_path
    )
    if checked.lalr is None:
        raise ValueError("checked artifact carries no static LALR adapter")
    return checked.lalr


def certify_static_lalr_adapter(
    adapter: StaticLalrAdapter,
    parser: Any,
    corpus: Sequence[str] = STATIC_LALR_CORPUS,
) -> StaticLalrCertificate:
    """Walk the live ``InteractiveParser`` and the adapter in lockstep.

    For every prefix of every corpus program (one prefix per lexed terminal),
    the adapter's accepts-set must equal ``InteractiveParser.accepts()`` and the
    adapter must admit the terminal the live lexer/parser just consumed.  The
    live parser stays the authority; this only reports whether the executable
    projection is faithful.
    """

    from lark.exceptions import UnexpectedInput

    mismatches: list[str] = []
    steps = 0
    visited: set[tuple[int, ...]] = set()

    for program in corpus:
        stack = adapter.start_stack()
        visited.add(stack)
        interactive = parser.parse_interactive(program)
        try:
            for token in interactive.iter_parse():
                # ``iter_parse`` yields before feeding, so both sides are at the
                # same prefix here.
                live = frozenset(interactive.accepts())
                static = adapter.accepts(stack)
                if live != static:
                    mismatches.append(
                        f"{program!r}@{steps}: accepts differ {sorted(live ^ static)}"
                    )
                    break
                advanced = adapter.step(stack, str(token.type))
                if advanced is None:
                    mismatches.append(
                        f"{program!r}@{steps}: adapter rejected {token.type}"
                    )
                    break
                stack = advanced
                visited.add(stack)
                steps += 1
            else:
                live = frozenset(interactive.accepts())
                static = adapter.accepts(stack)
                if live != static:
                    mismatches.append(
                        f"{program!r}@end: accepts differ {sorted(live ^ static)}"
                    )
        except UnexpectedInput as exc:  # pragma: no cover - corpus lexes cleanly
            mismatches.append(f"{program!r}: live parser rejected the corpus ({exc})")

    return StaticLalrCertificate(
        equal=not mismatches,
        programs=len(tuple(corpus)),
        steps=steps,
        states_visited=len({stack[-1] for stack in visited}),
        mismatches=tuple(mismatches),
    )


_STATIC_LALR_CERTIFICATES: dict[str, StaticLalrCertificate] = {}


def require_certified_static_lalr(
    tokenizer: Any = None,
    *,
    engine: OpenUIIncrementalEngine | None = None,
) -> StaticLalrCertificate:
    """Fail closed unless the executable adapter matches the live parser.

    Process-cached per artifact digest, following
    ``require_certified_static_projection``.  Raises ``ValueError`` on any
    lockstep divergence — an uncertified adapter is never handed back.
    """

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    eng = engine if engine is not None else OpenUIIncrementalEngine()
    tok = tokenizer if tokenizer is not None else DSLNativeTokenizer.build()
    checked = load_checked_completion_artifact(
        tok, eng._parser, grammar_path=eng.grammar_path
    )
    if checked.lalr is None:
        raise ValueError("checked artifact carries no static LALR adapter")
    cached = _STATIC_LALR_CERTIFICATES.get(checked.digest)
    if cached is None:
        cached = certify_static_lalr_adapter(checked.lalr, eng._parser)
        if not cached.equal:
            raise ValueError(
                "static LALR adapter is not certified against live Lark: "
                + "; ".join(cached.mismatches)
            )
        _STATIC_LALR_CERTIFICATES[checked.digest] = cached
    return cached


def candidate_multiset(
    domain: CompletionDomainV1 | DomainSnapshot,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Order-independent multiset of (kind, token_ids) pairs."""

    if isinstance(domain, DomainSnapshot):
        items = list(zip(domain.candidate_kinds, domain.candidates, strict=True))
    else:
        items = [
            (c.kind, tuple(int(t) for t in c.token_ids)) for c in domain.candidates
        ]
    return tuple(sorted(items, key=lambda item: (item[0], item[1])))


def _snapshot_from_domain(
    domain: CompletionDomainV1,
    *,
    authority: str,
    gamma_applied: bool,
    gamma_only_tightens: bool,
    certified_projection: bool,
    unknown_collapsed: bool = False,
) -> DomainSnapshot:
    coverage = "complete" if domain.status == "complete" else domain.status
    return DomainSnapshot(
        status=str(domain.status),
        coverage=coverage,
        reason=str(domain.reason or ""),
        candidates=tuple(tuple(int(t) for t in c.token_ids) for c in domain.candidates),
        candidate_kinds=tuple(str(c.kind) for c in domain.candidates),
        terminals=tuple(str(t) for t in (domain.terminals or ())),
        authority=authority,
        gamma_applied=gamma_applied,
        gamma_only_tightens=gamma_only_tightens,
        certified_projection=certified_projection,
        unknown_collapsed_to_unsupported=unknown_collapsed,
    )


def production_domain(
    request: CompletionDomainRequestV1,
    *,
    require_artifact: bool = True,
) -> CompletionDomainV1:
    """Production domain path: certified projection + packed session + Γ.

    Uses the pack-owned packed kernel (which already applies request-local
    semantic/Γ constraints through the completion forest). Optionally refuses
    to run when the artifact checker rejects.
    """

    from slm_training.dsl.pack import _openui_completion_domain

    if require_artifact:
        require_certified_static_projection(request.tokenizer)
    return _openui_completion_domain(request)


def production_domain_snapshot(
    prefix_ids: Sequence[int],
    tokenizer: Any,
    *,
    budget: int = 32,
    slot_contract: Sequence[str] = (":slot_0", ":slot_1"),
    require_artifact: bool = True,
    **kwargs: Any,
) -> DomainSnapshot:
    """E1 snapshot for the production (certified + Γ) path."""

    request = CompletionDomainRequestV1(
        prefix_ids=tuple(int(t) for t in prefix_ids),
        tokenizer=tokenizer,
        slot_contract=tuple(slot_contract),
        remaining_tokens=int(budget),
        **kwargs,
    )
    domain = production_domain(request, require_artifact=require_artifact)
    return _snapshot_from_domain(
        domain,
        authority="certified_static_plus_gamma",
        gamma_applied=True,
        gamma_only_tightens=True,
        certified_projection=bool(require_artifact),
    )


def oracle_lark_domain_snapshot(
    prefix_ids: Sequence[int],
    tokenizer: Any,
    *,
    budget: int = 32,
    slot_contract: Sequence[str] = (":slot_0", ":slot_1"),
    **kwargs: Any,
) -> DomainSnapshot:
    """Lark/reference oracle snapshot for E1 multiset equality."""

    from slm_training.dsl.pack import _openui_completion_domain_reference

    request = CompletionDomainRequestV1(
        prefix_ids=tuple(int(t) for t in prefix_ids),
        tokenizer=tokenizer,
        slot_contract=tuple(slot_contract),
        remaining_tokens=int(budget),
        **kwargs,
    )
    domain = _openui_completion_domain_reference(request)
    return _snapshot_from_domain(
        domain,
        authority="lark_reference_oracle",
        gamma_applied=True,
        gamma_only_tightens=True,
        certified_projection=False,
    )


def compare_domain_parity(
    production: DomainSnapshot,
    oracle: DomainSnapshot,
) -> DomainParityReport:
    """E1 equality: multisets, coverage/status, no UNKNOWN→UNSUPPORTED collapse."""

    reasons: list[str] = []
    multiset_match = candidate_multiset(production) == candidate_multiset(oracle)
    if not multiset_match:
        reasons.append("candidate_multiset_mismatch")
    coverage_match = production.coverage == oracle.coverage
    if not coverage_match:
        reasons.append("coverage_mismatch")
    status_match = production.status == oracle.status
    if not status_match:
        reasons.append("status_mismatch")
    # UNKNOWN must not silently become UNSUPPORTED on either side.
    no_unknown_collapse = (
        not production.unknown_collapsed_to_unsupported
        and not oracle.unknown_collapsed_to_unsupported
    )
    if production.status == "unsupported" and oracle.status == "incomplete":
        no_unknown_collapse = False
        reasons.append("unknown_collapsed_to_unsupported")
    if oracle.status == "unsupported" and production.status == "incomplete":
        # Opposite collapse is also a parity bug for E1.
        no_unknown_collapse = False
        reasons.append("oracle_unknown_collapsed")
    gamma_ok = production.gamma_only_tightens and oracle.gamma_only_tightens
    if not gamma_ok:
        reasons.append("gamma_widened")
    equal = (
        multiset_match
        and coverage_match
        and status_match
        and no_unknown_collapse
        and gamma_ok
        and production.reason == oracle.reason
    )
    if production.reason != oracle.reason:
        reasons.append("reason_mismatch")
    return DomainParityReport(
        equal=equal,
        production=production,
        oracle=oracle,
        multiset_match=multiset_match,
        coverage_match=coverage_match,
        status_match=status_match,
        no_unknown_collapse=no_unknown_collapse,
        gamma_only_tightens=gamma_ok,
        reasons=tuple(reasons),
    )


def assert_gamma_only_tightens(
    without_gamma_candidates: Iterable[tuple[int, ...]],
    with_gamma_candidates: Iterable[tuple[int, ...]],
) -> bool:
    """Γ leaf filters may only remove candidates, never add them."""

    base = {tuple(c) for c in without_gamma_candidates}
    filtered = {tuple(c) for c in with_gamma_candidates}
    return filtered.issubset(base)


_LITERAL_BOUNDARY_NAMES = frozenset({"LIT_STR", "LIT_NUM", "LIT_END"})


def dynamic_forced_closure(
    session: CompletionSession,
    state_id: int,
    room: int,
) -> tuple[tuple[int, ...], int, str]:
    """Dynamic path: session.forced_closure (literal/incomplete safe)."""

    return session.forced_closure(int(state_id), int(room))


def _path_crosses_literal(session: CompletionSession, token_ids: Sequence[int]) -> bool:
    """True when any token is a literal-boundary surface (never compress across)."""

    tok = session._tokenizer
    for token_id in token_ids:
        name = str(tok.id_to_token.get(int(token_id), ""))
        if name in _LITERAL_BOUNDARY_NAMES:
            return True
    return False


def walk_static_forced_macro(
    session: CompletionSession,
    state_id: int,
    room: int,
) -> tuple[tuple[int, ...], int, str, str]:
    """Independent AOT-style forced-macro walk — never calls ``forced_closure``.

    Returns ``(token_ids, terminal_state_id, coverage, stopped_on)``.
    Stopping rules (E4):

    * complete coverage required;
    * exactly one path;
    * stop on EOS / empty path / budget;
    * never advance a path containing LIT_STR / LIT_NUM / LIT_END;
    * stop on incomplete authority.
    """

    tokens: list[int] = []
    current = int(state_id)
    coverage = "complete"
    stopped_on = "empty"
    room = int(room)

    while len(tokens) < room:
        forest = session.outgoing(current)
        coverage = forest.coverage
        if coverage != "complete":
            stopped_on = "incomplete"
            break
        if len(forest.paths) != 1:
            stopped_on = "branch"
            break
        path = forest.paths[0]
        if path.kind == "eos" or not path.token_ids:
            stopped_on = "eos" if path.kind == "eos" else "empty"
            break
        remaining = room - len(tokens)
        if len(path.token_ids) > remaining:
            stopped_on = "budget"
            break
        if _path_crosses_literal(session, path.token_ids):
            stopped_on = "literal"
            break
        nxt = session.advance_path(current, path.token_ids)
        if nxt is None:
            stopped_on = "rejected"
            break
        tokens.extend(int(t) for t in path.token_ids)
        current = int(nxt)
        if len(tokens) >= room:
            stopped_on = "budget"
            break
    else:
        if tokens and stopped_on == "empty":
            stopped_on = "budget"

    return tuple(tokens), current, coverage, stopped_on


def extract_static_forced_macro(
    session: CompletionSession,
    state_id: int,
    room: int,
) -> ForcedMacroEdge:
    """Static walk + certify against the distinct dynamic forced_closure path.

    The walk itself never calls ``forced_closure``; comparison is a separate
    call so ``matched_dynamic`` is a real cross-implementation check, not a
    double-call of the same function.
    """

    token_ids, terminal_id, coverage, stopped_on = walk_static_forced_macro(
        session, state_id, room
    )
    dyn = dynamic_forced_closure(session, state_id, room)
    matched = dyn == (token_ids, terminal_id, coverage)
    return ForcedMacroEdge(
        token_ids=token_ids,
        terminal_state_id=int(terminal_id),
        coverage=str(coverage),
        matched_dynamic=matched,
        stopped_on=stopped_on,
    )


def static_forced_macro_edge(
    session: CompletionSession,
    state_id: int,
    room: int,
) -> ForcedMacroEdge:
    """E4 public API: static/AOT extractor certified against dynamic closure."""

    return extract_static_forced_macro(session, state_id, room)


def forest_expansion_count(session: CompletionSession) -> int:
    """Forest builds observed by the session (``edges_built``)."""

    return int(session.stats().get("edges_built", 0))
