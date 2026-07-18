"""Torch-free tri-state support oracle and replayable certificates.

This module implements VSS0-04 / SLM-60: the deterministic reference oracle that
answers whether a candidate participates in at least one bounded, verifier-accepted
completion, distinguishing a *proven* absence (``UNSUPPORTED``) from an incomplete
search (``UNKNOWN``). It is the first component permitted to emit ``UNSUPPORTED``.

The guarantee boundary, the tri-state table, the reference search order, and the
certificate/replay contract are owned by
``docs/design/verified-scope-solver.md`` (and the legality-vs-support distinction
by ``docs/design/lattice-recursive-search.md``); this module refers to those
documents rather than restating semantics.

Honesty invariants enforced here:

* ``UNKNOWN`` never becomes ``UNSUPPORTED`` -- budget/coverage/precondition gaps
  are terminal for the verdict, never converted into a proof of absence.
* ``UNSUPPORTED`` is asserted only after the reachable space is fully explored
  inside declared finite bounds *and* every observed coverage is ``complete``.
* Certificates embed no model logits, no timestamps, no secrets, and no raw
  user-region contents -- only a witness *digest* plus safe labels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleId,
    SolverBounds,
    SupportVerdict,
    _require_text,
    _strict_fields,
)

#: Certificate schema version; ``replay_support_certificate`` rejects other values.
SUPPORT_CERTIFICATE_SCHEMA_VERSION = 1

#: The deterministic, logit-independent exploration order (see the design doc).
REFERENCE_SEARCH_ORDER = "token-id-ascending.v1"

#: Coverage vocabulary shared with ``CompletionForest.coverage``.
_COVERAGE_VALUES = frozenset({"complete", "partial", "none"})


def _require_hex(value: Any, *, field: str, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{context} requires a SHA-256 hex {field}")
    return value


def _require_int(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} requires a non-negative {field}")
    return value


# --------------------------------------------------------------------------
# Injectable search seams
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifierOutcome:
    """Verifier decision for one decoded terminal plus its honest provenance.

    ``capability_available`` is ``False`` when a required pack capability is
    missing; such a path yields ``UNKNOWN``, never ``UNSUPPORTED``.
    """

    accepted: bool
    profile: str
    required_capabilities: tuple[str, ...] = ()
    capability_available: bool = True

    def __post_init__(self) -> None:
        context = "verifier outcome"
        if not isinstance(self.accepted, bool):
            raise ValueError(f"{context} accepted must be a bool")
        _require_text(self.profile, field="profile", context=context)
        caps = tuple(self.required_capabilities)
        if any(not isinstance(cap, str) or not cap for cap in caps):
            raise ValueError(f"{context} required_capabilities must be non-empty text")
        if not isinstance(self.capability_available, bool):
            raise ValueError(f"{context} capability_available must be a bool")
        object.__setattr__(self, "required_capabilities", caps)


@dataclass(frozen=True)
class ExpansionEdge:
    """One deterministic child continuation and the choice value that reaches it."""

    value: DomainValue
    state: FiniteDomainState

    def __post_init__(self) -> None:
        if not isinstance(self.value, DomainValue):
            raise ValueError("expansion edge requires a DomainValue")
        if not isinstance(self.state, FiniteDomainState):
            raise ValueError("expansion edge requires a FiniteDomainState child")


@dataclass(frozen=True)
class Expansion:
    """One node's bounded projection: child edges plus an optional terminal leaf.

    ``coverage`` mirrors ``CompletionForest.coverage``; a non-``complete`` value
    observed on any explored node forbids an ``UNSUPPORTED`` verdict.
    """

    coverage: str
    edges: tuple[ExpansionEdge, ...] = ()
    terminal: DomainValue | None = None

    def __post_init__(self) -> None:
        if self.coverage not in _COVERAGE_VALUES:
            raise ValueError("expansion coverage must be complete, partial, or none")
        edges = tuple(self.edges)
        if any(not isinstance(edge, ExpansionEdge) for edge in edges):
            raise ValueError("expansion edges must be ExpansionEdge instances")
        if self.terminal is not None and not isinstance(self.terminal, DomainValue):
            raise ValueError("expansion terminal must be a DomainValue or None")
        object.__setattr__(self, "edges", edges)


@runtime_checkable
class SupportExpander(Protocol):
    """Maps a projection state to its bounded child projections + terminal leaf."""

    def __call__(self, state: FiniteDomainState) -> Expansion: ...


@runtime_checkable
class SupportDecoder(Protocol):
    """Decodes a terminal leaf to its checkable form, or ``None`` if infeasible."""

    def __call__(self, terminal: DomainValue) -> str | None: ...


@runtime_checkable
class SupportVerifier(Protocol):
    """Validates a decoded terminal, returning an accept/reject :class:`VerifierOutcome`."""

    def __call__(self, terminal: str) -> VerifierOutcome: ...


# --------------------------------------------------------------------------
# Immutable JSON-safe result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchCounters:
    """Deterministic search accounting; no wall-clock time is ever recorded.

    ``steps`` is the injected deterministic step counter that stands in for a
    time budget (one increment per search iteration); it is never ``time.time()``.
    """

    nodes_expanded: int = 0
    tokens: int = 0
    depth: int = 0
    backtracks: int = 0
    verifier_calls: int = 0
    steps: int = 0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            _require_int(value, field=name, context="search counters")

    def to_dict(self) -> dict[str, int]:
        return {
            "nodes_expanded": self.nodes_expanded,
            "tokens": self.tokens,
            "depth": self.depth,
            "backtracks": self.backtracks,
            "verifier_calls": self.verifier_calls,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchCounters:
        expected = {
            "nodes_expanded",
            "tokens",
            "depth",
            "backtracks",
            "verifier_calls",
            "steps",
        }
        _strict_fields(data, expected, context="SearchCounters")
        return cls(**{name: data[name] for name in expected})


@dataclass(frozen=True)
class SupportQuery:
    """A support question: does ``candidate`` for ``hole_id`` reach a verified terminal?"""

    state_fingerprint: str
    hole_id: HoleId
    candidate: DomainValue

    def __post_init__(self) -> None:
        _require_hex(
            self.state_fingerprint, field="state_fingerprint", context="support query"
        )
        if not isinstance(self.hole_id, HoleId):
            raise ValueError("support query requires a HoleId")
        if not isinstance(self.candidate, DomainValue):
            raise ValueError("support query requires a DomainValue candidate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_fingerprint": self.state_fingerprint,
            "hole_id": self.hole_id.to_dict(),
            "candidate": self.candidate.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupportQuery:
        _strict_fields(
            data, {"state_fingerprint", "hole_id", "candidate"}, context="SupportQuery"
        )
        return cls(
            state_fingerprint=data["state_fingerprint"],
            hole_id=HoleId.from_dict(data["hole_id"]),
            candidate=DomainValue.from_dict(data["candidate"]),
        )


def _normalize_failure_counts(
    rows: Any, *, context: str
) -> tuple[tuple[str, int], ...]:
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for row in tuple(rows):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError(f"{context} failure_counts entries must be reason/count pairs")
        reason, count = row
        _require_text(reason, field="failure reason", context=context)
        if reason in seen:
            raise ValueError(f"{context} has duplicate failure reason {reason!r}")
        _require_int(count, field="failure count", context=context)
        seen.add(reason)
        normalized.append((reason, count))
    return tuple(sorted(normalized, key=lambda row: row[0]))


@dataclass(frozen=True)
class SupportCertificate:
    """Replayable justification for a support verdict.

    The certificate stores only replay-safe evidence: state/pack/version/bounds
    identity, the exploration trace by fingerprint, coverage observations, the
    verifier profile, structured failure counts, and -- for ``SUPPORTED`` -- a
    witness *digest* (never the raw witness). See
    ``docs/design/verified-scope-solver.md``. Cross-field honesty rules
    (e.g. ``UNSUPPORTED`` requires ``exhausted``) are enforced by
    :func:`replay_support_certificate`, not by construction, so that dishonest
    claims can be represented and then rejected on replay.
    """

    schema_version: int
    query: SupportQuery
    verdict: SupportVerdict
    problem_id: str
    pack_id: str
    constraint_version: str
    bounds: SolverBounds
    search_order: str
    explored_state_fingerprints: tuple[str, ...]
    coverage_observations: tuple[str, ...]
    verifier_profile: str
    witness_source: str | None = None
    witness_digest: str | None = None
    failure_counts: tuple[tuple[str, int], ...] = ()
    exhausted: bool = False
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        context = "support certificate"
        _require_int(self.schema_version, field="schema_version", context=context)
        if self.schema_version < 1:
            raise ValueError(f"{context} schema_version must be positive")
        if not isinstance(self.query, SupportQuery):
            raise ValueError(f"{context} requires a SupportQuery")
        object.__setattr__(self, "verdict", SupportVerdict(self.verdict))
        _require_text(self.problem_id, field="problem_id", context=context)
        _require_text(self.pack_id, field="pack_id", context=context)
        _require_text(
            self.constraint_version, field="constraint_version", context=context
        )
        if not isinstance(self.bounds, SolverBounds):
            raise ValueError(f"{context} requires SolverBounds")
        _require_text(self.search_order, field="search_order", context=context)
        explored = tuple(self.explored_state_fingerprints)
        for fingerprint in explored:
            _require_hex(fingerprint, field="explored fingerprint", context=context)
        object.__setattr__(self, "explored_state_fingerprints", explored)
        coverage = tuple(self.coverage_observations)
        if any(value not in _COVERAGE_VALUES for value in coverage):
            raise ValueError(f"{context} coverage_observations use an unknown tag")
        object.__setattr__(self, "coverage_observations", coverage)
        _require_text(self.verifier_profile, field="verifier_profile", context=context)
        if self.witness_source is not None:
            _require_text(self.witness_source, field="witness_source", context=context)
        if self.witness_digest is not None:
            _require_hex(self.witness_digest, field="witness_digest", context=context)
        if self.stop_reason is not None:
            _require_text(self.stop_reason, field="stop_reason", context=context)
        if not isinstance(self.exhausted, bool):
            raise ValueError(f"{context} exhausted must be a bool")
        object.__setattr__(
            self,
            "failure_counts",
            _normalize_failure_counts(self.failure_counts, context=context),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query": self.query.to_dict(),
            "verdict": self.verdict.value,
            "problem_id": self.problem_id,
            "pack_id": self.pack_id,
            "constraint_version": self.constraint_version,
            "bounds": self.bounds.to_dict(),
            "search_order": self.search_order,
            "explored_state_fingerprints": list(self.explored_state_fingerprints),
            "coverage_observations": list(self.coverage_observations),
            "verifier_profile": self.verifier_profile,
            "witness_source": self.witness_source,
            "witness_digest": self.witness_digest,
            "failure_counts": [[reason, count] for reason, count in self.failure_counts],
            "exhausted": self.exhausted,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupportCertificate:
        expected = {
            "schema_version",
            "query",
            "verdict",
            "problem_id",
            "pack_id",
            "constraint_version",
            "bounds",
            "search_order",
            "explored_state_fingerprints",
            "coverage_observations",
            "verifier_profile",
            "witness_source",
            "witness_digest",
            "failure_counts",
            "exhausted",
            "stop_reason",
        }
        _strict_fields(data, expected, context="SupportCertificate")
        return cls(
            schema_version=data["schema_version"],
            query=SupportQuery.from_dict(data["query"]),
            verdict=SupportVerdict(data["verdict"]),
            problem_id=data["problem_id"],
            pack_id=data["pack_id"],
            constraint_version=data["constraint_version"],
            bounds=SolverBounds.from_dict(data["bounds"]),
            search_order=data["search_order"],
            explored_state_fingerprints=tuple(data["explored_state_fingerprints"]),
            coverage_observations=tuple(data["coverage_observations"]),
            verifier_profile=data["verifier_profile"],
            witness_source=data["witness_source"],
            witness_digest=data["witness_digest"],
            failure_counts=tuple(tuple(row) for row in data["failure_counts"]),
            exhausted=data["exhausted"],
            stop_reason=data["stop_reason"],
        )


@dataclass(frozen=True)
class SupportResult:
    """A verdict, its replayable certificate, the search counters, and any witness.

    ``witness`` carries the accepted terminal for immediate downstream use; it is
    intentionally *not* persisted in the certificate, which stores only the digest.
    """

    verdict: SupportVerdict
    certificate: SupportCertificate
    counters: SearchCounters
    witness: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", SupportVerdict(self.verdict))
        if not isinstance(self.certificate, SupportCertificate):
            raise ValueError("support result requires a SupportCertificate")
        if not isinstance(self.counters, SearchCounters):
            raise ValueError("support result requires SearchCounters")
        if self.witness is not None and not isinstance(self.witness, str):
            raise ValueError("support result witness must be text or None")


# --------------------------------------------------------------------------
# Deterministic ordering helpers (logit-independent)
# --------------------------------------------------------------------------


def _value_token_ids(value: DomainValue) -> tuple[int, ...]:
    payload = value.payload
    if isinstance(payload, dict):
        raw = payload.get("token_ids")
        if isinstance(raw, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in raw
        ):
            return tuple(raw)
    return ()


def _order_key(value: DomainValue) -> tuple[tuple[int, ...], str, str]:
    """Token-id ascending, then tag, then canonical payload (REFERENCE_SEARCH_ORDER)."""
    return (_value_token_ids(value), value.tag, value.payload_json)


def _token_count(value: DomainValue) -> int:
    return len(_value_token_ids(value))


# --------------------------------------------------------------------------
# The reference enumerative backend
# --------------------------------------------------------------------------


class SupportOracle(Protocol):
    """Reference seam: classify one support query against a finite-domain state."""

    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult: ...


@dataclass(frozen=True)
class EnumerativeSupportOracle:
    """Deterministic bounded-enumeration support oracle (VSS0-04 reference backend).

    The three seams make the full tri-state logic drivable by tiny closed fixtures
    without the grammar. Real defaults wire ``build_completion_forest`` +
    ``completion_forest_state`` (expander), the production codec (decoder), and the
    configured :class:`~slm_training.dsl.pack.DslPack` oracle (verifier); build them
    with :func:`make_completion_forest_expander`, :func:`make_token_decoder`, and
    :func:`make_pack_verifier`. See ``docs/design/verified-scope-solver.md``.
    """

    expander: SupportExpander
    decoder: SupportDecoder
    verifier: SupportVerifier
    default_verifier_profile: str = "unverified"
    max_steps: int | None = None

    def __post_init__(self) -> None:
        for name in ("expander", "decoder", "verifier"):
            if not callable(getattr(self, name)):
                raise ValueError(f"support oracle {name} must be callable")
        _require_text(
            self.default_verifier_profile,
            field="default_verifier_profile",
            context="support oracle",
        )
        if self.max_steps is not None:
            _require_int(self.max_steps, field="max_steps", context="support oracle")

    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult:
        return _run_support_search(
            state,
            query,
            expander=self.expander,
            decoder=self.decoder,
            verifier=self.verifier,
            default_verifier_profile=self.default_verifier_profile,
            max_steps=self.max_steps,
        )


def _early_unknown(
    state: FiniteDomainState,
    query: SupportQuery,
    *,
    stop_reason: str,
    default_verifier_profile: str,
) -> SupportResult:
    certificate = SupportCertificate(
        schema_version=SUPPORT_CERTIFICATE_SCHEMA_VERSION,
        query=query,
        verdict=SupportVerdict.UNKNOWN,
        problem_id=state.problem_id,
        pack_id=state.pack_id,
        constraint_version=state.constraint_version,
        bounds=state.bounds,
        search_order=REFERENCE_SEARCH_ORDER,
        explored_state_fingerprints=(),
        coverage_observations=(),
        verifier_profile=default_verifier_profile,
        exhausted=False,
        stop_reason=stop_reason,
    )
    return SupportResult(
        verdict=SupportVerdict.UNKNOWN,
        certificate=certificate,
        counters=SearchCounters(),
    )


def _run_support_search(
    state: FiniteDomainState,
    query: SupportQuery,
    *,
    expander: SupportExpander,
    decoder: SupportDecoder,
    verifier: SupportVerifier,
    default_verifier_profile: str,
    max_steps: int | None,
) -> SupportResult:
    """Iterative token-id-ascending search implementing the reference invariants.

    Behaviour is specified exactly in ``docs/design/verified-scope-solver.md`` under
    "Reference support semantics" / "Default reference backend". No Python recursion
    is used; states are deduplicated by ``FiniteDomainState.fingerprint``.
    """
    if query.state_fingerprint != state.fingerprint:
        return _early_unknown(
            state,
            query,
            stop_reason="state_fingerprint_mismatch",
            default_verifier_profile=default_verifier_profile,
        )
    try:
        domain = state.domain(query.hole_id)
    except LookupError:
        return _early_unknown(
            state,
            query,
            stop_reason="hole_absent",
            default_verifier_profile=default_verifier_profile,
        )
    if query.candidate not in set(domain.values):
        return _early_unknown(
            state,
            query,
            stop_reason="candidate_absent_from_domain",
            default_verifier_profile=default_verifier_profile,
        )

    # Step 1: apply the queried candidate to the current projection.
    root = state.with_decision(query.hole_id, query.candidate)

    bounds = state.bounds
    explored: list[str] = []
    explored_set: set[str] = set()
    coverage_seen: set[str] = set()
    profiles_seen: set[str] = set()
    failures: dict[str, int] = {}
    witness: str | None = None
    witness_digest: str | None = None
    witness_source: str | None = None
    stop_reason: str | None = None
    verdict: SupportVerdict | None = None

    nodes = tokens = backtracks = verifier_calls = steps = max_depth = 0

    def bump_failure(reason: str) -> None:
        failures[reason] = failures.get(reason, 0) + 1

    # Step 2: iterative queue/stack -- no recursion. Push children reverse-sorted
    # so ``pop`` yields token-id-ascending order.
    stack: list[tuple[FiniteDomainState, int]] = [(root, 0)]
    while stack:
        steps += 1
        if max_steps is not None and steps > max_steps:
            stop_reason = "step_budget"
            break
        node, depth = stack.pop()
        if depth > bounds.max_depth:
            stop_reason = "max_depth"
            break
        fingerprint = node.fingerprint
        if fingerprint in explored_set:
            continue  # duplicate-state suppression
        explored_set.add(fingerprint)
        explored.append(fingerprint)
        nodes += 1
        if nodes > bounds.max_nodes:
            stop_reason = "max_nodes"
            break
        max_depth = max(max_depth, depth)

        expansion = expander(node)
        coverage_seen.add(expansion.coverage)

        # Steps 3-5: decode + verify the terminal; a verifier-accepted witness
        # short-circuits to SUPPORTED even if unexplored branches would be partial.
        if expansion.terminal is not None:
            tokens += _token_count(expansion.terminal)
            if tokens > bounds.max_tokens:
                stop_reason = "max_tokens"
                break
            decoded = decoder(expansion.terminal)
            if decoded is None:
                bump_failure("undecodable_terminal")
            else:
                verifier_calls += 1
                if verifier_calls > bounds.max_verifier_calls:
                    stop_reason = "max_verifier_calls"
                    break
                outcome = verifier(decoded)
                profiles_seen.add(outcome.profile)
                if not outcome.capability_available:
                    bump_failure("missing_capability")
                elif outcome.accepted:
                    witness = decoded
                    witness_digest = hashlib.sha256(decoded.encode()).hexdigest()
                    witness_source = "enumerative"
                    verdict = SupportVerdict.SUPPORTED
                    break
                else:
                    bump_failure("verifier_rejected")

        pushed = 0
        for edge in sorted(expansion.edges, key=lambda e: _order_key(e.value), reverse=True):
            tokens += _token_count(edge.value)
            if tokens > bounds.max_tokens:
                stop_reason = "max_tokens"
                break
            stack.append((edge.state, depth + 1))
            pushed += 1
        if stop_reason is not None:
            break

        if pushed == 0:
            # A dead-end (bottom or a leaf with no live continuation): retreat.
            backtracks += 1
            if backtracks > bounds.max_backtracks:
                stop_reason = "max_backtracks"
                break

    # Step 6-8: resolve the verdict. Budget/precondition stops are UNKNOWN and are
    # NEVER promoted to UNSUPPORTED. UNSUPPORTED requires full exhaustion with every
    # observed coverage complete and no undecodable/uncapable terminal.
    proof_incomplete = bool(
        (coverage_seen - {"complete"})
        or "undecodable_terminal" in failures
        or "missing_capability" in failures
    )
    exhausted = verdict is None and stop_reason is None
    if verdict is None:
        if stop_reason is not None:
            verdict = SupportVerdict.UNKNOWN
        elif proof_incomplete:
            verdict = SupportVerdict.UNKNOWN
            stop_reason = "incomplete_coverage"
        else:
            verdict = SupportVerdict.UNSUPPORTED

    verifier_profile = (
        ",".join(sorted(profiles_seen)) if profiles_seen else default_verifier_profile
    )
    counters = SearchCounters(
        nodes_expanded=nodes,
        tokens=tokens,
        depth=max_depth,
        backtracks=backtracks,
        verifier_calls=verifier_calls,
        steps=steps,
    )
    certificate = SupportCertificate(
        schema_version=SUPPORT_CERTIFICATE_SCHEMA_VERSION,
        query=query,
        verdict=verdict,
        problem_id=state.problem_id,
        pack_id=state.pack_id,
        constraint_version=state.constraint_version,
        bounds=bounds,
        search_order=REFERENCE_SEARCH_ORDER,
        explored_state_fingerprints=tuple(explored),
        coverage_observations=tuple(sorted(coverage_seen)),
        verifier_profile=verifier_profile,
        witness_source=witness_source,
        witness_digest=witness_digest,
        failure_counts=tuple(sorted(failures.items())),
        exhausted=exhausted,
        stop_reason=stop_reason,
    )
    return SupportResult(
        verdict=verdict,
        certificate=certificate,
        counters=counters,
        witness=witness,
    )


# --------------------------------------------------------------------------
# Certificate replay
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    """Structured replay outcome: ``ok`` iff ``violations`` is empty."""

    ok: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "violations": list(self.violations)}


def replay_support_certificate(
    certificate: SupportCertificate,
    state: FiniteDomainState,
    *,
    expander: SupportExpander,
    decoder: SupportDecoder,
    verifier: SupportVerifier,
    default_verifier_profile: str = "unverified",
    max_steps: int | None = None,
) -> ReplayResult:
    """Pure checker: does ``certificate`` honestly describe the deterministic search?

    Contract (``docs/design/verified-scope-solver.md``): verify state/query/
    fingerprint/pack/version/bounds identity; rerun the deterministic search;
    validate a ``SUPPORTED`` witness through the SAME verifier (digest match);
    REJECT ``UNSUPPORTED`` when ``exhausted`` is False, any coverage observation is
    not ``complete``, or any budget stop occurred; ACCEPT ``UNKNOWN`` as honest but
    never as pruning authority. Returns structured violation strings, never a bare
    bool.

    ``state`` is required because the certificate stores only the state's
    fingerprint (never its raw contents); an independent replay must be handed the
    same state it was computed against.
    """
    violations: list[str] = []

    if certificate.schema_version != SUPPORT_CERTIFICATE_SCHEMA_VERSION:
        violations.append("unsupported_schema_version")
    if certificate.search_order != REFERENCE_SEARCH_ORDER:
        violations.append("unknown_search_order")

    # Identity: the certificate must describe *this* state.
    if certificate.problem_id != state.problem_id:
        violations.append("problem_id_mismatch")
    if certificate.pack_id != state.pack_id:
        violations.append("pack_id_mismatch")
    if certificate.constraint_version != state.constraint_version:
        violations.append("stale_constraint_version")
    if certificate.bounds != state.bounds:
        violations.append("bounds_mismatch")
    if certificate.query.state_fingerprint != state.fingerprint:
        violations.append("state_fingerprint_mismatch")

    identity_ok = not violations

    # Structural honesty of a destructive (UNSUPPORTED) claim -- checked on the
    # certificate itself, before any rerun.
    if certificate.verdict is SupportVerdict.UNSUPPORTED:
        if not certificate.exhausted:
            violations.append("unsupported_not_exhausted")
        if any(tag != "complete" for tag in certificate.coverage_observations):
            violations.append("unsupported_incomplete_coverage")
        if certificate.stop_reason is not None:
            violations.append("unsupported_budget_stop")
    if certificate.verdict is SupportVerdict.SUPPORTED and certificate.witness_digest is None:
        violations.append("supported_without_witness_digest")

    # Rerun the deterministic search only when the certificate describes this state.
    if identity_ok:
        rerun = _run_support_search(
            state,
            certificate.query,
            expander=expander,
            decoder=decoder,
            verifier=verifier,
            default_verifier_profile=default_verifier_profile,
            max_steps=max_steps,
        )
        if certificate.verdict is SupportVerdict.SUPPORTED:
            if rerun.verdict is not SupportVerdict.SUPPORTED or rerun.witness is None:
                violations.append("supported_witness_not_reproduced")
            else:
                digest = hashlib.sha256(rerun.witness.encode()).hexdigest()
                if digest != certificate.witness_digest:
                    violations.append("witness_digest_mismatch")
        elif certificate.verdict is SupportVerdict.UNSUPPORTED:
            if rerun.verdict is not SupportVerdict.UNSUPPORTED:
                violations.append("unsupported_not_reproduced")
        # UNKNOWN is accepted as honest: it never authorizes pruning, so a
        # rerun that is more (or less) decisive is not a replay violation.

    return ReplayResult(ok=not violations, violations=tuple(violations))


# --------------------------------------------------------------------------
# Optional backend registry (ships only the enumerative backend)
# --------------------------------------------------------------------------

_BACKENDS: dict[str, Callable[..., SupportOracle]] = {}


def register_support_backend(name: str, factory: Callable[..., SupportOracle]) -> None:
    """Register an optional support backend factory behind the pluggable seam.

    Only the enumerative backend ships. A conforming backend must return exactly
    one of ``SUPPORTED``/``UNSUPPORTED``/``UNKNOWN`` with a replayable certificate
    and never assert ``UNSUPPORTED`` without complete-coverage exhaustion. No SMT
    backend and no new dependency is provided (see the design doc).
    """
    _require_text(name, field="name", context="support backend")
    if not callable(factory):
        raise ValueError("support backend factory must be callable")
    _BACKENDS[name] = factory


def get_support_backend(name: str) -> Callable[..., SupportOracle]:
    """Return a registered support backend factory by name."""
    if name not in _BACKENDS:
        raise KeyError(f"unknown support backend {name!r}; known={sorted(_BACKENDS)}")
    return _BACKENDS[name]


def support_backends() -> list[str]:
    return sorted(_BACKENDS)


register_support_backend("enumerative", EnumerativeSupportOracle)


# --------------------------------------------------------------------------
# Real default seams (reference wiring; lazily imported to stay Torch-free and
# not exercised by decode/generation under VSS0-04)
# --------------------------------------------------------------------------


def make_pack_verifier(*, dsl: str | None = None, pack: Any = None) -> SupportVerifier:
    """Verifier seam backed by the configured :class:`DslPack` G0-G12 oracle.

    A pack whose ``oracle`` slot is unavailable yields ``capability_available=False``
    (an ``UNKNOWN`` path), never a hard rejection. The recorded profile is the
    pack's honest ``reward_label`` (e.g. ``well_formed_not_behavioral``).
    """

    def verify(terminal: str) -> VerifierOutcome:
        from slm_training.dsl.pack import PackSlotUnavailable, get_pack

        resolved = pack if pack is not None else get_pack(dsl)
        profile = resolved.reward_label
        try:
            oracle = resolved.require("oracle")
        except PackSlotUnavailable:
            return VerifierOutcome(
                accepted=False,
                profile=profile,
                required_capabilities=("oracle",),
                capability_available=False,
            )
        report = oracle(terminal)
        return VerifierOutcome(
            accepted=bool(getattr(report, "ok", False)),
            profile=profile,
            required_capabilities=("oracle",),
            capability_available=True,
        )

    return verify


def make_token_decoder(*, tokenizer: Any) -> SupportDecoder:
    """Decoder seam that renders a terminal leaf to OpenUI source text.

    Accepts a terminal payload carrying either a decoded ``source`` string or the
    absolute ``token_ids`` of the completed program; anything else (or a decode
    failure) returns ``None`` -- an honest ``UNKNOWN`` contributor, not a rejection.
    """

    def decode(terminal: DomainValue) -> str | None:
        payload = terminal.payload
        if isinstance(payload, dict):
            source = payload.get("source")
            if isinstance(source, str):
                return source
            token_ids = payload.get("token_ids")
            if isinstance(token_ids, list):
                try:
                    return tokenizer.decode([int(item) for item in token_ids])
                except Exception:  # noqa: BLE001 - undecodable => UNKNOWN, never reject
                    return None
        return None

    return decode


class CompletionForestExpander:
    """Expander seam over ``build_completion_forest`` + ``completion_forest_state``.

    Threads the absolute prefix (which :class:`FiniteDomainState` intentionally
    elides) through an internal fingerprint map seeded at ``root_prefix``. A state
    whose prefix is unknown yields ``coverage="none"`` -- an honest ``UNKNOWN``,
    never a fabricated ``UNSUPPORTED``. This reference wiring is not invoked by
    decode/generation under VSS0-04.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        pack_id: str,
        constraint_version: str,
        bounds: SolverBounds,
        slot_contract: list[str] | None = None,
        root_prefix: tuple[int, ...] | list[int] = (),
    ) -> None:
        self._tokenizer = tokenizer
        self._pack_id = pack_id
        self._constraint_version = constraint_version
        self._bounds = bounds
        self._slot_contract = list(slot_contract) if slot_contract else None
        self._prefixes: dict[str, tuple[int, ...]] = {}
        self._root_state = self._register(tuple(int(item) for item in root_prefix))

    @property
    def root_state(self) -> FiniteDomainState:
        return self._root_state

    def _register(self, prefix: tuple[int, ...]) -> FiniteDomainState:
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            build_completion_forest,
        )
        from slm_training.dsl.solver.adapters import completion_forest_state

        forest = build_completion_forest(
            self._tokenizer, list(prefix), slot_contract=self._slot_contract
        )
        projection = completion_forest_state(
            prefix_ids=prefix,
            forest=forest,
            pack_id=self._pack_id,
            constraint_version=self._constraint_version,
            bounds=self._bounds,
        )
        self._prefixes[projection.fingerprint] = prefix
        # Register the decided-singleton variant per value so the oracle's
        # ``with_decision`` root maps back to the advanced prefix.
        hole = projection.holes[0]
        for value in hole.values:
            advanced = prefix + _value_token_ids(value)
            decided = projection.with_decision(hole.hole_id, value)
            self._prefixes.setdefault(decided.fingerprint, advanced)
        return projection

    def __call__(self, state: FiniteDomainState) -> Expansion:
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            build_completion_forest,
        )

        prefix = self._prefixes.get(state.fingerprint)
        if prefix is None:
            return Expansion(coverage="none")
        forest = build_completion_forest(
            self._tokenizer, list(prefix), slot_contract=self._slot_contract
        )
        eos = int(self._tokenizer.eos_id)
        edges: list[ExpansionEdge] = []
        terminal: DomainValue | None = None
        for path in forest.paths:
            token_ids = tuple(int(item) for item in path.token_ids)
            if token_ids and token_ids[0] == eos:
                terminal = DomainValue.create(
                    "openui_program", {"token_ids": list(prefix)}
                )
                continue
            child = self._register(prefix + token_ids)
            edges.append(
                ExpansionEdge(
                    value=DomainValue.create(
                        "completion_path",
                        {"kind": path.kind, "token_ids": list(token_ids)},
                    ),
                    state=child,
                )
            )
        return Expansion(coverage=forest.coverage, edges=tuple(edges), terminal=terminal)


def make_completion_forest_expander(**kwargs: Any) -> CompletionForestExpander:
    """Construct the reference :class:`CompletionForestExpander` seam."""
    return CompletionForestExpander(**kwargs)


__all__ = [
    "REFERENCE_SEARCH_ORDER",
    "SUPPORT_CERTIFICATE_SCHEMA_VERSION",
    "CompletionForestExpander",
    "EnumerativeSupportOracle",
    "Expansion",
    "ExpansionEdge",
    "ReplayResult",
    "SearchCounters",
    "SupportCertificate",
    "SupportDecoder",
    "SupportExpander",
    "SupportOracle",
    "SupportQuery",
    "SupportResult",
    "SupportVerifier",
    "VerifierOutcome",
    "get_support_backend",
    "make_completion_forest_expander",
    "make_pack_verifier",
    "make_token_decoder",
    "register_support_backend",
    "replay_support_certificate",
    "support_backends",
]
