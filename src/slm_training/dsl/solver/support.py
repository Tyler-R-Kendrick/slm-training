"""Exhaustive tri-state support oracle and replayable certificates (VSS0-04).

This is the first component allowed to produce ``UNSUPPORTED``. It answers, for a
single candidate at one hole of a :class:`FiniteDomainState`, whether that
candidate participates in at least one bounded, verifier-accepted completion:

* ``SUPPORTED`` — a verifier-accepted witness completion using the candidate was
  found (a witness is valid even if other branches are only partially covered);
* ``UNSUPPORTED`` — **every** reachable completion inside the declared finite
  bounds was exhausted with **complete** coverage and none verified;
* ``UNKNOWN`` — coverage was partial/none at a required expansion, a capability
  was unavailable, or a finite budget was exhausted before a witness. ``UNKNOWN``
  never licenses candidate removal, and a budget/timeout is never ``UNSUPPORTED``.

The search is deterministic and independent of model logits (values are explored
in the canonical order guaranteed by :class:`~slm_training.dsl.solver.state.HoleDomain`).
The problem-specific parts — how a chosen value expands to the next decision or a
terminal program, and how a terminal is verified — are injected through the
:class:`ProblemExpander` and :class:`Verifier` protocols, so the core is testable
against tiny closed fixtures and reused by the OpenUI adapter in
``openui_support.py``. Semantics are owned by
``docs/design/verified-scope-solver.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleId,
    SolverBounds,
    SupportVerdict,
)

CERTIFICATE_SCHEMA_VERSION = 1
SEARCH_ORDER = "canonical-domain-value-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _require_digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return value


# --------------------------------------------------------------------------- #
# Verifier + expander seams (problem-specific, injected)
# --------------------------------------------------------------------------- #


class VerifyStatus(str, Enum):
    """Outcome of validating one structurally-solved terminal program."""

    ACCEPT = "accept"  # verifier-accepted witness
    REJECT = "reject"  # hard rejection: this terminal is not a witness
    UNAVAILABLE = "unavailable"  # capability missing -> UNKNOWN, never UNSUPPORTED


@dataclass(frozen=True)
class VerifyOutcome:
    status: VerifyStatus
    detail: str = ""


class ExpandStatus(str, Enum):
    """How one chosen value advances the bounded search."""

    TERMINAL = "terminal"  # the choice completes a program to verify
    CONTINUE = "continue"  # the choice leads to the next decision state
    DEAD = "dead"  # the choice leads to bottom (no legal continuation)
    INCOMPLETE = "incomplete"  # coverage partial/none or capability unavailable


@dataclass(frozen=True)
class ExpandStep:
    """Result of applying one value to one hole of the current state."""

    status: ExpandStatus
    program: str | None = None
    next_state: FiniteDomainState | None = None
    coverage: str = "complete"
    detail: str = ""


class Verifier(Protocol):
    """Validates a structurally-solved terminal program."""

    @property
    def profile(self) -> str:
        """Stable identifier of the verifier stack/profile used."""
        ...

    def verify(self, program: str) -> VerifyOutcome: ...


class ProblemExpander(Protocol):
    """Deterministic bounded expansion of the choice/compiler search space."""

    @property
    def problem_id(self) -> str: ...

    @property
    def pack_id(self) -> str: ...

    @property
    def constraint_version(self) -> str: ...

    @property
    def bounds(self) -> SolverBounds: ...

    def successor(
        self, state: FiniteDomainState, hole_id: HoleId, value: DomainValue
    ) -> ExpandStep:
        """Apply ``value`` at ``hole_id`` and report the successor step.

        Must be deterministic and independent of model logits.
        """
        ...


# --------------------------------------------------------------------------- #
# Query / certificate / result contracts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SupportQuery:
    """Identifies the candidate whose support is being decided."""

    state_fingerprint: str
    hole_id: HoleId
    candidate: DomainValue

    def __post_init__(self) -> None:
        _require_digest(self.state_fingerprint, field="query state_fingerprint")
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
        return cls(
            state_fingerprint=str(data["state_fingerprint"]),
            hole_id=HoleId.from_dict(data["hole_id"]),
            candidate=DomainValue.from_dict(data["candidate"]),
        )


@dataclass(frozen=True)
class SearchCounters:
    """Bounded work counters; a frozen snapshot travels in the result."""

    nodes: int = 0
    tokens: int = 0
    depth: int = 0
    backtracks: int = 0
    verifier_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "nodes": self.nodes,
            "tokens": self.tokens,
            "depth": self.depth,
            "backtracks": self.backtracks,
            "verifier_calls": self.verifier_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchCounters:
        return cls(**{key: int(data[key]) for key in cls().to_dict()})


class _MutableCounters:
    """Internal running counters checked against bounds during search."""

    __slots__ = ("nodes", "tokens", "depth", "backtracks", "verifier_calls")

    def __init__(self) -> None:
        self.nodes = 0
        self.tokens = 0
        self.depth = 0
        self.backtracks = 0
        self.verifier_calls = 0

    def snapshot(self) -> SearchCounters:
        return SearchCounters(
            nodes=self.nodes,
            tokens=self.tokens,
            depth=self.depth,
            backtracks=self.backtracks,
            verifier_calls=self.verifier_calls,
        )

    def over_budget(self, bounds: SolverBounds) -> str | None:
        if self.nodes > bounds.max_nodes:
            return "budget:max_nodes"
        if self.tokens > bounds.max_tokens:
            return "budget:max_tokens"
        if self.depth > bounds.max_depth:
            return "budget:max_depth"
        if self.backtracks > bounds.max_backtracks:
            return "budget:max_backtracks"
        if self.verifier_calls > bounds.max_verifier_calls:
            return "budget:max_verifier_calls"
        return None


@dataclass(frozen=True)
class SupportCertificate:
    """Replayable justification for a support verdict.

    Contains no model logits, timestamps, secrets, or raw user-region contents;
    a witness travels only as ``witness_digest`` plus a source label.
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
            "failure_counts": [list(row) for row in self.failure_counts],
            "exhausted": self.exhausted,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupportCertificate:
        return cls(
            schema_version=int(data["schema_version"]),
            query=SupportQuery.from_dict(data["query"]),
            verdict=SupportVerdict(data["verdict"]),
            problem_id=str(data["problem_id"]),
            pack_id=str(data["pack_id"]),
            constraint_version=str(data["constraint_version"]),
            bounds=SolverBounds.from_dict(data["bounds"]),
            search_order=str(data["search_order"]),
            explored_state_fingerprints=tuple(
                str(item) for item in data["explored_state_fingerprints"]
            ),
            coverage_observations=tuple(
                str(item) for item in data["coverage_observations"]
            ),
            verifier_profile=str(data["verifier_profile"]),
            witness_source=(
                None if data.get("witness_source") is None else str(data["witness_source"])
            ),
            witness_digest=(
                None if data.get("witness_digest") is None else str(data["witness_digest"])
            ),
            failure_counts=tuple(
                (str(row[0]), int(row[1])) for row in data.get("failure_counts", ())
            ),
            exhausted=bool(data.get("exhausted", False)),
            stop_reason=(
                None if data.get("stop_reason") is None else str(data["stop_reason"])
            ),
        )

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class SupportResult:
    verdict: SupportVerdict
    certificate: SupportCertificate
    witness: str | None = None
    counters: SearchCounters = SearchCounters()


# --------------------------------------------------------------------------- #
# Enumerative reference oracle
# --------------------------------------------------------------------------- #


class SupportOracle(Protocol):
    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult: ...


class EnumerativeSupportOracle:
    """Deterministic bounded reference oracle over the choice/compiler path.

    Explores the candidate's completions iteratively (no Python recursion),
    deduplicating by the hard-state fingerprint, and returns exactly one of
    ``SUPPORTED``/``UNSUPPORTED``/``UNKNOWN`` with a replayable certificate.
    """

    def __init__(self, expander: ProblemExpander, verifier: Verifier) -> None:
        self._expander = expander
        self._verifier = verifier

    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult:
        if not isinstance(state, FiniteDomainState):
            raise ValueError("support oracle requires a FiniteDomainState")
        if query.state_fingerprint != state.fingerprint:
            raise ValueError(
                "stale support query: fingerprint does not match the state"
            )
        if (
            state.problem_id != self._expander.problem_id
            or state.pack_id != self._expander.pack_id
            or state.constraint_version != self._expander.constraint_version
            or state.bounds != self._expander.bounds
        ):
            raise ValueError("support query identity does not match the expander")
        domain = state.domain(query.hole_id)  # raises LookupError on unknown hole
        if query.candidate not in set(domain.values):
            raise ValueError("support query candidate is not in the hole domain")

        counters = _MutableCounters()
        explored: list[str] = []
        seen: set[str] = set()
        coverage_obs: set[str] = set()
        failures: dict[str, int] = {}
        witness: str | None = None
        witness_source: str | None = None
        incomplete = False
        stop_reason: str | None = None

        # Fix the queried candidate first, then explore its completions.
        stack: list[tuple[FiniteDomainState, HoleId, DomainValue, int]] = [
            (state, query.hole_id, query.candidate, 0)
        ]

        while stack:
            current, hole_id, value, depth = stack.pop()
            counters.nodes += 1
            counters.depth = max(counters.depth, depth)
            budget = counters.over_budget(self._expander.bounds)
            if budget is not None:
                stop_reason = budget
                incomplete = True
                break

            step = self._expander.successor(current, hole_id, value)
            counters.tokens += len(value.payload_json)
            coverage_obs.add(step.coverage)

            if step.status is ExpandStatus.INCOMPLETE or step.coverage in {"partial", "none"}:
                incomplete = True
                failures[f"incomplete:{step.detail or step.coverage}"] = (
                    failures.get(f"incomplete:{step.detail or step.coverage}", 0) + 1
                )
                continue

            if step.status is ExpandStatus.DEAD:
                counters.backtracks += 1
                failures[f"dead:{step.detail or 'bottom'}"] = (
                    failures.get(f"dead:{step.detail or 'bottom'}", 0) + 1
                )
                continue

            if step.status is ExpandStatus.TERMINAL:
                program = step.program or ""
                counters.verifier_calls += 1
                if counters.over_budget(self._expander.bounds) is not None:
                    stop_reason = "budget:max_verifier_calls"
                    incomplete = True
                    break
                outcome = self._verifier.verify(program)
                if outcome.status is VerifyStatus.ACCEPT:
                    witness = program
                    witness_source = step.detail or "terminal"
                    break
                if outcome.status is VerifyStatus.UNAVAILABLE:
                    incomplete = True
                    failures[f"verifier_unavailable:{outcome.detail}"] = (
                        failures.get(f"verifier_unavailable:{outcome.detail}", 0) + 1
                    )
                    continue
                # REJECT: a hard non-witness; keep searching.
                failures[f"reject:{outcome.detail or 'invalid'}"] = (
                    failures.get(f"reject:{outcome.detail or 'invalid'}", 0) + 1
                )
                continue

            # CONTINUE: expand the next decision state.
            child = step.next_state
            if child is None:
                incomplete = True
                failures["incomplete:missing_next_state"] = (
                    failures.get("incomplete:missing_next_state", 0) + 1
                )
                continue
            child_fp = child.fingerprint
            if child_fp in seen:
                continue  # deterministic dedup of equivalent hard states
            seen.add(child_fp)
            explored.append(child_fp)
            if child.is_bottom:
                counters.backtracks += 1
                failures["dead:child_bottom"] = failures.get("dead:child_bottom", 0) + 1
                continue
            # Push every live (hole, value) branch of the child in canonical
            # order (reversed so the smallest value is popped first). Prefer an
            # unresolved hole to avoid re-expanding a singleton decision path;
            # fall back to the first hole for expanders that emit a singleton
            # hole and expect the caller to drive it to terminal.
            unresolved = [h for h in child.holes if len(h.values) > 1]
            chosen_hole = unresolved[0] if unresolved else child.holes[0]
            child_hole = chosen_hole.hole_id
            for child_value in reversed(chosen_hole.values):
                stack.append((child, child_hole, child_value, depth + 1))

        verdict, exhausted = _decide(witness is not None, incomplete, stop_reason)
        certificate = SupportCertificate(
            schema_version=CERTIFICATE_SCHEMA_VERSION,
            query=query,
            verdict=verdict,
            problem_id=self._expander.problem_id,
            pack_id=self._expander.pack_id,
            constraint_version=self._expander.constraint_version,
            bounds=self._expander.bounds,
            search_order=SEARCH_ORDER,
            explored_state_fingerprints=tuple(explored),
            coverage_observations=tuple(sorted(coverage_obs)),
            verifier_profile=self._verifier.profile,
            witness_source=witness_source if verdict is SupportVerdict.SUPPORTED else None,
            witness_digest=(
                _sha256(witness) if witness is not None and verdict is SupportVerdict.SUPPORTED else None
            ),
            failure_counts=tuple(sorted(failures.items())),
            exhausted=exhausted,
            stop_reason=stop_reason,
        )
        return SupportResult(
            verdict=verdict,
            certificate=certificate,
            witness=witness if verdict is SupportVerdict.SUPPORTED else None,
            counters=counters.snapshot(),
        )


def _decide(
    found_witness: bool, incomplete: bool, stop_reason: str | None
) -> tuple[SupportVerdict, bool]:
    """Map (witness, incompleteness, budget) to a verdict + exhaustiveness flag."""
    if found_witness:
        return SupportVerdict.SUPPORTED, False
    if incomplete or stop_reason is not None:
        return SupportVerdict.UNKNOWN, False
    # Fully explored, complete coverage, no witness.
    return SupportVerdict.UNSUPPORTED, True


# --------------------------------------------------------------------------- #
# Certificate replay
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReplayResult:
    """Structured replay outcome; ``ok`` is true only when no violations remain."""

    ok: bool
    verdict: SupportVerdict
    violations: tuple[str, ...] = ()


def replay_support_certificate(
    certificate: SupportCertificate,
    *,
    state: FiniteDomainState,
    expander: ProblemExpander,
    verifier: Verifier,
) -> ReplayResult:
    """Independently re-check a certificate; never trusts a bare verdict.

    Re-derives identity, reruns the deterministic search, and enforces the
    honesty rules: ``UNSUPPORTED`` requires an exhausted, fully-covered search
    with no budget stop; ``SUPPORTED`` requires a re-verified witness whose digest
    matches; ``UNKNOWN`` is accepted as honest but never as pruning authority.
    """
    violations: list[str] = []

    if certificate.schema_version != CERTIFICATE_SCHEMA_VERSION:
        violations.append(
            f"schema_version {certificate.schema_version} != {CERTIFICATE_SCHEMA_VERSION}"
        )
    if certificate.search_order != SEARCH_ORDER:
        violations.append("search_order mismatch")
    if state.fingerprint != certificate.query.state_fingerprint:
        violations.append("state fingerprint does not match certificate query")
    for label, cert_value, live_value in (
        ("problem_id", certificate.problem_id, expander.problem_id),
        ("pack_id", certificate.pack_id, expander.pack_id),
        ("constraint_version", certificate.constraint_version, expander.constraint_version),
        ("verifier_profile", certificate.verifier_profile, verifier.profile),
    ):
        if cert_value != live_value:
            violations.append(f"{label} mismatch: {cert_value!r} != {live_value!r}")
    if certificate.bounds != expander.bounds:
        violations.append("bounds mismatch")

    # Rerun the search under the same deterministic oracle.
    recomputed: SupportResult | None = None
    if not violations:
        try:
            recomputed = EnumerativeSupportOracle(expander, verifier).check(
                state, certificate.query
            )
        except (ValueError, LookupError) as exc:
            violations.append(f"replay could not rerun search: {exc}")

    if recomputed is not None:
        if recomputed.verdict != certificate.verdict:
            violations.append(
                f"verdict changed on replay: {certificate.verdict.value} -> {recomputed.verdict.value}"
            )
        if certificate.verdict is SupportVerdict.UNSUPPORTED:
            if not certificate.exhausted:
                violations.append("UNSUPPORTED certificate is not marked exhausted")
            if certificate.stop_reason is not None:
                violations.append("UNSUPPORTED certificate has a budget/stop reason")
            if any(cov in {"partial", "none"} for cov in certificate.coverage_observations):
                violations.append("UNSUPPORTED certificate has incomplete coverage")
            if not recomputed.certificate.exhausted:
                violations.append("replay did not reach exhaustion for UNSUPPORTED")
        elif certificate.verdict is SupportVerdict.SUPPORTED:
            if certificate.witness_digest is None:
                violations.append("SUPPORTED certificate has no witness digest")
            elif recomputed.witness is None:
                violations.append("replay found no witness for SUPPORTED")
            else:
                if _sha256(recomputed.witness) != certificate.witness_digest:
                    violations.append("witness digest does not match replayed witness")
                if verifier.verify(recomputed.witness).status is not VerifyStatus.ACCEPT:
                    violations.append("replayed witness is not verifier-accepted")

    return ReplayResult(
        ok=not violations,
        verdict=certificate.verdict,
        violations=tuple(violations),
    )
