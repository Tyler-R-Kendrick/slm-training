"""Certificate-checked exact closure over finite domains (VSS1-01).

`exact_closure` is a deterministic, pure orchestration layer on top of the VSS0-04
support oracle. It repeatedly queries candidate support and removes **only**
candidates carrying a replay-valid ``UNSUPPORTED`` certificate, until it reaches a
fixed point, certified bottom, an all-singleton state, or an honest partial result
under a work budget. It never reimplements the support search; it drives a
:class:`SupportProvider` (the oracle plus its replay checker) and records every
destructive step as a :class:`CertifiedDeduction`.

Monotonicity and honesty:

* ``SUPPORTED`` and ``UNKNOWN`` values are always kept; only replay-valid
  ``UNSUPPORTED`` values are removed. A domain becomes **certified bottom** only
  when *every* value was removed with an accepted certificate.
* All deductions in one pass are computed against a single consistent pass-start
  state and applied together to produce one new immutable state; the next pass
  re-queries against the new fingerprint so a proof that went stale after a
  reduction is never reused.
* Soft scores/logits are not accepted anywhere in this API.

Semantics are owned by ``docs/design/verified-scope-solver.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SupportVerdict,
)
from slm_training.dsl.solver.support import (
    EnumerativeSupportOracle,
    ProblemExpander,
    ReplayResult,
    SupportCertificate,
    SupportQuery,
    SupportResult,
    Verifier,
    replay_support_certificate,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


# --------------------------------------------------------------------------- #
# Provider seam: an oracle that can also replay its own certificates
# --------------------------------------------------------------------------- #


class SupportProvider(Protocol):
    """A support oracle plus its replay checker and a stable backend version."""

    @property
    def backend_version(self) -> str: ...

    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult: ...

    def replay(
        self, certificate: SupportCertificate, *, state: FiniteDomainState
    ) -> ReplayResult: ...


class EnumerativeSupportProvider:
    """Bundles :class:`EnumerativeSupportOracle` with certificate replay."""

    def __init__(self, expander: ProblemExpander, verifier: Verifier) -> None:
        self._expander = expander
        self._verifier = verifier
        self._oracle = EnumerativeSupportOracle(expander, verifier)

    @property
    def backend_version(self) -> str:
        return f"enumerative/{self._verifier.profile}"

    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult:
        return self._oracle.check(state, query)

    def replay(
        self, certificate: SupportCertificate, *, state: FiniteDomainState
    ) -> ReplayResult:
        return replay_support_certificate(
            certificate, state=state, expander=self._expander, verifier=self._verifier
        )


# --------------------------------------------------------------------------- #
# Closure records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WitnessRef:
    """A kept ``SUPPORTED`` candidate and its certificate/witness digests."""

    hole_id: HoleId
    value: DomainValue
    certificate_id: str
    witness_digest: str | None
    witness_source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id.to_dict(),
            "value": self.value.to_dict(),
            "certificate_id": self.certificate_id,
            "witness_digest": self.witness_digest,
            "witness_source": self.witness_source,
        }


@dataclass(frozen=True)
class CertifiedDeduction:
    """One destructive domain reduction, each removal citing a certificate."""

    before_fingerprint: str
    after_fingerprint: str
    hole_id: HoleId
    removed: tuple[DomainValue, ...]
    certificate_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "hole_id": self.hole_id.to_dict(),
            "removed": [value.to_dict() for value in self.removed],
            "certificate_ids": list(self.certificate_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CertifiedDeduction:
        return cls(
            before_fingerprint=str(data["before_fingerprint"]),
            after_fingerprint=str(data["after_fingerprint"]),
            hole_id=HoleId.from_dict(data["hole_id"]),
            removed=tuple(DomainValue.from_dict(d) for d in data.get("removed", [])),
            certificate_ids=tuple(str(v) for v in data.get("certificate_ids", [])),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class ClosureCounters:
    passes: int = 0
    support_queries: int = 0
    cache_hits: int = 0
    supported: int = 0
    unsupported: int = 0
    unknown: int = 0
    candidates_removed: int = 0
    verifier_calls: int = 0
    expanded_nodes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "passes": self.passes,
            "support_queries": self.support_queries,
            "cache_hits": self.cache_hits,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "unknown": self.unknown,
            "candidates_removed": self.candidates_removed,
            "verifier_calls": self.verifier_calls,
            "expanded_nodes": self.expanded_nodes,
        }


@dataclass(frozen=True)
class ClosureResult:
    state: FiniteDomainState
    deductions: tuple[CertifiedDeduction, ...]
    unknown_queries: tuple[SupportQuery, ...]
    witnesses: tuple[WitnessRef, ...]
    counters: ClosureCounters
    reached_fixed_point: bool
    stop_reason: str | None = None


# --------------------------------------------------------------------------- #
# Query order
# --------------------------------------------------------------------------- #

QueryOrder = Callable[[FiniteDomainState], list[HoleDomain]]
DEFAULT_QUERY_ORDER = "smallest-domain-then-hole-id-v1"


def default_query_order(state: FiniteDomainState) -> list[HoleDomain]:
    """Unresolved holes: smallest live domain first, then canonical ``HoleId``."""
    unresolved = [hole for hole in state.holes if len(hole.values) > 1]
    return sorted(unresolved, key=lambda hole: (len(hole.values), hole.hole_id.sort_key))


class _MutCounters:
    __slots__ = (
        "passes", "support_queries", "cache_hits", "supported", "unsupported",
        "unknown", "candidates_removed", "verifier_calls", "expanded_nodes",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)

    def snapshot(self) -> ClosureCounters:
        return ClosureCounters(**{name: getattr(self, name) for name in self.__slots__})


def _cache_key(
    state: FiniteDomainState, hole_id: HoleId, value: DomainValue, backend_version: str
) -> str:
    # state.fingerprint already subsumes pack/constraint/bounds/hole domains.
    return _canonical_json(
        [state.fingerprint, hole_id.to_dict(), value.to_dict(), backend_version]
    )


# --------------------------------------------------------------------------- #
# Exact closure
# --------------------------------------------------------------------------- #


def exact_closure(
    state: FiniteDomainState,
    provider: SupportProvider,
    *,
    query_order: QueryOrder = default_query_order,
    cache: dict[str, SupportResult] | None = None,
    certificate_store: dict[str, SupportCertificate] | None = None,
    max_queries: int | None = None,
) -> ClosureResult:
    """Deterministic monotone fixed point that removes only certified candidates."""
    if not isinstance(state, FiniteDomainState):
        raise ValueError("exact_closure requires a FiniteDomainState")
    counters = _MutCounters()
    deductions: list[CertifiedDeduction] = []
    unknown_queries: list[SupportQuery] = []
    witnesses: list[WitnessRef] = []
    current = state
    stop_reason: str | None = None
    reached = False

    while True:
        if current.is_bottom:
            reached = True  # certified bottom (only certified removals got us here)
            break
        ordered = query_order(current)
        if not ordered:  # every domain singleton
            reached = True
            break
        counters.passes += 1
        pass_removals: dict[HoleId, tuple[tuple[DomainValue, ...], tuple[str, ...]]] = {}
        budget_hit = False

        for hole in ordered:
            removed: list[DomainValue] = []
            cert_ids: list[str] = []
            for value in hole.values:
                if max_queries is not None and counters.support_queries >= max_queries:
                    budget_hit = True
                    break
                query = SupportQuery(
                    state_fingerprint=current.fingerprint,
                    hole_id=hole.hole_id,
                    candidate=value,
                )
                key = _cache_key(current, hole.hole_id, value, provider.backend_version)
                if cache is not None and key in cache:
                    result = cache[key]
                    counters.cache_hits += 1
                else:
                    result = provider.check(current, query)
                    counters.support_queries += 1
                    counters.verifier_calls += result.counters.verifier_calls
                    counters.expanded_nodes += result.counters.nodes
                    if cache is not None:
                        cache[key] = result

                if result.verdict is SupportVerdict.SUPPORTED:
                    counters.supported += 1
                    witnesses.append(
                        WitnessRef(
                            hole_id=hole.hole_id,
                            value=value,
                            certificate_id=result.certificate.digest,
                            witness_digest=result.certificate.witness_digest,
                            witness_source=result.certificate.witness_source,
                        )
                    )
                    if certificate_store is not None:
                        certificate_store[result.certificate.digest] = result.certificate
                elif result.verdict is SupportVerdict.UNSUPPORTED:
                    counters.unsupported += 1
                    # Replay/validate against the exact pre-refinement state before
                    # removing anything. A stale or tampered proof does not remove.
                    replay = provider.replay(result.certificate, state=current)
                    if replay.ok and replay.verdict is SupportVerdict.UNSUPPORTED:
                        removed.append(value)
                        cert_ids.append(result.certificate.digest)
                        if certificate_store is not None:
                            certificate_store[result.certificate.digest] = result.certificate
                    else:
                        counters.unknown += 1
                        unknown_queries.append(query)
                else:  # UNKNOWN
                    counters.unknown += 1
                    unknown_queries.append(query)
            if removed:
                retained = tuple(v for v in hole.values if v not in set(removed))
                pass_removals[hole.hole_id] = (tuple(removed), tuple(cert_ids))
                del retained  # retained is recomputed below from the pass-start state
            if budget_hit:
                break

        if not pass_removals:
            reached = not budget_hit
            if budget_hit:
                stop_reason = "budget:max_queries"
            break

        # Apply every deduction against the single pass-start state.
        before_fp = current.fingerprint
        new_state = current
        for hole_id in sorted(pass_removals, key=lambda h: h.sort_key):
            removed, _cert_ids = pass_removals[hole_id]
            live = current.domain(hole_id).values
            retained = tuple(v for v in live if v not in set(removed))
            new_state = new_state.refine(hole_id, retained)
        after_fp = new_state.fingerprint
        for hole_id in sorted(pass_removals, key=lambda h: h.sort_key):
            removed, cert_ids_t = pass_removals[hole_id]
            counters.candidates_removed += len(removed)
            deductions.append(
                CertifiedDeduction(
                    before_fingerprint=before_fp,
                    after_fingerprint=after_fp,
                    hole_id=hole_id,
                    removed=removed,
                    certificate_ids=cert_ids_t,
                    reason="certified_unsupported",
                )
            )
        current = new_state

        if budget_hit:
            stop_reason = "budget:max_queries"
            reached = False
            break

    return ClosureResult(
        state=current,
        deductions=tuple(deductions),
        unknown_queries=tuple(unknown_queries),
        witnesses=tuple(witnesses),
        counters=counters.snapshot(),
        reached_fixed_point=reached,
        stop_reason=stop_reason,
    )
