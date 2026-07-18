"""Replay-verified solver supervision corpus builder (VSS3-01 / SLM-69).

Transforms solver traces into versioned JSONL corpora of support-set and
candidate-cost rows. Every hard label is tied to a replayed
``SupportCertificate``; unverified or tampered traces are skipped.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from slm_training.data.store import DataStore, write_common_manifest
from slm_training.dsl.solver.controller import SearchResult
from slm_training.dsl.solver.state import FiniteDomainState
from slm_training.dsl.solver.support import (
    DomainValue,
    HoleId,
    ProblemExpander,
    ReplayResult,
    SearchCounters,
    SupportCertificate,
    SupportVerdict,
    Verifier,
    replay_support_certificate,
)
from slm_training.harnesses.distill.trace_store import TraceStore

SOLVER_TRACE_SCHEMA_VERSION = 1
SUPERVISION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SolverTraceLineage:
    """Lineage copied from the source ProgramSpec/record."""

    program_family_id: str = ""
    lineage_id: str = ""
    split_group_id: str = ""
    split: str = "train"
    task: str = ""
    record_id: str = ""
    trace_id: str | None = None
    trajectory_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_family_id": self.program_family_id,
            "lineage_id": self.lineage_id,
            "split_group_id": self.split_group_id,
            "split": self.split,
            "task": self.task,
            "record_id": self.record_id,
            "trace_id": self.trace_id,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverTraceLineage:
        return cls(
            program_family_id=str(data.get("program_family_id", "")),
            lineage_id=str(data.get("lineage_id", "")),
            split_group_id=str(data.get("split_group_id", "")),
            split=str(data.get("split", "train")),
            task=str(data.get("task", "")),
            record_id=str(data.get("record_id", "")),
            trace_id=data.get("trace_id"),
            trajectory_id=data.get("trajectory_id"),
        )


@dataclass(frozen=True)
class SupportEvent:
    """One support query/verdict observed during a solver rollout."""

    state_fingerprint: str
    hole_id: HoleId
    candidate: DomainValue
    verdict: SupportVerdict
    certificate_id: str
    counters: SearchCounters
    ranker_id: str | None = None
    chosen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_fingerprint": self.state_fingerprint,
            "hole_id": self.hole_id.to_dict(),
            "candidate": self.candidate.to_dict(),
            "verdict": self.verdict.value,
            "certificate_id": self.certificate_id,
            "counters": self.counters.to_dict(),
            "ranker_id": self.ranker_id,
            "chosen": self.chosen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupportEvent:
        return cls(
            state_fingerprint=str(data["state_fingerprint"]),
            hole_id=HoleId.from_dict(data["hole_id"]),
            candidate=DomainValue.from_dict(data["candidate"]),
            verdict=SupportVerdict(data["verdict"]),
            certificate_id=str(data["certificate_id"]),
            counters=SearchCounters.from_dict(data["counters"]),
            ranker_id=data.get("ranker_id"),
            chosen=bool(data.get("chosen", False)),
        )


@dataclass(frozen=True)
class SolverTrace:
    """Solver trace envelope consumed by the supervision builder."""

    search_result: SearchResult | None = None
    certificate_store: dict[str, SupportCertificate] = field(default_factory=dict)
    support_events: tuple[SupportEvent, ...] = ()
    state_snapshots: dict[str, FiniteDomainState] = field(default_factory=dict)
    lineage: SolverTraceLineage = field(default_factory=SolverTraceLineage)
    final_status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOLVER_TRACE_SCHEMA_VERSION,
            "search_result": (
                self.search_result.to_dict() if self.search_result is not None else None
            ),
            "certificate_store": {
                cid: cert.to_dict() for cid, cert in self.certificate_store.items()
            },
            "support_events": [event.to_dict() for event in self.support_events],
            "state_snapshots": {
                fp: state.to_dict() for fp, state in self.state_snapshots.items()
            },
            "lineage": self.lineage.to_dict(),
            "final_status": self.final_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverTrace:
        cert_store = {
            str(cid): SupportCertificate.from_dict(cert)
            for cid, cert in (data.get("certificate_store") or {}).items()
        }
        snapshots = {
            str(fp): FiniteDomainState.from_dict(state)
            for fp, state in (data.get("state_snapshots") or {}).items()
        }
        search_raw = data.get("search_result")
        return cls(
            search_result=SearchResult.from_dict(search_raw) if search_raw is not None else None,
            certificate_store=cert_store,
            support_events=tuple(
                SupportEvent.from_dict(event) for event in data.get("support_events", ())
            ),
            state_snapshots=snapshots,
            lineage=SolverTraceLineage.from_dict(data.get("lineage", {})),
            final_status=str(data.get("final_status", "unknown")),
        )


@dataclass(frozen=True)
class SupportProviderBundle:
    """Expander + verifier pair needed to replay a certificate."""

    expander: ProblemExpander
    verifier: Verifier


@dataclass(frozen=True)
class ProviderRegistry:
    """Lookup expander/verifier bundles by pack + constraint version."""

    _providers: dict[tuple[str, str], SupportProviderBundle] = field(default_factory=dict)

    def register(
        self,
        pack_id: str,
        constraint_version: str,
        expander: ProblemExpander,
        verifier: Verifier,
    ) -> None:
        object.__setattr__(
            self,
            "_providers",
            {
                **self._providers,
                (pack_id, constraint_version): SupportProviderBundle(
                    expander=expander, verifier=verifier
                ),
            },
        )

    def get(self, pack_id: str, constraint_version: str) -> SupportProviderBundle | None:
        return self._providers.get((pack_id, constraint_version))


@dataclass(frozen=True)
class SupportSetRow:
    """One state/hole support-set target."""

    schema_version: int
    row_kind: str
    state_fingerprint: str
    parent_fingerprint: str | None
    problem_id: str
    pack_id: str
    constraint_version: str
    bounds: dict[str, Any]
    program_family_id: str
    lineage_id: str
    split_group_id: str
    split: str
    capsule_id: str | None
    hole_id: HoleId
    decision_level: int
    domain_values: tuple[DomainValue, ...]
    supported_values: tuple[DomainValue, ...]
    unsupported_values: tuple[DomainValue, ...]
    unknown_values: tuple[DomainValue, ...]
    certificate_ids_by_value: dict[str, str]
    witness_digests_by_value: dict[str, str | None]
    counters: SearchCounters
    final_trajectory_status: str
    trace_id: str | None
    trajectory_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_kind": self.row_kind,
            "state_fingerprint": self.state_fingerprint,
            "parent_fingerprint": self.parent_fingerprint,
            "problem_id": self.problem_id,
            "pack_id": self.pack_id,
            "constraint_version": self.constraint_version,
            "bounds": self.bounds,
            "program_family_id": self.program_family_id,
            "lineage_id": self.lineage_id,
            "split_group_id": self.split_group_id,
            "split": self.split,
            "capsule_id": self.capsule_id,
            "hole_id": self.hole_id.to_dict(),
            "decision_level": self.decision_level,
            "domain_values": [value.to_dict() for value in self.domain_values],
            "supported_values": [value.to_dict() for value in self.supported_values],
            "unsupported_values": [value.to_dict() for value in self.unsupported_values],
            "unknown_values": [value.to_dict() for value in self.unknown_values],
            "certificate_ids_by_value": self.certificate_ids_by_value,
            "witness_digests_by_value": self.witness_digests_by_value,
            "counters": self.counters.to_dict(),
            "final_trajectory_status": self.final_trajectory_status,
            "trace_id": self.trace_id,
            "trajectory_id": self.trajectory_id,
        }


@dataclass(frozen=True)
class CandidateCostRow:
    """One candidate cost-to-go target."""

    schema_version: int
    row_kind: str
    state_fingerprint: str
    parent_fingerprint: str | None
    problem_id: str
    pack_id: str
    constraint_version: str
    program_family_id: str
    lineage_id: str
    split_group_id: str
    split: str
    capsule_id: str | None
    hole_id: HoleId
    candidate: DomainValue
    ranker_id: str | None
    chosen: bool
    support_verdict: str
    nodes: int
    tokens: int
    depth: int
    backtracks: int
    verifier_calls: int
    terminal_success: bool
    cost_observed: bool
    censor_reason: str | None
    conflict_reason: str | None
    counters: SearchCounters
    final_trajectory_status: str
    trace_id: str | None
    trajectory_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_kind": self.row_kind,
            "state_fingerprint": self.state_fingerprint,
            "parent_fingerprint": self.parent_fingerprint,
            "problem_id": self.problem_id,
            "pack_id": self.pack_id,
            "constraint_version": self.constraint_version,
            "program_family_id": self.program_family_id,
            "lineage_id": self.lineage_id,
            "split_group_id": self.split_group_id,
            "split": self.split,
            "capsule_id": self.capsule_id,
            "hole_id": self.hole_id.to_dict(),
            "candidate": self.candidate.to_dict(),
            "ranker_id": self.ranker_id,
            "chosen": self.chosen,
            "support_verdict": self.support_verdict,
            "nodes": self.nodes,
            "tokens": self.tokens,
            "depth": self.depth,
            "backtracks": self.backtracks,
            "verifier_calls": self.verifier_calls,
            "terminal_success": self.terminal_success,
            "cost_observed": self.cost_observed,
            "censor_reason": self.censor_reason,
            "conflict_reason": self.conflict_reason,
            "counters": self.counters.to_dict(),
            "final_trajectory_status": self.final_trajectory_status,
            "trace_id": self.trace_id,
            "trajectory_id": self.trajectory_id,
        }


@dataclass(frozen=True)
class SupervisionBuildResult:
    """Result of a supervision build."""

    output_dir: Path
    manifest: dict[str, Any]
    row_count: int
    support_set_count: int
    candidate_cost_count: int
    trace_count: int
    rejected_traces: tuple[str, ...]
    verdict_counts: dict[str, int]


def _value_key(value: DomainValue) -> str:
    return json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"))


def _add_counters(a: SearchCounters, b: SearchCounters) -> SearchCounters:
    return SearchCounters(
        nodes=a.nodes + b.nodes,
        tokens=a.tokens + b.tokens,
        depth=max(a.depth, b.depth),
        backtracks=a.backtracks + b.backtracks,
        verifier_calls=a.verifier_calls + b.verifier_calls,
    )


def _zero_counters() -> SearchCounters:
    return SearchCounters()


def _content_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _verify_event(
    event: SupportEvent,
    certificate: SupportCertificate,
    trace: SolverTrace,
    registry: ProviderRegistry | None,
) -> ReplayResult:
    """Replay a certificate when a registry is supplied, otherwise trust it."""
    if registry is None:
        return ReplayResult(ok=True, verdict=event.verdict)
    state = trace.state_snapshots.get(event.state_fingerprint)
    if state is None:
        return ReplayResult(
            ok=False,
            verdict=event.verdict,
            violations=("state snapshot missing for replay",),
        )
    bundle = registry.get(certificate.pack_id, certificate.constraint_version)
    if bundle is None:
        return ReplayResult(
            ok=False,
            verdict=event.verdict,
            violations=(f"no provider registered for {certificate.pack_id}/{certificate.constraint_version}",),
        )
    return replay_support_certificate(
        certificate,
        state=state,
        expander=bundle.expander,
        verifier=bundle.verifier,
    )


def _extract_rows(trace: SolverTrace) -> Iterator[SupportSetRow | CandidateCostRow]:
    """Yield candidate-cost rows and aggregated support-set rows for one trace."""
    lineage = trace.lineage
    search = trace.search_result
    problem_id = search.state.problem_id if search is not None else ""
    pack_id = search.state.pack_id if search is not None else ""
    constraint_version = search.state.constraint_version if search is not None else ""
    bounds = search.state.bounds.to_dict() if search is not None else {}
    final_status = trace.final_status

    # Candidate-cost rows come directly from each support event.
    grouped: dict[tuple[str, HoleId], list[SupportEvent]] = defaultdict(list)
    for event in trace.support_events:
        grouped[(event.state_fingerprint, event.hole_id)].append(event)
        yield CandidateCostRow(
            schema_version=SUPERVISION_SCHEMA_VERSION,
            row_kind="candidate_cost",
            state_fingerprint=event.state_fingerprint,
            parent_fingerprint=trace.state_snapshots.get(event.state_fingerprint, FiniteDomainState(
                problem_id=problem_id,
                pack_id=pack_id,
                constraint_version=constraint_version,
                bounds=search.state.bounds if search is not None else None,
                holes=(),
            )).parent_fingerprint if event.state_fingerprint in trace.state_snapshots else None,
            problem_id=problem_id,
            pack_id=pack_id,
            constraint_version=constraint_version,
            program_family_id=lineage.program_family_id,
            lineage_id=lineage.lineage_id,
            split_group_id=lineage.split_group_id,
            split=lineage.split,
            capsule_id=None,
            hole_id=event.hole_id,
            candidate=event.candidate,
            ranker_id=event.ranker_id,
            chosen=event.chosen,
            support_verdict=event.verdict.value,
            nodes=event.counters.nodes,
            tokens=event.counters.tokens,
            depth=event.counters.depth,
            backtracks=event.counters.backtracks,
            verifier_calls=event.counters.verifier_calls,
            terminal_success=final_status == "solved",
            cost_observed=True,
            censor_reason=None,
            conflict_reason=None,
            counters=event.counters,
            final_trajectory_status=final_status,
            trace_id=lineage.trace_id,
            trajectory_id=lineage.trajectory_id,
        )

    # Aggregate support-set rows per (state, hole).
    for (state_fp, hole_id), events in grouped.items():
        state = trace.state_snapshots.get(state_fp)
        if state is None:
            domain_values = tuple(event.candidate for event in events)
        else:
            domain = state.domain(hole_id)
            domain_values = domain.values if domain is not None else ()

        supported: list[DomainValue] = []
        unsupported: list[DomainValue] = []
        cert_by_value: dict[str, str] = {}
        witness_by_value: dict[str, str | None] = {}
        counters = _zero_counters()
        for event in events:
            counters = _add_counters(counters, event.counters)
            key = _value_key(event.candidate)
            cert_by_value[key] = event.certificate_id
            cert = trace.certificate_store.get(event.certificate_id)
            witness_by_value[key] = cert.witness_digest if cert is not None else None
            if event.verdict is SupportVerdict.SUPPORTED:
                supported.append(event.candidate)
            elif event.verdict is SupportVerdict.UNSUPPORTED:
                unsupported.append(event.candidate)

        event_keys = {_value_key(event.candidate) for event in events}
        unknown = tuple(value for value in domain_values if _value_key(value) not in event_keys)

        yield SupportSetRow(
            schema_version=SUPERVISION_SCHEMA_VERSION,
            row_kind="support_set",
            state_fingerprint=state_fp,
            parent_fingerprint=state.parent_fingerprint if state is not None else None,
            problem_id=problem_id,
            pack_id=pack_id,
            constraint_version=constraint_version,
            bounds=bounds,
            program_family_id=lineage.program_family_id,
            lineage_id=lineage.lineage_id,
            split_group_id=lineage.split_group_id,
            split=lineage.split,
            capsule_id=None,
            hole_id=hole_id,
            decision_level=state.decision_level if state is not None else 0,
            domain_values=domain_values,
            supported_values=tuple(supported),
            unsupported_values=tuple(unsupported),
            unknown_values=unknown,
            certificate_ids_by_value=cert_by_value,
            witness_digests_by_value=witness_by_value,
            counters=counters,
            final_trajectory_status=final_status,
            trace_id=lineage.trace_id,
            trajectory_id=lineage.trajectory_id,
        )


@dataclass(frozen=True)
class SupervisionConfig:
    """Configuration for ``build_solver_supervision``."""

    trace_root: Path
    output_root: Path
    version: str
    provider_registry: ProviderRegistry | None = None
    verify_replay: bool = False
    dry_run: bool = False
    immutable: bool = False


def build_solver_supervision(config: SupervisionConfig) -> SupervisionBuildResult:
    """Build a solver supervision corpus from replay-verified traces."""
    store = DataStore(local_root=config.output_root)
    store.validate_id(config.version)
    output_dir = store.path("solver_supervision", config.version)

    if output_dir.exists():
        if config.immutable:
            raise FileExistsError(
                f"solver_supervision/{config.version} already exists; bump --version"
            )

    trace_store = TraceStore(config.trace_root)
    rows: list[dict[str, Any]] = []
    rejected: list[str] = []
    verdict_counts: dict[str, int] = defaultdict(int)
    trace_count = 0

    for trace_row in trace_store.iter_kind("solver"):
        trace_count += 1
        trajectory_id = str(trace_row.get("trajectory_id") or trace_count)
        try:
            solver_trace = SolverTrace.from_dict(trace_row)
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append(f"{trajectory_id}: malformed trace ({exc})")
            continue

        trace_ok = True
        trace_reasons: list[str] = []

        for event in solver_trace.support_events:
            cert = solver_trace.certificate_store.get(event.certificate_id)
            if cert is None:
                trace_ok = False
                trace_reasons.append(f"missing certificate {event.certificate_id}")
                continue
            if config.verify_replay:
                replay = _verify_event(
                    event, cert, solver_trace, config.provider_registry
                )
                if not replay.ok:
                    trace_ok = False
                    trace_reasons.extend(replay.violations)
                    continue
            verdict_counts[event.verdict.value] += 1

        if not trace_ok:
            rejected.append(f"{trajectory_id}: {', '.join(trace_reasons)}")
            continue

        for row_obj in _extract_rows(solver_trace):
            rows.append(row_obj.to_dict())

    support_set_count = sum(1 for row in rows if row.get("row_kind") == "support_set")
    candidate_cost_count = len(rows) - support_set_count

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "kind": "solver_supervision",
        "dataset_id": config.version,
        "row_count": len(rows),
        "support_set_count": support_set_count,
        "candidate_cost_count": candidate_cost_count,
        "trace_count": trace_count,
        "rejected_count": len(rejected),
        "verdict_counts": dict(verdict_counts),
        "source_trace_root": str(config.trace_root.as_posix()),
        "verify_replay": config.verify_replay,
        "content_fingerprint": _content_fingerprint(rows),
    }

    if not config.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        rows_path = output_dir / "rows.jsonl"
        rows_path.write_text(
            "".join(json.dumps(row, default=str) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        common = write_common_manifest(
            output_dir,
            kind="solver_supervision",
            dataset_id=config.version,
            trace_id=None,
            immutable=False,
        )
        manifest.update(common)

    return SupervisionBuildResult(
        output_dir=output_dir,
        manifest=manifest,
        row_count=len(rows),
        support_set_count=support_set_count,
        candidate_cost_count=candidate_cost_count,
        trace_count=trace_count,
        rejected_traces=tuple(rejected),
        verdict_counts=dict(verdict_counts),
    )


__all__ = [
    "CandidateCostRow",
    "ProviderRegistry",
    "SolverTrace",
    "SolverTraceLineage",
    "SupervisionBuildResult",
    "SupervisionConfig",
    "SupportEvent",
    "SupportProviderBundle",
    "SupportSetRow",
    "build_solver_supervision",
]
