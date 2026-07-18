"""Regression tests for the VSS3-01 solver supervision builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slm_training.data.store import DataStore
from slm_training.dsl.solver.controller import SearchResult, SearchStatus
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
)
from slm_training.dsl.solver.support import (
    SEARCH_ORDER,
    ExpandStatus,
    ExpandStep,
    SearchCounters,
    SupportCertificate,
    SupportOracle,
    SupportQuery,
    SupportResult,
    SupportVerdict,
    VerifyOutcome,
    VerifyStatus,
)
from slm_training.harnesses.distill.solver_supervision import (
    ProviderRegistry,
    SolverTrace,
    SolverTraceLineage,
    SupervisionConfig,
    SupportEvent,
    build_solver_supervision,
)
from slm_training.harnesses.distill.trace_store import TraceStore


class _FakeExpander:
    """ProblemExpander protocol stub for replay."""

    def __init__(self, problem_id: str, pack_id: str, constraint_version: str, bounds: SolverBounds):
        self.problem_id = problem_id
        self.pack_id = pack_id
        self.constraint_version = constraint_version
        self.bounds = bounds

    def successor(self, state, hole_id, value):
        payload = json.loads(value.payload_json)
        if payload == 1:
            return ExpandStep(
                status=ExpandStatus.TERMINAL,
                program="good",
                next_state=None,
                coverage="complete",
                detail="fake",
            )
        return ExpandStep(
            status=ExpandStatus.TERMINAL,
            program="bad",
            next_state=None,
            coverage="complete",
            detail="fake",
        )


class _FakeVerifier:
    """Verifier protocol stub for replay."""

    profile = "fake"

    def verify(self, source):
        if source is not None and source.startswith("good"):
            return VerifyOutcome(status=VerifyStatus.ACCEPT)
        return VerifyOutcome(status=VerifyStatus.REJECT)


class _FakeSupportOracle(SupportOracle):
    """Deterministic oracle: candidate 1 supported, others unsupported."""

    def __init__(self, expander, verifier):
        self.expander = expander
        self.verifier = verifier

    def check(self, state, query):
        payload = json.loads(query.candidate.payload_json)
        if payload == 1:
            witness = "good"
            return SupportResult(
                verdict=SupportVerdict.SUPPORTED,
                certificate=SupportCertificate(
                    schema_version=1,
                    query=query,
                    verdict=SupportVerdict.SUPPORTED,
                    problem_id=state.problem_id,
                    pack_id=state.pack_id,
                    constraint_version=state.constraint_version,
                    bounds=state.bounds,
                    search_order=SEARCH_ORDER,
                    explored_state_fingerprints=(),
                    coverage_observations=(),
                    verifier_profile=self.verifier.profile,
                    witness_source="fake",
                    witness_digest=hashlib.sha256(witness.encode("utf-8")).hexdigest(),
                ),
                witness=witness,
                counters=SearchCounters(nodes=2, tokens=1, depth=1, verifier_calls=1),
            )
        witness = None
        verdict = SupportVerdict.UNSUPPORTED
        return SupportResult(
            verdict=verdict,
            certificate=SupportCertificate(
                schema_version=1,
                query=query,
                verdict=verdict,
                problem_id=state.problem_id,
                pack_id=state.pack_id,
                constraint_version=state.constraint_version,
                bounds=state.bounds,
                search_order=SEARCH_ORDER,
                explored_state_fingerprints=(state.fingerprint,),
                coverage_observations=("complete",),
                verifier_profile=self.verifier.profile,
                exhausted=True,
            ),
            witness=witness,
            counters=SearchCounters(nodes=3, tokens=2, depth=1, verifier_calls=1),
        )


def _make_state(values=(1, 2, 3)) -> FiniteDomainState:
    bounds = SolverBounds(
        max_tokens=100,
        max_nodes=100,
        max_depth=10,
        max_backtracks=10,
        max_verifier_calls=10,
    )
    hole_id = HoleId(namespace="test", path=("a",), kind="stmt")
    domain_values = tuple(DomainValue(tag="int", payload_json=json.dumps(v)) for v in values)
    return FiniteDomainState(
        problem_id="p1",
        pack_id="fake-pack",
        constraint_version="v1",
        bounds=bounds,
        holes=(HoleDomain(hole_id, domain_values),),
    )


def _make_trace(values=(1, 2, 3), chosen=1, tamper: int | None = None) -> SolverTrace:
    state = _make_state(values)
    hole_id = state.holes[0].hole_id
    events: list[SupportEvent] = []
    cert_store: dict[str, SupportCertificate] = {}

    provider = _FakeSupportOracle(
        _FakeExpander(state.problem_id, state.pack_id, state.constraint_version, state.bounds),
        _FakeVerifier(),
    )

    for idx, value in enumerate(state.holes[0].values):
        query = SupportQuery(state.fingerprint, hole_id, value)
        result = provider.check(state, query)
        cert = result.certificate
        cert_id = f"cert-{value.payload_json}"
        if tamper is not None and idx == tamper:
            # Lie about the verdict so replay fails.
            cert = SupportCertificate(
                schema_version=cert.schema_version,
                query=cert.query,
                verdict=SupportVerdict.SUPPORTED,
                problem_id=cert.problem_id,
                pack_id=cert.pack_id,
                constraint_version=cert.constraint_version,
                bounds=cert.bounds,
                search_order=cert.search_order,
                explored_state_fingerprints=cert.explored_state_fingerprints,
                coverage_observations=cert.coverage_observations,
                verifier_profile=cert.verifier_profile,
                witness_source=cert.witness_source,
                witness_digest=cert.witness_digest,
            )
        cert_store[cert_id] = cert
        events.append(
            SupportEvent(
                state_fingerprint=state.fingerprint,
                hole_id=hole_id,
                candidate=value,
                verdict=cert.verdict,
                certificate_id=cert_id,
                counters=result.counters,
                chosen=(json.loads(value.payload_json) == chosen),
            )
        )

    search_result = SearchResult(
        status=SearchStatus.SOLVED,
        state=state,
        source="source",
        verifier_report={"status": "accept"},
        deductions=(),
        decisions=(),
        nogoods=(),
        counters=SearchCounters(nodes=5, tokens=3, depth=0, verifier_calls=2),
    )

    return SolverTrace(
        search_result=search_result,
        certificate_store=cert_store,
        support_events=tuple(events),
        state_snapshots={state.fingerprint: state},
        lineage=SolverTraceLineage(
            program_family_id="fam-1",
            lineage_id="line-1",
            split_group_id="sg-1",
            split="train",
            task="solver_smoke",
        ),
        final_status="solved",
    )


def _registry() -> ProviderRegistry:
    state = _make_state()
    expander = _FakeExpander(
        state.problem_id, state.pack_id, state.constraint_version, state.bounds
    )
    registry = ProviderRegistry()
    registry.register(state.pack_id, state.constraint_version, expander, _FakeVerifier())
    return registry


def _append_trace(trace_root: Path, trace: SolverTrace) -> None:
    store = TraceStore(trace_root)
    store.append({"version": 2, "kind": "solver", **trace.to_dict()})


def test_build_emits_support_set_and_candidate_cost_rows(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace = _make_trace()
    _append_trace(trace_root, trace)

    config = SupervisionConfig(
        trace_root=trace_root,
        output_root=tmp_path / "out",
        version="v1",
        provider_registry=_registry(),
        verify_replay=True,
    )
    result = build_solver_supervision(config)

    assert result.row_count == 4  # 3 candidate_cost + 1 support_set
    assert result.support_set_count == 1
    assert result.candidate_cost_count == 3
    assert result.rejected_traces == ()

    rows_path = result.output_dir / "rows.jsonl"
    assert rows_path.is_file()
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    support_rows = [r for r in rows if r["row_kind"] == "support_set"]
    cost_rows = [r for r in rows if r["row_kind"] == "candidate_cost"]
    assert len(support_rows) == 1
    assert len(cost_rows) == 3

    support = support_rows[0]
    assert len(support["supported_values"]) == 1
    assert len(support["unsupported_values"]) == 2
    assert support["unknown_values"] == []
    assert support["split"] == "train"
    assert support["program_family_id"] == "fam-1"

    chosen_cost = next(r for r in cost_rows if r["chosen"])
    assert chosen_cost["support_verdict"] == "supported"
    assert chosen_cost["cost_observed"] is True


def test_unknown_candidate_is_not_in_supported_or_unsupported(tmp_path: Path) -> None:
    # Only query candidates 1 and 2; candidate 3 stays unknown.
    trace_root = tmp_path / "traces"
    state = _make_state()
    hole_id = state.holes[0].hole_id
    values = state.holes[0].values
    provider = _FakeSupportOracle(
        _FakeExpander(state.problem_id, state.pack_id, state.constraint_version, state.bounds),
        _FakeVerifier(),
    )
    events = []
    cert_store = {}
    for value in values[:2]:
        query = SupportQuery(state.fingerprint, hole_id, value)
        result = provider.check(state, query)
        cert_id = f"cert-{value.payload_json}"
        cert_store[cert_id] = result.certificate
        events.append(
            SupportEvent(
                state_fingerprint=state.fingerprint,
                hole_id=hole_id,
                candidate=value,
                verdict=result.certificate.verdict,
                certificate_id=cert_id,
                counters=result.counters,
            )
        )
    trace = SolverTrace(
        search_result=SearchResult(
            status=SearchStatus.UNKNOWN,
            state=state,
            source=None,
            verifier_report=None,
            deductions=(),
            decisions=(),
            nogoods=(),
            counters=SearchCounters(),
            stop_reason="budget",
        ),
        certificate_store=cert_store,
        support_events=tuple(events),
        state_snapshots={state.fingerprint: state},
        lineage=SolverTraceLineage(split="held_out"),
        final_status="unknown",
    )
    _append_trace(trace_root, trace)

    config = SupervisionConfig(
        trace_root=trace_root,
        output_root=tmp_path / "out",
        version="v1",
        provider_registry=_registry(),
        verify_replay=True,
    )
    result = build_solver_supervision(config)
    rows_path = result.output_dir / "rows.jsonl"
    support = next(
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["row_kind"] == "support_set"
    )
    assert len(support["supported_values"]) == 1
    assert len(support["unsupported_values"]) == 1
    assert len(support["unknown_values"]) == 1


def test_tampered_certificate_is_rejected_with_verify_replay(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace = _make_trace(tamper=1)
    _append_trace(trace_root, trace)

    config = SupervisionConfig(
        trace_root=trace_root,
        output_root=tmp_path / "out",
        version="v1",
        provider_registry=_registry(),
        verify_replay=True,
    )
    result = build_solver_supervision(config)
    assert result.row_count == 0
    assert len(result.rejected_traces) == 1


def test_all_supported_alternatives_survive(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    state = _make_state(values=(1, 2))
    hole_id = state.holes[0].hole_id
    provider = _FakeSupportOracle(
        _FakeExpander(state.problem_id, state.pack_id, state.constraint_version, state.bounds),
        _FakeVerifier(),
    )
    events = []
    cert_store = {}
    for value in state.holes[0].values:
        query = SupportQuery(state.fingerprint, hole_id, value)
        result = provider.check(state, query)
        cert_id = f"cert-{value.payload_json}"
        # Override provider so both values are supported.
        cert = SupportCertificate(
            schema_version=1,
            query=query,
            verdict=SupportVerdict.SUPPORTED,
            problem_id=state.problem_id,
            pack_id=state.pack_id,
            constraint_version=state.constraint_version,
            bounds=state.bounds,
            search_order=SEARCH_ORDER,
            explored_state_fingerprints=(),
            coverage_observations=(),
            verifier_profile=provider.verifier.profile,
            witness_source="fake",
            witness_digest="5bf8aa57cf0b5814a51f2f72c7ed9dc0c2f6f44e08b4a2a4b5f93a41e9e3f0f6",
        )
        cert_store[cert_id] = cert
        events.append(
            SupportEvent(
                state_fingerprint=state.fingerprint,
                hole_id=hole_id,
                candidate=value,
                verdict=SupportVerdict.SUPPORTED,
                certificate_id=cert_id,
                counters=result.counters,
                chosen=(json.loads(value.payload_json) == 1),
            )
        )
    trace = SolverTrace(
        search_result=SearchResult(
            status=SearchStatus.SOLVED,
            state=state,
            source="source",
            verifier_report={"status": "accept"},
            deductions=(),
            decisions=(),
            nogoods=(),
            counters=SearchCounters(),
        ),
        certificate_store=cert_store,
        support_events=tuple(events),
        state_snapshots={state.fingerprint: state},
        lineage=SolverTraceLineage(),
        final_status="solved",
    )
    _append_trace(trace_root, trace)

    config = SupervisionConfig(
        trace_root=trace_root,
        output_root=tmp_path / "out",
        version="v1",
        provider_registry=_registry(),
        verify_replay=False,
    )
    result = build_solver_supervision(config)
    rows_path = result.output_dir / "rows.jsonl"
    support = next(
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["row_kind"] == "support_set"
    )
    assert len(support["supported_values"]) == 2
    assert support["unsupported_values"] == []
    assert support["unknown_values"] == []


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace = _make_trace()
    _append_trace(trace_root, trace)

    config = SupervisionConfig(
        trace_root=trace_root,
        output_root=tmp_path / "out",
        version="v1",
        dry_run=True,
    )
    result = build_solver_supervision(config)
    assert result.row_count == 4
    assert not result.output_dir.exists()


def test_datastore_can_resolve_solver_supervision(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace = _make_trace()
    _append_trace(trace_root, trace)

    config = SupervisionConfig(
        trace_root=trace_root,
        output_root=tmp_path / "out",
        version="v1",
    )
    _result = build_solver_supervision(config)

    store = DataStore(root=tmp_path, local_root=tmp_path / "out")
    ref = store.resolve("solver_supervision", "v1")
    assert ref.storage == "local"
    assert (ref.path / "rows.jsonl").is_file()
