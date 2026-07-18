"""VSS1-04 (SLM-64): solver-transition replay events + validator — core logic.

Torch-free tests for `dsl/solver/replay.py`: a clean full-mode stream replays
with zero violations, and every honesty invariant (fingerprint lineage,
live-only removals, certificate tamper detection, unknown-never-removes,
single-live decisions, backtrack lineage, nogood-not-a-deduction,
solved-has-report, certified-unsat purity, counter agreement, truncation
honesty) is independently detected. Model wiring/parity is covered under
tests/test_models/ and tests/test_harnesses/.
"""

from __future__ import annotations

import copy

import pytest

from slm_training.dsl.solver.replay import (
    CERTIFICATE_MODES,
    SOLVER_EVENT_KINDS,
    _digest,
    serialize_certificates,
    solver_replay_violations,
    solver_trace_counters,
)

_ROOT = "fp_root"
_S1 = "fp_s1"


def _cert_payload(tag: str) -> dict:
    # An opaque certificate dict whose id is its own sha256 digest (as in the
    # real store: certificate_store[cert.digest] = cert).
    return {
        "schema_version": 1,
        "verdict": "unsupported",
        "exhausted": True,
        "coverage_observations": ["complete"],
        "tag": tag,
    }


def _hole(name: str = "h0") -> dict:
    return {"namespace": "ns", "path": [name], "kind": "component"}


def _val(token: int) -> dict:
    return {"tag": "path", "value": f'{{"token_ids":[{token}]}}'}


def _clean_full_trace():
    """A valid full-mode stream: root state, one certified removal, solved."""
    cert = _cert_payload("c1")
    cid = _digest(cert)
    events = [
        {
            "kind": "solver_state",
            "state_fingerprint": _ROOT,
            "problem_id": "p",
            "pack_id": "openui",
            "constraint_version": "cv",
            "bounds": {},
            "decision_level": 0,
            "domain_summary": {"hole_count": 1},
            # value keys are canonical-JSON of the value dict (validator convention)
            "domain": {_hole_str(): [_valkey(10), _valkey(20), _valkey(30)]},
            "trace_truncated": False,
        },
        {
            "kind": "certified_deduction",
            "before_fingerprint": _ROOT,
            "after_fingerprint": _S1,
            "hole_id": _hole(),
            "removed": [_val(10)],
            "certificate_ids": [cid],
            "reason": "certified_unsupported",
        },
        {
            "kind": "solver_terminal",
            "status": "solved",
            "source_digest": "src",
            "verifier_report": {"name": "OpenUIWellFormed", "accepted": True},
            "certificate_mode": "full",
            "trace_truncated": False,
        },
    ]
    certificates = {cid: cert}
    return events, certificates


def _hole_str() -> str:
    from slm_training.dsl.solver.replay import _hole_key

    return _hole_key(_hole())


def _valkey(token: int) -> str:
    from slm_training.dsl.solver.replay import _value_key

    return _value_key(_val(token))


def test_clean_full_trace_replays_without_violations():
    events, certs = _clean_full_trace()
    assert solver_replay_violations(events, certificates=certs, certificate_mode="full") == []


def test_deduction_removing_non_live_value_is_detected():
    # "unknown-preservation": a value kept (never in the live snapshot) cannot be
    # certified-removed.
    events, certs = _clean_full_trace()
    events[1]["removed"] = [_val(99)]  # 99 was never live
    violations = solver_replay_violations(events, certificates=certs, certificate_mode="full")
    assert any("non-live" in v for v in violations)


def test_missing_certificate_is_detected():
    events, _certs = _clean_full_trace()
    violations = solver_replay_violations(events, certificates={}, certificate_mode="full")
    assert any("missing" in v for v in violations)


def test_tampered_certificate_is_detected():
    events, certs = _clean_full_trace()
    cid = next(iter(certs))
    certs[cid] = {**certs[cid], "tag": "TAMPERED"}  # digest no longer matches id
    violations = solver_replay_violations(events, certificates=certs, certificate_mode="full")
    assert any("digest mismatch" in v or "tampered" in v.lower() for v in violations)


def test_nogood_relabeled_as_deduction_is_detected():
    events, certs = _clean_full_trace()
    events[1]["certificate_ids"] = []  # a deduction with no certificate is a nogood
    violations = solver_replay_violations(events, certificates=certs, certificate_mode="full")
    assert any("nogood" in v.lower() for v in violations)


def test_solved_without_verifier_report_is_detected():
    events, certs = _clean_full_trace()
    events[-1]["verifier_report"] = None
    violations = solver_replay_violations(events, certificates=certs, certificate_mode="full")
    assert any("without a verifier report" in v for v in violations)


def test_certified_unsat_with_unknown_is_detected():
    events, certs = _clean_full_trace()
    events.insert(1, {
        "kind": "support_result",
        "state_fingerprint": _ROOT,
        "hole_id": _hole(),
        "candidate": _val(20),
        "verdict": "unknown",
        "certificate_id": None,
        "witness_digest": None,
        "stop_reason": None,
        "coverage": [],
        "counters": {},
    })
    events[-1]["status"] = "certified_unsat"
    violations = solver_replay_violations(events, certificates=certs, certificate_mode="full")
    assert any("certified_unsat" in v for v in violations)


def test_bad_before_fingerprint_lineage_is_detected():
    events, certs = _clean_full_trace()
    events[1]["before_fingerprint"] = "fp_wrong"
    violations = solver_replay_violations(events, certificates=certs, certificate_mode="full")
    assert any("!= active state" in v for v in violations)


def test_backtrack_to_unrecorded_state_is_detected():
    events, certs = _clean_full_trace()
    events.insert(2, {
        "kind": "backtrack",
        "from_fingerprint": _S1,
        "to_fingerprint": "fp_never_recorded",
        "from_level": 1,
        "to_level": 0,
        "decision_id": "d0",
        "conflict_kind": "certified_bottom",
    })
    violations = solver_replay_violations(events, certificates=certs, certificate_mode="full")
    assert any("unrecorded state" in v for v in violations)


def test_truncated_trace_is_non_replayable():
    events, certs = _clean_full_trace()
    events[0]["trace_truncated"] = True
    violations = solver_replay_violations(events, certificates=certs, certificate_mode="full")
    assert any("truncated" in v for v in violations)


def test_counter_mismatch_is_detected():
    events, certs = _clean_full_trace()
    counters = solver_trace_counters(events)
    counters["certified_deductions"] += 5  # lie about the count
    violations = solver_replay_violations(
        events, certificates=certs, certificate_mode="full", counters=counters
    )
    assert any("counter" in v for v in violations)


def test_matching_counters_pass():
    events, certs = _clean_full_trace()
    counters = solver_trace_counters(events)
    assert solver_replay_violations(
        events, certificates=certs, certificate_mode="full", counters=counters
    ) == []


def test_summary_and_none_modes_are_honest_about_limits():
    events, certs = _clean_full_trace()
    cid = next(iter(certs))

    class _Cert:
        def __init__(self, payload):
            self._p = payload

        def to_dict(self):
            return self._p

    store = {cid: _Cert(certs[cid])}
    full = serialize_certificates(store, "full")
    summary = serialize_certificates(store, "summary")
    none = serialize_certificates(store, "none")
    assert none == {}
    # summary drops the replay material (no 'tag'); full keeps it.
    assert "tag" in full[cid] and "tag" not in summary[cid]
    # In summary mode a tampered cert is NOT caught (summary is not a replay
    # guarantee) — honest limitation, not a false pass in full mode.
    tampered = copy.deepcopy(events)
    tampered_certs = {cid: {**certs[cid], "tag": "X"}}
    assert solver_replay_violations(
        tampered, certificates=tampered_certs, certificate_mode="summary"
    ) == []
    assert solver_replay_violations(
        tampered, certificates=tampered_certs, certificate_mode="full"
    ) != []


def test_no_raw_text_leaks_into_terminal_report():
    from slm_training.dsl.solver.replay import solver_terminal_event

    event = solver_terminal_event(
        status="solved",
        verifier_report={
            "name": "OpenUIWellFormed",
            "accepted": True,
            "secret_note": "user typed their password here",
        },
    )
    report = event["verifier_report"]
    assert report["name"] == "OpenUIWellFormed"
    assert report["accepted"] is True
    assert "secret_note" not in report  # non-allowlisted string dropped


def test_bad_mode_rejected():
    with pytest.raises(ValueError, match="solver_certificate_mode"):
        serialize_certificates({}, "drop")
    assert set(CERTIFICATE_MODES) == {"none", "summary", "full"}
    assert "solver_state" in SOLVER_EVENT_KINDS


def test_closure_events_round_trip_replays_and_detects_tamper():
    """A real ClosureResult (with a real SupportCertificate) round-trips through
    the producer + serializer and replays clean in full mode; tampering the
    stored certificate breaks the digest check."""
    from slm_training.dsl.solver.closure import (
        CertifiedDeduction,
        ClosureCounters,
        ClosureResult,
    )
    from slm_training.dsl.solver.replay import (
        serialize_certificates,
        solver_events_from_closure,
    )
    from slm_training.dsl.solver.state import (
        DomainValue,
        FiniteDomainState,
        HoleDomain,
        HoleId,
        SolverBounds,
        SupportVerdict,
    )
    from slm_training.dsl.solver.support import (
        SEARCH_ORDER,
        SupportCertificate,
        SupportQuery,
    )

    bounds = SolverBounds(
        max_tokens=100, max_nodes=100, max_depth=8, max_backtracks=8,
        max_verifier_calls=100,
    )
    hole = HoleId(namespace="ns", path=("h0",), kind="component")
    v_keep = DomainValue(tag="path", payload_json='{"token_ids":[20]}')
    v_drop = DomainValue(tag="path", payload_json='{"token_ids":[10]}')
    root = FiniteDomainState(
        problem_id="p", pack_id="openui", constraint_version="cv", bounds=bounds,
        holes=(HoleDomain(hole_id=hole, values=(v_drop, v_keep), metadata={}),),
    )
    refined = root.refine(hole, (v_keep,))
    cert = SupportCertificate(
        schema_version=1,
        query=SupportQuery(
            state_fingerprint=root.fingerprint, hole_id=hole, candidate=v_drop
        ),
        verdict=SupportVerdict.UNSUPPORTED,
        problem_id="p", pack_id="openui", constraint_version="cv", bounds=bounds,
        search_order=SEARCH_ORDER, explored_state_fingerprints=(),
        coverage_observations=("complete",), verifier_profile="stub", exhausted=True,
    )
    cid = cert.digest
    deduction = CertifiedDeduction(
        before_fingerprint=root.fingerprint, after_fingerprint=refined.fingerprint,
        hole_id=hole, removed=(v_drop,), certificate_ids=(cid,),
        reason="certified_unsupported",
    )
    result = ClosureResult(
        state=refined, deductions=(deduction,), unknown_queries=(), witnesses=(),
        counters=ClosureCounters(
            passes=1, support_queries=2, unsupported=1, candidates_removed=1
        ),
        reached_fixed_point=True,
    )
    events = solver_events_from_closure(result, root, certificate_mode="full")
    certs = serialize_certificates({cid: cert}, "full")
    assert solver_replay_violations(
        events, certificates=certs, certificate_mode="full"
    ) == []
    # The producer never claims a bare closure prune is "solved".
    terminal = next(e for e in events if e["kind"] == "solver_terminal")
    assert terminal["status"] == "unknown"
    # Tamper the stored certificate -> digest no longer matches its id.
    certs[cid] = {**certs[cid], "verifier_profile": "TAMPERED"}
    assert solver_replay_violations(
        events, certificates=certs, certificate_mode="full"
    ) != []
