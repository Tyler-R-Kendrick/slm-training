"""VSS3-01 (SLM-69): replay-verified solver supervision corpus builder tests.

Traces are constructed at the dict level (Torch-free) so the label rules and
honesty invariants are exercised directly: replay-before-emit, certificate-gated
UNSUPPORTED, all-supported-alternatives retention, cost censoring, nogood
non-relabeling, split inheritance + cross-split leak rejection, no raw text,
determinism, dry-run, and non-solver skipping.
"""

from __future__ import annotations

import hashlib
import json

from slm_training.harnesses.distill.solver_supervision import (
    SUPERVISION_SCHEMA_VERSION,
    SolverSupervisionBuilder,
    build_solver_supervision,
    write_corpus,
)


# --------------------------------------------------------------------------- #
# Trace construction helpers (mirror dsl.solver.replay canonicalization)
# --------------------------------------------------------------------------- #

BOUNDS = {
    "max_tokens": 8,
    "max_nodes": 4,
    "max_depth": 4,
    "max_backtracks": 2,
    "max_verifier_calls": 8,
}


def ck(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cid_for(payload) -> str:
    return hashlib.sha256(ck(payload).encode()).hexdigest()


def hole(kind="slot", ns="root", path=("a",)):
    return {"namespace": ns, "path": list(path), "kind": kind}


def val(tag, value):
    return {"tag": tag, "value": value}


def solver_state(fp, holes, *, level=0, problem="P1", pack="pack", truncated=False):
    domain = {ck(h): [ck(v) for v in vs] for h, vs in holes}
    return {
        "kind": "solver_state",
        "state_fingerprint": fp,
        "problem_id": problem,
        "pack_id": pack,
        "constraint_version": "c1",
        "bounds": BOUNDS,
        "decision_level": level,
        "domain_summary": {"holes": len(holes)},
        "domain": domain,
        "trace_truncated": truncated,
    }


def supported(fp, h, v, *, cert_id="scid", wd="wd"):
    return {
        "kind": "support_result",
        "state_fingerprint": fp,
        "hole_id": h,
        "candidate": v,
        "verdict": "supported",
        "certificate_id": cert_id,
        "witness_digest": wd,
    }


def unknown(fp, h, v, *, stop=None):
    return {
        "kind": "support_result",
        "state_fingerprint": fp,
        "hole_id": h,
        "candidate": v,
        "verdict": "unknown",
        "stop_reason": stop,
    }


def deduction(before, after, h, removed, cids):
    return {
        "kind": "certified_deduction",
        "before_fingerprint": before,
        "after_fingerprint": after,
        "hole_id": h,
        "removed": removed,
        "certificate_ids": cids,
    }


def decision(before, after, h, chosen, alts, *, level=0, ranker="r0"):
    return {
        "kind": "decision",
        "before_fingerprint": before,
        "after_fingerprint": after,
        "hole_id": h,
        "chosen": chosen,
        "alternatives": alts,
        "level": level,
        "ranker_id": ranker,
    }


def terminal(status, *, report=None, truncated=False):
    return {
        "kind": "solver_terminal",
        "status": status,
        "verifier_report": report,
        "certificate_mode": "full",
        "trace_truncated": truncated,
    }


def nogood(h, provenance):
    return {"kind": "nogood", "hole_id": h, "provenance": provenance}


def trace(events, *, certs=None, meta=None, final=None):
    envelope = {
        "version": 3,
        "meta": meta or {"checkpoint_sha": "sha", "decode_config_hash": "dch", "seed": 0},
        "steps": [],
        "events": events,
        "final": final or {},
        "solver": {"certificate_mode": "full", "certificates": certs or {}},
    }
    return envelope


def _clean_closure_trace():
    """FP0 with holeA domain {v1,v2,v3}: v1 supported, v2 certified-unsupported, v3 unknown."""
    h = hole()
    v1, v2, v3 = val("t", 1), val("t", 2), val("t", 3)
    cert_payload = {"schema_version": 1, "verdict": "unsupported", "exhausted": True}
    cid = cid_for(cert_payload)
    events = [
        solver_state("FP0", [(h, [v1, v2, v3])]),
        supported("FP0", h, v1),
        unknown("FP0", h, v3),
        deduction("FP0", "FP1", h, [v2], [cid]),
        terminal("unknown"),
    ]
    return trace(events, certs={cid: cert_payload}), (h, v1, v2, v3)


# --------------------------------------------------------------------------- #
# 1. clean full trace produces expected support/cost rows
# --------------------------------------------------------------------------- #


def test_clean_trace_produces_expected_support_rows():
    tr, (h, v1, v2, v3) = _clean_closure_trace()
    result = SolverSupervisionBuilder().build([tr])
    assert not result.rejected_traces
    assert len(result.support_rows) == 1
    row = result.support_rows[0]
    assert row.row_kind == "support_set"
    assert row.schema_version == SUPERVISION_SCHEMA_VERSION
    assert row.state_fingerprint == "FP0"
    assert row.hole_kind == "slot"
    assert row.checkpoint_sha == "sha"
    assert row.final_trajectory_status == "unknown"
    assert row.domain_values == [v1, v2, v3]
    assert row.supported_values == [v1]
    assert row.unsupported_values == [v2]
    assert row.unknown_values == [v3]
    # UNSUPPORTED carries the certificate that licensed its removal.
    assert row.certificate_ids_by_value[ck(v2)]


# --------------------------------------------------------------------------- #
# 2. unknown candidate is not in supported/unsupported sets
# --------------------------------------------------------------------------- #


def test_unknown_not_in_supported_or_unsupported():
    tr, (h, v1, v2, v3) = _clean_closure_trace()
    row = SolverSupervisionBuilder().build([tr]).support_rows[0]
    assert v3 in row.unknown_values
    assert v3 not in row.supported_values
    assert v3 not in row.unsupported_values


# --------------------------------------------------------------------------- #
# 3. tampered unsupported certificate rejects the trace
# --------------------------------------------------------------------------- #


def test_tampered_certificate_rejects_trace():
    h = hole()
    v1, v2 = val("t", 1), val("t", 2)
    real_payload = {"schema_version": 1, "verdict": "unsupported", "exhausted": True}
    cid = cid_for(real_payload)
    tampered = {"schema_version": 1, "verdict": "unsupported", "exhausted": False}
    events = [
        solver_state("FP0", [(h, [v1, v2])]),
        supported("FP0", h, v1),
        deduction("FP0", "FP1", h, [v2], [cid]),
        terminal("unknown"),
    ]
    tr = trace(events, certs={cid: tampered})  # digest(tampered) != cid
    result = SolverSupervisionBuilder().build([tr])
    assert result.support_rows == []
    assert result.rejected_traces
    assert result.rejected_traces[0]["reason"] == "replay_violations"


# --------------------------------------------------------------------------- #
# 4. truncated/timeout suffix is censored, not assigned a low/high cost
# --------------------------------------------------------------------------- #


def test_budget_stopped_cost_is_censored():
    h = hole()
    v1, v2 = val("t", 1), val("t", 2)
    events = [
        solver_state("FP0", [(h, [v1, v2])]),
        decision("FP0", "FP1", h, v1, [v2]),
        terminal("budget_exhausted"),
    ]
    result = SolverSupervisionBuilder().build([trace(events)])
    assert result.cost_rows
    for row in result.cost_rows:
        assert row.cost_observed is False
        assert row.censor_reason == "budget_exhausted"
        assert row.terminal_success is False


def test_solved_suffix_cost_is_observed():
    h = hole()
    v1, v2 = val("t", 1), val("t", 2)
    events = [
        solver_state("FP0", [(h, [v1, v2])]),
        supported("FP0", h, v1),
        supported("FP0", h, v2),
        decision("FP0", "FP1", h, v1, [v2]),
        terminal("solved", report={"status": "ok"}),
    ]
    result = SolverSupervisionBuilder().build([trace(events)])
    assert result.cost_rows
    for row in result.cost_rows:
        assert row.cost_observed is True
        assert row.censor_reason is None
        assert row.terminal_success is True


def test_truncated_trace_censored_when_replay_check_off():
    # With replay verification off, a truncated trace still censors its cost.
    h = hole()
    v1, v2 = val("t", 1), val("t", 2)
    events = [
        solver_state("FP0", [(h, [v1, v2])], truncated=True),
        decision("FP0", "FP1", h, v1, [v2]),
        terminal("unknown", truncated=True),
    ]
    result = SolverSupervisionBuilder(verify_replay=False).build([trace(events)])
    assert result.cost_rows
    assert all(r.cost_observed is False for r in result.cost_rows)
    assert all(r.censor_reason == "truncated_suffix" for r in result.cost_rows)


# --------------------------------------------------------------------------- #
# 5. all supported alternatives survive even when one is chosen
# --------------------------------------------------------------------------- #


def test_all_supported_alternatives_retained():
    h = hole()
    v1, v4 = val("t", 1), val("t", 4)
    events = [
        solver_state("FP0", [(h, [v1, v4])]),
        supported("FP0", h, v1),
        supported("FP0", h, v4),
        decision("FP0", "FP1", h, v1, [v4]),  # chose v1
        terminal("solved", report={"status": "ok"}),
    ]
    result = SolverSupervisionBuilder().build([trace(events)])
    row = result.support_rows[0]
    # The chosen value does not eclipse the other supported alternative.
    assert v1 in row.supported_values
    assert v4 in row.supported_values


# --------------------------------------------------------------------------- #
# 6. local nogood is not a global UNSUPPORTED relabel
# --------------------------------------------------------------------------- #


def test_nogood_is_not_global_unsupported():
    h = hole()
    v1, v2 = val("t", 1), val("t", 2)
    events = [
        solver_state("FP0", [(h, [v1, v2])]),
        nogood(h, {"kind": "conflict", "value": v2}),
        terminal("unknown"),
    ]
    result = SolverSupervisionBuilder().build([trace(events)])
    row = result.support_rows[0]
    # v2 lost a local branch but was never certified UNSUPPORTED.
    assert row.unsupported_values == []
    assert v2 in row.domain_values


# --------------------------------------------------------------------------- #
# 7. split inheritance and cross-split duplicate rejection
# --------------------------------------------------------------------------- #


def test_split_lineage_inherited_from_meta():
    h = hole()
    v1 = val("t", 1)
    meta = {
        "program_family_id": "famA",
        "lineage_id": "linA",
        "split_group_id": "grpA",
        "split": "val",
        "checkpoint_sha": "sha",
        "decode_config_hash": "dch",
        "seed": 7,
    }
    events = [solver_state("FPv", [(h, [v1])]), supported("FPv", h, v1), terminal("unknown")]
    result = SolverSupervisionBuilder().build([trace(events, meta=meta)])
    row = result.support_rows[0]
    assert row.split == "val"
    assert row.program_family_id == "famA"
    assert row.lineage_id == "linA"
    assert row.split_group_id == "grpA"
    assert row.seed == 7
    assert result.derived_lineage_traces == []


def test_cross_split_state_leak_rejected():
    h = hole()
    v1 = val("t", 1)

    def one(fp, split):
        meta = {
            "program_family_id": "fam",
            "lineage_id": "lin",
            "split_group_id": "grp",
            "split": split,
            "checkpoint_sha": "sha",
        }
        return trace(
            [solver_state(fp, [(h, [v1])]), terminal("unknown")], meta=meta
        )

    # Same state fingerprint appears first in train, then in held-out test.
    result = SolverSupervisionBuilder().build([one("SHARED", "train"), one("SHARED", "test")])
    splits = {r.split for r in result.support_rows}
    assert splits == {"train"}  # the held-out duplicate was dropped
    assert any(r["reason"] == "cross_split_state_leak" for r in result.rejected_rows)


# --------------------------------------------------------------------------- #
# 8. raw opaque-region / final source text is absent from rows
# --------------------------------------------------------------------------- #


def test_final_source_text_absent_from_rows(tmp_path):
    tr, _ = _clean_closure_trace()
    tr["final"] = {"text": "SECRET_USER_TEXT", "canvas": [1, 2, 3]}
    tr["meta"] = {**tr["meta"], "raw_prompt": "SECRET_USER_TEXT"}
    out = tmp_path / "corpus"
    write_corpus(SolverSupervisionBuilder().build([tr]), out)
    blob = "".join(
        p.read_text(encoding="utf-8") for p in (out / "rows").glob("*.jsonl")
    )
    assert "SECRET_USER_TEXT" not in blob


# --------------------------------------------------------------------------- #
# 9. deterministic row / manifest hashes and ordering
# --------------------------------------------------------------------------- #


def test_deterministic_hashes_and_ordering(tmp_path):
    tr, _ = _clean_closure_trace()
    m1 = write_corpus(SolverSupervisionBuilder().build([tr]), tmp_path / "a")
    m2 = write_corpus(SolverSupervisionBuilder().build([tr]), tmp_path / "b")
    assert m1["artifacts"] == m2["artifacts"]
    for name in m1["artifacts"]:
        assert m1["artifacts"][name]["content_sha256"] == m2["artifacts"][name][
            "content_sha256"
        ]
    a = (tmp_path / "a" / "rows").glob("*.jsonl")
    for path in a:
        peer = tmp_path / "b" / "rows" / path.name
        assert path.read_bytes() == peer.read_bytes()


# --------------------------------------------------------------------------- #
# 10. dry run writes nothing
# --------------------------------------------------------------------------- #


def test_dry_run_writes_nothing(tmp_path):
    tr, _ = _clean_closure_trace()
    out = tmp_path / "corpus"
    summary = build_solver_supervision([tr], output_dir=out, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["counts"]["support_set_rows"] == 1
    assert not out.exists() or not any(out.iterdir())


# --------------------------------------------------------------------------- #
# 11. historical non-solver traces are skipped with a reason, not crashed
# --------------------------------------------------------------------------- #


def test_non_solver_trace_skipped_with_reason():
    decode_only = {
        "version": 2,
        "meta": {},
        "steps": [{"step": 0, "canvas": [5, 0], "commits": [{"t": 0, "id": 5}]}],
        "events": [],
        "final": {"canvas": [5, 2], "text": "x"},
    }
    result = SolverSupervisionBuilder().build([decode_only])
    assert result.support_rows == []
    assert result.cost_rows == []
    assert result.rejected_traces[0]["reason"] == "non_solver_trace"
