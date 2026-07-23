from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.semantic_floor_gate import (
    SemanticFloorGateV1,
    build_semantic_floor_gate,
    decide_verdict,
    render_markdown,
    require_floor_gate,
    validate_gate_references,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_current_sde5_bundle_is_inconclusive(repo_root: Path) -> None:
    gate = build_semantic_floor_gate(repo_root=repo_root)
    assert gate.verdict == "inconclusive"
    assert gate.sample_sizes["strict_meaning_v2"] == 0
    assert gate.seeds == (0, 1, 2)
    assert gate.anti_gaming["status"] == "scheduled_not_executed"
    assert gate.agentv_evaluation["status"] == "missing"


@pytest.mark.parametrize(
    ("strict", "n", "proxy_moved", "expected"),
    [
        (None, 8, True, "inconclusive"),  # syntax-only / unmeasured
        (0.0, 8, True, "proxy_only"),  # meaning-v1/proxy moved, strict stayed at floor
        (0.2, 7, True, "inconclusive"),  # n < 8
    ],
)
def test_nonqualifying_evidence_cannot_escape_floor(
    strict: float | None, n: int, proxy_moved: bool, expected: str
) -> None:
    assert (
        decide_verdict(
            strict_meaning_v2=strict,
            eval_n=n,
            paired_reproducible=True,
            anti_gaming_passed=True,
            identities_resolved=True,
            agentv_contradiction=False,
            proxy_moved=proxy_moved,
        )
        == expected
    )


def test_floor_escape_requires_every_condition() -> None:
    assert (
        decide_verdict(
            strict_meaning_v2=0.125,
            eval_n=8,
            paired_reproducible=True,
            anti_gaming_passed=True,
            identities_resolved=True,
            agentv_contradiction=False,
            proxy_moved=True,
        )
        == "floor_escaped"
    )


def test_gaming_and_unresolved_identity_fail_closed() -> None:
    common = dict(
        strict_meaning_v2=0.25,
        eval_n=8,
        paired_reproducible=True,
        anti_gaming_passed=False,
        agentv_contradiction=False,
        proxy_moved=True,
    )
    assert decide_verdict(**common, identities_resolved=True, gaming_explains_gain=True) == "rejected"
    assert decide_verdict(**common, identities_resolved=False) == "inconclusive"


def test_gate_hash_changes_with_evidence_or_evaluator_version(repo_root: Path) -> None:
    gate = build_semantic_floor_gate(repo_root=repo_root)
    changed_eval = replace(gate, evaluator_versions={**gate.evaluator_versions, "@agentv/core": "next"})
    changed_evidence = replace(
        gate,
        evidence=(replace(gate.evidence[0], sha256="0" * 64), *gate.evidence[1:]),
    )
    assert gate.gate_hash != changed_eval.gate_hash
    assert gate.gate_hash != changed_evidence.gate_hash


def test_claim_authorization_blocks_semantics_but_allows_diagnostics(repo_root: Path) -> None:
    gate = build_semantic_floor_gate(repo_root=repo_root)
    assert require_floor_gate(gate, "diagnostic") is gate
    for claim in ("semantic_prediction", "semantic_causal", "learned_latent"):
        with pytest.raises(PermissionError, match=gate.gate_hash):
            require_floor_gate(gate, claim)
    with pytest.raises(ValueError, match="unknown"):
        require_floor_gate(gate, "semantic_magic")


def test_round_trip_hash_and_narrative(repo_root: Path) -> None:
    gate = build_semantic_floor_gate(repo_root=repo_root)
    restored = SemanticFloorGateV1.from_dict(gate.to_dict())
    assert restored.gate_hash == gate.gate_hash
    narrative = render_markdown(restored)
    assert f"**Verdict:** **{gate.verdict}**" in narrative
    assert gate.gate_hash in narrative


def test_fresh_checkout_references_resolve(repo_root: Path) -> None:
    gate = build_semantic_floor_gate(repo_root=repo_root)
    assert validate_gate_references(gate, repo_root=repo_root) == []


def test_reference_validator_detects_tamper(repo_root: Path, tmp_path: Path) -> None:
    gate = build_semantic_floor_gate(repo_root=repo_root)
    relative = gate.evidence[0].path
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    failures = validate_gate_references(replace(gate, evidence=(gate.evidence[0],)), repo_root=tmp_path)
    assert "SHA-256 mismatch" in failures[0]
