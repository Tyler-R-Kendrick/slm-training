"""SFF formal obligations + golden-vector bridges to Lean cores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slm_training.data.progspec.semantic_evidence import (
    FactorRepresentation,
    build_factor_representation,
    project_program_factors,
    reconstruct_membership,
    shuffle_port_roles,
)
from slm_training.dsl.grammar.fastpath.residual_support import rank_inside_legal_residual
from slm_training.harnesses.experiments.anti_e237_semantic_factor_frontier import (
    build_campaign,
    lock_campaign,
)
from slm_training.autoresearch.schemas import FormalPreflightV1
from slm_training.harnesses.experiments.semantic_factor_formal import (
    SFF_FORMAL_DIR,
    SFF_FORMAL_TEMPLATES,
    SFFFormalError,
    load_or_build_sff_formal_obligations,
    load_sff_formal_obligations,
    validate_committed_preflights,
    verify_sff_formal_sources,
)
from slm_training.models.semantic_factor_propagation import column_stochastic_S
from slm_training.models.semantic_residual_scorer import (
    SemanticResidualScorerConfig,
    SemanticResidualScorerV1,
    assert_score_keys_legal,
)

_GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "slm_training"
    / "resources"
    / "experiments"
    / "semantic_factor_frontier"
    / "golden_vectors.v1.json"
)


def _golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def test_formal_obligations_nonempty_and_lockable() -> None:
    obligations = load_sff_formal_obligations()
    assert len(obligations) == len(SFF_FORMAL_TEMPLATES)
    assert len({o.template_id for o in obligations}) == len(obligations)
    campaign = build_campaign(source_commit="a" * 40, source_dirty=False)
    assert campaign.formal_obligations
    assert any(a.kind == "formal_preflight" for a in campaign.artifact_requirements)
    lock = lock_campaign(campaign)
    assert lock.manifest_sha256


def test_committed_preflights_are_formal_preflight_v1() -> None:
    preflights = validate_committed_preflights()
    assert len(preflights) == len(SFF_FORMAL_TEMPLATES)
    for pf in preflights:
        assert isinstance(pf, FormalPreflightV1)
        assert pf.status == "proved"
        assert pf.mathlib_version == "none"
        assert pf.lean_version
        assert len(pf.build_output_sha256) == 64
        assert pf.checker_contract and "leverproof" in pf.checker_contract
        # Round-trip raw JSON through the schema.
        path = SFF_FORMAL_DIR / f"{pf.template_id.replace('.', '_')}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        FormalPreflightV1.model_validate(raw)


def test_formal_sources_present() -> None:
    report = verify_sff_formal_sources(run_lean=False)
    assert len(report["templates"]) == len(SFF_FORMAL_TEMPLATES)


def test_load_or_build_alias_is_fail_closed() -> None:
    # write_if_missing must not silently re-stamp proved without Lean.
    obs = load_or_build_sff_formal_obligations(write_if_missing=True)
    assert len(obs) == len(SFF_FORMAL_TEMPLATES)


def test_stale_preflight_raises(tmp_path, monkeypatch) -> None:
    from slm_training.harnesses.experiments import semantic_factor_formal as sff

    # Point formal dir at empty temp → missing preflight.
    monkeypatch.setattr(sff, "SFF_FORMAL_DIR", tmp_path)
    with pytest.raises(SFFFormalError, match="missing formal preflight"):
        sff.load_sff_formal_obligations()


def test_invalid_preflight_schema_raises(tmp_path, monkeypatch) -> None:
    from slm_training.harnesses.experiments import semantic_factor_formal as sff

    monkeypatch.setattr(sff, "SFF_FORMAL_DIR", tmp_path)
    # Write a minimal invalid file for the first template id.
    tid = SFF_FORMAL_TEMPLATES[0]["template_id"]
    path = tmp_path / f"{tid.replace('.', '_')}.json"
    path.write_text('{"schema_version": "FormalPreflightV1"}\n', encoding="utf-8")
    with pytest.raises(SFFFormalError, match="FormalPreflightV1"):
        sff.validate_committed_preflights()


def test_golden_filter_legal_matches_lean_guard() -> None:
    g = _golden()["filter_legal"]
    legal = set(g["legal"])
    keys_in = list(g["keys_in"])
    keys_out = [k for k in keys_in if k in legal]
    assert keys_out == g["keys_out"]
    assert_score_keys_legal({str(k): 1.0 for k in keys_out}, [str(x) for x in legal])


def test_golden_scorer_control_plane() -> None:
    g = _golden()
    scorer = SemanticResidualScorerV1(
        SemanticResidualScorerConfig(
            representation=FactorRepresentation.NONE,
            decode_apply=True,
            alpha=1.5,
        )
    )
    snap = project_program_factors(
        program_id="s",
        components=(("root", "Card"),),
        legal_candidate_ids=("=",),
        coverage="complete",
    )
    out = scorer.score(
        legal_candidate_ids=("=",),
        coverage="complete",
        snapshot=snap,
        expected_source_digest=snap.source_digest,
        baseline_scores={"=": 1.0},
    )
    assert out.singleton_skip
    assert out.applications == 0

    out2 = scorer.score(
        legal_candidate_ids=("a", "b"),
        coverage="incomplete",
        snapshot=None,
        baseline_scores={},
    )
    assert out2.incomplete_skip
    assert out2.unknown_preserved
    assert out2.applications == 0

    scorer_off = SemanticResidualScorerV1(
        SemanticResidualScorerConfig(
            representation=FactorRepresentation.DIRECT_FACTORS,
            decode_apply=False,
            alpha=1.5,
        )
    )
    snap2 = project_program_factors(
        program_id="m",
        components=(("r", "Card"), ("g", "Button"), ("d", "List")),
        binder_edges=(("r", "g", "USE"),),
        legal_candidate_ids=("g", "d", "n"),
        coverage="complete",
    )
    out3 = scorer_off.score(
        legal_candidate_ids=("g", "d", "n"),
        coverage="complete",
        snapshot=snap2,
        expected_source_digest=snap2.source_digest,
        baseline_scores={"g": 0.1, "d": 0.4, "n": 0.05},
    )
    assert out3.applications == g["decode_off_zero_apps"]["non_singleton_expected_apps"]


def test_golden_incidence_degrees_and_column_stochastic() -> None:
    g = _golden()["incidence_golden_B"]
    cols = g["columns"]
    B = np.array(cols, dtype=np.float64).T
    assert B.shape == (3, 2)
    d_e = B.sum(axis=0)
    d_v = B.sum(axis=1)
    assert list(d_e.astype(int)) == g["col_degrees"]
    assert list(d_v.astype(int)) == g["row_degrees"]
    # Derived target matches Lean sColumnSumTarget = ∏d_e · ∏d_v
    target = int(np.prod(d_e) * np.prod(d_v))
    assert target == g["s_column_sum_target"]
    assert g["s_column_sum_nums"] == [target] * B.shape[0]
    S, col_sums, _ = column_stochastic_S(B)
    assert np.allclose(col_sums, 1.0, atol=1e-9)
    assert np.allclose(S.sum(axis=0), 1.0, atol=1e-9)


def test_golden_soft_token_collision() -> None:
    g = _golden()["soft_token"]
    a = sum(int(x) * int(c) for x, c in zip(g["a"], g["coeffs"], strict=True))
    b = sum(int(x) * int(c) for x, c in zip(g["b"], g["coeffs"], strict=True))
    assert (a == b) == g["collision"]
    assert g["a"] != g["b"]


def test_golden_rank_inside_legal_matches_lean_guard() -> None:
    g = _golden()["rank_inside_legal"]
    winner, _ = rank_inside_legal_residual(g["scores"], g["legal_mask"])
    assert winner == g["winner_index"]
    none_w, _ = rank_inside_legal_residual(
        g["all_illegal_scores"], g["all_illegal_mask"]
    )
    assert none_w is g["all_illegal_winner"]


def test_membership_roundtrip_matches_golden_encode_reconstruct() -> None:
    """Assert the exact Lean reconstruct_encode_example fixture."""

    g = _golden()["membership_roundtrip"]
    factors = g["factors"]
    # Pure encode: (node, factor_id) pairs in factor order.
    enc: list[list[int]] = []
    for f in factors:
        for n in f["nodes"]:
            enc.append([int(n), int(f["factor_id"])])
    assert enc == g["encode_incidence"]
    # Pure reconstruct from incidence.
    reconstructed: dict[str, list[int]] = {}
    for node, fid in enc:
        reconstructed.setdefault(str(fid), []).append(int(node))
    assert reconstructed == g["reconstructed_nodes"]
    for f in factors:
        assert reconstructed[str(f["factor_id"])] == f["nodes"]


def test_membership_roundtrip_via_factor_node_view() -> None:
    components = (("1", "A"), ("2", "B"), ("3", "C"))
    snap = project_program_factors(
        program_id="rt",
        components=components,
        binder_edges=(("1", "2", "USE"), ("2", "3", "USE")),
        legal_candidate_ids=("1", "2", "3"),
        coverage="complete",
    )
    view = build_factor_representation(
        snap, FactorRepresentation.FACTOR_NODE, seed=0
    )
    assert view is not None
    reconstructed = reconstruct_membership(view)
    expected = {f.factor_id: f.membership() for f in snap.factors}
    assert reconstructed == expected


def test_role_shuffle_preserves_membership_like_lean() -> None:
    g = _golden()["role_shuffle"]
    # Lean model: rotate roles, nodes fixed.
    ports = list(g["ports"])
    k = int(g["k"])
    roles = [p["role"] for p in ports]
    nodes = [p["node"] for p in ports]
    kk = k % len(roles)
    roles_after = roles[kk:] + roles[:kk]
    assert roles_after == g["roles_after"]
    assert nodes == g["membership_after"]
    # Python harness shuffle_port_roles preserves membership multiset.
    snap = project_program_factors(
        program_id="sh",
        components=(("a", "A"), ("b", "B"), ("c", "C")),
        binder_edges=(("a", "b", "USE"), ("b", "c", "DEFINITION")),
        legal_candidate_ids=("a", "b", "c"),
        coverage="complete",
    )
    before = {f.factor_id: f.membership() for f in snap.factors}
    shuffled = shuffle_port_roles(snap.factors, seed=1)
    after = {f.factor_id: f.membership() for f in shuffled}
    assert before == after
