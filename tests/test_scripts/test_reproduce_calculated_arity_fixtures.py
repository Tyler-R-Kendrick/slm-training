"""Regression tests for the CAP5 deterministic fixture reproduction script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reproduce_calculated_arity_fixtures import main


@pytest.fixture
def summary(tmp_path: Path) -> dict:
    out = tmp_path / "cap_repro"
    rc = main(["--out", str(out), "--verify-expected"])
    assert rc == 0, "reproduce script should exit 0"
    return json.loads((out / "cap_repro_summary.json").read_text())


def test_summary_schema(summary: dict) -> None:
    assert summary["schema"] == "cap5_repro_summary/v1"
    assert summary["ok"] is True
    assert summary["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert "analysis.arity.diffusion_graph" in summary["version_stamp"]["components"]


def test_exact_arity_matches_committed_fixture(summary: dict) -> None:
    arity = summary["sections"]["exact_arity"]
    assert arity["ok"] is True
    counts = arity["counts"]
    assert counts["canonical_asts"] == 400
    assert counts["trie_states"] == 844
    assert counts["minimized_states"] == 28
    assert counts["action_alphabet"] == 8
    assert counts["scope_signatures"] == 3
    assert counts["min_k"] == 3


def test_coding_constructions_verified(summary: dict) -> None:
    coding = summary["sections"]["coding"]
    assert coding["ok"] is True
    assert coding["mds_7_4_2_3"]["ok"] is True
    assert coding["hamming_7_4_3"]["ok"] is True
    assert coding["singleton_bound_q7_n4_d3"] == 49
    assert coding["sphere_packing_M49_q7_n4_t1"] is True
    assert coding["ternary_ecoc_width_8_detect"] == 3
    assert coding["minimum_margin_trit_planes_4_1"] == 3


def test_task_quotient_fixture(summary: dict) -> None:
    tq = summary["sections"]["task_quotient"]
    assert tq["ok"] is True
    assert tq["state_count"] == 3
    assert tq["quotient_size"] > 0


def test_conditional_rate_fixture(summary: dict) -> None:
    cr = summary["sections"]["conditional_rate"]
    assert cr["ok"] is True
    assert 0 <= cr["fano_lower_bound_error"] <= 1
    assert cr["posterior_max"] >= cr["posterior_mean"]


def test_durable_certificates_indexed(summary: dict) -> None:
    certs = summary["sections"]["durable_certificates"]
    assert certs["ok"] is True
    assert any("cap" in c["path"] for c in certs["certificates"])
