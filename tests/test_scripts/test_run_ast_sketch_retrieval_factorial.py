from __future__ import annotations

from scripts import run_ast_sketch_retrieval_factorial


def test_plan_only_mode(tmp_path) -> None:
    out = tmp_path / "plan"
    assert (
        run_ast_sketch_retrieval_factorial.main(
            ["--mode", "plan-only", "--output-dir", str(out), "--seeds", "0"]
        )
        == 0
    )
    assert (out / "ast_sketch_retrieval_manifest.json").exists()
    assert (out / "ast_sketch_retrieval_report.json").exists()
    assert (out / "ast_sketch_retrieval_report.md").exists()


def test_fixture_mode_with_parent(tmp_path) -> None:
    out = tmp_path / "fixture"
    assert (
        run_ast_sketch_retrieval_factorial.main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(out),
                "--seeds",
                "0,1",
                "--include-controls",
                "--parent-checkpoint-uri",
                "hf://bucket/checkpoint/ref.json",
            ]
        )
        == 0
    )
    assert "slm133_fixture" in (out / "ast_sketch_retrieval_report.json").read_text()


def test_frontier_mode_emits_partial_fixture(tmp_path) -> None:
    out = tmp_path / "frontier"
    assert (
        run_ast_sketch_retrieval_factorial.main(
            [
                "--mode",
                "frontier",
                "--output-dir",
                str(out),
                "--seeds",
                "0",
                "--parent-checkpoint-uri",
                "hf://bucket/checkpoint/ref.json",
            ]
        )
        == 0
    )
    assert (out / "ast_sketch_retrieval_report.json").exists()


def test_local_verified_patch_diagnostic_writes_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        run_ast_sketch_retrieval_factorial,
        "publish_agentv_evaluation",
        lambda *_args, **_kwargs: {"criteria": {"pass": True}},
    )
    assert (
        run_ast_sketch_retrieval_factorial.main(
            [
                "--mode",
                "verified-patch-local",
                "--output-dir",
                str(tmp_path / "out"),
                "--docs-dir",
                str(tmp_path / "docs"),
            ]
        )
        == 0
    )
    assert (tmp_path / "out" / "slm295_verified_patch_v1.json").exists()
    assert (tmp_path / "docs" / "iter-slm295-verified-patch-v1-20260725.md").exists()
