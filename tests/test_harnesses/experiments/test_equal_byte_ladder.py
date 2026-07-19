"""CAP3-05: equal-byte width × precision ladder planner tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from slm_training.harnesses.experiments.ladder import (
    LadderPoint,
    estimate_bytes,
    model_build_config_for_point,
    plan_equal_byte_ladder,
)


def test_estimate_bytes_scales_with_width_and_format() -> None:
    """Modeled bytes should grow with width and shrink with lower bit width."""
    fp16_32 = estimate_bytes(32, 2, 1, 2, "fp16")
    fp16_64 = estimate_bytes(64, 2, 1, 2, "fp16")
    int4_64 = estimate_bytes(64, 2, 1, 2, "int4")
    assert fp16_64 > fp16_32
    assert int4_64 < fp16_64


def test_plan_equal_byte_ladder_finds_feasible_points() -> None:
    ladders = plan_equal_byte_ladder(
        byte_budgets=(80_000,),
        formats=("int4", "int8"),
        widths=(32, 64, 96, 128),
        horizons=(1.0,),
        tolerance=0.10,
    )
    assert len(ladders) == 2
    for lad in ladders:
        assert len(lad.points) == 1
        point = lad.points[0]
        assert point.byte_budget == 80_000
        assert point.precision_format in ("int4", "int8")
        assert point.actual_bytes is not None and point.actual_bytes > 0
        assert point.budget_delta is not None
        assert point.status in ("feasible", "infeasible")
        # At least one format should be feasible at 80 KB with the default grid.
    statuses = {lad.points[0].status for lad in ladders}
    assert "feasible" in statuses


def test_equal_byte_point_id_suffixes() -> None:
    point = LadderPoint(
        d_model=64,
        n_heads=2,
        context_layers=1,
        denoiser_layers=2,
        target_token_budget=1_000,
        byte_budget=64_000,
        precision_format="int4",
    )
    assert "_b64000" in point.point_id
    assert "_pint4" in point.point_id


def test_infeasible_format_when_budget_too_small() -> None:
    ladders = plan_equal_byte_ladder(
        byte_budgets=(1_000,),
        formats=("fp16",),
        widths=(32, 64),
        horizons=(1.0,),
        tolerance=0.01,
    )
    assert len(ladders) == 1
    point = ladders[0].points[0]
    assert point.status == "infeasible"
    assert point.actual_bytes is not None


def test_model_build_config_receives_equal_byte_metadata(tmp_path: Path) -> None:
    point = LadderPoint(
        d_model=32,
        n_heads=2,
        context_layers=1,
        denoiser_layers=2,
        target_token_budget=1_000,
        byte_budget=64_000,
        precision_format="int4",
    )
    ladder = plan_equal_byte_ladder(
        byte_budgets=(64_000,),
        formats=("int4",),
        widths=(32,),
        horizons=(1.0,),
    )[0]
    cfg = model_build_config_for_point(
        point,
        ladder,
        train_dir=tmp_path,
        test_dir=None,
        run_root=tmp_path / "runs",
        seed=0,
        steps=10,
        batch_size=2,
    )
    assert cfg.quant_format == "int4"
    assert cfg.use_dynamic_quant is True
    assert cfg.byte_budget == 64_000


def test_plan_only_cli_writes_manifest(tmp_path: Path) -> None:
    out = tmp_path / "plans"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_scaling_ladder",
            "--family",
            "equal-byte-precision",
            "--byte-budgets",
            "64KB",
            "--precision-formats",
            "int4",
            "--widths",
            "32,64",
            "--plan-only",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
    )
    manifests = list(out.glob("equal_byte_plan_*.json"))
    assert manifests, result.stderr
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["family"] == "equal-byte-precision"
    assert manifest["n_ladders"] == 1
    assert len(manifest["ladders"][0]["points"]) == 1
    point = manifest["ladders"][0]["points"][0]
    assert point["byte_budget"] == 64 * 1024
    assert point["precision_format"] == "int4"
    assert point["status"] in ("feasible", "infeasible")
