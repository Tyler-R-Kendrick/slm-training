from pathlib import Path

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.feature_flags import (
    catalog,
    load_snapshot,
    resolve,
    save_snapshot,
)


def test_all_behavior_config_fields_have_generated_openfeature_flags() -> None:
    rows = catalog()["flags"]
    keys = {row["key"] for row in rows}
    assert "slm.grammar_ltr_primary" in keys
    assert "slm.semantic_role_decode_weight" in keys
    assert "slm.run_id" not in keys
    assert len(rows) > 200
    assert all("field" not in row for row in rows)


def test_openfeature_snapshot_round_trips_typed_levers(tmp_path: Path) -> None:
    config = ModelBuildConfig(
        train_dir=tmp_path / "train",
        run_root=tmp_path / "runs",
        run_id="flags",
        grammar_ltr_primary=True,
        lr=0.0002,
        diffusion_length_buckets=(32, 64),
    )
    resolved, snapshot = resolve(config, phase="training")

    assert resolved.grammar_ltr_primary is True
    assert resolved.lr == 0.0002
    assert resolved.diffusion_length_buckets == (32, 64)
    assert all("field" not in row for row in snapshot["flags"])
    saved = save_snapshot(config.run_dir, snapshot)
    assert saved["snapshots"]["training"]["registry_revision"] == catalog()["revision"]
    assert load_snapshot(config.run_dir) == saved
