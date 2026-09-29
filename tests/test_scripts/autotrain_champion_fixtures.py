"""Storage-only champion fixtures with explicit synthetic exposure; no trained capability."""

import json
from pathlib import Path
from types import SimpleNamespace


def _p8_control_run(
    root: Path,
    campaign_id: str,
    *,
    steps: int = 22,
    stopped_on: str = "steps",
    train_dir: Path | None = None,
    with_checkpoint: bool = True,
) -> Path:
    prefix = campaign_id.replace("continuous-loop-", "c")
    run = root / campaign_id / "runs" / f"{prefix}-control"
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if with_checkpoint:
        from slm_training.harness_core.checkpoint_bundle import stage_checkpoint_bundle
        from tests.test_harnesses.model_build.test_checkpoint_bundle import _checkpoint

        source = _checkpoint(run / "checkpoints/source", campaign_id.encode())
        bundle_root = run / "checkpoints/trial"
        digest = stage_checkpoint_bundle(bundle_root, source, {
            "role": "trial_cursor", "ancestor_exposure": [], "exposure": [{
                "snapshot_sha256": "f" * 64, "record_count": 101,
                "consumed_examples": steps * 2, "example_exposure_epochs": steps * 2 / 101,
                "tokens": {"prompt": 0, "target": 0},
            }],
        })
        checkpoint = bundle_root / "bundles" / digest / "last.pt"
        (run / "checkpoints/last.pt").write_bytes(source.read_bytes())
    (run / "train_summary.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "stopped_on": stopped_on,
                "elapsed_wall_seconds": 3.36,
                "record_count": 101,
                "train_dir": str(train_dir) if train_dir else None,
                "data_manifest_sha": "f" * 64,
                "initialized_weight_count": 0,
                "track": {"trainable_params": 1_608_962},
            }
        ),
        encoding="utf-8",
    )
    return run


def _p8_policy(**warm: object) -> SimpleNamespace:
    block: dict[str, object] = {"enabled": True, "max_cumulative_epochs": 50}
    block.update(warm)
    return SimpleNamespace(
        measurement={
            "warm_start": block,
            "thrash_timing": {"min_train_floor_seconds": 20},
        }
    )

