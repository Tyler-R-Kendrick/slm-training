"""Canonical lineage manifest fixture shared by lifecycle tests."""

from __future__ import annotations

from slm_training.lineage.records import RunManifest


def run_manifest(run_id: str, *, parent_ids: tuple[str, ...] = ()) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        track="twotower",
        parent_ids=parent_ids,
        base_model_id="base",
        base_model_revision="abc123",
        architecture_sha="arch",
        tokenizer_sha="tok",
        parameter_shapes_sha="shapes",
        data_snapshot_sha="data",
        eval_snapshot_sha="eval",
        recipe_sha="recipe",
        code_sha="code",
        seed=1,
        hardware={"device": "cpu"},
        artifact_uris=(),
        metrics={},
        lifecycle_state="running",
        initialization="parent" if parent_ids else "scratch",
        recipe={"lr": 1e-3},
        created_at="2026-01-01T00:00:00Z",
    )
