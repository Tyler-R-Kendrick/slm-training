from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.model_cycle import main as model_cycle_main
from slm_training.integrations.nemo_rl import (
    BUCKET_MOUNT,
    NEMO_RL_GIT_SHA,
    NEMO_RL_IMAGE,
    NEMO_RL_VERSION,
    build_entrypoint_script,
    build_hf_jobs_command,
    parse_job_status,
    score_openui,
    validate_train_summary,
    write_train_summary,
)
from slm_training.lineage.records import RunManifest
from slm_training.lineage.store import LineageStore
from slm_training.lineage.tracks import CAUSAL_BASE_CANDIDATES


GOLD = (
    'root = Stack([title, body], "column")\n'
    'title = TextContent(":alert.title")\n'
    'body = TextContent(":alert.body")'
)


def causal_manifest(run_id: str = "nemo-smoke") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        track="causal_lm",
        parent_ids=(),
        base_model_id="Qwen/Qwen3-0.6B",
        base_model_revision=CAUSAL_BASE_CANDIDATES["Qwen/Qwen3-0.6B"],
        architecture_sha="arch",
        tokenizer_sha="tokenizer",
        parameter_shapes_sha="shapes",
        data_snapshot_sha="data",
        eval_snapshot_sha="eval",
        recipe_sha="recipe",
        code_sha="a" * 40,
        seed=7,
        hardware={},
        artifact_uris=(),
        metrics={},
        lifecycle_state="running",
        initialization="scratch",
        recipe={"rank": 16},
        created_at="2026-07-14T00:00:00Z",
    )


def test_openui_reward_uses_visible_slot_contract() -> None:
    reward = score_openui(
        GOLD,
        gold_openui=GOLD,
        slot_inventory=(":alert.title", ":alert.body"),
    )
    assert reward.to_dict() == {
        "parse": 1.0,
        "placeholder_fidelity": 1.0,
        "structural_similarity": 1.0,
        "composite": 1.0,
    }
    assert (
        score_openui(
            "not OpenUI", gold_openui=GOLD, slot_inventory=(":alert.title",)
        ).composite
        == 0.0
    )


def test_hf_job_recipe_is_pinned_and_secret_safe() -> None:
    script = build_entrypoint_script(
        run_id="nemo-smoke",
        base_model_id="Qwen/Qwen3-0.6B",
        base_model_revision="b" * 40,
        repo_url="https://example.com/repo.git",
        revision="a" * 40,
        data_path="fixtures/nemo_rl/openui_smoke.jsonl",
        checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
        seed=7,
    )
    command = build_hf_jobs_command(entrypoint=script)
    assert NEMO_RL_GIT_SHA in script
    assert "grpo.max_num_steps=1" in script
    assert "lora_cfg.enabled=true" in script
    assert command[command.index("--secrets") + 1] == "HF_TOKEN"
    assert NEMO_RL_IMAGE in command
    assert f"hf://buckets/TKendrick/OpenUI:{BUCKET_MOUNT}" in command
    with pytest.raises(ValueError, match="exact 40-character"):
        build_entrypoint_script(
            run_id="nemo-smoke",
            base_model_id="Qwen/Qwen3-0.6B",
            base_model_revision="b" * 40,
            repo_url="https://example.com/repo.git",
            revision="main",
            data_path="fixtures/nemo_rl/openui_smoke.jsonl",
            checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
            seed=7,
        )


def test_summary_and_hf_status_validation(tmp_path: Path) -> None:
    output = tmp_path / "run"
    checkpoint = output / "checkpoints" / "step-1" / "adapter.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"adapter")
    summary_path = output / "train_summary.json"
    write_train_summary(
        summary_path,
        run_id="nemo-smoke",
        bucket_uri="hf://buckets/TKendrick/OpenUI/checkpoints/nemo-smoke/nemo_rl",
    )
    payload = json.loads(summary_path.read_text())
    validate_train_summary(payload, run_id="nemo-smoke")
    assert payload["nemo_rl_version"] == NEMO_RL_VERSION
    assert parse_job_status({"status": {"stage": "COMPLETED"}}) == "completed"
    with pytest.raises(ValueError, match="checkpoint"):
        validate_train_summary({**payload, "checkpoint_written": False}, run_id="nemo-smoke")


def test_lineage_dry_run_and_reconcile(tmp_path: Path, capsys) -> None:
    root = tmp_path / "lineage"
    store = LineageStore(root)
    store.create_run(causal_manifest())
    assert (
        model_cycle_main(
            [
                "--lineage-root",
                str(root),
                "submit-nemo",
                "--run-id",
                "nemo-smoke",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["kind"] == "hardware_smoke"
    assert plan["nemo_rl_git_sha"] == NEMO_RL_GIT_SHA

    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"status": {"stage": "COMPLETED"}}))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": "nemo-smoke",
                "kind": "hardware_smoke",
                "nemo_rl_version": NEMO_RL_VERSION,
                "nemo_rl_git_sha": NEMO_RL_GIT_SHA,
                "checkpoint_bucket": "hf://buckets/TKendrick/OpenUI/checkpoints/nemo-smoke/nemo_rl",
                "checkpoint_written": True,
            }
        )
    )
    assert (
        model_cycle_main(
            [
                "--lineage-root",
                str(root),
                "reconcile-nemo",
                "--run-id",
                "nemo-smoke",
                "--job-id",
                "job-1",
                "--status-json",
                str(status_path),
                "--summary",
                str(summary_path),
            ]
        )
        == 0
    )
    reconciled = store.load_run("nemo-smoke")
    assert reconciled.lifecycle_state == "screened"
    assert reconciled.legacy_kind == "hardware_smoke"
    assert reconciled.recipe["nemo_rl"]["claim"] == "hardware_smoke"


def test_metadata_revisions_are_immutable_and_hardware_smoke_cannot_promote(
    tmp_path: Path,
) -> None:
    store = LineageStore(tmp_path)
    store.create_run(causal_manifest())
    updated = store.record_run_metadata(
        "nemo-smoke",
        recipe={"nemo_rl": {"job_id": "job-1"}},
        artifact_uris=("hf-job://job-1",),
        legacy_kind="hardware_smoke",
    )
    assert updated.recipe_sha != "recipe"
    assert (tmp_path / "runs/nemo-smoke/manifest.json").exists()
    validated = RunManifest(
        **{
            **causal_manifest("validated-smoke").to_dict(),
            "lifecycle_state": "validated",
            "legacy_kind": "hardware_smoke",
        }
    )
    store.create_run(validated)
    with pytest.raises(ValueError, match="hardware-smoke"):
        model_cycle_main(
            [
                "--lineage-root",
                str(tmp_path),
                "promote",
                "--run-id",
                "validated-smoke",
                "--report",
                "missing",
            ]
        )
