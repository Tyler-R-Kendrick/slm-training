from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.model_cycle import main as model_cycle_main
from slm_training.integrations.molt_rl import (
    BUCKET_MOUNT,
    MOLT_GIT_SHA,
    MOLT_IMAGE,
    MOLT_VERSION,
    build_entrypoint_script,
    build_hf_jobs_command,
    decode_label,
    encode_label,
    normalize_experience,
    prepare_prompt_dataset,
    validate_trace_file,
    validate_train_summary,
    write_train_summary,
)
from slm_training.integrations.nemo_rl import score_openui as nemo_score_openui
from slm_training.integrations.openui_rl import score_openui
from slm_training.lineage.records import RunManifest
from slm_training.lineage.store import LineageStore
from slm_training.lineage.tracks import CAUSAL_BASE_CANDIDATES

GOLD = 'root = TextContent(":profile.name")'


def causal_manifest(run_id: str = "molt-smoke") -> RunManifest:
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


def test_molt_uses_shared_reward_and_label_contract() -> None:
    encoded = encode_label(GOLD, [":profile.name"])
    assert decode_label(encoded) == {
        "gold_openui": GOLD,
        "slot_inventory": [":profile.name"],
    }
    assert score_openui(
        GOLD, gold_openui=GOLD, slot_inventory=[":profile.name"]
    ) == nemo_score_openui(
        GOLD, gold_openui=GOLD, slot_inventory=[":profile.name"]
    )


def test_prompt_adapter_and_experience_normalizer(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Build a profile"}],
                "gold_openui": GOLD,
                "slot_inventory": [":profile.name"],
            }
        )
        + "\n"
    )
    prepared = tmp_path / "prepared.jsonl"
    assert prepare_prompt_dataset(source, prepared) == 1
    assert json.loads(prepared.read_text())["input"] == "Build a profile"

    class Tokenizer:
        @staticmethod
        def decode(tokens, *, skip_special_tokens):
            assert skip_special_tokens is True
            return "completion:" + ",".join(str(item) for item in tokens)

    experience = SimpleNamespace(
        sequences=[10, 11, 12, 13],
        action_mask=[False, True, True],
        attention_mask=[1, 1, 1, 1],
        labels=[encode_label(GOLD, [":profile.name"])],
        prompts=["Build a profile"],
        info={
            "parse": [1.0],
            "placeholder_fidelity": [0.5],
            "structural_similarity": [0.75],
            "composite": [0.7875],
        },
        rewards=[0.7875],
        truncated=[False],
        group_ids=["group-1"],
        rollout_ids=["rollout-1"],
        rollout_log_probs=[0.0, -0.1, -0.2],
    )
    row = normalize_experience(
        experience, tokenizer=Tokenizer(), run_id="molt-smoke", step=1
    )
    assert row["action_token_ids"] == [12, 13]
    assert row["completion"] == "completion:12,13"
    assert row["prompt_tokens"] == 2
    assert row["completion_tokens"] == 2
    assert row["rollout_logprobs"] == [-0.1, -0.2]
    trace = tmp_path / "rl_traces.jsonl"
    trace.write_text(json.dumps(row) + "\n")
    assert validate_trace_file(trace, run_id="molt-smoke") == 1


def test_molt_job_recipe_is_pinned_and_secret_safe(approved_rl_report) -> None:
    script = build_entrypoint_script(
        run_id="molt-smoke",
        base_model_id="Qwen/Qwen3-0.6B",
        base_model_revision="b" * 40,
        repo_url="https://example.com/repo.git",
        revision="a" * 40,
        data_path="fixtures/nemo_rl/openui_smoke.jsonl",
        checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
        seed=7,
        rl_readiness_report=approved_rl_report.model_dump(mode="json"),
    )
    command = build_hf_jobs_command(entrypoint=script)
    assert MOLT_GIT_SHA in script
    assert "python -m scripts.run_molt_rl" in script
    assert "trap sync_artifacts EXIT" in script
    assert command[command.index("--secrets") + 1] == "HF_TOKEN"
    assert MOLT_IMAGE in command
    assert f"hf://buckets/TKendrick/OpenUI:{BUCKET_MOUNT}" in command
    with pytest.raises(ValueError, match="exact 40-character"):
        build_entrypoint_script(
            run_id="molt-smoke",
            base_model_id="Qwen/Qwen3-0.6B",
            base_model_revision="b" * 40,
            repo_url="https://example.com/repo.git",
            revision="main",
            data_path="fixtures/nemo_rl/openui_smoke.jsonl",
            checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
            seed=7,
            rl_readiness_report=approved_rl_report.model_dump(mode="json"),
        )


def test_summary_validation(tmp_path: Path) -> None:
    output = tmp_path / "run"
    checkpoint = output / "checkpoints" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    summary = output / "train_summary.json"
    write_train_summary(
        summary,
        run_id="molt-smoke",
        bucket_uri="hf://buckets/TKendrick/OpenUI/checkpoints/molt-smoke/molt_rl",
        trace_count=4,
    )
    payload = json.loads(summary.read_text())
    validate_train_summary(payload, run_id="molt-smoke")
    assert payload["molt_version"] == MOLT_VERSION
    assert payload["recipe"]["update"] == "full_parameter_fsdp"
    with pytest.raises(ValueError, match="traces"):
        validate_train_summary({**payload, "trace_count": 0}, run_id="molt-smoke")


def test_lineage_dry_run_paid_guard_and_reconcile(
    tmp_path: Path, capsys, approved_rl_report_path
) -> None:
    root = tmp_path / "lineage"
    store = LineageStore(root)
    store.create_run(causal_manifest())
    base = [
        "--lineage-root",
        str(root),
        "submit-molt",
        "--run-id",
        "molt-smoke",
        "--rl-readiness-report",
        str(approved_rl_report_path),
    ]
    assert model_cycle_main([*base, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["kind"] == "hardware_smoke"
    assert plan["molt_git_sha"] == MOLT_GIT_SHA
    with pytest.raises(ValueError, match="ack-paid-gpu"):
        model_cycle_main(base)

    status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": {"stage": "COMPLETED"}}))
    bucket_uri = "hf://buckets/TKendrick/OpenUI/checkpoints/molt-smoke/molt_rl"
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_id": "molt-smoke",
                "kind": "hardware_smoke",
                "molt_version": MOLT_VERSION,
                "molt_git_sha": MOLT_GIT_SHA,
                "molt_image": MOLT_IMAGE,
                "checkpoint_bucket": bucket_uri,
                "checkpoint_written": True,
                "raw_traces": f"{bucket_uri}/traces/raw/",
                "normalized_traces": f"{bucket_uri}/rl_traces.jsonl",
                "trace_count": 1,
            }
        )
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "engine": "molt",
                "run_id": "molt-smoke",
                "rewards": {},
                "action_token_ids": [],
            }
        )
        + "\n"
    )
    run_root = tmp_path / "runs"
    assert (
        model_cycle_main(
            [
                "--lineage-root",
                str(root),
                "reconcile-molt",
                "--run-id",
                "molt-smoke",
                "--job-id",
                "job-1",
                "--status-json",
                str(status),
                "--summary",
                str(summary),
                "--trace",
                str(trace),
                "--run-root",
                str(run_root),
            ]
        )
        == 0
    )
    reconciled = store.load_run("molt-smoke")
    assert reconciled.lifecycle_state == "screened"
    assert reconciled.recipe["molt_rl"]["claim"] == "hardware_smoke"
    trace_ref = json.loads((run_root / "molt-smoke" / "trace.json").read_text())
    assert reconciled.trace_id == trace_ref["trace_id"]
    assert (
        Path(trace_ref["bundle"]) / "domain" / "molt" / "rl_traces.jsonl"
    ).is_file()
