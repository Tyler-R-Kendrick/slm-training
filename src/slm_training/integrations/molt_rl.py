"""Pinned Molt RL job recipe, dataset adapter, and durable trace normalization."""

from __future__ import annotations

import base64
import json
import re
import shlex
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.integrations.nemo_rl import (
    BUCKET_MOUNT,
    DEFAULT_BUCKET,
    DEFAULT_REPO,
    parse_job_status,
)

MOLT_VERSION = "0.1.2"
MOLT_GIT_SHA = "21c1b8921b73f5c8317b5fc9e359e9a1b7d255d2"
MOLT_REPO = "https://github.com/NVIDIA-NeMo/labs-molt.git"
MOLT_IMAGE = (
    "hijkzzz/molt:0.1.2@"
    "sha256:b9c82365b0c65e9cd4daf0addc34c9a5eba89cfc4593fa2e480246dc7c1dfcd2"
)
DEFAULT_FLAVOR = "h200x2"
TRACE_SCHEMA_VERSION = 1

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_STEP_RE = re.compile(r"rollout_step(\d+)\.pt$")


def encode_label(gold_openui: str, slot_inventory: Iterable[str]) -> str:
    return json.dumps(
        {"gold_openui": gold_openui, "slot_inventory": list(slot_inventory)},
        sort_keys=True,
    )


def decode_label(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("gold_openui"), str):
        raise ValueError("Molt label must contain gold_openui")
    slots = value.get("slot_inventory") or []
    if not isinstance(slots, list) or not all(isinstance(item, str) for item in slots):
        raise ValueError("Molt label slot_inventory must be a string list")
    return {"gold_openui": value["gold_openui"], "slot_inventory": slots}


def prepare_prompt_dataset(source: Path, destination: Path) -> int:
    """Convert the shared OpenUI smoke rows to Molt's input/label columns."""
    rows: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            messages = value.get("messages") or []
            if not messages or not isinstance(messages[0].get("content"), str):
                raise ValueError("OpenUI RL row must contain messages[0].content")
            rows.append(
                json.dumps(
                    {
                        "input": messages[0]["content"],
                        "label": encode_label(
                            str(value["gold_openui"]),
                            value.get("slot_inventory") or [],
                        ),
                    },
                    sort_keys=True,
                )
            )
    if not rows:
        raise ValueError("OpenUI RL dataset is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def build_entrypoint_script(
    *,
    run_id: str,
    base_model_id: str,
    base_model_revision: str,
    repo_url: str,
    revision: str,
    data_path: str,
    checkpoint_bucket: str,
    seed: int,
    rl_readiness_report: Mapping[str, Any],
) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"unsafe run id: {run_id!r}")
    if not _GIT_SHA_RE.fullmatch(revision):
        raise ValueError("revision must be an exact 40-character lowercase git SHA")
    if not _GIT_SHA_RE.fullmatch(base_model_revision):
        raise ValueError("base model revision must be an exact git SHA")
    if "/" not in checkpoint_bucket.removeprefix("hf://buckets/"):
        raise ValueError("checkpoint bucket must be hf://buckets/<owner>/<name>")
    q = shlex.quote
    output_root = f"/workspace/molt-runs/{run_id}"
    bucket_uri = f"{checkpoint_bucket}/checkpoints/{run_id}/molt_rl"
    readiness_b64 = base64.b64encode(
        json.dumps(dict(rl_readiness_report), sort_keys=True).encode()
    ).decode()
    return f"""set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/slm-training:${{PYTHONPATH:-}}
export SLM_MOLT_RUN_ID={q(run_id)}
export SLM_MOLT_OUTPUT_ROOT={q(output_root)}
export SLM_MOLT_BUCKET_URI={q(bucket_uri)}
export SLM_MOLT_BASE_MODEL_ID={q(base_model_id)}
export SLM_MOLT_BASE_MODEL_REVISION={q(base_model_revision)}
export SLM_MOLT_CODE_REVISION={q(revision)}
export SLM_MOLT_DATA_PATH={q('/workspace/slm-training/' + data_path)}
export SLM_MOLT_SEED={int(seed)}
export SLM_MOLT_RL_READINESS_B64={q(readiness_b64)}

rm -rf /workspace/slm-training /workspace/labs-molt
git clone --filter=blob:none {q(repo_url)} /workspace/slm-training
git -C /workspace/slm-training checkout {q(revision)}
test "$(git -C /workspace/slm-training rev-parse HEAD)" = {q(revision)}
git clone --filter=blob:none {q(MOLT_REPO)} /workspace/labs-molt
git -C /workspace/labs-molt checkout {q(MOLT_GIT_SHA)}
test "$(git -C /workspace/labs-molt rev-parse HEAD)" = {q(MOLT_GIT_SHA)}
python -m pip install --no-deps -e /workspace/labs-molt -e /workspace/slm-training

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={base_model_id!r},
    revision={base_model_revision!r},
    local_dir="/workspace/base-model",
)
PY

DEST={q(BUCKET_MOUNT + '/checkpoints/' + run_id + '/molt_rl')}
mkdir -p {q(output_root)} "$DEST"
sync_artifacts() {{ rsync -a {q(output_root + '/')} "$DEST/" || true; }}
trap sync_artifacts EXIT
python -m scripts.run_molt_rl
test -f {q(output_root + '/train_summary.json')}
test -f {q(output_root + '/rl_traces.jsonl')}
echo "Molt RL smoke artifacts: {bucket_uri}"
""".strip()


def build_hf_jobs_command(
    *,
    entrypoint: str,
    image: str = MOLT_IMAGE,
    flavor: str = DEFAULT_FLAVOR,
    timeout: str = "3m",
    checkpoint_bucket: str = DEFAULT_BUCKET,
) -> list[str]:
    if image != MOLT_IMAGE:
        raise ValueError(f"Molt image must remain pinned to {MOLT_IMAGE}")
    if timeout != "3m":
        raise ValueError("Molt RL timeout must be 3m")
    return [
        "hf",
        "jobs",
        "run",
        "--detach",
        "--flavor",
        flavor,
        "--timeout",
        timeout,
        "--secrets",
        "HF_TOKEN",
        "--label",
        "workflow=molt-rl-openui-smoke",
        "--volume",
        f"{checkpoint_bucket}:{BUCKET_MOUNT}",
        image,
        "bash",
        "-lc",
        entrypoint,
    ]


def _first(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and value.numel() == 1:
        return value.item()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value


def _vector(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return list(value or [])


def normalize_experience(
    experience: Any,
    *,
    tokenizer: Any,
    run_id: str,
    step: int,
    trace_id: str | None = None,
) -> dict[str, Any]:
    sequences = [int(item) for item in _vector(experience.sequences)]
    action_mask = [bool(item) for item in _vector(experience.action_mask)]
    action_ids = [token for token, keep in zip(sequences[1:], action_mask) if keep]
    attention_tokens = sum(bool(item) for item in _vector(experience.attention_mask))
    label = decode_label(str(_first(experience.labels)))
    info = getattr(experience, "info", {}) or {}
    rewards = {
        key: float(_first(info.get(key)) or 0.0)
        for key in (
            "parse",
            "placeholder_fidelity",
            "structural_similarity",
            "composite",
        )
    }
    if not rewards["composite"]:
        rewards["composite"] = float(_first(experience.rewards) or 0.0)
    rollout_logprobs = getattr(experience, "rollout_log_probs", None)
    selected_logprobs = None
    if rollout_logprobs is not None:
        values = _vector(rollout_logprobs)
        selected_logprobs = [
            float(value) for value, keep in zip(values, action_mask) if keep
        ]
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "engine": "molt",
        "run_id": run_id,
        "trace_id": trace_id,
        "step": step,
        "group_id": str(_first(getattr(experience, "group_ids", None)) or ""),
        "rollout_id": str(_first(getattr(experience, "rollout_ids", None)) or ""),
        "prompt": str(_first(experience.prompts) or ""),
        "completion": tokenizer.decode(action_ids, skip_special_tokens=True),
        "gold_openui": label["gold_openui"],
        "slot_inventory": label["slot_inventory"],
        "rewards": rewards,
        "prompt_tokens": max(0, attention_tokens - len(action_ids)),
        "completion_tokens": len(action_ids),
        "truncated": bool(_first(experience.truncated)),
        "action_token_ids": action_ids,
        "rollout_logprobs": selected_logprobs,
    }


def normalize_rollout_dumps(
    raw_dir: Path,
    destination: Path,
    *,
    tokenizer: Any,
    run_id: str,
    trace_id: str | None = None,
) -> int:
    import torch

    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("rollout_step*.pt")):
        match = _STEP_RE.search(path.name)
        if not match:
            continue
        experiences = torch.load(path, map_location="cpu", weights_only=False)
        rows.extend(
            normalize_experience(
                experience,
                tokenizer=tokenizer,
                run_id=run_id,
                step=int(match.group(1)),
                trace_id=trace_id,
            )
            for experience in experiences
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def validate_trace(row: Mapping[str, Any], *, run_id: str) -> None:
    if row.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported RL trace schema_version")
    if row.get("engine") != "molt" or row.get("run_id") != run_id:
        raise ValueError("RL trace identity mismatch")
    if not isinstance(row.get("rewards"), Mapping):
        raise ValueError("RL trace lacks rewards")
    if not isinstance(row.get("action_token_ids"), list):
        raise ValueError("RL trace lacks action_token_ids")


def validate_trace_file(path: Path, *, run_id: str) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            validate_trace(json.loads(line), run_id=run_id)
            count += 1
    if count < 1:
        raise ValueError("RL trace file is empty")
    return count


def write_train_summary(
    path: Path,
    *,
    run_id: str,
    bucket_uri: str,
    trace_count: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    checkpoint_dir = path.parent / "checkpoints"
    payload = {
        "run_id": run_id,
        "kind": "hardware_smoke",
        "molt_version": MOLT_VERSION,
        "molt_git_sha": MOLT_GIT_SHA,
        "molt_image": MOLT_IMAGE,
        "checkpoint_bucket": bucket_uri,
        "checkpoint_written": checkpoint_dir.is_dir()
        and any(item.is_file() for item in checkpoint_dir.rglob("*")),
        "raw_traces": f"{bucket_uri}/traces/raw/",
        "normalized_traces": f"{bucket_uri}/rl_traces.jsonl",
        "trace_count": trace_count,
        "recipe": {
            "algorithm": "grpo",
            "steps": 1,
            "prompts_per_step": 2,
            "generations_per_prompt": 2,
            "actor_gpus": 1,
            "vllm_gpus": 1,
            "update": "full_parameter_fsdp",
        },
    }
    payload.update(metadata or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_train_summary(payload: Mapping[str, Any], *, run_id: str) -> None:
    required = {
        "run_id": run_id,
        "kind": "hardware_smoke",
        "molt_version": MOLT_VERSION,
        "molt_git_sha": MOLT_GIT_SHA,
        "molt_image": MOLT_IMAGE,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"invalid train summary {key}: {payload.get(key)!r}")
    if not str(payload.get("checkpoint_bucket", "")).startswith("hf://buckets/"):
        raise ValueError("train summary lacks a durable checkpoint_bucket URI")
    bucket_uri = str(payload["checkpoint_bucket"])
    if not str(payload.get("raw_traces", "")).startswith(bucket_uri + "/"):
        raise ValueError("train summary lacks a durable raw trace URI")
    if not str(payload.get("normalized_traces", "")).startswith(bucket_uri + "/"):
        raise ValueError("train summary lacks a durable normalized trace URI")
    if not payload.get("checkpoint_written"):
        raise ValueError("train summary does not confirm a checkpoint")
    if int(payload.get("trace_count") or 0) < 1:
        raise ValueError("train summary does not confirm normalized RL traces")


__all__ = [
    "BUCKET_MOUNT",
    "DEFAULT_BUCKET",
    "DEFAULT_FLAVOR",
    "DEFAULT_REPO",
    "MOLT_GIT_SHA",
    "MOLT_IMAGE",
    "MOLT_VERSION",
    "build_entrypoint_script",
    "build_hf_jobs_command",
    "decode_label",
    "encode_label",
    "normalize_experience",
    "normalize_rollout_dumps",
    "parse_job_status",
    "prepare_prompt_dataset",
    "validate_trace_file",
    "validate_train_summary",
    "write_train_summary",
]
