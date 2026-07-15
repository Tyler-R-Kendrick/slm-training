"""Pinned NeMo RL job recipe and OpenUI reward helpers."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.harnesses.model_build.eval_runner import structural_similarity
from slm_training.harnesses.preference import grammar_score

NEMO_RL_VERSION = "0.6.0"
NEMO_RL_GIT_SHA = "c339070fa3bfa83a5ac58ff80d73518911e14b81"
NEMO_RL_IMAGE = "nvcr.io/nvidia/nemo-rl:v0.6.0"
NEMO_RL_ROOT = "/opt/nemo-rl"
DEFAULT_REPO = "https://github.com/Tyler-R-Kendrick/slm-training.git"
DEFAULT_BUCKET = "hf://buckets/TKendrick/OpenUI"
BUCKET_MOUNT = "/mnt/openui-bucket"
DEFAULT_FLAVOR = "a10g-large"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class OpenUIReward:
    parse: float
    placeholder_fidelity: float
    structural_similarity: float
    composite: float

    def to_dict(self) -> dict[str, float]:
        return {
            "parse": self.parse,
            "placeholder_fidelity": self.placeholder_fidelity,
            "structural_similarity": self.structural_similarity,
            "composite": self.composite,
        }


def score_openui(
    prediction: str,
    *,
    gold_openui: str,
    slot_inventory: Sequence[str],
) -> OpenUIReward:
    """Score only visible structure and slot-contract information."""
    parse = grammar_score(prediction)
    predicted = set(extract_placeholders(prediction))
    expected = set(slot_inventory)
    if expected:
        fidelity = len(predicted & expected) / len(expected)
    else:
        fidelity = 1.0 if not predicted else 0.5
    structure = structural_similarity(prediction, gold_openui)
    composite = 0.0
    if parse > 0.0:
        composite = 0.45 * parse + 0.30 * fidelity + 0.25 * structure
    return OpenUIReward(
        parse=round(parse, 4),
        placeholder_fidelity=round(fidelity, 4),
        structural_similarity=round(structure, 4),
        composite=round(composite, 4),
    )


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
) -> str:
    """Build the hermetic one-step LoRA GRPO smoke command."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"unsafe run id: {run_id!r}")
    if not _GIT_SHA_RE.fullmatch(revision):
        raise ValueError("revision must be an exact 40-character lowercase git SHA")
    bucket_path = checkpoint_bucket.removeprefix("hf://buckets/")
    if "/" not in bucket_path:
        raise ValueError("checkpoint bucket must be hf://buckets/<owner>/<name>")
    q = shlex.quote
    expected_code = revision.split("+", 1)[0]
    output_root = f"/workspace/nemo-runs/{run_id}"
    bucket_uri = f"{checkpoint_bucket}/checkpoints/{run_id}/nemo_rl"
    return f"""set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/slm-training:${{PYTHONPATH:-}}
export NEMO_RL_ROOT={q(NEMO_RL_ROOT)}
export SLM_NEMO_RUN_ID={q(run_id)}
export SLM_NEMO_OUTPUT_ROOT={q(output_root)}
export SLM_NEMO_BUCKET_URI={q(bucket_uri)}
export SLM_NEMO_BASE_MODEL_ID={q(base_model_id)}
export SLM_NEMO_BASE_MODEL_REVISION={q(base_model_revision)}
export SLM_NEMO_CODE_REVISION={q(revision)}
export SLM_NEMO_DATA_PATH={q(data_path)}
export SLM_NEMO_SEED={int(seed)}

test "$(git -C {q(NEMO_RL_ROOT)} rev-parse HEAD)" = {q(NEMO_RL_GIT_SHA)}
rm -rf /workspace/slm-training
git clone --filter=blob:none {q(repo_url)} /workspace/slm-training
git -C /workspace/slm-training checkout {q(revision)}
test "$(git -C /workspace/slm-training rev-parse HEAD)" = {q(expected_code)}
python -m pip install --no-deps -e /workspace/slm-training

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={base_model_id!r},
    revision={base_model_revision!r},
    local_dir="/workspace/base-model",
)
PY

cd {q(NEMO_RL_ROOT)}
python -m scripts.run_nemo_rl \
  --config examples/configs/grpo_math_1B.yaml \
  grpo.max_num_steps=1 \
  grpo.num_prompts_per_step=2 \
  grpo.num_generations_per_prompt=2 \
  grpo.seed={int(seed)} \
  data.shuffle=false \
  data.train.dataset_name=ResponseDataset \
  data.train.data_path={q('/workspace/slm-training/' + data_path)} \
  data.train.split_validation_size=0 \
  data.validation=null \
  data.default.dataset_name=ResponseDataset \
  data.default.prompt_file=null \
  data.default.system_prompt_file=null \
  data.default.processor=openui_hf_data_processor \
  data.default.env_name=openui \
  +env.openui.num_workers=1 \
  policy.model_name=/workspace/base-model \
  policy.tokenizer.name=/workspace/base-model \
  policy.max_total_sequence_length=512 \
  policy.train_global_batch_size=4 \
  policy.train_micro_batch_size=1 \
  policy.dtensor_cfg.lora_cfg.enabled=true \
  policy.dtensor_cfg.lora_cfg.match_all_linear=false \
  policy.dtensor_cfg.lora_cfg.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  policy.dtensor_cfg.lora_cfg.dim=16 \
  policy.dtensor_cfg.lora_cfg.alpha=32 \
  policy.dtensor_cfg.lora_cfg.dropout=0.05 \
  cluster.gpus_per_node=1 \
  cluster.num_nodes=1 \
  logger.log_dir={q(output_root + '/logs')} \
  logger.wandb_enabled=false \
  checkpointing.enabled=true \
  checkpointing.save_period=1 \
  checkpointing.keep_top_k=1 \
  checkpointing.save_optimizer=false \
  checkpointing.save_consolidated=true \
  checkpointing.model_save_format=safetensors \
  checkpointing.checkpoint_dir={q(output_root + '/checkpoints')}

DEST={q(BUCKET_MOUNT + '/checkpoints/' + run_id + '/nemo_rl')}
mkdir -p "$DEST"
rsync -a {q(output_root + '/')} "$DEST/"
test -f "$DEST/train_summary.json"
echo "NeMo RL smoke artifacts: {bucket_uri}"
""".strip()


def build_hf_jobs_command(
    *,
    entrypoint: str,
    image: str = NEMO_RL_IMAGE,
    flavor: str = DEFAULT_FLAVOR,
    timeout: str = "2h",
    checkpoint_bucket: str = DEFAULT_BUCKET,
) -> list[str]:
    if image != NEMO_RL_IMAGE:
        raise ValueError(f"NeMo RL image must remain pinned to {NEMO_RL_IMAGE}")
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
        "workflow=nemo-rl-openui-smoke",
        "--volume",
        f"{checkpoint_bucket}:{BUCKET_MOUNT}",
        image,
        "bash",
        "-lc",
        entrypoint,
    ]


def parse_job_status(payload: Mapping[str, Any]) -> str:
    """Normalize the small status variations used by HF Jobs JSON output."""
    for key in ("status", "stage", "state"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, Mapping):
            for nested in ("stage", "status", "name"):
                if isinstance(value.get(nested), str):
                    return str(value[nested]).lower()
    raise ValueError("HF Jobs response did not contain a status")


def validate_train_summary(payload: Mapping[str, Any], *, run_id: str) -> None:
    required = {
        "run_id": run_id,
        "nemo_rl_version": NEMO_RL_VERSION,
        "nemo_rl_git_sha": NEMO_RL_GIT_SHA,
        "kind": "hardware_smoke",
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"invalid train summary {key}: {payload.get(key)!r}")
    uri = str(payload.get("checkpoint_bucket", ""))
    if not uri.startswith("hf://buckets/"):
        raise ValueError("train summary lacks a durable checkpoint_bucket URI")
    if not payload.get("checkpoint_written"):
        raise ValueError("train summary does not confirm a checkpoint")


def write_train_summary(
    path: Path,
    *,
    run_id: str,
    bucket_uri: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    checkpoint_dir = path.parent / "checkpoints"
    checkpoint_written = checkpoint_dir.is_dir() and any(
        item.is_file() for item in checkpoint_dir.rglob("*")
    )
    payload = {
        "run_id": run_id,
        "kind": "hardware_smoke",
        "nemo_rl_version": NEMO_RL_VERSION,
        "nemo_rl_git_sha": NEMO_RL_GIT_SHA,
        "checkpoint_bucket": bucket_uri,
        "checkpoint_written": checkpoint_written,
        "recipe": {
            "algorithm": "grpo",
            "steps": 1,
            "prompts_per_step": 2,
            "generations_per_prompt": 2,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
    }
    payload.update(metadata or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
