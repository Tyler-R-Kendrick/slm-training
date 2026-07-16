#!/usr/bin/env python3
"""Run the pinned one-step Molt OpenUI smoke and persist normalized traces."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

from slm_training.integrations.molt_rl import (
    normalize_rollout_dumps,
    prepare_prompt_dataset,
    validate_trace_file,
    write_train_summary,
)


def main() -> int:
    from slm_training.autoresearch.rl_gate import assert_rl_ready
    from slm_training.autoresearch.schemas import RLReadinessReport

    encoded = os.environ.get("SLM_MOLT_RL_READINESS_B64")
    if not encoded:
        raise ValueError("RL is locked: missing SLM_MOLT_RL_READINESS_B64")
    readiness = RLReadinessReport.model_validate(
        json.loads(base64.b64decode(encoded).decode())
    )
    assert_rl_ready(readiness)

    run_id = os.environ["SLM_MOLT_RUN_ID"]
    output_root = Path(os.environ["SLM_MOLT_OUTPUT_ROOT"])
    model_path = Path("/workspace/base-model")
    data_path = Path(os.environ["SLM_MOLT_DATA_PATH"])
    seed = int(os.environ["SLM_MOLT_SEED"])
    prepared = output_root / "molt_prompts.jsonl"
    record_count = prepare_prompt_dataset(data_path, prepared)
    raw_traces = output_root / "traces" / "raw"

    command = [
        "python",
        "-m",
        "molt.cli.train_rl_ray",
        "--actor.model_name_or_path",
        str(model_path),
        "--data.prompt_dataset",
        str(prepared),
        "--data.input_key",
        "input",
        "--data.label_key",
        "label",
        "--data.max_samples",
        str(record_count),
        "--data.max_len",
        "512",
        "--train.agent_path",
        "/workspace/slm-training/src/slm_training/integrations/molt_rl_agent.py",
        "--vllm.num_engines",
        "1",
        "--vllm.tensor_parallel_size",
        "1",
        "--vllm.enforce_eager",
        "--rollout.batch_size",
        "2",
        "--rollout.micro_batch_size",
        "1",
        "--rollout.n_samples_per_prompt",
        "2",
        "--rollout.max_new_tokens",
        "256",
        "--rollout.max_tokens_per_gpu",
        "1024",
        "--train.batch_size",
        "4",
        "--train.micro_batch_size",
        "1",
        "--train.max_tokens_per_gpu",
        "1024",
        "--train.max_epochs",
        "1",
        "--train.num_episodes",
        "1",
        "--train.seed",
        str(seed),
        "--train.force_sync_mode",
        "--train.rollout_dump_dir",
        str(raw_traces),
        "--algo.advantage.estimator",
        "grpo",
        "--algo.kl.init_coef",
        "0",
        "--actor.num_nodes",
        "1",
        "--actor.num_gpus_per_node",
        "1",
        "--actor.gradient_checkpoint",
        "none",
        "--fsdp.attn_implementation",
        "sdpa",
        "--ckpt.output_dir",
        str(output_root / "checkpoints"),
        "--ckpt.path",
        str(output_root / "resumable"),
        "--ckpt.save_steps",
        "1",
        "--logger.tensorboard_dir",
        str(output_root / "logs" / "tensorboard"),
    ]
    error: subprocess.CalledProcessError | None = None
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        error = exc
    finally:
        if raw_traces.exists():
            from transformers import AutoTokenizer

            normalize_rollout_dumps(
                raw_traces,
                output_root / "rl_traces.jsonl",
                tokenizer=AutoTokenizer.from_pretrained(model_path),
                run_id=run_id,
                trace_id=os.environ.get("SLM_TRACE_ID"),
            )
    if error is not None:
        raise error

    trace_path = output_root / "rl_traces.jsonl"
    trace_count = validate_trace_file(trace_path, run_id=run_id)
    write_train_summary(
        output_root / "train_summary.json",
        run_id=run_id,
        bucket_uri=os.environ["SLM_MOLT_BUCKET_URI"],
        trace_count=trace_count,
        metadata={
            "base_model_id": os.environ["SLM_MOLT_BASE_MODEL_ID"],
            "base_model_revision": os.environ["SLM_MOLT_BASE_MODEL_REVISION"],
            "code_revision": os.environ["SLM_MOLT_CODE_REVISION"],
            "data_path": str(data_path),
            "record_count": record_count,
            "seed": seed,
            "rl_readiness_report_id": readiness.report_id,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
