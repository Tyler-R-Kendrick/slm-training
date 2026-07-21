#!/usr/bin/env python3
"""Submit a bounded TwoTower checkpoint smoke on Hugging Face Jobs.

ZeroGPU Spaces are **not** used for full training — short ``@spaces.GPU`` quotas
and no ``torch.compile``. Prefer HF Jobs (this launcher) or multi-farm pods
(``scripts.remote_train``). See ``docs/design/hf-jobs-train.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess

from slm_training.levers import HF_JOB_TIMEOUT, MAX_RUN_MINUTES, MAX_RUN_SECONDS

DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_FLAVOR = "a10g-large"
DEFAULT_BUCKET = "hf://buckets/TKendrick/OpenUI"
DEFAULT_REPO = "https://github.com/Tyler-R-Kendrick/slm-training.git"
# Mount point for the durable checkpoint bucket inside the Job container.
BUCKET_MOUNT = "/mnt/openui-bucket"
JOB_TIMEOUT = HF_JOB_TIMEOUT


def build_entrypoint_script(
    *,
    repo_url: str,
    branch: str,
    run_id: str,
    steps: int,
    context_backend: str,
    checkpoint_bucket: str,
    sync_checkpoints: bool,
    skip_eval: bool,
    extra_train_args: str,
) -> str:
    """Bash body executed inside the Jobs container after image start."""
    sync_flag = (
        "--no-sync-checkpoints"
        if not sync_checkpoints
        else (
            f"--sync-checkpoints --checkpoint-bucket {shlex.quote(checkpoint_bucket)}"
        )
    )
    eval_block = ""
    if not skip_eval:
        ckpt = f"outputs/runs/{run_id}/checkpoints/last.pt"
        eval_block = f"""
python -m scripts.evaluate_model --train-dir outputs/data/train/v1 --test-dir outputs/data/eval/v1 --run-id {shlex.quote(run_id)} --ship-gates
python -m scripts.export_cactus --checkpoint {shlex.quote(ckpt)} --out-dir outputs/cactus/bundle
python -m scripts.bench_cactus --checkpoint {shlex.quote(ckpt)} --with-design-md
""".rstrip()

    extra = f" {extra_train_args.strip()}" if extra_train_args.strip() else ""

    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export SLM_FAST_TRAIN=1
export HF_JOBS_FAST_TRAIN=1
export SLM_MAX_WALL_MINUTES={MAX_RUN_MINUTES}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

echo "[hf-jobs] GPU probe"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo "[hf-jobs] install system deps (git + node for grammar bridges)"
apt-get update -qq
apt-get install -y -qq git curl ca-certificates
# Node 20 LTS — OpenUI / DESIGN.md bridges need npm ci
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -qq nodejs
fi
node -v && npm -v

WORK=/workspace/slm-training
mkdir -p /workspace
rm -rf "$WORK"
git clone --depth 1 --branch {shlex.quote(branch)} {shlex.quote(repo_url)} "$WORK"
cd "$WORK"

python -m pip install -U pip
python -m pip install -e '.[torch,hf,rico,dev]'
(cd src/apps/openui_bridge && npm ci)
(cd src/apps/design_md_bridge && npm ci)

python -m scripts.build_train_data --source all --version v1 --synthesizer quality --max-openui-chars 600 --max-components 10
python -m scripts.build_test_data --source both --version v1 --train-manifest outputs/data/train/v1/manifest.json

echo "[hf-jobs] train run_id={shlex.quote(run_id)} steps={int(steps)}"
python -m scripts.train_model \\
  --train-dir outputs/data/train/v1 \\
  --run-id {shlex.quote(run_id)} \\
  --steps {int(steps)} \\
  --device auto \\
  --context-backend {shlex.quote(context_backend)} \\
  --fast-train \\
  --compile-mode reduce-overhead \\
  --ltr-loss-weight 1.0 \\
  --grammar-ltr-primary \\
  {sync_flag}{extra}
{eval_block}

echo "[hf-jobs] done"
if [ -d {shlex.quote(BUCKET_MOUNT)} ]; then
  echo "[hf-jobs] bucket mount present at {BUCKET_MOUNT}"
  ls -la {shlex.quote(BUCKET_MOUNT)} | head -n 20 || true
fi
""".strip()


def build_hf_jobs_command(
    *,
    flavor: str,
    timeout: str,
    image: str,
    entrypoint: str,
    checkpoint_bucket: str,
    mount_bucket: bool,
    env: dict[str, str] | None = None,
) -> list[str]:
    if timeout != JOB_TIMEOUT:
        raise ValueError(f"HF Jobs timeout must be {JOB_TIMEOUT}")
    cmd: list[str] = [
        "hf",
        "jobs",
        "run",
        "--flavor",
        flavor,
        "--timeout",
        timeout,
        "--secrets",
        "HF_TOKEN",
        "--env",
        "SLM_FAST_TRAIN=1",
        "--env",
        "HF_JOBS_FAST_TRAIN=1",
    ]
    for key, value in (env or {}).items():
        cmd.extend(["--env", f"{key}={value}"])
    if mount_bucket:
        # Buckets are read-write by default; local fsspec sync still uses HF_TOKEN.
        bucket_id = checkpoint_bucket.removeprefix("hf://")
        cmd.extend(["--volume", f"hf://{bucket_id}:{BUCKET_MOUNT}"])
    cmd.extend([image, "bash", "-lc", entrypoint])
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", default=DEFAULT_FLAVOR, help="Jobs hardware flavor")
    parser.add_argument(
        "--timeout",
        choices=(JOB_TIMEOUT,),
        default=JOB_TIMEOUT,
        help="Hard-capped Job wall-clock timeout.",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image for hf jobs run")
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--run-id", default="hf_jobs_run")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--context-backend", choices=("scratch", "hf"), default="hf")
    parser.add_argument("--checkpoint-bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--no-sync-checkpoints",
        action="store_true",
        help="Skip HF Bucket upload (ephemeral Job disk only — usually wrong).",
    )
    parser.add_argument(
        "--no-mount-bucket",
        action="store_true",
        help="Do not volume-mount the checkpoint bucket into the Job.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train only (skip evaluate / export_cactus / bench).",
    )
    parser.add_argument(
        "--extra-train-args",
        default="",
        help="Appended to train_model (quoted as a single string).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the hf jobs command + script JSON; do not submit.",
    )
    parser.add_argument(
        "--print-script",
        action="store_true",
        help="Print only the entrypoint bash (for Hub upload / debugging).",
    )
    args = parser.parse_args(argv)

    entrypoint = build_entrypoint_script(
        repo_url=args.repo_url,
        branch=args.branch,
        run_id=args.run_id,
        steps=args.steps,
        context_backend=args.context_backend,
        checkpoint_bucket=args.checkpoint_bucket,
        sync_checkpoints=not args.no_sync_checkpoints,
        skip_eval=args.skip_eval,
        extra_train_args=args.extra_train_args,
    )
    if args.print_script:
        print(entrypoint)
        return 0

    cmd = build_hf_jobs_command(
        flavor=args.flavor,
        timeout=args.timeout,
        image=args.image,
        entrypoint=entrypoint,
        checkpoint_bucket=args.checkpoint_bucket,
        mount_bucket=not args.no_mount_bucket,
    )
    plan = {
        "command": cmd,
        "command_pretty": " ".join(shlex.quote(c) for c in cmd[:20]) + " …",
        "flavor": args.flavor,
        "timeout": args.timeout,
        "image": args.image,
        "run_id": args.run_id,
        "steps": args.steps,
        "checkpoint_bucket": args.checkpoint_bucket,
        "mount_bucket": not args.no_mount_bucket,
        "entrypoint": entrypoint,
        "note": (
            "ZeroGPU Spaces are unsuitable for this bounded checkpoint smoke; "
            "use HF Jobs (this command) or scripts.remote_train pods. "
            "Docs: https://huggingface.co/docs/hub/jobs-quickstart"
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        # hf CLI may still read ~/.cache/huggingface/token — warn only.
        print(
            json.dumps(
                {
                    "warning": "HF_TOKEN unset; relying on hf auth login cache",
                }
            )
        )

    try:
        proc = subprocess.run(cmd, check=False, timeout=MAX_RUN_SECONDS)
    except subprocess.TimeoutExpired:
        return 124
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
