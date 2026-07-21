#!/usr/bin/env python3
"""Remote train on a launched GPU pod (SSH human-in-the-loop automation)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from slm_training.levers import (
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS as KILL_AFTER_SECONDS,
    MAX_RUN_MINUTES,
    MAX_RUN_SECONDS,
)


def _ssh_base(
    host: str, user: str | None, identity: Path | None, port: int
) -> list[str]:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port)]
    if identity:
        cmd.extend(["-i", str(identity)])
    target = f"{user}@{host}" if user else host
    cmd.append(target)
    return cmd


def _shell_path(path: str) -> str:
    """Quote a path, but keep leading ~/ unquoted so the remote shell expands it."""
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    if path == "~":
        return "~"
    return shlex.quote(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Pod SSH host / connect address")
    parser.add_argument("--user", default=None)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument(
        "--repo-url",
        default="https://github.com/Tyler-R-Kendrick/slm-training.git",
    )
    parser.add_argument("--branch", default="main")
    parser.add_argument("--remote-dir", default="~/slm-training")
    parser.add_argument("--run-id", default="remote_run")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--context-backend", choices=("scratch", "hf"), default="hf")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pull-dir",
        type=Path,
        default=Path("outputs/remote_pull"),
        help="Local directory to scp checkpoints into after training.",
    )
    parser.add_argument(
        "--checkpoint-bucket",
        default="hf://buckets/TKendrick/OpenUI",
        help="HF Bucket for durable checkpoints (synced after train).",
    )
    parser.add_argument(
        "--no-sync-checkpoints",
        action="store_true",
        help="Skip HF Bucket upload (still scp locally).",
    )
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id) is None or args.run_id in {".", ".."}:
        parser.error(
            "--run-id may contain only letters, digits, dot, underscore, and dash"
        )

    remote_dir = _shell_path(args.remote_dir)
    run_id = shlex.quote(args.run_id)
    ckpt = f"outputs/runs/{args.run_id}/checkpoints/last.pt"
    ckpt_q = shlex.quote(ckpt)

    sync_flag = (
        "--no-sync-checkpoints"
        if args.no_sync_checkpoints
        else f"--sync-checkpoints --checkpoint-bucket {shlex.quote(args.checkpoint_bucket)}"
    )
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_export = (
        f"export HF_TOKEN={shlex.quote(hf_token)}"
        if hf_token and not args.no_sync_checkpoints
        else "# HF_TOKEN not forwarded (bucket sync disabled or unset locally)"
    )
    # Forward HF write auth into the remote train (required for bucket sync).
    # --fast-train + reduce-overhead match HF Jobs CUDA defaults (TF32 / AMP / compile).
    run_script = f"""
set -euo pipefail
{token_export}
export SLM_FAST_TRAIN=1
export SLM_MAX_WALL_MINUTES={MAX_RUN_MINUTES}
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
mkdir -p {remote_dir}
cd {remote_dir}
if [ ! -d .git ]; then
  git clone --branch {shlex.quote(args.branch)} {shlex.quote(args.repo_url)} .
else
  git fetch origin {shlex.quote(args.branch)} && git checkout {shlex.quote(args.branch)} && git pull --ff-only
fi
python -m pip install -e '.[torch,hf,rico,dev]'
(cd src/apps/openui_bridge && npm ci)
(cd src/apps/design_md_bridge && npm ci)
python -m scripts.build_train_data --source all --version v1 --synthesizer quality --max-openui-chars 600 --max-components 10
python -m scripts.build_test_data --source both --version v1 --train-manifest outputs/data/train/v1/manifest.json
python -m scripts.train_model --train-dir outputs/data/train/v1 --run-id {run_id} --steps {int(args.steps)} --device auto --context-backend {shlex.quote(args.context_backend)} --fast-train --compile-mode reduce-overhead --ltr-loss-weight 1.0 --grammar-ltr-primary {sync_flag}
python -m scripts.evaluate_model --train-dir outputs/data/train/v1 --test-dir outputs/data/eval/v1 --run-id {run_id} --ship-gates
python -m scripts.export_cactus --checkpoint {ckpt_q} --out-dir outputs/cactus/bundle
python -m scripts.bench_cactus --checkpoint {ckpt_q} --with-design-md
""".strip()
    remote_script = (
        f"timeout --signal=INT --kill-after={KILL_AFTER_SECONDS}s "
        f"{INTERRUPT_AFTER_SECONDS}s bash -lc {shlex.quote(run_script)}"
    )

    ssh = _ssh_base(args.host, args.user, args.identity, args.port)
    plan = {
        "ssh": ssh,
        "remote_script": remote_script,
        "pull_dir": str(args.pull_dir),
        "checkpoint": ckpt,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    deadline = time.monotonic() + MAX_RUN_SECONDS

    def remaining() -> float:
        return max(0.001, deadline - time.monotonic())

    try:
        proc = subprocess.run(
            ssh + ["bash", "-lc", remote_script],
            check=False,
            timeout=remaining(),
        )
    except subprocess.TimeoutExpired:
        return 124
    if proc.returncode != 0:
        return int(proc.returncode)

    args.pull_dir.mkdir(parents=True, exist_ok=True)
    try:
        resolve_proc = subprocess.run(
            ssh + ["bash", "-lc", f"cd {remote_dir} && pwd -P"],
            check=False,
            capture_output=True,
            text=True,
            timeout=remaining(),
        )
    except subprocess.TimeoutExpired:
        return 124
    resolved_dir = (resolve_proc.stdout or "").strip()
    if resolve_proc.returncode != 0 or not resolved_dir:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "failed to resolve remote training directory",
                    "stderr": (resolve_proc.stderr or "").strip(),
                },
                indent=2,
            )
        )
        return int(resolve_proc.returncode or 1)

    remote_ckpt = f"{resolved_dir}/outputs/runs/{args.run_id}/checkpoints/"
    scp = ["scp", "-r", "-P", str(args.port)]
    if args.identity:
        scp.extend(["-i", str(args.identity)])
    target = f"{args.user}@{args.host}" if args.user else args.host
    scp.extend([f"{target}:{shlex.quote(remote_ckpt)}", str(args.pull_dir)])
    try:
        copy_proc = subprocess.run(scp, check=False, timeout=remaining())
    except subprocess.TimeoutExpired:
        return 124
    if copy_proc.returncode != 0:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "checkpoint copy failed",
                    "returncode": copy_proc.returncode,
                },
                indent=2,
            )
        )
        return int(copy_proc.returncode)
    pulled = list(args.pull_dir.rglob("last.pt"))
    if not pulled:
        print(
            json.dumps(
                {"ok": False, "error": "copy completed but last.pt was not found"},
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {"ok": True, "pulled_to": str(args.pull_dir), "checkpoint": str(pulled[0])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
