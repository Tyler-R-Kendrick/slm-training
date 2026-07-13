#!/usr/bin/env python3
"""Remote train on a launched GPU pod (SSH human-in-the-loop automation)."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def _ssh_base(host: str, user: str | None, identity: Path | None, port: int) -> list[str]:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port)]
    if identity:
        cmd.extend(["-i", str(identity)])
    target = f"{user}@{host}" if user else host
    cmd.append(target)
    return cmd


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
    args = parser.parse_args(argv)

    remote_script = f"""
set -euo pipefail
mkdir -p {shlex.quote(args.remote_dir)}
cd {shlex.quote(args.remote_dir)}
if [ ! -d .git ]; then
  git clone --branch {shlex.quote(args.branch)} {shlex.quote(args.repo_url)} .
else
  git fetch origin {shlex.quote(args.branch)} && git checkout {shlex.quote(args.branch)} && git pull --ff-only
fi
python -m pip install -e '.[torch,hf,rico,dev]'
(cd tools/openui_bridge && npm ci)
(cd tools/design_md_bridge && npm ci)
python -m scripts.build_train_data --source rico --rico-limit 500
python -m scripts.build_test_data --source rico --rico-limit 100 --train-manifest outputs/train_data/v0/manifest.json
python -m scripts.train_model --run-id {shlex.quote(args.run_id)} --steps {int(args.steps)} --context-backend {shlex.quote(args.context_backend)}
python -m scripts.evaluate_model --run-id {shlex.quote(args.run_id)} --suite smoke
python -m scripts.export_cactus --checkpoint outputs/runs/{shlex.quote(args.run_id)}/checkpoints/model.pt --out-dir outputs/cactus/bundle
python -m scripts.bench_cactus --checkpoint outputs/runs/{shlex.quote(args.run_id)}/checkpoints/model.pt --with-design-md
""".strip()

    ssh = _ssh_base(args.host, args.user, args.identity, args.port)
    plan = {
        "ssh": ssh,
        "remote_script": remote_script,
        "pull_dir": str(args.pull_dir),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    proc = subprocess.run(ssh + ["bash", "-lc", remote_script], check=False)
    if proc.returncode != 0:
        return int(proc.returncode)

    args.pull_dir.mkdir(parents=True, exist_ok=True)
    remote_ckpt = f"{args.remote_dir}/outputs/runs/{args.run_id}/checkpoints/"
    scp = ["scp", "-r", "-P", str(args.port)]
    if args.identity:
        scp.extend(["-i", str(args.identity)])
    target = f"{args.user}@{args.host}" if args.user else args.host
    scp.extend([f"{target}:{remote_ckpt}", str(args.pull_dir)])
    subprocess.run(scp, check=False)
    print(json.dumps({"ok": True, "pulled_to": str(args.pull_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
