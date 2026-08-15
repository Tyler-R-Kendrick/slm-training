#!/usr/bin/env python3
"""In-repo continuous thrash supervisor.

Replaces fragile /tmp bash supervisors. After every supervised cycle (success
or failure) runs ``self_heal_unblock_loop`` so soft thrash blockers never need
a human or chat prompt.

Usage::

    python -m scripts.run_autotrain_supervisor \\
      --loop-id continuous-openui-local \\
      --train-version wf_smoke_v2 --steps 20

Hard pending (true harness crash, formal, foreign dirt, deliver_stack) logs and
backs off; soft failures heal and immediately continue. Parked
``rebuild_data`` is a local-CPU heal, not a no-op.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _load_continuous():
    script = Path(__file__).resolve().parent / "run_autotrain_continuous.py"
    spec = importlib.util.spec_from_file_location("run_autotrain_continuous", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-id", default="continuous-openui-local")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/autoresearch"),
        help="Campaign bundle root (relative to cwd)",
    )
    parser.add_argument("--train-version", default="wf_smoke_v2")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--primary-metric",
        default="smoke.structural_similarity",
    )
    parser.add_argument(
        "--hard-backoff-seconds",
        type=float,
        default=30.0,
        help="Sleep after hard_pending before retrying",
    )
    parser.add_argument(
        "--soft-backoff-seconds",
        type=float,
        default=2.0,
        help="Sleep after soft heal before next cycle",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="0 = unbounded supervised restarts",
    )
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    root = args.root if args.root.is_absolute() else cwd / args.root
    root.mkdir(parents=True, exist_ok=True)
    continuous = _load_continuous()
    log_dir = root / "loops" / args.loop_id
    log_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = log_dir / "supervisor.jsonl"

    def log_event(event: dict) -> None:
        event = {
            **event,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loop_id": args.loop_id,
        }
        with supervisor_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
        print(json.dumps(event, sort_keys=True), flush=True)

    py = sys.executable
    cycle = 0
    while args.max_cycles == 0 or cycle < args.max_cycles:
        cycle += 1
        # Heal first (including local-CPU rebuild_data). Park is not a stop
        # while a locally executable rebuild is pending.
        try:
            report = continuous.self_heal_unblock_loop(
                cwd=cwd,
                root=root,
                loop_id=args.loop_id,
            )
            log_event({"event": "pre_cycle_unblock", "cycle": cycle, **report})
        except Exception as exc:  # noqa: BLE001
            log_event({"event": "pre_cycle_unblock_error", "error": repr(exc)})
            report = {}
        parked = continuous._check_regime_parked(root=root, loop_id=args.loop_id)
        if parked:
            log_event(
                {
                    "event": "regime_parked",
                    "status": parked,
                    "cycle": cycle,
                    "soft_healed": list(report.get("soft_healed") or []),
                }
            )
            return 0
        if report.get("hard_pending"):
            log_event(
                {
                    "event": "hard_pending_backoff",
                    "cycle": cycle,
                    "hard_pending": report["hard_pending"],
                }
            )
            time.sleep(max(1.0, float(args.hard_backoff_seconds)))
            continue

        cmd = [
            py,
            "-m",
            "scripts.run_autotrain_continuous",
            "--loop-id",
            args.loop_id,
            "--root",
            str(args.root),
            "--supervised",
            "--max-cycles",
            "1",
            "--train-version",
            args.train_version,
            "--steps",
            str(args.steps),
            "--primary-metric",
            args.primary_metric,
        ]
        env = os.environ.copy()
        src = cwd / "src"
        if src.is_dir():
            env["PYTHONPATH"] = (
                str(src)
                if not env.get("PYTHONPATH")
                else f"{src}{os.pathsep}{env['PYTHONPATH']}"
            )
        log_event({"event": "start_driver", "cycle": cycle, "cmd": cmd})
        proc = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        log_event(
            {
                "event": "driver_exit",
                "cycle": cycle,
                "returncode": int(proc.returncode),
            }
        )
        # Post-cycle unblock regardless of exit code.
        try:
            report = continuous.self_heal_unblock_loop(
                cwd=cwd,
                root=root,
                loop_id=args.loop_id,
            )
            log_event({"event": "post_cycle_unblock", "cycle": cycle, **report})
            if report.get("hard_pending"):
                time.sleep(max(1.0, float(args.hard_backoff_seconds)))
            else:
                time.sleep(max(0.5, float(args.soft_backoff_seconds)))
        except Exception as exc:  # noqa: BLE001
            log_event({"event": "post_cycle_unblock_error", "error": repr(exc)})
            time.sleep(max(1.0, float(args.soft_backoff_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
