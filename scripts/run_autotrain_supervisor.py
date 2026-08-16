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


def _handle_hard_pending(
    hard_pending: list[dict],
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str,
    max_heal_attempts: int,
    playbooks_enabled: bool,
    log_event,
) -> dict:
    """Dispatch heal playbooks + escalation governance for hard blockers.

    Returns ``{"any_healed": bool, "sleep_seconds": float, "outcomes": [...]}``.
    Never raises: heal-layer bugs degrade to the legacy fixed backoff.
    """
    try:
        from slm_training.autoresearch import heal
        from slm_training.autoresearch.heal.escalation import EscalationLedger

        blockers = [
            {**entry, "_root": root, "_loop_id": loop_id}
            for entry in hard_pending
        ]
        receipts = ()
        if playbooks_enabled:
            receipts = heal.run_playbooks(
                root=root,
                loop_id=loop_id,
                campaign_id=campaign_id or "unknown",
                blockers=blockers,
                cwd=cwd,
                max_attempts_per_fingerprint=max_heal_attempts,
            )
        any_healed = any(r.outcome == "healed" for r in receipts)
        # Cross-link a fresh quarantine stash SHA into the escalation note so
        # the next agent can find the quarantined evidence (skeptic O6.5).
        # stash@{0} is only unambiguous when exactly one quarantine healed in
        # this dispatch; with several, annotating would name the wrong stash,
        # so leave attribution to the per-receipt stash message instead.
        ledger = EscalationLedger.load(root, loop_id)
        quarantined = [
            r
            for r in receipts
            if r.outcome == "healed"
            and r.playbook_id.startswith("quarantine_dirt")
        ]
        if len(quarantined) == 1:
            stash_sha = _stash_head_sha(cwd)
            if stash_sha:
                ledger.resolve(
                    quarantined[0].blocker_fingerprint,
                    note=(
                        f"quarantined_stash_sha={stash_sha} "
                        f"restore=git stash apply {stash_sha}"
                    ),
                )
        ledger.save()
        return {
            "any_healed": any_healed,
            "sleep_seconds": ledger.sleep_seconds(default=30.0),
            "outcomes": [r.outcome for r in receipts],
            "open_escalations": len(ledger.open_records()),
        }
    except Exception as exc:  # noqa: BLE001 — heal bugs never kill supervision
        log_event({"event": "hard_pending_heal_error", "error": repr(exc)})
        return {"any_healed": False, "sleep_seconds": 30.0, "outcomes": []}


def _stash_head_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "stash@{0}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


def _write_family_closures(log_event) -> None:
    """Fail-soft post-cycle conclusion writer (WP-4 production caller)."""
    try:
        from slm_training.autoresearch.heal.conclusion_writer import (
            write_family_closures,
        )

        appended = write_family_closures()
        if appended:
            log_event(
                {
                    "event": "family_closures_appended",
                    "families": [r.family_key for r in appended],
                }
            )
    except Exception as exc:  # noqa: BLE001 — a writer bug never kills a cycle
        log_event({"event": "family_closures_error", "error": repr(exc)})


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
    parser.add_argument(
        "--max-heal-attempts",
        type=int,
        default=2,
        help="Playbook attempts per blocker fingerprint before escalation",
    )
    parser.add_argument(
        "--no-playbooks",
        action="store_true",
        help="Disable heal-playbook dispatch (ledger + governed backoff only)",
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
            outcome = _handle_hard_pending(
                report["hard_pending"],
                cwd=cwd,
                root=root,
                loop_id=args.loop_id,
                campaign_id=str(report.get("predecessor_campaign_id") or ""),
                max_heal_attempts=int(args.max_heal_attempts),
                playbooks_enabled=not args.no_playbooks,
                log_event=log_event,
            )
            log_event(
                {
                    "event": "hard_pending_heal",
                    "cycle": cycle,
                    "hard_pending": report["hard_pending"],
                    **outcome,
                }
            )
            if outcome.get("any_healed"):
                # Floor sleep: a blocker whose reason text shifts each cycle
                # mints fresh fingerprints, so the attempt budget alone does
                # not bound a heal-spin — never loop with zero delay.
                time.sleep(max(0.5, float(args.soft_backoff_seconds)))
                continue  # re-run the unblock loop
            time.sleep(
                max(
                    1.0,
                    float(
                        outcome.get("sleep_seconds")
                        or args.hard_backoff_seconds
                    ),
                )
            )
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
        _write_family_closures(log_event)
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
