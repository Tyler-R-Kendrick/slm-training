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
import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path

from scripts.autotrain_supervisor_operations import (
    run_operation as _run_operation,
)
from scripts.autotrain_supervision import (
    _MAX_HARD_RETRY_SECONDS as _MAX_HARD_RETRY_SECONDS,
    _NO_CAMPAIGN_THRESHOLD as _NO_CAMPAIGN_THRESHOLD,
    _STALL_KIND as _STALL_KIND,
    watchdog_no_campaign as _watchdog_no_campaign,
)


def _source_identity(cwd: Path) -> str:
    from slm_training.harness_core.execution_release import runtime_source_identity
    from scripts.merge_verification_evidence import source_identity

    return runtime_source_identity(cwd) or source_identity(cwd)


def _load_continuous():
    script = Path(__file__).resolve().parent / "run_autotrain_continuous.py"
    spec = importlib.util.spec_from_file_location("run_autotrain_continuous", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
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
            {**entry, "_root": root, "_loop_id": loop_id} for entry in hard_pending
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
            if r.outcome == "healed" and r.playbook_id.startswith("quarantine_dirt")
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
            "sleep_seconds": min(
                _MAX_HARD_RETRY_SECONDS, ledger.sleep_seconds(default=30.0)
            ),
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


_POLICY_PRIMARY_FALLBACK = "smoke.eval_nll"


def _default_primary_metric() -> str:
    """Driver and supervisor screen on the policy's ``screening_primary.metric``."""
    try:
        from slm_training.autoresearch.climb_policy import load_climb_policy

        return str(load_climb_policy().screening_primary["metric"])
    except Exception as exc:  # noqa: BLE001 — a policy load bug must not stop supervision
        print(
            f"SUPERVISOR_POLICY_WARN primary metric fallback={_POLICY_PRIMARY_FALLBACK} "
            f"error={exc!r}",
            file=sys.stderr,
            flush=True,
        )
        return _POLICY_PRIMARY_FALLBACK


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


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--continuation-grant", help="Explicit ResourceGrant JSON shared by the driver and locked campaign")
    parser.add_argument(
        "--primary-metric",
        default=_default_primary_metric(),
        help="Defaults to climb policy screening_primary.metric (same as the driver)",
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
        "--park-backoff-seconds",
        type=float,
        default=300.0,
        help="Sleep between park re-checks (park is a wait state, not exit)",
    )
    parser.add_argument(
        "--exit-on-park",
        action="store_true",
        help="Legacy: exit 0 on regime park instead of waiting for recovery",
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
    parser.add_argument("--operation-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--operation-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-pass", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--delivery-config", type=Path,
                        help="Trusted GitHub connector host configuration")
    parser.add_argument(
        "--repair-config",
        type=Path,
        help="Explicit preapproved local repair grant/recipes; absent means a scoped capability wait",
    )
    return parser


def _operation_main(request_path: Path, output_path: Path) -> int:
    from scripts.autotrain_supervisor_operations import operation_main

    return operation_main(
        request_path,
        output_path,
        source_identity=_source_identity,
        load_continuous=_load_continuous,
        handle_hard_pending=_handle_hard_pending,
        write_family_closures=_write_family_closures,
    )


def _supervise(args, runtime, common: dict) -> int:
    from scripts.autotrain_supervision import supervise

    return supervise(
        args,
        runtime,
        common,
        run_operation=_run_operation,
        watchdog=_watchdog_no_campaign,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.operation_request or args.operation_output:
        if not args.operation_request or not args.operation_output:
            raise ValueError("operation requires both request and output")
        return _operation_main(args.operation_request, args.operation_output)
    from scripts.merge_verification_evidence import digest, environment_identity
    from slm_training.autoresearch.runtime.activity_runtime import (
        ActivityRuntime,
        ControllerBusy,
    )
    from slm_training.autoresearch.storage import CampaignStore
    from scripts.autotrain_repair_activation import (
        VerifiedRestart, recover_release, restart_supervisor,
    )

    cwd = Path.cwd().resolve()
    root = args.root.resolve()
    if Path(args.loop_id).name != args.loop_id or args.loop_id in {".", ".."}:
        raise ValueError("loop-id must be a single path component")
    store = CampaignStore("runtime", root / "loops" / args.loop_id)
    try:
        # Lease ownership is acquired before importing or invoking the driver/heal.
        with ActivityRuntime(store) as runtime:
            passes = sum(e["event_type"] == "supervisor_pass" for e in store.verify_event_chain())
            if args.max_cycles and args.stop_after_pass is None:
                args.stop_after_pass = passes + args.max_cycles
            common = {
                "cwd": str(cwd),
                "root": str(root),
                "loop_id": args.loop_id,
                "source_digest": _source_identity(cwd),
                "environment_digest": digest(environment_identity()),
                "repair_config": str(args.repair_config.resolve())
                if args.repair_config
                else None,
                "repair_config_digest": hashlib.sha256(
                    args.repair_config.read_bytes()
                ).hexdigest()
                if args.repair_config
                else None,
                "delivery_config": str(args.delivery_config.resolve())
                if args.delivery_config else None,
                "delivery_config_digest": hashlib.sha256(args.delivery_config.read_bytes()).hexdigest()
                if args.delivery_config else None,
            }
            previous = {
                sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
            }
            for sig in previous:
                signal.signal(sig, lambda *_: runtime.cancel_event.set())
            try:
                recovered = recover_release(
                    runtime, common, sequence=passes, log_event=lambda e: print(json.dumps(e)),
                    run_operation=_run_operation,
                )
                return 10 if recovered == "waiting_delivery" else _supervise(args, runtime, common)
            finally:
                if runtime.cancel_event.is_set():
                    runtime.cancel_all(reason="explicit supervisor stop")
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    except ControllerBusy:
        print("supervisor already owned by a current controller", file=sys.stderr)
        return 2
    except VerifiedRestart as restart:
        restart_supervisor(args, restart.handoff)
        raise AssertionError("execve returned without replacing the controller")


if __name__ == "__main__":
    raise SystemExit(main())
