"""Zero-minute local merge gate mirroring the dormant ``python-static`` CI job.

Hosted GitHub Actions triggers are disabled for cost
(`docs/design/ci-minutes-and-speed-plan-20260806.md`), which leaves a hole:
a PR with red hooks must not merge just because GHA is silent. This script is
the single local gate command that closes that hole. It mirrors the exact
check invocations of the `.github/workflows/ci.yml` ``python-static`` job (in
the same order), then collects conservative source/resource/contract-owned
tests and executes bounded resumable shards. The fast hook is not release proof.

`merge_gate_steps` owns the mirrored commands; the merge-ready tests certify
their parity with `.github/workflows/ci.yml`. Full verification continues
independent obligations after failure, retaining scoped repair/budget waits.
The fast feedback profile still skips dependent execution after static failure.
Timeouts are incomplete evidence, never passing release obligations.

Rule: red ``verify_merge_ready`` ⇒ no merge. GHA silence is not approval.

Run: ``python -m scripts.verify_merge_ready [--fast] [--json]``
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS

ROOT = Path(__file__).resolve().parents[1]
# Enough tail to show the failing assertion plus surrounding context without
# flooding an agent's terminal or the JSON summary.
TAIL_CHARS = 4000
BLOCK_RULE = "red verify_merge_ready => no merge; GHA silence is not approval"

@dataclass(frozen=True)
class Step:
    """One mirrored gate invocation."""

    name: str
    cmd: tuple[str, ...]
    # Static steps are seconds-cheap certificates; the non-static changed-test
    # step actually executes the selected pytest fan-out.
    static: bool = True

def _ruff_cmd(python: str) -> tuple[str, ...]:
    """CI runs bare ``ruff check .``; prefer the interpreter's own install."""
    sibling = Path(python).parent / "ruff"
    if sibling.is_file():
        return (str(sibling), "check", ".")
    path_ruff = shutil.which("ruff")
    if path_ruff:
        return (path_ruff, "check", ".")
    return (python, "-m", "ruff", "check", ".")

def merge_gate_steps(*, fast: bool = False) -> tuple[Step, ...]:
    """The gate, in `.github/workflows/ci.yml` ``python-static`` order."""
    python = sys.executable
    steps = [
        Step("repo_policy", (python, "-m", "scripts.repo_policy")),
        Step("decode_invariants", (python, "-m", "scripts.verify_decode_invariants")),
        Step("agent_surfaces", (python, "-m", "scripts.verify_agent_surfaces")),
        Step("ownership_map", (python, "-m", "scripts.verify_ownership_map")),
        Step(
            "research_citation_catalog",
            (python, "-m", "scripts.verify_research_citation_catalog"),
        ),
        Step(
            "research_experiment_preregistry",
            (python, "-m", "scripts.verify_research_experiment_preregistry"),
        ),
        Step("extract_test_cases", (python, "-m", "scripts.extract_test_cases")),
        Step(
            "refresh_test_cases",
            (python, "-m", "scripts.refresh_test_cases", "--check", "--changed"),
        ),
        Step("ruff", _ruff_cmd(python)),
        Step("code_quality", (python, "-m", "scripts.verify_code_quality")),
        Step(
            "compileall", (python, "-m", "compileall", "-q", "src", "scripts", "tests")
        ),
        Step(
            "checkpoint_references",
            (python, "-m", "scripts.verify_checkpoint_references", "--check"),
        ),
        Step(
            "version_stamps",
            (python, "-m", "scripts.verify_version_stamps", "--check"),
        ),
        Step(
            "formal_evidence_mutations",
            (
                python,
                "-m",
                "scripts.verify_formal_evidence_mutations",
                "--check",
            ),
        ),
        Step(
            "evidence_ledger",
            (python, "-m", "scripts.build_evidence_ledger", "--check"),
        ),
        Step(
            "integ06_adversarial_acceptance",
            (
                python,
                "-m",
                "scripts.verify_integ06_adversarial_acceptance",
                "--check",
            ),
        ),
        Step(
            "integ07_activation_preflight_recall",
            (
                python,
                "-m",
                "scripts.verify_integ07_activation_preflight",
                "--check",
            ),
        ),
        Step(
            "integ10_release_matrix",
            (
                python,
                "-m",
                "scripts.verify_integ10_release_matrix",
                "--check",
            ),
        ),
    ]
    if not fast:
        steps.append(
            Step(
                "changed_tests",
                (python, "-m", "scripts.check_changed"),
                static=False,
            )
        )
    return tuple(steps)

def run_step(step: Step, *, budget_seconds: float, root: Path = ROOT) -> dict:
    """Execute one step and return its machine-readable record."""
    if (
        not math.isfinite(budget_seconds)
        or not 0 < budget_seconds <= INTERRUPT_AFTER_SECONDS
    ):
        raise ValueError("step budget exceeds canonical interrupt cap")
    result = run_bounded_process(
        list(step.cmd),
        cwd=root,
        interrupt_after_seconds=budget_seconds,
        kill_grace_seconds=KILL_GRACE_SECONDS,
    )
    status = "ok" if result.returncode == 0 else "failed"
    if result.timed_out or result.interrupted or result.killed:
        status = "timeout"
    output = result.stdout + result.stderr
    record = {
        "name": step.name,
        "cmd": shlex.join(step.cmd),
        "status": status,
        "exit_code": result.returncode,
        "seconds": round(result.duration_seconds, 2),
    }
    if status != "ok":
        record["output_tail"] = output[-TAIL_CHARS:]
    return record

def run_gate(
    steps: tuple[Step, ...],
    *,
    budget_seconds: float,
    root: Path = ROOT,
    echo=None,
) -> dict:
    """Run every static step, then the test step only if statics are green."""
    records: list[dict] = []
    static_failed = False
    deadline = time.monotonic() + INTERRUPT_AFTER_SECONDS - KILL_GRACE_SECONDS
    for step in steps:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            records.append({"name": step.name, "status": "pending", "seconds": 0.0})
            continue
        if not step.static and static_failed:
            records.append(
                {
                    "name": step.name,
                    "cmd": shlex.join(step.cmd),
                    "status": "skipped",
                    "seconds": 0.0,
                    "output_tail": "skipped: a static step already failed",
                }
            )
            continue
        record = run_step(
            step, budget_seconds=min(budget_seconds, remaining), root=root
        )
        records.append(record)
        if record["status"] != "ok" and step.static:
            static_failed = True
        if echo:
            echo(f"[{record['status']}] {record['name']} ({record['seconds']}s)")
            if record["status"] not in ("ok", "skipped"):
                echo(record.get("output_tail", ""))
    ok = bool(records) and all(record["status"] == "ok" for record in records)
    return {
        "gate": "verify_merge_ready",
        "mirrors": ".github/workflows/ci.yml#python-static (+ changed tests)",
        "rule": BLOCK_RULE,
        "max_step_seconds": budget_seconds,
        "ok": ok,
        "steps": records,
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--identity", help="controller-locked verification binding")
    parser.add_argument("--require-js-runtime", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="static checks only; skip the changed-test execution (pre-push mode)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the machine-readable summary as JSON on stdout",
    )
    parser.add_argument(
        "--max-step-seconds",
        type=float,
        default=120.0,
        help=(
            "per-step wall budget in seconds (default: 120; maximum: "
            f"{INTERRUPT_AFTER_SECONDS}, from slm_training.levers)"
        ),
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="target branch/base tree for affected coverage (default: origin/main)",
    )
    parser.add_argument(
        "--state-dir", type=Path, help="private controller cache outside candidate"
    )
    parser.add_argument(
        "--local-feedback",
        action="store_true",
        help="explicit same-user diagnostics only; never release evidence",
    )
    parser.add_argument(
        "--runtime-root",
        action="append",
        type=Path,
        default=[],
        help="approved read-only runtime installation (default: interpreter prefix)",
    )
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.max_step_seconds)
        or not 0 < args.max_step_seconds <= INTERRUPT_AFTER_SECONDS
    ):
        parser.error(
            f"--max-step-seconds must be positive and at most {INTERRUPT_AFTER_SECONDS}"
        )

    # In JSON mode keep stdout pure JSON; stream progress to stderr instead.
    stream = sys.stderr if args.json else sys.stdout

    def echo(line: str) -> None:
        print(line, file=stream, flush=True)

    if args.fast:
        summary = run_gate(
            merge_gate_steps(fast=True),
            budget_seconds=args.max_step_seconds,
            root=args.source.resolve(),
            echo=echo,
        )
        summary["verification_complete"] = False
        summary["release_authorized"] = False
    else:
        import os
        import tempfile

        from scripts.merge_verification import run_locked_release_gate
        from scripts.merge_verification_evidence import digest
        from slm_training.autoresearch.heal.isolation import IsolationUnavailable
        from scripts.merge_verification_runtime import (
            approved_runtime_roots,
            preflight_js,
        )

        if args.state_dir is None:
            args.state_dir = (
                Path(tempfile.gettempdir())
                / f"slm-merge-{os.getuid()}-{digest(str(ROOT))[:16]}"
            )
        try:
            preflight_js(args)
            summary = run_locked_release_gate(
                args.identity,
                dict(
                    steps=merge_gate_steps(),
                    root=args.source.resolve(),
                    base_ref=args.base_ref,
                    state_dir=args.state_dir,
                    step_seconds=args.max_step_seconds,
                    run_step=run_step,
                    local_feedback=args.local_feedback,
                    runtime_roots=approved_runtime_roots(
                        args.source, args.runtime_root
                    ),
                    runtime_digest=os.environ.get(
                        "MERGE_VERIFICATION_RUNTIME_IDENTITY"
                    ),
                ),
            )
        except IsolationUnavailable as exc:
            summary = {
                "status": "waiting_capability",
                "reason": str(exc),
                "steps": [],
                "verification_complete": False,
                "release_authorized": False,
                "unmet_predicate": "isolated_verifier_available",
                "wake_source": "isolation_capability_probe",
            }
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            summary = {
                "status": "waiting_dependency"
                if isinstance(exc, BlockingIOError)
                else "waiting_environment"
                if isinstance(exc, OSError)
                else "invalid_evidence",
                "reason": str(exc),
                "steps": [],
                "verification_complete": False,
            }
        summary["ok"] = summary["verification_complete"]
    summary["fast"] = args.fast
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _render_summary(summary, echo)
    if summary.get("status") == "pending":
        return 10
    if summary.get("status", "").startswith("waiting_"):
        return 20
    return 0 if summary["ok"] else 1

def _render_summary(summary: dict, echo) -> None:
    if "phase_progress" in summary:
        echo(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "status",
                        "phase_progress",
                        "next_action",
                        "spent_seconds",
                    )
                },
                sort_keys=True,
            )
        )
        return
    if summary["ok"]:
        mode = "fast/static" if summary["fast"] else "full"
        echo(
            f"merge-ready: local checks passed ({mode}); autonomous release requires independent verification"
        )
    else:
        failed = [
            s.get("name", s.get("kind", "workload"))
            for s in summary["steps"]
            if s["status"] != "ok"
        ]
        echo(f"merge-ready: BLOCKED ({', '.join(failed)})")
        if summary.get("reason"):
            echo(summary["reason"])
        echo(BLOCK_RULE)

if __name__ == "__main__":
    raise SystemExit(main())
