"""Zero-minute local merge gate mirroring the dormant ``python-static`` CI job.

Hosted GitHub Actions triggers are disabled for cost
(`docs/design/ci-minutes-and-speed-plan-20260806.md`), which leaves a hole:
a PR with red hooks must not merge just because GHA is silent. This script is
the single local gate command that closes that hole. It mirrors the exact
check invocations of the `.github/workflows/ci.yml` ``python-static`` job (in
the same order), then finishes with the changed-regression test fan-out from
the ``python`` job (`scripts.check_changed --changed-tests-only`).

Mirrored steps (source of truth: `.github/workflows/ci.yml` ``python-static``;
`tests/test_scripts/test_merge_ready_gate.py` certifies the parity):

1.  ``python -m scripts.repo_policy``
2.  ``python -m scripts.verify_decode_invariants``
3.  ``python -m scripts.verify_agent_surfaces``
4.  ``python -m scripts.verify_ownership_map``
5.  ``python -m scripts.extract_test_cases`` (read-only sweep; no ``--write``)
6.  ``python -m scripts.refresh_test_cases --check --changed`` (local analogue
    of CI's ``--check --base-ref <pr-base>``)
7.  ``ruff check .``
8.  ``python -m compileall -q src scripts tests``
9.  ``python -m scripts.verify_checkpoint_references --check``
10. ``python -m scripts.verify_version_stamps --check``
11. ``python -m scripts.check_changed --changed-tests-only`` (skipped by
    ``--fast``)

Failure behavior (documented choice): every *static* step always runs even
after a failure — each is seconds-cheap, and collecting all red steps in one
pass beats re-running the gate once per fix. The final changed-test step is
the only expensive one, so it is skipped when any static step failed; the
gate exits non-zero either way. A step that exceeds its wall budget is
recorded as ``timeout`` and counts as failure — a timed out run is never
evidence (`slm_training.levers.MAX_RUN_MINUTES`).

Rule: red ``verify_merge_ready`` ⇒ no merge. GHA silence is not approval.

Run: ``python -m scripts.verify_merge_ready [--fast] [--json]``
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.repo_policy import canonical_run_minutes

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
        Step("extract_test_cases", (python, "-m", "scripts.extract_test_cases")),
        Step(
            "refresh_test_cases",
            (python, "-m", "scripts.refresh_test_cases", "--check", "--changed"),
        ),
        Step("ruff", _ruff_cmd(python)),
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
                (python, "-m", "scripts.check_changed", "--changed-tests-only"),
                static=False,
            )
        )
    return tuple(steps)


def run_step(step: Step, *, budget_seconds: float, root: Path = ROOT) -> dict:
    """Execute one step and return its machine-readable record."""
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(step.cmd),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=budget_seconds,
        )
        status = "ok" if result.returncode == 0 else "failed"
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        output = "".join(
            part.decode(errors="replace") if isinstance(part, bytes) else (part or "")
            for part in (exc.stdout, exc.stderr)
        )
    record = {
        "name": step.name,
        "cmd": shlex.join(step.cmd),
        "status": status,
        "seconds": round(time.monotonic() - started, 2),
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
    for step in steps:
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
        record = run_step(step, budget_seconds=budget_seconds, root=root)
        records.append(record)
        if record["status"] != "ok" and step.static:
            static_failed = True
        if echo:
            echo(f"[{record['status']}] {record['name']} ({record['seconds']}s)")
            if record["status"] not in ("ok", "skipped"):
                echo(record.get("output_tail", ""))
    ok = all(record["status"] == "ok" for record in records)
    return {
        "gate": "verify_merge_ready",
        "mirrors": ".github/workflows/ci.yml#python-static (+ changed tests)",
        "rule": BLOCK_RULE,
        "max_step_seconds": budget_seconds,
        "ok": ok,
        "steps": records,
    }


def main(argv: list[str] | None = None) -> int:
    max_run_minutes = canonical_run_minutes()
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=float(max_run_minutes * 60),
        help=(
            "per-step wall budget in seconds (default and maximum: "
            f"{max_run_minutes * 60}, from slm_training.levers.MAX_RUN_MINUTES)"
        ),
    )
    args = parser.parse_args(argv)
    if not 0 < args.max_step_seconds <= max_run_minutes * 60:
        parser.error(
            f"--max-step-seconds must be positive and at most {max_run_minutes * 60}"
        )

    # In JSON mode keep stdout pure JSON; stream progress to stderr instead.
    stream = sys.stderr if args.json else sys.stdout

    def echo(line: str) -> None:
        print(line, file=stream, flush=True)

    summary = run_gate(
        merge_gate_steps(fast=args.fast),
        budget_seconds=args.max_step_seconds,
        echo=echo,
    )
    summary["fast"] = args.fast
    if args.json:
        print(json.dumps(summary, indent=2))
    elif summary["ok"]:
        mode = "fast/static" if args.fast else "full"
        echo(f"merge-ready: ok ({mode})")
    else:
        failed = [s["name"] for s in summary["steps"] if s["status"] != "ok"]
        echo(f"merge-ready: BLOCKED ({', '.join(failed)})")
        echo(BLOCK_RULE)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
