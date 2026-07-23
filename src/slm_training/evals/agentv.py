"""AgentEvals criteria authoring with AgentV runner publication."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from slm_training.bridge_utils import checkout_roots


def _agentv_runtime(repo_root: Path) -> tuple[Path, Path]:
    """Resolve the pinned SDK from this checkout or its Git common checkout."""
    override = os.getenv("AGENTV_RUNNER")
    if override:
        runner = Path(override).resolve()
        return runner, runner.parents[1]

    for root in checkout_roots(repo_root):
        runner = root / "scripts" / "run_agentv_eval.mjs"
        sdk = root / "node_modules" / "@agentv" / "core" / "package.json"
        if runner.is_file() and sdk.is_file():
            return runner, root
    raise RuntimeError(
        "AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER"
    )


def publish_agentevals_evaluation(
    run_dir: Path | str,
    *,
    name: str,
    claim: str,
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Write authoritative AgentEvals assertions and run them with AgentV."""
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not slug or not cases:
        raise ValueError("AgentEvals evaluation requires a name and at least one case")

    # AgentV may execute from the Git common checkout while artifacts belong to
    # an isolated worktree. Absolute paths keep publication attached to the run.
    root = (Path(run_dir) / "evals").resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root / f"{slug}.eval.jsonl"
    output_dir = root / slug
    repo_root = Path(__file__).resolve().parents[3]
    grader = repo_root / "scripts" / "grade_eval_criterion.py"
    rows = []
    for case in cases:
        case_id = str(case["id"])
        payload = {
            "claim": claim,
            "result": case.get("result"),
        }
        criteria = list(case.get("assertions") or [])
        if not criteria:
            # Compatibility for non-ship diagnostic publishers. Their domain
            # harness supplies the raw boolean, but AgentEvals still owns the
            # required assertion and final verdict.
            criteria = [
                {
                    "id": f"{case_id}:domain_criterion",
                    "actual": case.get("pass"),
                    "operator": "eq",
                    "expected": True,
                }
            ]
        rows.append(
            json.dumps(
                {
                    "id": case_id,
                    "criteria": str(case["criteria"]),
                    "input": json.dumps(payload, sort_keys=True),
                    "metadata": {
                        "claim": claim,
                        **dict(case.get("metadata") or {}),
                    },
                    "assert": [
                        {
                            "name": str(criterion["id"]),
                            "type": "code-grader",
                            "command": ["python3", str(grader)],
                            "required": True,
                            "min_score": 1,
                            "config": dict(criterion),
                        }
                        for criterion in criteria
                    ],
                },
                sort_keys=True,
            )
        )
    spec_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    runner, runtime_root = _agentv_runtime(repo_root)
    completed = subprocess.run(
        [
            "node",
            str(runner),
            "--spec",
            str(spec_path),
            "--output-dir",
            str(output_dir),
            "--experiment",
            slug,
        ],
        cwd=runtime_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"AgentV runner failed: {detail}")
    try:
        published = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"AgentV runner returned invalid JSON: {completed.stdout!r}"
        ) from exc
    criterion_results = [
        {
            "id": score.get("details", {}).get("criterion_id", score.get("name")),
            "pass": score.get("score") == 1,
            **dict(score.get("details") or {}),
        }
        for result in published.get("results", [])
        for score in result.get("scores", [])
    ]
    passed = sum(item["pass"] for item in criterion_results)
    execution_errors = int(published.get("summary", {}).get("executionErrors", 0))
    return {
        "format": "AgentEvals JSONL",
        "authority": "AgentEvals assertions",
        "sdk": "@agentv/core",
        "criteria": {
            "pass": bool(criterion_results)
            and passed == len(criterion_results)
            and execution_errors == 0,
            "passed": passed,
            "failed": len(criterion_results) - passed,
            "total": len(criterion_results),
            "failures": [
                str(item["id"]) for item in criterion_results if not item["pass"]
            ],
            "results": criterion_results,
        },
        "runner": {
            "name": "AgentV",
            "sdk": "@agentv/core",
            "execution_errors": execution_errors,
        },
        "spec": str(spec_path),
        **published,
    }


def publish_agentv_evaluation(
    run_dir: Path | str,
    *,
    name: str,
    claim: str,
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compatibility alias for diagnostic publishers using the AgentV runner."""
    return publish_agentevals_evaluation(
        run_dir,
        name=name,
        claim=claim,
        cases=cases,
    )


def model_ship_gate_cases(
    suites: dict[str, dict[str, Any]], *, include_missing_suites: bool = True
) -> list[dict[str, Any]]:
    """Lower ship policy evidence to raw AgentEvals assertion cases."""
    from slm_training.harnesses.model_build.ship_gates import (
        DEFAULT_MIN_SUITE_N,
        DEFAULT_SHIP_GATES,
        _slim_suite,
    )
    from slm_training.harness_core.gate_engine import build_gate_criteria

    cases = []
    selected = (
        DEFAULT_SHIP_GATES
        if include_missing_suites
        else {
            suite: DEFAULT_SHIP_GATES[suite]
            for suite in suites
            if suite in DEFAULT_SHIP_GATES
        }
    )
    actual, criteria = build_gate_criteria(
        suites,
        selected,
        normalize_suite=_slim_suite,
        default_min_n=DEFAULT_MIN_SUITE_N,
    )
    for suite, thresholds in selected.items():
        assertions = [item for item in criteria if item["suite"] == suite]
        cases.append(
            {
                "id": suite,
                "criteria": (
                    f"Meet the canonical honest ship thresholds for {suite}; "
                    "a production claim still requires every policy suite."
                ),
                "assertions": assertions,
                "result": {
                    "actual": actual.get(suite),
                    "thresholds": thresholds,
                },
                "metadata": {
                    "suite": suite,
                    "honesty": "canonical_ship_gates",
                    "gate_authority": "AgentEvals assertions",
                },
            }
        )
    return cases


def publish_model_evaluation(
    run_dir: Path | str,
    suites: dict[str, dict[str, Any]],
    *,
    include_missing_suites: bool = True,
) -> dict[str, Any]:
    stamp = next(
        (
            str(metrics["evaluated_at"])
            for metrics in suites.values()
            if metrics.get("evaluated_at")
        ),
        "run",
    )
    return publish_agentevals_evaluation(
        run_dir,
        name=f"openui-model-ship-gates-{stamp}",
        claim="honest_multi_suite_ship_gate",
        cases=model_ship_gate_cases(
            suites, include_missing_suites=include_missing_suites
        ),
    )
