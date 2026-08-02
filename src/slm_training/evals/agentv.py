"""AgentEvals JSONL authoring and AgentV SDK result publication."""

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

    checkout_runner = repo_root / "scripts" / "run_agentv_eval.mjs"
    for root in checkout_roots(repo_root):
        runner = (
            checkout_runner
            if checkout_runner.is_file()
            else root / "scripts" / "run_agentv_eval.mjs"
        )
        sdk = root / "node_modules" / "@agentv" / "core" / "package.json"
        if runner.is_file() and sdk.is_file():
            return runner, root
    raise RuntimeError(
        "AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER"
    )


def _sanitized_node_env() -> dict[str, str]:
    # Session environments may inject NODE_OPTIONS entries (e.g. --import tsx)
    # that this Node build rejects with exit 9, silently killing the runner.
    # Mirrors slm_training.dsl.grammar.backends.graphql_js._sanitized_env.
    env = dict(os.environ)
    env["NODE_OPTIONS"] = ""
    return env


def publish_agentv_evaluation(
    run_dir: Path | str,
    *,
    name: str,
    claim: str,
    cases: Sequence[dict[str, Any]],
    version_stamp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write authoritative AgentEvals assertions and run them with AgentV."""
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not slug or not cases:
        raise ValueError("AgentV evaluation requires a name and at least one case")

    # AgentV may execute from the Git common checkout while artifacts belong to
    # an isolated worktree. Absolute paths keep publication attached to the run.
    root = (Path(run_dir) / "agentv").resolve()
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
            criteria = [{
                "id": f"{case_id}:domain_criterion",
                "actual": case.get("pass"),
                "operator": "eq",
                "expected": True,
            }]
        metadata = {
            "claim": claim,
            **dict(case.get("metadata") or {}),
        }
        if version_stamp is not None:
            metadata["version_stamp"] = version_stamp
        rows.append(
            json.dumps(
                {
                    "id": case_id,
                    "criteria": str(case["criteria"]),
                    "input": json.dumps(payload, sort_keys=True),
                    "metadata": metadata,
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
    trace_id = _run_trace_id(Path(run_dir))
    command = [
        "node",
        str(runner),
        "--spec",
        str(spec_path),
        "--output-dir",
        str(output_dir),
        "--experiment",
        slug,
        "--sdk-root",
        str(runtime_root),
    ]
    if trace_id is not None:
        command.extend(("--trace-id", trace_id, "--run-id", Path(run_dir).name))
    completed = subprocess.run(
        command,
        cwd=runtime_root,
        check=False,
        capture_output=True,
        text=True,
        env=_sanitized_node_env(),
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"AgentV SDK evaluation failed: {detail}")
    try:
        published = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"AgentV SDK returned invalid JSON: {completed.stdout!r}"
        ) from exc
    if version_stamp is not None:
        if version_stamp.get("stamp_schema") != "version_stamp/v1":
            raise ValueError("AgentV version stamp must use version_stamp/v1")
        _stamp_agentv_artifacts(output_dir, version_stamp)
    criterion_results = _read_agentv_criterion_results(published)
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
            "failures": [str(item["id"]) for item in criterion_results if not item["pass"]],
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


def _run_trace_id(run_dir: Path) -> str | None:
    """Return a valid W3C trace ID if the AgentV call is inside a run trace."""
    path = run_dir / "trace.json"
    if not path.is_file():
        return None
    try:
        trace_id = str(json.loads(path.read_text(encoding="utf-8")).get("trace_id", ""))
    except (OSError, json.JSONDecodeError):
        return None
    return trace_id if re.fullmatch(r"[0-9a-f]{32}", trace_id) else None


def _read_agentv_criterion_results(published: dict[str, Any]) -> list[dict[str, Any]]:
    """Read per-criterion verdicts from AgentV's canonical published index."""
    index_path = Path(str(published.get("artifacts", {}).get("indexPath", "")))
    if not index_path.is_file():
        return []

    results = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for score in row.get("scores", []):
            details = dict(score.get("details") or {})
            results.append(
                {
                    "id": details.get("criterion_id", score.get("name")),
                    "pass": score.get("score") == 1,
                    **details,
                }
            )
    return results


def _stamp_agentv_artifacts(
    output_dir: Path, version_stamp: dict[str, Any]
) -> None:
    """Attach the canonical experiment stamp to generated JSON result files."""
    for path in output_dir.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["version_stamp"] = version_stamp
            path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
    for path in output_dir.rglob("*.jsonl"):
        stamped = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payload["version_stamp"] = version_stamp
            stamped.append(json.dumps(payload, sort_keys=True))
        path.write_text("\n".join(stamped) + "\n", encoding="utf-8")


def model_ship_gate_cases(
    suites: dict[str, dict[str, Any]],
    *,
    include_missing_suites: bool = True,
    suite_reachability: dict[str, float] | None = None,
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
        suite_reachability=suite_reachability,
    )
    for suite, thresholds in selected.items():
        cases.append(
            {
                "id": suite,
                "criteria": (
                    f"Meet the canonical honest ship thresholds for {suite}; "
                    "a production claim still requires every policy suite."
                ),
                "assertions": [item for item in criteria if item["suite"] == suite],
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
    suite_reachability: dict[str, float] | None = None,
) -> dict[str, Any]:
    stamp = next(
        (
            str(metrics["evaluated_at"])
            for metrics in suites.values()
            if metrics.get("evaluated_at")
        ),
        "run",
    )
    return publish_agentv_evaluation(
        run_dir,
        name=f"openui-model-ship-gates-{stamp}",
        claim="honest_multi_suite_ship_gate",
        cases=model_ship_gate_cases(
            suites,
            include_missing_suites=include_missing_suites,
            suite_reachability=suite_reachability,
        ),
    )
