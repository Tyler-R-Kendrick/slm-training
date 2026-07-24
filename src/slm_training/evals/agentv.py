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


def publish_agentv_evaluation(
    run_dir: Path | str,
    *,
    name: str,
    claim: str,
    cases: Sequence[dict[str, Any]],
    version_stamp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write AgentEvals JSONL and evaluate it with the pinned AgentV SDK."""
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not slug or not cases:
        raise ValueError("AgentV evaluation requires a name and at least one case")

    # AgentV may execute from the Git common checkout while artifacts belong to
    # an isolated worktree. Absolute paths keep publication attached to the run.
    root = (Path(run_dir) / "agentv").resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root / f"{slug}.eval.jsonl"
    output_dir = root / slug
    rows = []
    for case in cases:
        case_id = str(case["id"])
        payload = {
            "agentv_pass": case.get("pass") is True,
            "claim": claim,
            "checks": dict(case.get("checks") or {}),
            "failures": list(case.get("failures") or []),
            "result": case.get("result"),
        }
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
                    "assert": [{"type": "is-json", "required": True}],
                },
                sort_keys=True,
            )
        )
    spec_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
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
            "--sdk-root",
            str(runtime_root),
        ],
        cwd=runtime_root,
        check=False,
        capture_output=True,
        text=True,
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
    return {
        "format": "AgentEvals JSONL",
        "sdk": "@agentv/core",
        "spec": str(spec_path),
        **published,
    }


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
    suites: dict[str, dict[str, Any]], *, include_missing_suites: bool = True
) -> list[dict[str, Any]]:
    """Lower ship gates to AgentV cases for a full or selected suite set."""
    from slm_training.harnesses.model_build.ship_gates import (
        DEFAULT_SHIP_GATES,
        evaluate_ship_gates,
    )

    gates = evaluate_ship_gates(suites)
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
    for suite, thresholds in selected.items():
        prefix = f"{suite}:"
        checks = {
            key: passed
            for key, passed in gates["gates"].items()
            if key.startswith(prefix)
        }
        failures = [item for item in gates["failures"] if item.startswith(prefix)]
        cases.append(
            {
                "id": suite,
                "criteria": (
                    f"Meet the canonical honest ship thresholds for {suite}; "
                    "a production claim still requires every policy suite."
                ),
                "pass": bool(checks) and all(checks.values()),
                "failures": failures,
                "result": {
                    "actual": gates["actual"].get(suite),
                    "checks": checks,
                    "thresholds": thresholds,
                },
                "metadata": {"suite": suite, "honesty": "canonical_ship_gates"},
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
    return publish_agentv_evaluation(
        run_dir,
        name=f"openui-model-ship-gates-{stamp}",
        claim="honest_multi_suite_ship_gate",
        cases=model_ship_gate_cases(
            suites, include_missing_suites=include_missing_suites
        ),
    )
