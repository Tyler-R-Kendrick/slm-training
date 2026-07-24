"""AgentEvals JSONL authoring and AgentV SDK result publication."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from slm_training.bridge_utils import checkout_roots


MODEL_QUALITY_METRICS = (
    "parse_rate", "meaningful_program_rate", "binding_aware_meaningful_v2_rate_strict",
    "binding_aware_meaningful_v2_rate_coverage_conditioned", "binding_aware_meaningful_v2_coverage",
    "syntax_parse_rate", "raw_syntax_validity", "contract_precision", "contract_recall",
    "placeholder_fidelity", "placeholder_fidelity_normalized", "placeholder_validity",
    "exact_match", "structural_similarity", "tree_edit_similarity", "component_type_recall",
    "reward_score", "ast_node_f1", "ast_edge_f1", "language_validity", "canonical_exact",
    "ref_graph_exact", "target_correctness", "target_efficiency", "target_composite",
)


def _agentv_runtime(repo_root: Path) -> tuple[Path, Path]:
    """Resolve the pinned SDK from this checkout or its Git common checkout."""
    override = os.getenv("AGENTV_RUNNER")
    if override:
        runner = Path(override).resolve()
        return runner, runner.parents[1]

    roots = checkout_roots(repo_root)
    runner = next(
        (root / "scripts" / "run_agentv_eval.mjs" for root in roots
         if (root / "scripts" / "run_agentv_eval.mjs").is_file()),
        None,
    )
    sdk_root = next(
        (root for root in roots
         if (root / "node_modules" / "@agentv" / "core" / "package.json").is_file()),
        None,
    )
    if runner is not None and sdk_root is not None:
        return runner, sdk_root
    raise RuntimeError(
        "AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER"
    )


def publish_agentv_evaluation(
    run_dir: Path | str,
    *,
    name: str,
    claim: str,
    cases: Sequence[dict[str, Any]],
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
            "claim": claim,
            "suite": case.get("metadata", {}).get("suite", case_id),
            "result": case.get("result"),
        }
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
    return {
        "format": "AgentEvals JSONL",
        "sdk": "@agentv/core",
        "spec": str(spec_path),
        **published,
    }


def _metric_defined_n(metrics: dict[str, Any], metric: str) -> int:
    defined = (metrics.get("metric_defined_n") or {}).get(metric)
    if isinstance(defined, int) and defined >= 0:
        return defined
    value = metrics.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0
    document_n = metrics.get("document_n")
    n = document_n if isinstance(document_n, int) else metrics.get("n", 0)
    return n if isinstance(n, int) and n >= 0 else 0


def _model_metric_result(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": {metric: metrics.get(metric) for metric in MODEL_QUALITY_METRICS},
        "metric_defined_n": {
            metric: _metric_defined_n(metrics, metric) for metric in MODEL_QUALITY_METRICS
        },
    }


def apply_agentv_metric_results(
    metrics: dict[str, Any], publication: dict[str, Any], suite: str
) -> None:
    """Make the named SDK grader outputs the recorded model metrics."""
    execution_errors = int((publication.get("summary") or {}).get("executionErrors", 0))
    if execution_errors:
        raise RuntimeError(f"AgentV SDK reported {execution_errors} execution errors")
    sdk_metrics = (publication.get("metric_results") or {}).get(suite)
    if not isinstance(sdk_metrics, dict):
        raise RuntimeError(f"AgentV SDK returned no named metrics for suite {suite!r}")
    local_denominators = metrics.setdefault("metric_defined_n", {})
    for metric in MODEL_QUALITY_METRICS:
        observed = sdk_metrics.get(metric)
        if not isinstance(observed, dict):
            raise RuntimeError(f"AgentV SDK omitted metric {metric!r} for {suite!r}")
        expected = metrics.get(metric)
        actual = observed.get("value")
        if expected is None:
            if actual is not None:
                raise RuntimeError(f"AgentV SDK defined unavailable metric {metric!r}")
        elif not isinstance(expected, (int, float)) or isinstance(expected, bool):
            raise RuntimeError(f"metric {metric!r} is not numeric or null")
        elif not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"AgentV SDK disagreed on metric {metric!r}")
        metrics[metric] = actual
        local_denominators[metric] = int(observed.get("defined_n") or 0)
    metrics["metric_evaluator"] = {
        "sdk": "@agentv/core",
        "metrics": list(MODEL_QUALITY_METRICS),
        "execution_errors": execution_errors,
    }
    metrics["evaluation_artifacts"] = {
        "format": publication.get("format"),
        "sdk": "@agentv/core",
        "spec": publication.get("spec"),
        "artifacts": publication.get("artifacts"),
    }


def model_ship_gate_cases(
    suites: dict[str, dict[str, Any]], *, include_missing_suites: bool = True
) -> list[dict[str, Any]]:
    """Publish named domain-metric cases for a full or selected suite set."""
    from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES

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
    for suite in selected:
        cases.append(
            {
                "id": suite,
                "criteria": (
                    f"Meet the canonical honest ship thresholds for {suite}; "
                    "a production claim still requires every policy suite."
                ),
                "result": _model_metric_result(suites.get(suite, {})),
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
