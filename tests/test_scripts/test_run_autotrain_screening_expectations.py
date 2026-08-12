"""Exact Lean expectation binding for decision-bearing screening."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_continuous.py"
)
_SPEC = importlib.util.spec_from_file_location("run_autotrain_continuous", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def test_screening_manifest_locks_only_exact_zero_expectations() -> None:
    manifest = _mod._manifest(
        "continuous-loop-c1",
        {
            "experiment_id": "c-screen",
            "hypothesis": "A size-matched lever changes fixture decode behavior.",
            "knobs": {"seed": 7},
        },
        "a" * 40,
        role="screening",
    )
    expectations = json.loads(
        _mod.screening_expectations_path().read_text(encoding="utf-8")
    )

    assert (
        manifest.metric_expectations_sha256
        == _mod.locked_screening_expectations_sha256()
    )
    assert manifest.formal_obligations == ()
    assert {metric["metric_id"] for metric in expectations["metrics"]} == {
        "invalid_grammar_count",
        "uncertified_fallback_count",
        "singleton_neural_forward_count",
        "missing_observation_count",
        "runtime_timeout_count",
    }
    assert all(metric["authority"] == "theorem" for metric in expectations["metrics"])
    assert all(metric["dependencies"] == [] for metric in expectations["metrics"])
    assert all(
        metric["program"]
        == [{"op": "constant", "value": {"numerator": 0, "denominator": 1}}]
        for metric in expectations["metrics"]
    )
