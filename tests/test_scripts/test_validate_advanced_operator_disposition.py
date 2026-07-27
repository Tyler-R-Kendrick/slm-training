"""Regression tests for the DSH5-12 (SLM-420) disposition schema validator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_advanced_operator_disposition import validate

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "docs/design/dsh5-12-advanced-operator-disposition-20260727-local/report.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_published_report_is_structurally_valid() -> None:
    assert validate(_report()) == []


def test_validator_rejects_missing_claim() -> None:
    report = _report()
    report["disposition"]["claims"] = report["disposition"]["claims"][:-1]
    violations = validate(report)
    assert violations
    assert any("reconstruct" in v or "exactly" in v for v in violations)


def test_validator_rejects_unauthorized_default_enable() -> None:
    report = _report()
    report["disposition"]["advanced_path_enabled_by_default"] = True
    violations = validate(report)
    assert violations
    assert any("advanced_path_enabled_by_default" in v or "no advanced path" in v for v in violations)


def test_validator_rejects_supported_claim_backed_only_by_unrun_dimensions() -> None:
    """No conditional/unrun issue may be counted as delivered value.

    Enforced by ``AdvancedOperatorClaimV1.__post_init__`` itself (a SUPPORTED
    headline must equal one of its own dimension verdicts, so it can never
    have every dimension undelivered -- that state is unconstructable, not
    merely unchecked); ``validate()`` surfaces the failure via its
    reconstruction guard rather than a separate check.
    """
    report = _report()
    claims = report["disposition"]["claims"]
    target = next(item for item in claims if item["claim"] == "crossover_work")
    tampered = copy.deepcopy(target)
    tampered["verdict"] = "supported"
    tampered["dimension_verdicts"] = {dim: "unrun_conditional" for dim in tampered["dimension_verdicts"]}
    index = claims.index(target)
    claims[index] = tampered
    violations = validate(report)
    assert violations
    assert any("must equal one of" in v for v in violations)


def test_validator_rejects_bad_recommendation() -> None:
    report = _report()
    report["disposition"]["recommendation"] = "ship_it"
    violations = validate(report)
    assert violations
    assert any("unsupported recommendation" in v or "not one of the five allowed options" in v for v in violations)
