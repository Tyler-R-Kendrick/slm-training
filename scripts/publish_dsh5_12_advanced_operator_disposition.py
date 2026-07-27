#!/usr/bin/env python3
"""Publish SLM-420's (DSH5-12) advanced-operator disposition from durable evidence.

Unlike ``publish_dsh3_33_operator_policy_disposition.py``, this script does not
call ``publish_agentv_evaluation``: the AgentV/AgentEvals Node.js SDK
(``@agentv/core``) is not installed in every environment this script must run
in (no committed lockfile install, and installing it here would pull
Playwright browser binaries this task does not need). The disposition's own
structural invariants -- exactly eleven claims, every claim addressing all
four dimensions, no advanced path enabled by default, a rejected inherited
policy carrying no allowed inventory -- are already enforced by
``AdvancedOperatorDispositionV1.__post_init__`` and independently re-checked
by ``scripts/validate_advanced_operator_disposition.py``, which is this
issue's required schema validator. The ``self_check`` block below records the
same pass/fail facts an AgentEvals case would have asserted, computed in pure
Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.evals.advanced_operator_disposition import (
    AdvancedOperatorClaim,
    AdvancedOperatorDimension,
    build_advanced_operator_disposition,
)
from slm_training.harness_core.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "dsh3_33_report": "docs/design/dsh3-33-rebased-cap2-disposition-20260726-local/report.json",
    "dsh5_03_report": "docs/design/dsh5-03-bulk-operator-crossover-20260726-local/report.json",
    "dsh5_09_report": "docs/design/dsh5-09-adaptive-router-20260726-local/report.json",
}

# Issue -> the real registry component that issue's own change owns. SLM-419
# (DSH5-11) is a repository-wide audit that touched no watched component, so
# it gets a bare stamp (empty ``components``) rather than a fabricated owner.
_COMPONENT_BY_ISSUE = {
    "SLM-409": "dsl.operators.selectors",
    "SLM-410": "dsl.operators.bulk",
    "SLM-412": "dsl.operators.transactions",
    "SLM-413": "dsl.operators.transaction_executor",
    "SLM-414": "harness.experiments.operator_transaction_policy",
    "SLM-415": "dsl.operators.sequence_merge",
    "SLM-416": "dsl.operators.control_actions",
    "SLM-418": "dsl.operators.replay_preference",
}


def _component_version_stamps() -> dict[str, dict[str, Any]]:
    stamps = {issue: build_version_stamp(component) for issue, component in _COMPONENT_BY_ISSUE.items()}
    stamps["SLM-419"] = build_version_stamp()
    return stamps


def _self_check(disposition: dict[str, Any]) -> dict[str, Any]:
    claims = {item["claim"]: item for item in disposition["claims"]}
    cases = [
        {
            "id": "complete-claim-ledger",
            "criteria": "Every one of the eleven required advanced-operator claims has one typed disposition.",
            "pass": len(claims) == len(AdvancedOperatorClaim),
        },
        {
            "id": "every-claim-addresses-every-dimension",
            "criteria": "Every claim states an explicit verdict for all four independent dimensions.",
            "pass": all(
                set(claim["dimension_verdicts"]) == {d.value for d in AdvancedOperatorDimension} for claim in claims.values()
            ),
        },
        {
            "id": "negative-and-unavailable-retained",
            "criteria": "Unrun/unavailable crossover, routing, template, and systems verdicts remain non-promotional.",
            "pass": (
                claims["crossover_work"]["verdict"] == "unavailable"
                and claims["adaptive_routing"]["verdict"] == "unavailable"
                and claims["parameterized_templates"]["verdict"] == "unavailable"
                and claims["systems_efficiency"]["verdict"] == "unavailable"
            ),
        },
        {
            "id": "control-plane-execution-not-conflated-with-learning",
            "criteria": "control_plane_execution_learning separates a supported execution axis from an unrun learning axis.",
            "pass": (
                claims["control_plane_execution_learning"]["dimension_verdicts"]["runtime_correctness"] == "supported"
                and claims["control_plane_execution_learning"]["dimension_verdicts"]["learned_semantic_benefit"]
                == "unrun_conditional"
            ),
        },
        {
            "id": "no-advanced-path-enabled-by-default",
            "criteria": "This disposition never flips an advanced-operator path on by default.",
            "pass": disposition["advanced_path_enabled_by_default"] is False,
        },
        {
            "id": "inherited-policy-inventory-empty",
            "criteria": "The DSH3-33/SLM-408 allowed learned head/objective/action inventory is inherited, not re-derived, and is empty.",
            "pass": (
                disposition["inherited_policy"]["dsh5_may_start"] is False
                and not disposition["inherited_policy"]["allowed_heads"]
                and not disposition["inherited_policy"]["allowed_objectives"]
                and not disposition["inherited_policy"]["allowed_actions"]
            ),
        },
        {
            "id": "cap2-retained-not-advanced",
            "criteria": "CAP2 certificate posture is recorded as retained unchanged, never advanced.",
            "pass": next(item for item in disposition["retention"] if item["capability"] == "CAP2")["verdict"] == "supported",
        },
    ]
    passed = sum(1 for case in cases if case["pass"])
    return {
        "runner": "pure_python_self_check",
        "note": (
            "AgentV/AgentEvals (@agentv/core) publication was intentionally skipped: not installed in "
            "this environment and installing it was judged not worth the disk/dependency risk for this "
            "issue. This block re-checks the same pass/fail facts in pure Python; the disposition's own "
            "dataclasses already enforce these invariants structurally via __post_init__, and "
            "scripts/validate_advanced_operator_disposition.py re-validates the published artifact."
        ),
        "cases": cases,
        "summary": {"passed": passed, "total": len(cases), "pass": passed == len(cases)},
    }


def _markdown(disposition: dict[str, Any], self_check: dict[str, Any]) -> str:
    lines = [
        "# DSH5-12 advanced-operator disposition",
        "",
        "Status: **DSH5 advanced abstractions dispositioned** — compiler-owned "
        "runtime utilities (selectors, bulk atomicity, transaction contracts/"
        "execution, sequence merge, control-plane execution) are supported; "
        "no learned-benefit, routing, template, or systems-efficiency claim "
        "is supported.",
        "",
        "| Claim | Verdict | Evidence |",
        "| --- | --- | --- |",
    ]
    for claim in disposition["claims"]:
        lines.append(f"| `{claim['claim']}` | `{claim['verdict']}` | {', '.join(claim['evidence_ids']) or 'none'} |")
    lines.extend(
        [
            "",
            "| Retention | Verdict |",
            "| --- | --- |",
        ]
    )
    for item in disposition["retention"]:
        lines.append(f"| `{item['capability']}` | `{item['verdict']}` |")
    lines.extend(
        [
            "",
            f"**Inherited policy inventory (SLM-408 / DSH3-33):** may_start="
            f"`{disposition['inherited_policy']['dsh5_may_start']}`, allowed_heads="
            f"`{disposition['inherited_policy']['allowed_heads']}`, allowed_objectives="
            f"`{disposition['inherited_policy']['allowed_objectives']}`, allowed_actions="
            f"`{disposition['inherited_policy']['allowed_actions']}`.",
            "",
            f"**Recommendation:** `{disposition['recommendation']}` — "
            f"{disposition['recommendation_reason']}",
            "",
            f"Self-check: {self_check['summary']['passed']}/{self_check['summary']['total']} "
            "pure-Python structural cases pass (AgentV publication intentionally skipped; see "
            "`self_check.note` in report.json).",
            "",
            "The historical DSH3-33 (SLM-408) disposition remains unchanged and authoritative for "
            "the CAP2 learned-policy line this program rebased from. No checkpoint, model-card "
            "checkpoint-roster change, remote run, human-rating gate, production change, or "
            "advanced-operator default-on authorization follows from this disposition.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=ROOT / default)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = {name: json.loads(getattr(args, name).read_text(encoding="utf-8")) for name in DEFAULTS}
    stamp = build_version_stamp("evals.advanced_operator_disposition")
    disposition = build_advanced_operator_disposition(
        **reports,
        version_stamp=stamp,
        component_version_stamps=_component_version_stamps(),
    ).to_dict()

    self_check = _self_check(disposition)
    payload = {
        "schema": "dsh5_12_advanced_operator_disposition_report/v1",
        "disposition": disposition,
        "self_check": self_check,
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(_markdown(disposition, self_check), encoding="utf-8")
    return 0 if self_check["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
