"""Fail-closed local activation audit for SLM-268 (VSD2-03)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

SCHEMA = "verified_data_scaling_activation/v1"
EXPERIMENT_ID = "slm268-verified-data-scaling-activation"

# These are the prerequisite evidence records required by the issue.  Their
# contents are audited, not trusted merely because the Linear issues are Done.
GATES = (
    ("SLM-260", "docs/design/oracle-scoring-replay.md", "fixture_only"),
    ("SLM-261", "docs/design/training-loss-ledger.md", "missing_or_unqualified"),
    ("SLM-262", "docs/design/iter-slm262-gpu-reference-run-20260721.json", "environment_incompatible"),
    ("SLM-263", "docs/design/iter-slm263-semantic-failure-census-20260724.json", "insufficient_evidence"),
    ("SLM-264", "docs/design/iter-slm264-semantic-collapse-telemetry-20260724.json", "insufficient_evidence"),
    ("SLM-265", "docs/design/iter-slm265-domain-shift-audit-20260724.json", "insufficient_evidence"),
    ("SLM-266", "docs/design/iter-slm266-local-yield-disposition-20260724.json", "budget_or_yield_blocked"),
    ("SLM-267", "docs/design/iter-slm267-programspec-coverage-scaling-20260725.json", "inconclusive"),
)

def _sha(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def build_activation_report(root: Path) -> dict[str, Any]:
    """Return a self-hashed no-spend report; no corpus or training is launched."""
    gates = []
    for issue, relative, reason in GATES:
        path = root / relative
        gates.append({"issue": issue, "path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing", "qualified": False, "reason": reason})
    report: dict[str, Any] = {
        "schema": SCHEMA, "experiment_id": EXPERIMENT_ID, "claim_class": "diagnostic",
        "activation_status": "blocked", "activation_verdict": "corpus_or_provenance_blocked",
        "protocol": None, "launchable_rungs": [], "gates": gates,
        "arms": {arm: {"available": False, "reason": "activation_blocked"} for arm in ("current_verified_unique_10k", "teacher_verified_10k", "compiler_targeted_10k", "hybrid_teacher_compiler_10k")},
        "successor_conditions": ["qualified full scorer replay", "qualified trainer protocol and durable local cost path", "frozen full evaluation strata", "10k qualified teacher and compiler manifests"],
        "honesty": "No train, eval, checkpoint, corpus publication, or ship claim was made.",
        "version_stamp": build_version_stamp("harness.experiments", "harness.experiments.slm268_verified_data_activation"),
    }
    report["manifest_sha256"] = _sha({key: value for key, value in report.items() if key != "version_stamp"})
    return report

def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(f"| {g['issue']} | `{g['reason']}` | `{g['sha256']}` |" for g in report["gates"])
    return f"# SLM-268 verified-data-scaling activation\n\n**Status:** `{report['activation_status']}` — `{report['activation_verdict']}`\n\n{report['honesty']}\n\n| prerequisite | disposition | evidence SHA-256 |\n| --- | --- | --- |\n{rows}\n\nNo rung is launchable.\n"
