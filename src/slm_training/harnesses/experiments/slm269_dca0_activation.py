"""Fail-closed activation audit for SLM-269 (DCA0-01).

DCA0-01 ("Compare canonical-only targets with certified equivalent-surface
augmentation at fixed semantic exposure") is blocked by three prerequisites
whose artifacts must be audited, not trusted merely because the Linear
issues are Done:

* SLM-260 — equivalent variants must pass oracle replay (full scorer, not a
  fixture);
* SLM-261 — trainer and loss accounting qualified (LossLedgerV1);
* SLM-268 — a selected verified-data regime OR an explicit large-scale
  negative result.

SLM-268 itself closed with ``activation_status=blocked`` /
``corpus_or_provenance_blocked`` (no selected regime, no negative result),
so no matched corpus arm can be built honestly and the canonical-vs-
equivalent training matrix cannot launch. This audit emits that
machine-readable disposition; it launches no training and builds no corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

SCHEMA = "dca0_target_policy_activation/v1"
EXPERIMENT_ID = "slm269-dca0-01-target-policy-activation"

# (issue, artifact path, audit disposition when unqualified)
GATES = (
    ("SLM-260", "docs/design/oracle-scoring-replay.md", "fixture_only_not_full_scorer"),
    ("SLM-261", "docs/design/training-loss-ledger.md", "missing_or_unqualified"),
    (
        "SLM-268",
        "docs/design/iter-slm268-verified-data-scaling-activation-20260725.json",
        "blocked_no_selected_regime_no_negative_result",
    ),
)


def _sha(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_activation_report(root: Path) -> dict[str, Any]:
    """Return a self-hashed no-spend report; no corpus or training is launched."""
    gates = []
    for issue, relative, reason in GATES:
        path = root / relative
        gates.append(
            {
                "issue": issue,
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else "missing",
                "qualified": False,
                "reason": reason,
            }
        )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "claim_class": "diagnostic",
        "activation_status": "blocked",
        "activation_verdict": "prerequisite_evidence_unqualified",
        "disposition": "inconclusive",
        "gates": gates,
        "corpus_arms": {
            arm: {"available": False, "reason": "activation_blocked"}
            for arm in (
                "canonical_only",
                "equivalent_random_one",
                "equivalent_all_matched",
                "equivalent_all_extra_exposure",
                "shuffled_variant_set",
            )
        },
        "successor_conditions": [
            "qualified full scorer oracle replay (SLM-260 successor)",
            "qualified trainer memorization + LossLedgerV1 (SLM-261 successor)",
            "SLM-268 successor: selected verified-data regime or explicit "
            "large-scale negative result",
            "then file the EquivalentTargetSetV1 contract + applicability "
            "audit + matched training matrix",
        ],
        "honesty": "No train, eval, corpus build, checkpoint, or ship claim was made.",
        "version_stamp": build_version_stamp(
            "harness.experiments", "harness.experiments.slm269_dca0_activation"
        ),
    }
    report["manifest_sha256"] = _sha(
        {key: value for key, value in report.items() if key != "version_stamp"}
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {g['issue']} | `{g['reason']}` | `{g['sha256']}` |" for g in report["gates"]
    )
    return (
        "# SLM-269 DCA0-01 target-policy activation\n\n"
        f"**Status:** `{report['activation_status']}` — `{report['activation_verdict']}`\n\n"
        f"Disposition: `{report['disposition']}`\n\n"
        f"{report['honesty']}\n\n"
        "| prerequisite | disposition | evidence SHA-256 |\n"
        "| --- | --- | --- |\n"
        f"{rows}\n\n"
        "No corpus arm or matrix cell is launchable. The canonical-vs-equivalent "
        "question (DCA-H1a/H1b) remains open; only this launch is gated.\n"
    )
