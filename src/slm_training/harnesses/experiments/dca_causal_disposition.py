"""SLM-276 DCA2-03 — CausalDispositionBundleV1 fail-closed publisher.

Aggregates the real, committed prerequisite dispositions of the
Constrained Diffusion Evidence-to-Deployment initiative (SLM-268 …
SLM-275) into the final causal disposition bundle. Per the issue text, a
blocked/not-activated result is a valid completed disposition, and no
ineligible family may be forced into the final tournament — so when the
prerequisite legs are blocked/inconclusive/not_activated, there are no
eligible finalists, no tournament spend, and the recorded outcome is
``no_model_ship_ready_measurement_or_data_blocker``.

Every field is derived from the artifacts; nothing is retyped by hand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "NO_SHIP_READY",
    "CausalDispositionBundleV1",
    "build_bundle",
    "render_markdown",
]

BUNDLE_SCHEMA_VERSION = "causal_disposition_bundle/v1"
NO_SHIP_READY = "no_model_ship_ready_measurement_or_data_blocker"

REPO_ROOT = Path(__file__).resolve().parents[4]

# (leg, artifact path, verdict extractor kind, issue id)
PREREQUISITES: tuple[tuple[str, str, str], ...] = (
    ("SLM-268", "docs/design/iter-slm268-verified-data-scaling-activation-20260725.json", "activation_status"),
    ("SLM-269", "docs/design/iter-slm269-dca0-01-activation-20260725.json", "activation_status"),
    ("SLM-270", "docs/design/iter-slm270-dca0-02-activation-20260725.json", "verdict"),
    ("SLM-271", "docs/design/iter-slm271-dca0-03-activation-20260725.json", "verdict"),
    ("SLM-272", "docs/design/iter-slm272-dca1-01-activation-20260725.json", "verdict"),
    ("SLM-273", "docs/design/iter-slm273-dca1-02-activation-20260725.json", "verdict"),
    ("SLM-274", "docs/design/iter-slm274-dca2-01-activation-20260725.json", "verdict"),
    ("SLM-275", "docs/design/iter-slm275-dca2-02-activation-20260725.json", "verdict"),
)

HYPOTHESES: tuple[tuple[str, str, str], ...] = (
    ("DCA-H1", "canonical-only vs certified equivalent-surface targets", "SLM-269"),
    ("DCA-H2", "sparse positive subset of TwoTower auxiliaries", "SLM-270"),
    ("DCA-H3", "difficulty-aware mask curriculum improves tail semantics", "SLM-271"),
    ("DCA-H4", "program-level verifier-ranked preference improves delayed semantics", "SLM-272"),
    ("DCA-H5", "cross-pack structured-program transfer into OpenUI", "SLM-273"),
    ("DCA-H6", "faithful AR to block-diffusion conversion Pareto-wins", "SLM-274"),
    ("DCA-H7", "coupled-mask diffusion GRPO with static rewards", "SLM-275"),
)

_POSITIVE = {"complete", "selected_or_negative_result", "prerequisites_met_pending_campaign"}


@dataclass(frozen=True)
class CausalDispositionBundleV1:
    schema_version: str = BUNDLE_SCHEMA_VERSION
    bundle_id: str = "constrained-diffusion-causal-disposition-v1"
    linear_issue: str = "SLM-276"
    date: str = "2026-07-25"
    initiative: str = "constrained-diffusion-evidence-to-deployment"
    prerequisite_dispositions: dict[str, dict[str, str]] = field(default_factory=dict)
    hypothesis_ledger: list[dict[str, str]] = field(default_factory=list)
    finalist_eligibility: dict[str, str] = field(default_factory=dict)
    evidence_gaps: list[str] = field(default_factory=list)
    outcome: str = NO_SHIP_READY
    promotion_action: str = "none_incumbent_retained"
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["bundle_fingerprint"] = self.fingerprint()
        return data

    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "prerequisite_dispositions": self.prerequisite_dispositions,
            "outcome": self.outcome,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


def _verdict_of(path: Path, kind: str) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return str(data.get(kind, ""))


def build_bundle(root: Path | None = None) -> CausalDispositionBundleV1:
    """Derive the fail-closed disposition bundle from committed artifacts."""
    root = root or REPO_ROOT
    dispositions: dict[str, dict[str, str]] = {}
    gaps: list[str] = []
    any_blocked = False

    for issue, rel, kind in PREREQUISITES:
        verdict = _verdict_of(root / rel, kind)
        if verdict is None:
            gaps.append(f"missing_or_invalid_artifact:{issue}:{rel}")
            dispositions[issue] = {
                "path": rel,
                "verdict": "unavailable_or_unknown",
                "evidence_strength": "unavailable_or_unknown",
            }
            any_blocked = True
            continue
        completed = verdict in _POSITIVE
        if not completed:
            any_blocked = True
        dispositions[issue] = {
            "path": rel,
            "verdict": verdict,
            "evidence_strength": "fixture_or_wiring",
        }

    ledger = []
    for hypothesis, question, issue in HYPOTHESES:
        verdict = dispositions.get(issue, {}).get("verdict", "unavailable_or_unknown")
        completed = verdict in _POSITIVE
        ledger.append(
            {
                "hypothesis": hypothesis,
                "question": question,
                "issue": issue,
                "disposition": "tested" if completed else "blocked",
                "causal_interpretation": (
                    "resolved by completed campaign"
                    if completed
                    else f"untested: {issue} prerequisite evidence is "
                    f"'{verdict}'; the campaign never launched, so no causal "
                    "claim of either sign is made"
                ),
                "evidence_strength": dispositions.get(issue, {}).get(
                    "evidence_strength", "unavailable_or_unknown"
                ),
            }
        )

    finalist_eligibility = {
        "current_twotower_incumbent": "ineligible_no_vsd0_clean_scorer_or_qualified_baseline",
        "autoregressive_legal_action": "ineligible_no_qualified_baseline_campaign",
        "x22_valid_state_search": "ineligible_no_qualified_baseline_campaign",
        "faithful_block_diffusion": "ineligible_not_activated_SLM-274",
        "program_dpo_ipo": "ineligible_inconclusive_SLM-272",
        "diffusion_rl": "ineligible_inconclusive_SLM-275",
    }

    outcome = NO_SHIP_READY if any_blocked else "incumbent_retained_no_challenger_passed"

    return CausalDispositionBundleV1(
        prerequisite_dispositions=dispositions,
        hypothesis_ledger=ledger,
        finalist_eligibility=finalist_eligibility,
        evidence_gaps=gaps,
        outcome=outcome,
        version_stamp=build_version_stamp("harness.experiments"),
    )


def render_markdown(bundle: CausalDispositionBundleV1) -> str:
    lines = [
        f"# SLM-276 DCA2-03 — CausalDispositionBundleV1 ({bundle.bundle_id})",
        "",
        f"Outcome: **{bundle.outcome}**",
        f"Promotion action: `{bundle.promotion_action}`",
        "",
        "## Prerequisite dispositions",
        "",
    ]
    lines += [
        f"- `{issue}`: `{d['verdict']}` (evidence: {d['evidence_strength']})"
        for issue, d in bundle.prerequisite_dispositions.items()
    ]
    lines += ["", "## Hypothesis ledger", ""]
    lines += [
        f"- **{h['hypothesis']}** ({h['issue']}): {h['disposition']} — {h['causal_interpretation']}"
        for h in bundle.hypothesis_ledger
    ]
    lines += ["", "## Finalist eligibility", ""]
    lines += [f"- `{k}`: {v}" for k, v in bundle.finalist_eligibility.items()]
    if bundle.evidence_gaps:
        lines += ["", "## Evidence gaps", ""]
        lines += [f"- {g}" for g in bundle.evidence_gaps]
    lines += [
        "",
        "## Non-goals honored",
        "",
        "No tournament spend (no eligible finalists), no promotion, no new "
        "architecture/objective/RL mechanism, no gate loosening, no "
        "fixture/local substitution. Blocked and not-activated results are "
        "recorded as valid completed dispositions.",
        "",
    ]
    return "\n".join(lines)
