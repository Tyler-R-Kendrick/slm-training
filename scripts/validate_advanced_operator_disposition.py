#!/usr/bin/env python3
"""Validate a published DSH5-12 (SLM-420) advanced-operator disposition.

Re-parses a ``report.json`` produced by
``scripts/publish_dsh5_12_advanced_operator_disposition.py`` into
``AdvancedOperatorDispositionV1`` and re-checks the issue's acceptance
criteria that are not already enforced by the dataclass's own
``__post_init__`` (which this reconstruction still exercises, since a
malformed payload will fail to build): that no conditional/unrun claim is
being counted as delivered value, and that the recommendation is drawn from
the issue's exact five-option vocabulary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slm_training.evals.advanced_operator_disposition import (
    ALLOWED_RECOMMENDATIONS,
    AdvancedOperatorClaim,
    AdvancedOperatorClaimV1,
    AdvancedOperatorDimension,
    AdvancedOperatorDispositionV1,
    AdvancedOperatorEvidenceV1,
    AdvancedOperatorVerdict,
    InheritedPolicyInventoryV1,
    RetentionClaimV1,
)


def _load_evidence(payload: dict[str, Any]) -> AdvancedOperatorEvidenceV1:
    return AdvancedOperatorEvidenceV1(
        evidence_id=payload["evidence_id"],
        issue_alias=payload["issue_alias"],
        linear_id=payload["linear_id"],
        artifact_path=payload["artifact_path"],
        code_commit=payload["code_commit"],
        version_stamp=payload["version_stamp"],
        data_identity=payload["data_identity"],
        suite_identity=payload["suite_identity"],
        checkpoint_identity=payload["checkpoint_identity"],
        tokenizer_identity=payload["tokenizer_identity"],
        registry_identity=payload["registry_identity"],
        schema_identity=payload["schema_identity"],
        objective_identity=payload["objective_identity"],
        decode_policy=payload["decode_policy"],
        hardware_identity=payload["hardware_identity"],
        result_identity=payload["result_identity"],
        schema=payload.get("schema", "advanced_operator_evidence/v1"),
    )


def _load_claim(payload: dict[str, Any]) -> AdvancedOperatorClaimV1:
    return AdvancedOperatorClaimV1(
        claim=AdvancedOperatorClaim(payload["claim"]),
        verdict=AdvancedOperatorVerdict(payload["verdict"]),
        reason=payload["reason"],
        dimension_verdicts={
            AdvancedOperatorDimension(dim): AdvancedOperatorVerdict(verdict)
            for dim, verdict in payload["dimension_verdicts"].items()
        },
        dimension_reasons={AdvancedOperatorDimension(dim): reason for dim, reason in payload["dimension_reasons"].items()},
        evidence_ids=tuple(payload["evidence_ids"]),
        schema=payload.get("schema", "advanced_operator_claim/v1"),
    )


def load_disposition(payload: dict[str, Any]) -> AdvancedOperatorDispositionV1:
    """Reconstruct the typed disposition from a published ``report.json``'s
    ``disposition`` block, re-running every structural invariant."""
    inherited = payload["inherited_policy"]
    return AdvancedOperatorDispositionV1(
        evidence=tuple(_load_evidence(item) for item in payload["evidence"]),
        claims=tuple(_load_claim(item) for item in payload["claims"]),
        inherited_policy=InheritedPolicyInventoryV1(
            source_disposition=inherited["source_disposition"],
            source_artifact_path=inherited["source_artifact_path"],
            dsh5_may_start=inherited["dsh5_may_start"],
            allowed_heads=tuple(inherited["allowed_heads"]),
            allowed_objectives=tuple(inherited["allowed_objectives"]),
            allowed_actions=tuple(inherited["allowed_actions"]),
            reason=inherited["reason"],
        ),
        retention=tuple(
            RetentionClaimV1(
                capability=item["capability"],
                verdict=AdvancedOperatorVerdict(item["verdict"]),
                reason=item["reason"],
                evidence_ids=tuple(item["evidence_ids"]),
            )
            for item in payload["retention"]
        ),
        advanced_path_enabled_by_default=payload["advanced_path_enabled_by_default"],
        recommendation=payload["recommendation"],
        recommendation_reason=payload["recommendation_reason"],
        historical_disposition=payload["historical_disposition"],
        version_stamp=payload["version_stamp"],
    )


def validate(report: dict[str, Any]) -> list[str]:
    """Return a list of human-readable violations; empty means valid."""
    violations: list[str] = []
    disposition_payload = report.get("disposition")
    if disposition_payload is None:
        return ["report.json has no top-level 'disposition' object"]

    try:
        disposition = load_disposition(disposition_payload)
    except (KeyError, ValueError, TypeError) as exc:
        return [f"disposition failed to reconstruct/validate: {exc}"]

    # The claim-count, recommendation, and default-enable checks below are
    # defense-in-depth: `load_disposition` above already exercises
    # `AdvancedOperatorDispositionV1.__post_init__`, which rejects a wrong
    # claim count, an unknown recommendation, and `advanced_path_enabled_by_default=True`
    # before this function ever reaches them. They stay reachable only if
    # those dataclass invariants are ever weakened; the checks further below
    # (inherited-policy inventory, historical-disposition reference, and
    # SUPPORTED-with-no-evidence) add genuinely new coverage today.
    if len(disposition.claims) != len(AdvancedOperatorClaim):
        violations.append(
            f"expected exactly {len(AdvancedOperatorClaim)} claims, found {len(disposition.claims)}"
        )

    if disposition.recommendation not in ALLOWED_RECOMMENDATIONS:
        violations.append(f"recommendation {disposition.recommendation!r} is not one of the five allowed options")

    if disposition.advanced_path_enabled_by_default:
        violations.append("advanced_path_enabled_by_default must be False")

    # Acceptance: "No conditional/unrun issue is counted as delivered value."
    # `AdvancedOperatorClaimV1.__post_init__` (exercised by `load_disposition`
    # above) already enforces the load-bearing half of this: a SUPPORTED
    # headline must equal one of its own dimension verdicts, so a SUPPORTED
    # claim can never have every dimension undelivered -- that state is
    # unconstructable, not merely unchecked. This loop only re-asserts the
    # other half the dataclass does not already cover: cited evidence.
    for claim in disposition.claims:
        if claim.verdict is AdvancedOperatorVerdict.SUPPORTED and not claim.evidence_ids:
            violations.append(f"claim {claim.claim.value!r} is SUPPORTED with no cited evidence")

    # Acceptance: DSH5 default-on / policy inventory only follows explicit
    # authorization from the inherited (DSH3-33/SLM-408) disposition.
    if not disposition.inherited_policy.dsh5_may_start and (
        disposition.inherited_policy.allowed_heads
        or disposition.inherited_policy.allowed_objectives
        or disposition.inherited_policy.allowed_actions
    ):
        violations.append("inherited policy inventory is non-empty despite dsh5_may_start=False")

    # Historical evidence is referenced, never rewritten.
    if "dsh3-33" not in disposition.historical_disposition.lower():
        violations.append("historical_disposition must reference the DSH3-33 (SLM-408) artifact, not omit it")

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to a published dsh5-12 report.json")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    violations = validate(report)
    if violations:
        for violation in violations:
            print(f"FAIL: {violation}", file=sys.stderr)
        return 1

    disposition = report["disposition"]
    print(
        json.dumps(
            {
                "decision": "structurally_valid",
                "claim_count": len(disposition["claims"]),
                "recommendation": disposition["recommendation"],
                "dsh5_may_start": disposition["inherited_policy"]["dsh5_may_start"],
                "advanced_path_enabled_by_default": disposition["advanced_path_enabled_by_default"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
