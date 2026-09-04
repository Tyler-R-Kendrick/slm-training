"""The formal claims a promotion carries.

One responsibility: the claim record itself -- binding a fresh replay's claims,
restoring frozen ones, and reading or writing the preflight status they roll up
to.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.autotrain_io import read_json
from slm_training.autoresearch.experiment_campaign import (
    ExperimentCampaignV1,
)
from slm_training.autoresearch.formal import formal_obligation_id
from slm_training.autoresearch.schemas import (
    FormalClaimV1,
    FormalObligationV1,
    FormalPreflightV1,
)

PROMOTE_FORMAL_TEMPLATE_ID = "metrics.structural_similarity_monotone"


def formal_preflight_status(camp_dir: Path) -> str | None:
    """Read only a cache-validated formal preflight status for promote gate."""
    path = camp_dir / "formal_preflight_status.json"
    if path.is_file():
        data = read_json(path)
        status = data.get("status")
        if status != "proved":
            return str(status) if status is not None else None
        expected_sha = data.get("preflight_sha256")
        validated_sha = data.get("binding_validated_sha256")
        if expected_sha and validated_sha == expected_sha:
            return "proved"
    return None




def promote_formal_claim_dict() -> dict[str, str]:
    """Canonical required formal claim payload for promote experiment specs."""
    return {
        "template_id": PROMOTE_FORMAL_TEMPLATE_ID,
        "claim": (
            "Structural similarity is monotone under declared component "
            "inequalities for continuous promote."
        ),
        "policy": "required",
    }


def restore_frozen_formal_claims(
    camp_dir: Path,
    experiment: dict[str, Any],
    manifest: ExperimentCampaignV1,
) -> None:
    """Recover claims omitted by an older replay from its proved artifacts."""

    if experiment.get("formal_claims") or not manifest.formal_obligations:
        return
    claims: list[dict[str, str]] = []
    for obligation in manifest.formal_obligations:
        path = (
            camp_dir
            / "artifacts"
            / "formal_preflights"
            / f"{obligation.preflight_sha256}.json"
        )
        if not path.is_file():
            raise RuntimeError(
                "frozen replay formal preflight is missing: "
                f"{obligation.preflight_sha256}"
            )
        preflight = FormalPreflightV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        claim = FormalClaimV1(
            template_id=preflight.template_id,
            claim=preflight.claim,
            policy=preflight.policy,
        )
        if (
            preflight.campaign_id != manifest.campaign_id
            or preflight.experiment_id != manifest.experiment_id
            or preflight.template_id != obligation.template_id
            or preflight.policy != obligation.policy
            or (claim.policy == "required" and preflight.status != "proved")
            or formal_obligation_id(manifest.campaign_id, manifest.experiment_id, claim)
            != preflight.obligation_id
        ):
            raise RuntimeError(
                "frozen replay formal claim recovery mismatch: "
                f"{obligation.obligation_id}"
            )
        claims.append(claim.model_dump())
    experiment["formal_claims"] = claims


def bind_fresh_replay_formal_preflight(
    successor: ExperimentCampaignV1,
    frozen: ExperimentCampaignV1,
    *,
    preflight_sha256: str,
    formal_claims: list[dict[str, str]],
) -> ExperimentCampaignV1:
    """Bind frozen claim policy to current identities and the fresh proof."""

    claims = tuple(FormalClaimV1.model_validate(claim) for claim in formal_claims)
    expected = sorted(
        (obligation.template_id, obligation.policy)
        for obligation in frozen.formal_obligations
    )
    actual = sorted((claim.template_id, claim.policy) for claim in claims)
    if actual != expected:
        raise RuntimeError("frozen replay formal claim policy mismatch")
    obligations = tuple(
        FormalObligationV1(
            obligation_id=formal_obligation_id(
                successor.campaign_id, successor.experiment_id, claim
            ),
            template_id=claim.template_id,
            policy=claim.policy,
            preflight_sha256=preflight_sha256,
        )
        for claim in claims
    )
    rebound = successor.model_copy(update={"formal_obligations": obligations})
    return ExperimentCampaignV1.model_validate(rebound.model_dump(mode="json"))
