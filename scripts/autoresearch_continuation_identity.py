"""Physical release/environment identity; never aliases a historical source."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ContinuationGrant:
    """Controller-resolved identity and finite logical wall allowance."""

    execution_identity: str
    total_wall_seconds: float
    max_attempts: int | None = None
    interrupt_seconds: float | None = None
    finalization_reserve_seconds: float | None = None


def resolved_continuation_grant(
    root: Path,
    total_seconds: float,
    max_attempts=None,
    *,
    interrupt_seconds: float | None = None,
    finalization_reserve_seconds: float | None = None,
) -> ContinuationGrant:
    """Reuse the release/environment identity owners, excluding incidental clocks."""
    from scripts.merge_verification_evidence import (
        environment_identity,
        source_identity,
    )
    from slm_training.harness_core.activity_contract import contract_digest
    from slm_training.harness_core.execution_release import runtime_source_identity

    identity_root = Path(root).resolve()
    if not (identity_root / ".autonomy-release.json").is_file():
        try:
            identity_root = Path(
                subprocess.run(
                    ["git", "-C", str(identity_root), "rev-parse", "--show-toplevel"],
                    check=True, capture_output=True, text=True, timeout=10,
                ).stdout.strip()
            ).resolve()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    identity = contract_digest(
        {
            "source": runtime_source_identity(identity_root) or source_identity(identity_root),
            "environment": environment_identity(),
        }
    )
    return ContinuationGrant(
        identity, total_seconds, max_attempts,
        interrupt_seconds, finalization_reserve_seconds,
    )


def resolve_campaign_grant(campaign, root, resolver):
    """Resolve current physical identity while preserving configured logical budget."""
    logical_grant = campaign.budget.continuation_grant
    try:
        return resolver(
            root,
            campaign.budget.logical_seconds,
            logical_grant.max_attempts if logical_grant else None,
            interrupt_seconds=logical_grant.interrupt_seconds if logical_grant else None,
            finalization_reserve_seconds=(
                logical_grant.finalization_reserve_seconds if logical_grant else None
            ),
        )
    except TypeError as exc:
        # Preserve compatibility with older injected/test resolvers while the
        # canonical resolver carries the full grant contract.
        if "unexpected keyword argument" not in str(exc):
            raise
        return resolver(
            root, campaign.budget.logical_seconds,
            logical_grant.max_attempts if logical_grant else None,
        )
