"""Controller-only integration with the independent isolated verification owner."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from slm_training.autoresearch.heal.dispatch import accept_verification
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    private_snapshot,
    tree_manifest,
)
from slm_training.autoresearch.heal.repair_contracts import (
    RepairDispatchResult,
    RepairProposal,
    RepairRequest,
    RepairVerification,
    VerificationBinding,
)
from slm_training.autoresearch.heal.repair_verifier import (
    VerificationCheck,
    VerificationRequest,
    verify_candidate,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    MAX_RUN_SECONDS,
)
from slm_training.lineage.records import canonical_json


@dataclass(frozen=True)
class VerificationWorkspace:
    base: Path
    candidate: Path
    runtime_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SourceVerificationGate:
    """Controller-selected canonical merge journal, never a worker-supplied file.

    ``root`` is the source-aware verifier's private Git materialization. Its
    source snapshot must equal the independently checked candidate byte/mode
    manifest. Git's comparison base may be older than the repair input, but its
    selected changes must cover every actual repair change. The locked original
    input is independently reproduced; the two base identities are not conflated.

    The controller must keep this journal/issuer outside every workload mount.
    MAC validation is integrity, not an alternative to that OS boundary. These
    lazy imports reuse the canonical script-owned evidence implementation; no
    candidate module is imported or alternate verdict is constructed here.
    """

    root: Path
    state_dir: Path
    base_ref: str
    identity: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, str) or len(self.identity) != 64
            or any(char not in "0123456789abcdef" for char in self.identity)
            or not isinstance(self.base_ref, str) or not self.base_ref.strip()
        ):
            raise ValueError("source verification requires a locked identity and base")

    def read(self, workspace: VerificationWorkspace) -> dict | None:
        from scripts.merge_verification import _summary, verification_binding
        from scripts.merge_verification_evidence import (
            ReceiptCache, digest, validate_cached_state,
        )
        from scripts.verify_merge_ready import merge_gate_steps

        runtimes = workspace.runtime_roots or (Path(sys.prefix),)
        protected = self.state_dir.resolve()
        for exposed in (self.root, workspace.base, workspace.candidate, *runtimes):
            exposed = exposed.resolve()
            if protected.is_relative_to(exposed) or exposed.is_relative_to(protected):
                raise ValueError("source_verification_issuer_exposed_to_workload")
        # Do not create a fresh signing key merely to discover missing evidence.
        # Materialization/collection belongs to the separately scheduled verifier;
        # its root need not exist yet when this dependency is first registered.
        if not (self.state_dir / "issuer.key").exists():
            return None
        state = ReceiptCache(self.state_dir, self.root).load(self.identity)
        if state is None:
            return None
        binding = verification_binding(
            self.root, self.base_ref, merge_gate_steps(),
            isolated=True, runtimes=runtimes,
        )
        if digest(binding) != self.identity:
            raise ValueError("source_verification_binding_changed")
        validate_cached_state(state, binding)
        summary = _summary(state)
        if not summary["verification_complete"]:
            return None
        with tempfile.TemporaryDirectory(prefix="slm-source-binding-") as temporary:
            source = private_snapshot(self.root, Path(temporary) / "source")
            if tree_manifest(source) != tree_manifest(workspace.candidate):
                raise ValueError("source_verification_candidate_mismatch")
        before, after = tree_manifest(workspace.base), tree_manifest(workspace.candidate)
        changed = {
            path for path in before.keys() | after.keys()
            if before.get(path) != after.get(path)
            and not ((workspace.candidate / path).is_dir() and path not in before)
        }
        if not changed <= set(binding["changed_paths"]):
            raise ValueError("source_verification_omits_repair_changes")
        return summary

    def authorize(self, summary: dict, evidence) -> bool:
        from scripts.merge_verification_evidence import authorize_release, digest

        # Only observations returned directly by the trusted independent runner
        # reach this call. A serialized worker receipt has no entrypoint here.
        independent = {
            "verification_identity": self.identity,
            "evidence_sha256": digest(summary),
            "scope_passed": evidence.accepted,
            "original_reproducer_passed": evidence.accepted,
            "isolation_enforced": evidence.evidence_class == "independent_isolated_process",
        }
        return authorize_release(
            summary, expected_identity=self.identity,
            independent_verification=independent,
        )


def check_manifest_digest(
    original: VerificationCheck,
    checks: tuple[VerificationCheck, ...],
    equivalence_checks: tuple[VerificationCheck, ...] = (),
) -> str:
    """Pre-patch locked check identity: candidate hashes are bound separately."""
    payload = {
        "original": asdict(original),
        "checks": [asdict(check) for check in checks],
        "equivalence_checks": [asdict(check) for check in equivalence_checks],
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def proposal_patch_digest(base: Path, candidate: Path) -> str:
    """Hash exact changed content/mode entries, independent of Git metadata."""
    return patch_manifest_digest(tree_manifest(base), tree_manifest(candidate))


def patch_manifest_digest(before: dict, after: dict) -> str:
    # JSON transport turns manifest tuples into lists; normalize both sides.
    before = {path: list(entry) for path, entry in before.items()}
    after = {path: list(entry) for path, entry in after.items()}
    changes = {
        path: {"before": before.get(path), "after": after.get(path)}
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    }
    return hashlib.sha256(canonical_json(changes).encode()).hexdigest()


def _reserve_verification(
    request: RepairRequest,
    spec: VerificationRequest,
    journal: CampaignStore,
    binding: VerificationBinding | None = None,
) -> str | None:
    assert request.grant is not None
    starts = [
        event
        for event in journal.verify_event_chain()
        if event["event_type"] in {"repair_started", "repair_verification_started"}
        and event.get("detail", {}).get("fingerprint") == request.blocker.fingerprint()
    ]
    if any(
        event["event_type"] == "repair_verification_started"
        and event["detail"].get("request_digest") == request.digest()
        and event["detail"].get("authority_binding_digest")
        == (binding.digest() if binding else None)
        for event in starts
    ):
        return "verification_attempt_requires_reconciliation"
    count = 2 + len(spec.checks) + 2 * len(spec.equivalence_checks)
    reserve = min(MAX_RUN_SECONDS, count * (spec.timeout_seconds + KILL_GRACE_SECONDS))
    spent = sum(float(event["detail"]["reserved_seconds"]) for event in starts)
    spent += sum(
        float(event["detail"]["reserved_seconds"])
        for event in journal.verify_event_chain()
        if event["event_type"] == "operation_diagnosis_started"
        and event["detail"]["grant_digest"] == request.grant.digest()
    )
    if spent + reserve > request.grant.total_seconds:
        return "verification_grant_exhausted"
    journal.append_event(
        "repair_verification_started",
        experiment_id=request.activity_id,
        detail={
            "fingerprint": request.blocker.fingerprint(),
            "request_digest": request.digest(),
            "fence": spec.fencing_token,
            "authority_binding_digest": binding.digest() if binding else None,
            "reserved_seconds": reserve,
        },
    )
    return None


def verify_repair(
    request: RepairRequest,
    proposal: RepairProposal,
    verification_request: VerificationRequest,
    *,
    workspace: VerificationWorkspace,
    journal: CampaignStore,
    fence_valid: Callable[[str], bool],
    binding: VerificationBinding | None = None,
    source_verification: SourceVerificationGate | None = None,
) -> RepairDispatchResult:
    """Invoke original reproduction and independent tests; authenticate in memory.

    This function only runs in the controller's immutable trusted release. There
    is no API taking a worker-supplied verification file. Isolation's verifier
    computes actual candidate content/scope and observes check outcomes itself.
    """
    deadline = time.monotonic() + INTERRUPT_AFTER_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS
    current_fence = binding.fence if binding else request.fence
    if request.grant is None or request.grant.expires_at <= time.time() or not fence_valid(current_fence):
        raise ValueError("verification requires a current grant and fence")
    if proposal.request_digest != request.digest():
        raise ValueError("proposal identity mismatch")
    if proposal.patch_digest != proposal_patch_digest(
        workspace.base, workspace.candidate
    ):
        raise ValueError("proposal patch digest mismatch")
    original, checks = verification_request.original, verification_request.checks
    if (
        check_manifest_digest(original, checks, verification_request.equivalence_checks)
        != request.verification_manifest_digest
    ):
        raise ValueError("independent checks differ from locked manifest")
    expected = VerificationRequest(
        request_digest=request.digest(),
        blocker_digest=request.blocker.fingerprint(),
        source_digest=request.blocker.source_digest,
        candidate_digest=proposal.tree_digest,
        environment_digest=request.blocker.environment_digest,
        verifier_release=verification_request.verifier_release,
        authority_digest=request.grant.digest(),
        fencing_token=current_fence,
        allowed_paths=request.allowed_paths,
        original=original,
        checks=checks,
        failure_returncode=request.failure_returncode,
        failure_stdout_sha256=request.failure_stdout_sha256,
        failure_stderr_sha256=request.failure_stderr_sha256,
        semantics_preserving_paths=request.semantics_preserving_paths,
        equivalence_checks=verification_request.equivalence_checks,
        timeout_seconds=request.grant.interrupt_seconds,
    )
    if verification_request != expected:
        raise ValueError("independent verifier identity mismatch")
    source_evidence = source_verification.read(workspace) if source_verification else None
    if source_evidence is None:
        waiting = RepairDispatchResult(
            status="waiting_verification", request_digest=request.digest(),
            reason="source_verification_pending" if source_verification else "source_verification_not_configured",
            proposal=proposal,
        )
        artifact = journal.write_artifact("repair_source_verification_wait", waiting)
        journal.append_event(
            "repair_source_verification_wait", artifact_sha256=artifact.stem,
            idempotency_key="source-verification-wait:" + artifact.stem,
            detail={"request_digest": request.digest(), "proposal_digest": proposal.digest(),
                    "unmet_predicate": "complete_current_source_verification",
                    "wake_source": "source_verification_completed",
                    "required_capability": "isolated_source_verifier"},
        )
        return waiting
    pending = _reserve_verification(request, verification_request, journal, binding)
    if pending:
        return RepairDispatchResult(
            status="waiting_diagnosis",
            request_digest=request.digest(),
            reason=pending,
            proposal=proposal,
        )
    evidence = verify_candidate(
        verification_request,
        workspace.base,
        workspace.candidate,
        runtime_roots=workspace.runtime_roots,
        deadline=deadline,
    )
    assert source_verification is not None
    if source_verification.read(workspace) != source_evidence:
        raise ValueError("source_verification_changed_during_independent_checks")
    accepted = source_verification.authorize(source_evidence, evidence)
    artifact = journal.write_artifact("repair_verification", {
        "schema_version": "source_and_predicate_verification/v1",
        "independent_verification": asdict(evidence),
        "source_verification": source_evidence,
        "source_snapshot_digest": manifest_digest(tree_manifest(workspace.base)),
        "candidate_snapshot_digest": manifest_digest(tree_manifest(workspace.candidate)),
        "source_release_authorized": accepted,
    })
    receipt = RepairVerification(
        request_digest=request.digest(),
        proposal_digest=proposal.digest(),
        verifier_release=verification_request.verifier_release,
        manifest_digest=request.verification_manifest_digest,
        source_digest=request.blocker.source_digest,
        environment_digest=request.blocker.environment_digest,
        input_digest=request.blocker.input_digest,
        grant_id=request.grant.grant_id,
        fence=current_fence,
        authority_binding_digest=binding.digest() if binding else None,
        original_failure_reproduced=bool(evidence.observations) and evidence.reason
        != "original_failure_not_reproduced",
        original_predicate_restored=evidence.accepted,
        required_checks_passed=accepted,
        protected_surfaces_unchanged=evidence.accepted,
        release_digest=proposal.tree_digest,
        evidence_digest=artifact.stem,
    )
    return accept_verification(
        request,
        proposal,
        receipt,
        journal=journal,
        authenticated=lambda supplied: supplied is receipt,
        fence_valid=fence_valid,
        binding=binding,
    )
