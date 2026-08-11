"""KERN-11 / SLM-539: fixture-level proof-trace refinement checker.

Checks sealed INTEG-01 ``CanonicalProofTraceV1`` compact events against
abstract decode/search owners (KERN-01 Judgment, KERN-02 CompleteDomain,
KERN-06 EventTrace cost). The runtime is untrusted — only its compact trace
is admitted.

Scope is an explicit bounded fixture subset. Successful checks bind into
``formal_authority/v2``; this module never claims full PyTorch semantics or
physical latency equivalence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.formal.authority import (
    CheckerResultRefV1,
    FormalAuthorityV2,
    from_formal_object_v1,
)
from slm_training.formal.event_trace import (
    DECODE_UNIT_WORK_MODEL,
    DecodeUnitWorkCounts,
)
from slm_training.formal.judgment import (
    ClassificationInputsV1,
    JudgmentOutcome,
    classify,
)
from slm_training.formal.objects import export_lean_claim
from slm_training.formal.proof_trace import (
    CANONICAL_FIELDS,
    PROOF_TRACE_SCHEMA,
    CanonicalProofTraceV1,
    ProofTraceEventField,
    ProofTraceEventV1,
    frozen_fixture_rows,
    project_runtime_evidence,
    proof_trace_integrity_ok,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.models.decode_stats import (
    DecodeIdentityV1,
    DecodeStats,
    build_mechanism_activation,
)
from slm_training.runtime.telemetry.replay_bundle import build_replay_bundle

SCHEMA = "proof_trace_refinement/v1"
CERTIFICATE_SCHEMA = "proof_trace_refinement_certificate/v1"
ISSUE = "SLM-539"
KERN = "KERN-11"

THEOREM_FIXTURE_REFINES = (
    "ProofTraceRefinement.fixture_bundle_stats_mechanism_refines"
)
THEOREM_ILLEGAL_COMMIT = "ProofTraceRefinement.injected_illegal_commit_fails"
THEOREM_SCOPE = "ProofTraceRefinement.scope_is_fixture_subset"

SUPPORTED_FIXTURE_IDS: frozenset[str] = frozenset({"bundle_stats_mechanism"})

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "formal"
    / "trace_refinement_fixtures.v1.json"
)


class RefinementRejectReason(str, Enum):
    INTEGRITY = "integrity"
    UNSUPPORTED_FIXTURE = "unsupported_fixture"
    REORDERED_EVENTS = "reordered_events"
    MISMATCHED_IDENTITY = "mismatched_identity"
    FABRICATED_COMPLETENESS = "fabricated_completeness"
    FABRICATED_REFUTATION = "fabricated_refutation"
    ILLEGAL_COMMIT = "illegal_commit"
    STALE_STATE_DOMAIN = "stale_state_domain"
    OMITTED_MODEL_FORWARDS = "omitted_model_forwards"
    COST_MISMATCH = "cost_mismatch"
    ASSUMPTION_VIOLATION = "assumption_violation"


@dataclass(frozen=True)
class FixtureAssumptionsV1:
    """Explicit assumptions for the supported fixture subset."""

    complete_requires_digest: bool = True
    legacy_coverage_never_authorizes: bool = True
    dense_ordinals: bool = True
    identity_must_match: bool = True
    cost_forwards_require_invocation: bool = True
    unknown_never_removes: bool = True
    claims_full_pytorch_semantics: bool = False
    claims_physical_latency_equivalence: bool = False
    fixture_subset_scoped: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete_requires_digest": self.complete_requires_digest,
            "legacy_coverage_never_authorizes": self.legacy_coverage_never_authorizes,
            "dense_ordinals": self.dense_ordinals,
            "identity_must_match": self.identity_must_match,
            "cost_forwards_require_invocation": self.cost_forwards_require_invocation,
            "unknown_never_removes": self.unknown_never_removes,
            "claims_full_pytorch_semantics": self.claims_full_pytorch_semantics,
            "claims_physical_latency_equivalence": (
                self.claims_physical_latency_equivalence
            ),
            "fixture_subset_scoped": self.fixture_subset_scoped,
        }


DEFAULT_ASSUMPTIONS = FixtureAssumptionsV1()


@dataclass(frozen=True)
class AbstractStateV1:
    identity_digest: str
    current_state_digest: str = ""
    domain_digest: str = ""
    domain_status: str = ""
    domain_size: int = 0
    declared_forwards: int = 0
    declared_cost: int = 0
    saw_ranker_invocation: bool = False
    saw_observed_work: bool = False


@dataclass(frozen=True)
class RefinementCheckResultV1:
    schema: str
    accepted: bool
    fixture_id: str | None
    reject_reason: str | None
    expected_cost: int | None
    expected_forwards: int | None
    assumptions: FixtureAssumptionsV1
    theorem_ref: str
    trace_hash: str | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "fixture_id": self.fixture_id,
            "reject_reason": self.reject_reason,
            "expected_cost": self.expected_cost,
            "expected_forwards": self.expected_forwards,
            "assumptions": self.assumptions.to_dict(),
            "theorem_ref": self.theorem_ref,
            "trace_hash": self.trace_hash,
            "detail": self.detail,
            "issue": ISSUE,
            "key": KERN,
            "claims_full_pytorch_semantics": False,
            "claims_physical_latency_equivalence": False,
        }


@dataclass(frozen=True)
class RefinementCertificateV1:
    """Successful fixture-scoped refinement binding."""

    schema: str
    fixture_id: str
    theorem_ref: str
    trace_hash: str
    check: RefinementCheckResultV1
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fixture_id": self.fixture_id,
            "theorem_ref": self.theorem_ref,
            "trace_hash": self.trace_hash,
            "check": self.check.to_dict(),
            "content_sha256": self.content_sha256,
            "issue": ISSUE,
            "key": KERN,
            "claims_abstract_refinement": True,
            "fixture_subset_scoped": True,
            "claims_full_pytorch_semantics": False,
            "claims_physical_latency_equivalence": False,
        }


def _envelope_identity(trace: CanonicalProofTraceV1) -> str:
    identity = trace.identity or {}
    request_id = identity.get("request_id")
    if request_id:
        return str(request_id)
    return content_sha(identity) if identity else ""


def _event_identity(event: ProofTraceEventV1, envelope: str) -> str:
    refs = event.refs
    if "request_id" in refs:
        return str(refs["request_id"])
    if "identity_digest" in refs:
        return str(refs["identity_digest"])
    return envelope if event.field == ProofTraceEventField.STATE_INPUT_IDENTITY.value else ""


def _judgment_tag(refs: Mapping[str, Any]) -> str:
    raw = refs.get("judgment") or refs.get("outcome")
    if raw is None:
        kind = str(refs.get("kind") or "")
        if kind in {"certified_deduction", "support_result", "solver_terminal"}:
            return JudgmentOutcome.WITNESSED.value
        if kind == "nogood":
            return JudgmentOutcome.REFUTED.value
        return "none"
    return str(raw)


def _ordinals_ok(events: Sequence[ProofTraceEventV1]) -> bool:
    return all(event.ordinal == index for index, event in enumerate(events))


def _reject(
    *,
    reason: RefinementRejectReason,
    assumptions: FixtureAssumptionsV1,
    fixture_id: str | None,
    trace_hash: str | None,
    detail: str,
    expected_cost: int | None = None,
    expected_forwards: int | None = None,
) -> RefinementCheckResultV1:
    return RefinementCheckResultV1(
        schema=SCHEMA,
        accepted=False,
        fixture_id=fixture_id,
        reject_reason=reason.value,
        expected_cost=expected_cost,
        expected_forwards=expected_forwards,
        assumptions=assumptions,
        theorem_ref=THEOREM_FIXTURE_REFINES,
        trace_hash=trace_hash,
        detail=detail,
    )


def check_trace_refinement(
    trace: CanonicalProofTraceV1,
    *,
    fixture_id: str | None = None,
    assumptions: FixtureAssumptionsV1 = DEFAULT_ASSUMPTIONS,
    expected_cost: int | None = None,
    expected_forwards: int | None = None,
) -> RefinementCheckResultV1:
    """Replay a sealed canonical proof trace under abstract semantics."""

    if not proof_trace_integrity_ok(trace):
        return _reject(
            reason=RefinementRejectReason.INTEGRITY,
            assumptions=assumptions,
            fixture_id=fixture_id,
            trace_hash=getattr(trace, "trace_hash", None),
            detail="trace failed integrity / hash seal",
        )
    if fixture_id is not None and fixture_id not in SUPPORTED_FIXTURE_IDS:
        return _reject(
            reason=RefinementRejectReason.UNSUPPORTED_FIXTURE,
            assumptions=assumptions,
            fixture_id=fixture_id,
            trace_hash=trace.trace_hash,
            detail=f"fixture {fixture_id!r} outside supported subset",
        )
    if assumptions.dense_ordinals and not _ordinals_ok(trace.events):
        return _reject(
            reason=RefinementRejectReason.REORDERED_EVENTS,
            assumptions=assumptions,
            fixture_id=fixture_id,
            trace_hash=trace.trace_hash,
            detail="event ordinals are not dense 0..n-1",
        )

    envelope = _envelope_identity(trace)
    state = AbstractStateV1(identity_digest=envelope)

    for event in trace.events:
        if not event.observed:
            continue
        refs = event.refs
        event_id = _event_identity(event, envelope)
        if (
            assumptions.identity_must_match
            and event_id
            and event_id != envelope
        ):
            return _reject(
                reason=RefinementRejectReason.MISMATCHED_IDENTITY,
                assumptions=assumptions,
                fixture_id=fixture_id,
                trace_hash=trace.trace_hash,
                detail=f"identity digest mismatch at ordinal {event.ordinal}",
            )

        field = event.field
        if field == ProofTraceEventField.LEGAL_DOMAIN.value:
            status = str(refs.get("legal_domain_status") or "")
            digest = str(refs.get("legal_domain_digest") or "")
            size = int(refs.get("legal_domain_size") or 0)
            if (
                assumptions.complete_requires_digest
                and status == "complete"
                and not digest
            ):
                return _reject(
                    reason=RefinementRejectReason.FABRICATED_COMPLETENESS,
                    assumptions=assumptions,
                    fixture_id=fixture_id,
                    trace_hash=trace.trace_hash,
                    detail="complete domain without digest",
                )
            if (
                state.domain_digest
                and digest
                and digest != state.domain_digest
                and state.current_state_digest
                and str(refs.get("state_fingerprint") or "")
                in {"", state.current_state_digest}
            ):
                return _reject(
                    reason=RefinementRejectReason.STALE_STATE_DOMAIN,
                    assumptions=assumptions,
                    fixture_id=fixture_id,
                    trace_hash=trace.trace_hash,
                    detail="legal_domain digest changed under stale state",
                )
            state = replace(
                state,
                domain_digest=digest,
                domain_status=status,
                domain_size=size,
                current_state_digest=(
                    str(refs["state_fingerprint"])
                    if refs.get("state_fingerprint")
                    else state.current_state_digest
                ),
            )
        elif field == ProofTraceEventField.STATE_TRANSITION.value:
            state = replace(
                state,
                current_state_digest=str(refs.get("state_fingerprint") or ""),
            )
        elif field == ProofTraceEventField.CHOSEN_ACTION.value:
            if (
                state.domain_status == "complete"
                and state.domain_digest
                and state.domain_size == 0
            ):
                return _reject(
                    reason=RefinementRejectReason.ILLEGAL_COMMIT,
                    assumptions=assumptions,
                    fixture_id=fixture_id,
                    trace_hash=trace.trace_hash,
                    detail="chosen_action against empty complete domain",
                )
        elif field == ProofTraceEventField.VERIFIER_SOLVER_CERTIFICATE.value:
            removed = refs.get("removed") or []
            removed_count = (
                len(removed) if isinstance(removed, (list, tuple)) else int(removed or 0)
            )
            tag = _judgment_tag(refs)
            if (
                assumptions.unknown_never_removes
                and removed_count > 0
                and tag
                in {
                    JudgmentOutcome.UNKNOWN.value,
                    JudgmentOutcome.INVALID.value,
                    "none",
                }
            ):
                return _reject(
                    reason=RefinementRejectReason.FABRICATED_REFUTATION,
                    assumptions=assumptions,
                    fixture_id=fixture_id,
                    trace_hash=trace.trace_hash,
                    detail="removal under unknown/invalid judgment",
                )
        elif field == ProofTraceEventField.RANKER_MODEL_INVOCATION.value:
            state = replace(
                state,
                declared_forwards=int(refs.get("forwards_count") or 0),
                saw_ranker_invocation=True,
            )
        elif field == ProofTraceEventField.OBSERVED_WORK.value:
            counts = refs.get("counts") or {}
            neural = int(counts.get("neural_forward") or refs.get("neural_forward") or 0)
            cost = int(refs.get("abstract_cost") or 0)
            state = replace(
                state,
                declared_cost=cost,
                declared_forwards=max(state.declared_forwards, neural),
                saw_observed_work=True,
            )

    cost_projection = trace.cost_projection or {}
    projected_cost = cost_projection.get("abstract_cost")
    if expected_cost is None:
        expected_cost = (
            int(projected_cost)
            if projected_cost is not None
            else state.declared_cost
        )
    if expected_forwards is None:
        expected_forwards = state.declared_forwards

    if state.declared_cost != expected_cost:
        return _reject(
            reason=RefinementRejectReason.COST_MISMATCH,
            assumptions=assumptions,
            fixture_id=fixture_id,
            trace_hash=trace.trace_hash,
            detail=(
                f"declared cost {state.declared_cost} != expected {expected_cost}"
            ),
            expected_cost=expected_cost,
            expected_forwards=expected_forwards,
        )
    if state.declared_forwards != expected_forwards:
        return _reject(
            reason=RefinementRejectReason.COST_MISMATCH,
            assumptions=assumptions,
            fixture_id=fixture_id,
            trace_hash=trace.trace_hash,
            detail=(
                f"declared forwards {state.declared_forwards} != "
                f"expected {expected_forwards}"
            ),
            expected_cost=expected_cost,
            expected_forwards=expected_forwards,
        )
    if (
        assumptions.cost_forwards_require_invocation
        and expected_forwards > 0
        and not (state.saw_ranker_invocation and state.saw_observed_work)
    ):
        return _reject(
            reason=RefinementRejectReason.OMITTED_MODEL_FORWARDS,
            assumptions=assumptions,
            fixture_id=fixture_id,
            trace_hash=trace.trace_hash,
            detail="neural_forward cost without ranker_model_invocation",
            expected_cost=expected_cost,
            expected_forwards=expected_forwards,
        )

    return RefinementCheckResultV1(
        schema=SCHEMA,
        accepted=True,
        fixture_id=fixture_id,
        reject_reason=None,
        expected_cost=expected_cost,
        expected_forwards=expected_forwards,
        assumptions=assumptions,
        theorem_ref=THEOREM_FIXTURE_REFINES,
        trace_hash=trace.trace_hash,
        detail="accepted under fixture-subset assumptions",
    )


def emit_refinement_certificate(
    trace: CanonicalProofTraceV1,
    *,
    fixture_id: str,
    assumptions: FixtureAssumptionsV1 = DEFAULT_ASSUMPTIONS,
    expected_cost: int | None = None,
    expected_forwards: int | None = None,
) -> RefinementCertificateV1:
    """Seal a successful refinement check; fails closed on reject."""

    result = check_trace_refinement(
        trace,
        fixture_id=fixture_id,
        assumptions=assumptions,
        expected_cost=expected_cost,
        expected_forwards=expected_forwards,
    )
    if not result.accepted:
        raise ValueError(
            f"refinement rejected ({result.reject_reason}): {result.detail}"
        )
    payload = {
        "schema": CERTIFICATE_SCHEMA,
        "fixture_id": fixture_id,
        "theorem_ref": THEOREM_FIXTURE_REFINES,
        "trace_hash": trace.trace_hash,
        "check": result.to_dict(),
        "issue": ISSUE,
        "key": KERN,
    }
    digest = content_sha(payload)
    return RefinementCertificateV1(
        schema=CERTIFICATE_SCHEMA,
        fixture_id=fixture_id,
        theorem_ref=THEOREM_FIXTURE_REFINES,
        trace_hash=trace.trace_hash,
        check=result,
        content_sha256=digest,
    )


def bind_refinement_authority(
    certificate: RefinementCertificateV1,
) -> FormalAuthorityV2:
    """Bind successful refinement evidence into ``formal_authority/v2``."""

    obj = export_lean_claim(THEOREM_FIXTURE_REFINES)
    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="supported",
            has_witness_digest=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    return from_formal_object_v1(
        obj,
        judgment=judgment,
        checker_results=(
            CheckerResultRefV1(
                backend="proof_trace_refinement",
                status="pass",
                detail=f"fixture {certificate.fixture_id} refined",
                result_digest=certificate.content_sha256,
            ),
        ),
        certificate_digest=certificate.content_sha256,
        replay_digest=certificate.trace_hash,
        empirical_remainder_claim_ids=(
            "physical_latency",
            "full_pytorch_semantics",
        ),
        empirical_remainder_notes=(
            "KERN-11 is fixture/subset-scoped; physical latency and full "
            "PyTorch semantics remain empirical remainder."
        ),
    )



def reseal_trace(trace: CanonicalProofTraceV1) -> CanonicalProofTraceV1:
    """Recompute ``trace_hash`` after an adversarial mutation (test / injector aid).

    Production traces arrive pre-sealed from INTEG-01; resealing is only for
    proving the *semantic* checker rejects illegal shapes independent of hash
    integrity.
    """

    from slm_training.formal import proof_trace as _pt

    return _pt._seal(
        replace(
            trace,
            claims_abstract_refinement=False,
            kern11_deferred=True,
            trace_hash="",
        )
    )

# --------------------------------------------------------------------------- #
# Adversarial injectors (acceptance: invalid traces fail)
# --------------------------------------------------------------------------- #


def inject_illegal_commit(trace: CanonicalProofTraceV1) -> CanonicalProofTraceV1:
    """Force a chosen_action against an empty complete domain (unsealed)."""

    events: list[ProofTraceEventV1] = []
    for event in trace.events:
        if event.field == ProofTraceEventField.LEGAL_DOMAIN.value and event.observed:
            refs = dict(event.refs)
            refs["legal_domain_size"] = 0
            refs["legal_domain_status"] = "complete"
            events.append(replace(event, refs=refs))
        else:
            events.append(event)
    return replace(trace, events=tuple(events))


def inject_fabricated_completeness(
    trace: CanonicalProofTraceV1,
) -> CanonicalProofTraceV1:
    events: list[ProofTraceEventV1] = []
    for event in trace.events:
        if event.field == ProofTraceEventField.LEGAL_DOMAIN.value and event.observed:
            refs = dict(event.refs)
            refs["legal_domain_status"] = "complete"
            refs["legal_domain_digest"] = ""
            events.append(replace(event, refs=refs))
        else:
            events.append(event)
    return replace(trace, events=tuple(events))


def inject_fabricated_refutation(
    trace: CanonicalProofTraceV1,
) -> CanonicalProofTraceV1:
    events: list[ProofTraceEventV1] = []
    touched = False
    for event in trace.events:
        if (
            not touched
            and event.field == ProofTraceEventField.VERIFIER_SOLVER_CERTIFICATE.value
            and event.observed
        ):
            refs = dict(event.refs)
            refs["removed"] = [1]
            refs["judgment"] = JudgmentOutcome.UNKNOWN.value
            events.append(replace(event, refs=refs))
            touched = True
        else:
            events.append(event)
    if not touched:
        raise ValueError("no verifier_solver_certificate to fabricate")
    return replace(trace, events=tuple(events))


def inject_reordered_events(trace: CanonicalProofTraceV1) -> CanonicalProofTraceV1:
    """Swap the first two events while keeping their original ordinals."""

    if len(trace.events) < 2:
        raise ValueError("need at least two events")
    swapped = list(trace.events)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    return replace(trace, events=tuple(swapped))


def inject_omitted_model_forwards(
    trace: CanonicalProofTraceV1,
) -> CanonicalProofTraceV1:
    """Drop ranker_model_invocation while keeping neural_forward cost."""

    kept = [
        event
        for event in trace.events
        if event.field != ProofTraceEventField.RANKER_MODEL_INVOCATION.value
    ]
    renumbered = tuple(
        replace(event, ordinal=index) for index, event in enumerate(kept)
    )
    return replace(trace, events=renumbered)


def inject_mismatched_identity(
    trace: CanonicalProofTraceV1, *, stale: str = "forged-request"
) -> CanonicalProofTraceV1:
    events: list[ProofTraceEventV1] = []
    touched = False
    for event in trace.events:
        if (
            not touched
            and event.field == ProofTraceEventField.STATE_INPUT_IDENTITY.value
            and event.observed
        ):
            refs = dict(event.refs)
            refs["request_id"] = stale
            events.append(replace(event, refs=refs))
            touched = True
        else:
            events.append(event)
    if not touched:
        raise ValueError("no state_input_identity to mismatch")
    return replace(trace, events=tuple(events))


def inject_stale_state_domain(trace: CanonicalProofTraceV1) -> CanonicalProofTraceV1:
    """Rewrite legal_domain digest after a state transition, same fingerprint."""

    events = list(trace.events)
    state_fp = None
    for event in events:
        if (
            event.field == ProofTraceEventField.STATE_TRANSITION.value
            and event.observed
            and "state_fingerprint" in event.refs
        ):
            state_fp = str(event.refs["state_fingerprint"])
            break
    if state_fp is None:
        raise ValueError("no state_transition fingerprint")
    updated: list[ProofTraceEventV1] = []
    # Append a second legal_domain after state transition with a new digest.
    inserted = False
    for event in events:
        updated.append(event)
        if (
            not inserted
            and event.field == ProofTraceEventField.STATE_TRANSITION.value
            and event.observed
        ):
            updated.append(
                ProofTraceEventV1(
                    ordinal=event.ordinal + 1,
                    field=ProofTraceEventField.LEGAL_DOMAIN.value,
                    observed=True,
                    source_kind="injected.stale_domain",
                    refs={
                        "legal_domain_size": 4,
                        "legal_domain_status": "complete",
                        "legal_domain_digest": "e" * 64,
                        "state_fingerprint": state_fp,
                    },
                )
            )
            inserted = True
    # Renumber dense ordinals so only the stale-domain rule fires.
    renumbered = tuple(
        replace(event, ordinal=index) for index, event in enumerate(updated)
    )
    return replace(trace, events=renumbered)


def build_supported_fixture_trace() -> CanonicalProofTraceV1:
    """Build the frozen ``bundle_stats_mechanism`` runtime trace."""

    identity = DecodeIdentityV1(
        contract_version=2,
        pack_id="openui",
        tokenizer_id="dsl-native/v5",
        artifact_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        evaluator_version="eval/v1",
        evaluator_hash="c" * 64,
        code_commit="deadbeef",
    )
    bundle = build_replay_bundle(
        request_id="integ01-fixture",
        identity=identity,
        seed=7,
        decoder_choices=(
            {
                "kind": "solver_state",
                "state_fingerprint": "fp0",
                "decision_level": 0,
                "domain": {},
            },
            {"kind": "decoder_choice", "action": "select", "token_id": 3},
            {
                "kind": "certified_deduction",
                "before_fingerprint": "fp0",
                "after_fingerprint": "fp1",
                "removed": [],
                "certificate_id": "cert-1",
            },
        ),
        legal_domain_size=4,
        legal_domain_status="complete",
        legal_domain_digest="d" * 64,
        artifact_state={"cache": "warm"},
    )
    stats = DecodeStats(
        tokens_emitted=2,
        dfa_sync_count=3,
        solver_expanded_nodes=5,
        solver_verifier_calls=7,
        forwards_count=11,
    )
    activation = build_mechanism_activation(
        mechanism_id="goal_support",
        eligible=True,
        invoked=True,
        forwards=1,
        verifier_calls=1,
        evidence_class="fixture",
    )
    return project_runtime_evidence(
        bundle=bundle, stats=stats, activations=(activation,)
    )


def frozen_refinement_fixture_rows() -> list[dict[str, Any]]:
    """Deterministic rows for ``trace_refinement_fixtures.v1.json``."""

    trace = build_supported_fixture_trace()
    check = check_trace_refinement(
        trace,
        fixture_id="bundle_stats_mechanism",
        expected_cost=28,
        expected_forwards=11,
    )
    cert = emit_refinement_certificate(
        trace,
        fixture_id="bundle_stats_mechanism",
        expected_cost=28,
        expected_forwards=11,
    )
    counts = DecodeUnitWorkCounts(
        tokenize=2,
        grammar_transition=3,
        solver_expansion=5,
        certificate_check=7,
        neural_forward=11,
    )
    integ_rows = frozen_fixture_rows()
    return [
        {
            "id": "bundle_stats_mechanism",
            "supported": True,
            "trace_hash": trace.trace_hash,
            "integ01_row_id": integ_rows[0]["id"],
            "accepted": check.accepted,
            "expected_cost": 28,
            "expected_forwards": 11,
            "decode_unit_work_cost": counts.unit_work_cost(),
            "machine_model": DECODE_UNIT_WORK_MODEL.id,
            "theorem_ref": THEOREM_FIXTURE_REFINES,
            "certificate_sha256": cert.content_sha256,
            "assumptions": DEFAULT_ASSUMPTIONS.to_dict(),
            "canonical_fields": list(CANONICAL_FIELDS),
            "proof_trace_schema": PROOF_TRACE_SCHEMA,
        }
    ]


def load_fixture_payload() -> dict[str, Any]:
    raw = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(
            f"fixture schema mismatch: got {raw.get('schema')!r}, expected {SCHEMA!r}"
        )
    return raw


__all__ = [
    "CERTIFICATE_SCHEMA",
    "DEFAULT_ASSUMPTIONS",
    "ISSUE",
    "KERN",
    "SCHEMA",
    "SUPPORTED_FIXTURE_IDS",
    "THEOREM_FIXTURE_REFINES",
    "THEOREM_ILLEGAL_COMMIT",
    "THEOREM_SCOPE",
    "AbstractStateV1",
    "FixtureAssumptionsV1",
    "RefinementCertificateV1",
    "RefinementCheckResultV1",
    "RefinementRejectReason",
    "bind_refinement_authority",
    "build_supported_fixture_trace",
    "check_trace_refinement",
    "emit_refinement_certificate",
    "frozen_refinement_fixture_rows",
    "inject_fabricated_completeness",
    "inject_fabricated_refutation",
    "inject_illegal_commit",
    "inject_mismatched_identity",
    "inject_omitted_model_forwards",
    "inject_reordered_events",
    "inject_stale_state_domain",
    "load_fixture_payload",
    "reseal_trace",
]
