"""INTEG-01 / SLM-528: canonical proof-trace projection from existing evidence.

Projects ``ReplayBundleV1``, ``DecodeStats`` / ``DecodeStatsRecordV1``,
``MechanismActivationV1``, and ``VerifierWitnessV1`` into one ordered abstract
trace that reuses KERN-06 ``Event`` / cost vocabulary for observed work.
Does **not** introduce a second runtime recorder and does **not** claim
abstract refinement (KERN-11 / SLM-539 owns that proof separately).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from slm_training.evals.semantic_failure import VerifierWitnessV1
from slm_training.formal.event_trace import (
    ProjectedDecodeTraceV1,
    build_event_trace_evidence,
    project_decode_stats,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.models.decode_stats import (
    DecodeStats,
    DecodeStatsRecordV1,
    MechanismActivationV1,
)
from slm_training.runtime.telemetry.replay_bundle import (
    ReplayBundleV1,
    replay_bundle_integrity_ok,
)

PROOF_TRACE_SCHEMA = "canonical_proof_trace/v1"
PROOF_TRACE_ISSUE = "SLM-528"
PROOF_TRACE_KEY = "INTEG-01"

# Minimal canonical field vocabulary (issue contract).
CANONICAL_FIELDS: tuple[str, ...] = (
    "state_input_identity",
    "legal_domain",
    "ranker_model_invocation",
    "chosen_action",
    "verifier_solver_certificate",
    "cache_mechanism",
    "state_transition",
    "observed_work",
)


class ProofTraceEventField(str, Enum):
    STATE_INPUT_IDENTITY = "state_input_identity"
    LEGAL_DOMAIN = "legal_domain"
    RANKER_MODEL_INVOCATION = "ranker_model_invocation"
    CHOSEN_ACTION = "chosen_action"
    VERIFIER_SOLVER_CERTIFICATE = "verifier_solver_certificate"
    CACHE_MECHANISM = "cache_mechanism"
    STATE_TRANSITION = "state_transition"
    OBSERVED_WORK = "observed_work"


# Owner decoder_choice / solver kinds → canonical field (projection only).
_SOURCE_KIND_TO_FIELD: dict[str, ProofTraceEventField] = {
    "solver_state": ProofTraceEventField.STATE_TRANSITION,
    "decision": ProofTraceEventField.CHOSEN_ACTION,
    "decoder_choice": ProofTraceEventField.CHOSEN_ACTION,
    "support_result": ProofTraceEventField.VERIFIER_SOLVER_CERTIFICATE,
    "certified_deduction": ProofTraceEventField.VERIFIER_SOLVER_CERTIFICATE,
    "nogood": ProofTraceEventField.VERIFIER_SOLVER_CERTIFICATE,
    "backtrack": ProofTraceEventField.STATE_TRANSITION,
    "solver_terminal": ProofTraceEventField.VERIFIER_SOLVER_CERTIFICATE,
}


def _omit_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop None values — never fabricate placeholders for unobserved fields."""

    return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class ProofTraceEventV1:
    """One observed (or explicitly unobserved) step in the canonical trace."""

    ordinal: int
    field: str
    observed: bool
    source_kind: str | None
    refs: dict[str, Any]

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if self.field not in CANONICAL_FIELDS:
            raise ValueError(f"unknown canonical field: {self.field!r}")
        if not self.observed and self.refs:
            raise ValueError("unobserved events must not carry fabricated refs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "field": self.field,
            "observed": self.observed,
            "source_kind": self.source_kind,
            "refs": dict(self.refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofTraceEventV1:
        return cls(
            ordinal=int(payload["ordinal"]),
            field=str(payload["field"]),
            observed=bool(payload["observed"]),
            source_kind=payload.get("source_kind"),
            refs=dict(payload.get("refs") or {}),
        )


@dataclass(frozen=True)
class CanonicalProofTraceV1:
    """Deterministic projection envelope — hash-bound, refinement-claim-free."""

    schema_version: str
    identity: dict[str, Any]
    events: tuple[ProofTraceEventV1, ...]
    cost_projection: dict[str, Any] | None
    source_digests: dict[str, Any]
    unobserved_fields: tuple[str, ...]
    claims_abstract_refinement: bool
    kern11_deferred: bool
    trace_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": dict(self.identity),
            "events": [event.to_dict() for event in self.events],
            "cost_projection": (
                dict(self.cost_projection) if self.cost_projection is not None else None
            ),
            "source_digests": dict(self.source_digests),
            "unobserved_fields": list(self.unobserved_fields),
            "claims_abstract_refinement": self.claims_abstract_refinement,
            "kern11_deferred": self.kern11_deferred,
            "trace_hash": self.trace_hash,
            "issue": PROOF_TRACE_ISSUE,
            "key": PROOF_TRACE_KEY,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalProofTraceV1:
        if payload.get("schema_version") != PROOF_TRACE_SCHEMA:
            raise ValueError(
                f"unsupported proof trace schema: {payload.get('schema_version')!r}"
            )
        if payload.get("claims_abstract_refinement") is True:
            raise ValueError(
                "canonical proof traces cannot claim abstract refinement "
                "(KERN-11 / SLM-539 proves that separately)"
            )
        trace = cls(
            schema_version=str(payload["schema_version"]),
            identity=dict(payload.get("identity") or {}),
            events=tuple(
                ProofTraceEventV1.from_dict(item) for item in payload.get("events") or ()
            ),
            cost_projection=(
                dict(payload["cost_projection"])
                if payload.get("cost_projection") is not None
                else None
            ),
            source_digests=dict(payload.get("source_digests") or {}),
            unobserved_fields=tuple(payload.get("unobserved_fields") or ()),
            claims_abstract_refinement=bool(
                payload.get("claims_abstract_refinement", False)
            ),
            kern11_deferred=bool(payload.get("kern11_deferred", True)),
            trace_hash=str(payload.get("trace_hash") or ""),
        )
        if _trace_digest(trace) != trace.trace_hash:
            raise ValueError("proof trace hash mismatch (tampered or corrupted payload)")
        return trace


def _trace_digest(trace: CanonicalProofTraceV1) -> str:
    payload = trace.to_dict()
    payload.pop("trace_hash", None)
    return content_sha(payload)


def proof_trace_integrity_ok(trace: CanonicalProofTraceV1) -> bool:
    return _trace_digest(trace) == trace.trace_hash


def _seal(trace: CanonicalProofTraceV1) -> CanonicalProofTraceV1:
    if trace.claims_abstract_refinement:
        raise ValueError("INTEG-01 traces must not claim abstract refinement")
    return replace(trace, trace_hash=_trace_digest(trace))


def _identity_from_bundle(bundle: ReplayBundleV1) -> dict[str, Any]:
    ident = dict(bundle.identity)
    return _omit_none(
        {
            "request_id": bundle.request_id or None,
            "seed": bundle.seed,
            "contract_version": ident.get("contract_version"),
            "pack_id": ident.get("pack_id"),
            "tokenizer_id": ident.get("tokenizer_id"),
            "artifact_sha256": ident.get("artifact_sha256"),
            "checkpoint_sha256": ident.get("checkpoint_sha256"),
            "evaluator_version": ident.get("evaluator_version"),
            "evaluator_hash": ident.get("evaluator_hash"),
            "code_commit": ident.get("code_commit"),
        }
    )


def _choice_refs(choice: Mapping[str, Any], field: ProofTraceEventField) -> dict[str, Any]:
    """Select observed keys only — leave the rest out rather than inventing."""

    kind = str(choice.get("kind") or "")
    if field is ProofTraceEventField.STATE_TRANSITION:
        return _omit_none(
            {
                "state_fingerprint": choice.get("state_fingerprint"),
                "before_fingerprint": choice.get("before_fingerprint"),
                "after_fingerprint": choice.get("after_fingerprint"),
                "decision_level": choice.get("decision_level"),
                "problem_id": choice.get("problem_id"),
                "pack_id": choice.get("pack_id"),
                "trace_truncated": choice.get("trace_truncated"),
            }
        )
    if field is ProofTraceEventField.CHOSEN_ACTION:
        return _omit_none(
            {
                "action": choice.get("action"),
                "token": choice.get("token"),
                "token_id": choice.get("token_id"),
                "selected": choice.get("selected"),
                "hole": choice.get("hole"),
                "value": choice.get("value"),
                "decision_level": choice.get("decision_level"),
            }
        )
    if field is ProofTraceEventField.VERIFIER_SOLVER_CERTIFICATE:
        report = choice.get("verifier_report")
        report_name = None
        if isinstance(report, Mapping):
            report_name = report.get("name") or report.get("verifier")
        return _omit_none(
            {
                "certificate_id": choice.get("certificate_id"),
                "certificate_mode": choice.get("certificate_mode"),
                "support": choice.get("support"),
                "status": choice.get("status"),
                "verdict": choice.get("verdict"),
                "verifier": report_name or choice.get("verifier"),
                "before_fingerprint": choice.get("before_fingerprint"),
                "after_fingerprint": choice.get("after_fingerprint"),
                "removed": choice.get("removed"),
                "kind": kind or None,
            }
        )
    return _omit_none({key: choice.get(key) for key in ("kind",) if key in choice})


def _events_from_bundle(bundle: ReplayBundleV1) -> list[ProofTraceEventV1]:
    events: list[ProofTraceEventV1] = []
    ordinal = 0

    identity_refs = _identity_from_bundle(bundle)
    events.append(
        ProofTraceEventV1(
            ordinal=ordinal,
            field=ProofTraceEventField.STATE_INPUT_IDENTITY.value,
            observed=bool(identity_refs),
            source_kind="replay_bundle.identity",
            refs=identity_refs if identity_refs else {},
        )
    )
    ordinal += 1

    legal_refs = _omit_none(
        {
            "legal_domain_size": bundle.legal_domain_size,
            "legal_domain_status": bundle.legal_domain_status,
            "legal_domain_digest": bundle.legal_domain_digest,
        }
    )
    # status "unknown" with no size/digest is still an explicit observation of unknown.
    legal_observed = bool(
        bundle.legal_domain_size is not None
        or bundle.legal_domain_digest is not None
        or bundle.legal_domain_status
    )
    events.append(
        ProofTraceEventV1(
            ordinal=ordinal,
            field=ProofTraceEventField.LEGAL_DOMAIN.value,
            observed=legal_observed,
            source_kind="replay_bundle.legal_domain",
            refs=legal_refs if legal_observed else {},
        )
    )
    ordinal += 1

    for choice in bundle.decoder_choices:
        kind = str(choice.get("kind") or "")
        field = _SOURCE_KIND_TO_FIELD.get(kind)
        if field is None:
            # Preserve unknown kinds as unobserved placeholders under cache_mechanism
            # only when the payload itself is empty; otherwise attach as residual refs
            # under cache_mechanism without inventing a semantic field.
            events.append(
                ProofTraceEventV1(
                    ordinal=ordinal,
                    field=ProofTraceEventField.CACHE_MECHANISM.value,
                    observed=True,
                    source_kind=kind or "unknown_decoder_choice",
                    refs=_omit_none({"residual_kind": kind or None, "keys": sorted(choice)}),
                )
            )
            ordinal += 1
            continue
        refs = _choice_refs(choice, field)
        events.append(
            ProofTraceEventV1(
                ordinal=ordinal,
                field=field.value,
                observed=True,
                source_kind=kind,
                refs=refs,
            )
        )
        ordinal += 1

    cache_payload = dict(bundle.artifact_state or {})
    if cache_payload:
        events.append(
            ProofTraceEventV1(
                ordinal=ordinal,
                field=ProofTraceEventField.CACHE_MECHANISM.value,
                observed=True,
                source_kind="replay_bundle.artifact_state",
                refs=_omit_none({"artifact_state": cache_payload}),
            )
        )
    return events


def _append_work_events(
    events: list[ProofTraceEventV1],
    *,
    stats: DecodeStats | Mapping[str, Any] | DecodeStatsRecordV1 | None,
) -> ProjectedDecodeTraceV1 | None:
    if stats is None:
        return None
    if isinstance(stats, DecodeStatsRecordV1):
        projection = project_decode_stats(stats.stats, record=stats)
    else:
        projection = project_decode_stats(stats)
    evidence = build_event_trace_evidence(
        stats if not isinstance(stats, DecodeStatsRecordV1) else stats
    )
    ordinal = len(events)
    forwards = projection.counts.neural_forward
    events.append(
        ProofTraceEventV1(
            ordinal=ordinal,
            field=ProofTraceEventField.RANKER_MODEL_INVOCATION.value,
            observed=True,
            source_kind="decode_stats.forwards_count",
            refs={"forwards_count": forwards, "invoked": forwards > 0},
        )
    )
    events.append(
        ProofTraceEventV1(
            ordinal=ordinal + 1,
            field=ProofTraceEventField.OBSERVED_WORK.value,
            observed=True,
            source_kind="event_trace.project_decode_stats",
            refs={
                "counts": projection.counts.to_dict(),
                "abstract_cost": projection.abstract_cost,
                "machine_model": f"{projection.model.id}.v{projection.model.version}",
                "evidence_sha256": evidence.certificate_sha256(),
                "unobserved_event_kinds": list(projection.unobserved_event_kinds),
            },
        )
    )
    return projection


def _append_mechanism_events(
    events: list[ProofTraceEventV1],
    activations: Sequence[MechanismActivationV1],
) -> list[str]:
    digests: list[str] = []
    for activation in activations:
        digests.append(activation.activation_hash)
        events.append(
            ProofTraceEventV1(
                ordinal=len(events),
                field=ProofTraceEventField.CACHE_MECHANISM.value,
                observed=True,
                source_kind="mechanism_activation",
                refs=_omit_none(
                    {
                        "mechanism_id": activation.mechanism_id,
                        "eligible": activation.eligible,
                        "invoked": activation.invoked,
                        "abstained": activation.abstained,
                        "no_op": activation.no_op,
                        "changed_top_choice": activation.changed_top_choice,
                        "changed_final_program": activation.changed_final_program,
                        "forwards": activation.forwards,
                        "verified_states": activation.verified_states,
                        "verifier_calls": activation.verifier_calls,
                        "evidence_class": activation.evidence_class,
                        "activation_hash": activation.activation_hash,
                        "failure_reason": activation.failure_reason,
                    }
                ),
            )
        )
    return digests


def _append_witness_events(
    events: list[ProofTraceEventV1],
    witnesses: Sequence[VerifierWitnessV1],
) -> list[str]:
    digests: list[str] = []
    for witness in witnesses:
        digests.append(witness.witness_digest)
        events.append(
            ProofTraceEventV1(
                ordinal=len(events),
                field=ProofTraceEventField.VERIFIER_SOLVER_CERTIFICATE.value,
                observed=True,
                source_kind="verifier_witness",
                refs=_omit_none(
                    {
                        "witness_digest": witness.witness_digest,
                        "source_trace_fingerprint": witness.source_trace_fingerprint,
                        "evaluator_version": witness.evaluator_version,
                        "evaluator_hash": witness.evaluator_hash,
                        "first_failed_gate": witness.first_failed_gate,
                        "localization_count": len(witness.localizations),
                        "unresolved_count": len(witness.unresolved_localizations),
                    }
                ),
            )
        )
    return digests


def _unobserved(events: Sequence[ProofTraceEventV1]) -> tuple[str, ...]:
    seen = {event.field for event in events if event.observed}
    return tuple(field for field in CANONICAL_FIELDS if field not in seen)


def project_runtime_evidence(
    *,
    bundle: ReplayBundleV1 | None = None,
    stats: DecodeStats | Mapping[str, Any] | DecodeStatsRecordV1 | None = None,
    activations: Sequence[MechanismActivationV1] = (),
    witnesses: Sequence[VerifierWitnessV1] = (),
) -> CanonicalProofTraceV1:
    """Deterministically project existing owners into a canonical proof trace.

    Absent owners leave their canonical fields in ``unobserved_fields`` rather
    than inventing events. Always seals with ``claims_abstract_refinement=False``.
    """

    if bundle is None and stats is None and not activations and not witnesses:
        raise ValueError("at least one evidence owner is required")

    events: list[ProofTraceEventV1] = []
    identity: dict[str, Any] = {}
    source_digests: dict[str, Any] = {
        "replay_bundle_hash": None,
        "decode_record_digest": None,
        "mechanism_activation_hashes": [],
        "verifier_witness_digests": [],
        "cost_projection_sha256": None,
    }

    if bundle is not None:
        if not replay_bundle_integrity_ok(bundle):
            raise ValueError("replay bundle failed integrity check")
        identity = _identity_from_bundle(bundle)
        events.extend(_events_from_bundle(bundle))
        source_digests["replay_bundle_hash"] = bundle.bundle_hash

    projection = _append_work_events(events, stats=stats)
    if projection is not None:
        source_digests["cost_projection_sha256"] = projection.certificate_sha256()
        if isinstance(stats, DecodeStatsRecordV1):
            source_digests["decode_record_digest"] = stats.record_digest
            # Prefer record identity when bundle identity was empty.
            if not identity:
                identity = _omit_none(dict(stats.identity))

    source_digests["mechanism_activation_hashes"] = _append_mechanism_events(
        events, activations
    )
    source_digests["verifier_witness_digests"] = _append_witness_events(events, witnesses)

    cost_payload = projection.to_dict() if projection is not None else None
    sealed = _seal(
        CanonicalProofTraceV1(
            schema_version=PROOF_TRACE_SCHEMA,
            identity=identity,
            events=tuple(events),
            cost_projection=cost_payload,
            source_digests=source_digests,
            unobserved_fields=_unobserved(events),
            claims_abstract_refinement=False,
            kern11_deferred=True,
            trace_hash="",
        )
    )
    return sealed


def project_replay_bundle(
    bundle: ReplayBundleV1,
    *,
    stats: DecodeStats | Mapping[str, Any] | DecodeStatsRecordV1 | None = None,
    activations: Sequence[MechanismActivationV1] = (),
    witnesses: Sequence[VerifierWitnessV1] = (),
) -> CanonicalProofTraceV1:
    """Adapter: ``ReplayBundleV1`` (+ optional co-evidence) → canonical trace."""

    return project_runtime_evidence(
        bundle=bundle, stats=stats, activations=activations, witnesses=witnesses
    )


def match_proof_trace_to_evidence(
    trace: CanonicalProofTraceV1,
    *,
    bundle: ReplayBundleV1 | None = None,
    stats: DecodeStats | Mapping[str, Any] | DecodeStatsRecordV1 | None = None,
    activations: Sequence[MechanismActivationV1] = (),
    witnesses: Sequence[VerifierWitnessV1] = (),
) -> bool:
    """Re-project current evidence and compare sealed hashes (replay match)."""

    if not proof_trace_integrity_ok(trace):
        return False
    if trace.claims_abstract_refinement:
        return False
    expected = project_runtime_evidence(
        bundle=bundle, stats=stats, activations=activations, witnesses=witnesses
    )
    return expected.trace_hash == trace.trace_hash and expected.to_dict() == trace.to_dict()


def replay_proof_trace(trace: CanonicalProofTraceV1) -> CanonicalProofTraceV1:
    """Offline replay: re-parse + re-hash; refuse forged refinement claims."""

    return CanonicalProofTraceV1.from_dict(trace.to_dict())


# --------------------------------------------------------------------------- #
# Mutation detectors (acceptance: reorder / omit / stale-state fail closed)
# --------------------------------------------------------------------------- #


def mutate_reorder_events(trace: CanonicalProofTraceV1) -> CanonicalProofTraceV1:
    """Adversarial: swap first two events' ordinals/order without resealing."""

    if len(trace.events) < 2:
        raise ValueError("need at least two events to reorder")
    swapped = list(trace.events)
    swapped[0], swapped[1] = (
        replace(swapped[1], ordinal=0),
        replace(swapped[0], ordinal=1),
    )
    # Keep the original (now-stale) hash so integrity fails.
    return replace(trace, events=tuple(swapped))


def mutate_omit_event(trace: CanonicalProofTraceV1, *, ordinal: int = 0) -> CanonicalProofTraceV1:
    """Adversarial: drop one event and renumber without resealing."""

    kept = [event for event in trace.events if event.ordinal != ordinal]
    if len(kept) == len(trace.events):
        raise ValueError(f"ordinal {ordinal} not present")
    renumbered = tuple(
        replace(event, ordinal=index) for index, event in enumerate(kept)
    )
    return replace(trace, events=renumbered)


def mutate_stale_state_fingerprint(
    trace: CanonicalProofTraceV1, *, stale: str = "stale-state-fingerprint"
) -> CanonicalProofTraceV1:
    """Adversarial: rewrite a state fingerprint ref; leave hash stale."""

    updated: list[ProofTraceEventV1] = []
    touched = False
    for event in trace.events:
        if (
            not touched
            and event.field == ProofTraceEventField.STATE_TRANSITION.value
            and event.observed
            and "state_fingerprint" in event.refs
        ):
            refs = dict(event.refs)
            refs["state_fingerprint"] = stale
            updated.append(replace(event, refs=refs))
            touched = True
        else:
            updated.append(event)
    if not touched:
        raise ValueError("no state_transition with state_fingerprint to stale")
    return replace(trace, events=tuple(updated))


def mutation_rejects_reorder(trace: CanonicalProofTraceV1) -> bool:
    mutated = mutate_reorder_events(trace)
    return not proof_trace_integrity_ok(mutated)


def mutation_rejects_omit(trace: CanonicalProofTraceV1) -> bool:
    mutated = mutate_omit_event(trace)
    return not proof_trace_integrity_ok(mutated)


def mutation_rejects_stale_state(trace: CanonicalProofTraceV1) -> bool:
    mutated = mutate_stale_state_fingerprint(trace)
    return not proof_trace_integrity_ok(mutated)


def frozen_fixture_rows() -> list[dict[str, Any]]:
    """Deterministic rows for resources/formal/proof_trace_fixtures.v1.json."""

    from slm_training.models.decode_stats import DecodeIdentityV1, build_mechanism_activation
    from slm_training.runtime.telemetry.replay_bundle import build_replay_bundle

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
    trace = project_runtime_evidence(
        bundle=bundle, stats=stats, activations=(activation,)
    )
    return [
        {
            "id": "bundle_stats_mechanism",
            "trace_hash": trace.trace_hash,
            "event_count": len(trace.events),
            "claims_abstract_refinement": trace.claims_abstract_refinement,
            "kern11_deferred": trace.kern11_deferred,
            "unobserved_fields": list(trace.unobserved_fields),
            "abstract_cost": (
                trace.cost_projection or {}).get("abstract_cost"),
            "machine_model": ((trace.cost_projection or {}).get("machine_model") or {}).get(
                "id"
            ),
        }
    ]


__all__ = [
    "CANONICAL_FIELDS",
    "PROOF_TRACE_ISSUE",
    "PROOF_TRACE_KEY",
    "PROOF_TRACE_SCHEMA",
    "CanonicalProofTraceV1",
    "ProofTraceEventField",
    "ProofTraceEventV1",
    "frozen_fixture_rows",
    "match_proof_trace_to_evidence",
    "mutate_omit_event",
    "mutate_reorder_events",
    "mutate_stale_state_fingerprint",
    "mutation_rejects_omit",
    "mutation_rejects_reorder",
    "mutation_rejects_stale_state",
    "project_replay_bundle",
    "project_runtime_evidence",
    "proof_trace_integrity_ok",
    "replay_proof_trace",
]
