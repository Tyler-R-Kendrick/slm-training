"""INTEG-08 / SLM-562: corpus-local dominance + durable skip-decision evidence.

Emits sealed evidence that binds baseline/treatment/corpus/model/cost identities
to a KERN-08 ``DominanceCertificateV1`` (or an explicit non-certification).
Dominated treatments may be recorded into the existing exhausted-knob ledger and
mechanism-disposition owner so equivalent knob signatures are not re-spent.

Dispositions (fail closed):

- ``skip_dominated`` — output-equivalent on the locked corpus and no cheaper
  under the declared cost model (KERN-08 dominance certificate emits).
- ``no_effect_cheaper`` — no-effect holds but treatment is strictly cheaper on
  at least one row (not dominated; do not skip-as-dominated).
- ``unknown`` — incomplete equivalence, cost, scan, or theorem evidence;
  never certify.

Dominance is local to the exact identity tuple and never generalizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from slm_training.formal.mechanism_trigger import (
    CACHE_REUSE,
    CLOSURE_REMOVAL,
    DOMINANCE_SCHEMA,
    FORCED_SPAN,
    NEURAL_FORWARD_ONLY,
    SINGLETON_BYPASS,
    THEOREM_NECESSARY,
    UNCONSTRAINED_RERANK,
    CertificateError,
    DominanceCertificateV1,
    Mechanism,
    NamedCostModel,
    NoEffectCertificateV1,
    Observation,
    TriggerEvidence,
    dominates_locally,
    emit_dominance_certificate,
    emit_no_effect_certificate,
    outputs_equal,
    trigger_evidence_complete,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.mechanism_disposition_report import (
    build_mechanism_disposition_record,
)
from slm_training.harnesses.experiments.slm160_spv_disposition import Disposition

try:
    from slm_training.autoresearch.preflight import PreflightVerdict
except ImportError:  # pragma: no cover
    from typing import Literal as _Lit

    from pydantic import BaseModel, Field

    class PreflightVerdict(BaseModel):  # type: ignore[no-redef]
        check_id: str
        verdict: _Lit["pass", "warn", "block"]
        reasons: list[str]
        data: dict = Field(default_factory=dict)

CHECK_ID = "mechanism_dominance"
EVIDENCE_SCHEMA = "corpus_local_dominance_evidence/v1"
ADMISSION_SCHEMA = "mechanism_dominance_admission/v1"
FEEDBACK_SCHEMA = "mechanism_dominance_disposition_feedback/v1"
ISSUE = "SLM-562"
INTEG = "INTEG-08"
KERNEL = "slm_training.formal.mechanism_trigger"

DispositionKind = Literal["skip_dominated", "no_effect_cheaper", "unknown"]
AggregateRelation = Literal["dominated", "cheaper", "unknown"]

SUPPORTED_MECHANISMS: dict[str, Mechanism] = {
    SINGLETON_BYPASS.id: SINGLETON_BYPASS,
    CLOSURE_REMOVAL.id: CLOSURE_REMOVAL,
    CACHE_REUSE.id: CACHE_REUSE,
    FORCED_SPAN.id: FORCED_SPAN,
}

_UNSAFE_MECHANISMS: dict[str, Mechanism] = {
    UNCONSTRAINED_RERANK.id: UNCONSTRAINED_RERANK,
}

DOMINANCE_SKIP_REASON = "corpus_local_dominance"


@dataclass(frozen=True)
class DominanceScopeIdentityV1:
    """Exact local scope — mutation of any field invalidates the certificate."""

    baseline_id: str
    treatment_id: str
    corpus_id: str
    corpus_version: str
    model_id: str
    cost_model: NamedCostModel
    mechanism_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "treatment_id": self.treatment_id,
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "model_id": self.model_id,
            "cost_model": self.cost_model.tag,
            "mechanism_id": self.mechanism_id,
        }

    @property
    def data_eval_identity(self) -> str:
        """Corpus/model/cost identity used by ExhaustedKnobLedger matching."""

        return content_sha(self.to_dict())


@dataclass(frozen=True)
class CorpusLocalDominanceEvidenceV1:
    """Durable dominance/skip evidence — replay reconstructs the decision."""

    schema: str
    disposition: DispositionKind
    scope: DominanceScopeIdentityV1
    scan_complete: bool
    theorem_ref: str | None
    claim_class: str
    output_equivalence: dict[str, Any]
    per_example_costs: tuple[dict[str, Any], ...]
    aggregate: dict[str, Any]
    empirical_remainder: str | None
    reasons: tuple[str, ...]
    certificate: dict[str, Any] | None
    no_effect_certificate: dict[str, Any] | None
    generalizes_beyond_corpus: bool
    campaign_id: str | None
    experiment_id: str | None
    manifest_sha256: str | None
    knob_signature_sha256: str | None
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "disposition": self.disposition,
            "scope": self.scope.to_dict(),
            "scan_complete": self.scan_complete,
            "theorem_ref": self.theorem_ref,
            "claim_class": self.claim_class,
            "output_equivalence": dict(self.output_equivalence),
            "per_example_costs": [dict(row) for row in self.per_example_costs],
            "aggregate": dict(self.aggregate),
            "empirical_remainder": self.empirical_remainder,
            "reasons": list(self.reasons),
            "certificate": self.certificate,
            "no_effect_certificate": self.no_effect_certificate,
            "generalizes_beyond_corpus": self.generalizes_beyond_corpus,
            "campaign_id": self.campaign_id,
            "experiment_id": self.experiment_id,
            "manifest_sha256": self.manifest_sha256,
            "knob_signature_sha256": self.knob_signature_sha256,
            "content_sha256": self.content_sha256,
            "integ": INTEG,
            "issue": ISSUE,
            "kernel": KERNEL,
        }


@dataclass(frozen=True)
class MechanismDominanceAdmissionV1:
    schema: str
    disposition: DispositionKind
    evidence: CorpusLocalDominanceEvidenceV1
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "disposition": self.disposition,
            "evidence": self.evidence.to_dict(),
            "content_sha256": self.content_sha256,
        }


def _as_observation(row: Observation | Mapping[str, Any]) -> Observation:
    if isinstance(row, Observation):
        return row
    return Observation.from_dict(row)


def _per_example_costs(rows: Sequence[Observation]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "input_id": row.input_id,
            "baseline_cost": row.baseline_cost,
            "enabled_cost": row.enabled_cost,
            "outputs_equal": outputs_equal(row),
            "dominates_locally": dominates_locally(row),
            "trigger": row.trigger.value,
        }
        for row in rows
    )


def _aggregate(rows: Sequence[Observation]) -> dict[str, Any]:
    baseline_total = sum(row.baseline_cost for row in rows)
    treatment_total = sum(row.enabled_cost for row in rows)
    all_equal = bool(rows) and all(outputs_equal(row) for row in rows)
    all_dominated = bool(rows) and all(dominates_locally(row) for row in rows)
    any_cheaper = any(row.enabled_cost < row.baseline_cost for row in rows)
    if all_dominated:
        relation: AggregateRelation = "dominated"
    elif all_equal and any_cheaper:
        relation = "cheaper"
    else:
        relation = "unknown"
    return {
        "baseline_total": baseline_total,
        "treatment_total": treatment_total,
        "delta_treatment_minus_baseline": treatment_total - baseline_total,
        "relation": relation,
        "row_count": len(rows),
    }


def _output_equivalence(
    rows: Sequence[Observation],
    *,
    theorem_ref: str | None,
    no_effect: NoEffectCertificateV1 | None,
) -> dict[str, Any]:
    complete = bool(rows) and all(
        trigger_evidence_complete(row.trigger) for row in rows
    )
    all_equal = bool(rows) and all(outputs_equal(row) for row in rows)
    return {
        "complete": complete and all_equal,
        "all_outputs_equal": all_equal,
        "trigger_evidence_complete": complete,
        "theorem_ref": theorem_ref,
        "no_effect_certificate_digest": (
            content_sha(no_effect.to_dict()) if no_effect is not None else None
        ),
        "proof": "kern08_no_effect" if no_effect is not None else None,
    }


def _seal_evidence(
    *,
    disposition: DispositionKind,
    scope: DominanceScopeIdentityV1,
    scan_complete: bool,
    theorem_ref: str | None,
    claim_class: str,
    rows: tuple[Observation, ...],
    reasons: tuple[str, ...],
    certificate: DominanceCertificateV1 | None,
    no_effect: NoEffectCertificateV1 | None,
    empirical_remainder: str | None,
    campaign_id: str | None,
    experiment_id: str | None,
    manifest_sha256: str | None,
    knob_signature_sha256: str | None,
) -> CorpusLocalDominanceEvidenceV1:
    body = {
        "schema": EVIDENCE_SCHEMA,
        "disposition": disposition,
        "scope": scope.to_dict(),
        "scan_complete": scan_complete,
        "theorem_ref": theorem_ref,
        "claim_class": claim_class,
        "output_equivalence": _output_equivalence(
            rows, theorem_ref=theorem_ref, no_effect=no_effect
        ),
        "per_example_costs": list(_per_example_costs(rows)),
        "aggregate": _aggregate(rows),
        "empirical_remainder": empirical_remainder,
        "reasons": list(reasons),
        "certificate": certificate.to_dict() if certificate is not None else None,
        "no_effect_certificate": (
            no_effect.to_dict() if no_effect is not None else None
        ),
        "generalizes_beyond_corpus": False,
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
        "manifest_sha256": manifest_sha256,
        "knob_signature_sha256": knob_signature_sha256,
        "integ": INTEG,
        "issue": ISSUE,
        "kernel": KERNEL,
    }
    digest = content_sha(body)
    return CorpusLocalDominanceEvidenceV1(
        schema=EVIDENCE_SCHEMA,
        disposition=disposition,
        scope=scope,
        scan_complete=scan_complete,
        theorem_ref=theorem_ref,
        claim_class=claim_class,
        output_equivalence=body["output_equivalence"],
        per_example_costs=tuple(body["per_example_costs"]),
        aggregate=body["aggregate"],
        empirical_remainder=empirical_remainder,
        reasons=reasons,
        certificate=body["certificate"],
        no_effect_certificate=body["no_effect_certificate"],
        generalizes_beyond_corpus=False,
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        manifest_sha256=manifest_sha256,
        knob_signature_sha256=knob_signature_sha256,
        content_sha256=digest,
    )


def emit_dominance_evidence(
    *,
    mechanism_id: str,
    baseline_id: str,
    treatment_id: str,
    corpus_id: str,
    corpus_version: str,
    model_id: str,
    observations: Sequence[Observation | Mapping[str, Any]],
    scan_complete: bool,
    cost_model: NamedCostModel | None = None,
    theorem_ref: str = THEOREM_NECESSARY,
    claim_class: str = "fixture",
    campaign_id: str | None = None,
    experiment_id: str | None = None,
    manifest_sha256: str | None = None,
    knob_signature_sha256: str | None = None,
) -> CorpusLocalDominanceEvidenceV1:
    """Classify and seal corpus-local dominance evidence (fail closed)."""

    mid = str(mechanism_id)
    model = cost_model or NEURAL_FORWARD_ONLY
    rows = tuple(_as_observation(item) for item in observations)
    scope = DominanceScopeIdentityV1(
        baseline_id=str(baseline_id),
        treatment_id=str(treatment_id),
        corpus_id=str(corpus_id),
        corpus_version=str(corpus_version),
        model_id=str(model_id),
        cost_model=model,
        mechanism_id=mid,
    )

    def _unknown(reasons: tuple[str, ...], remainder: str) -> CorpusLocalDominanceEvidenceV1:
        return _seal_evidence(
            disposition="unknown",
            scope=scope,
            scan_complete=bool(scan_complete),
            theorem_ref=None,
            claim_class=claim_class,
            rows=rows,
            reasons=reasons,
            certificate=None,
            no_effect=None,
            empirical_remainder=remainder,
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            knob_signature_sha256=knob_signature_sha256,
        )

    unsafe = _UNSAFE_MECHANISMS.get(mid)
    if unsafe is not None:
        return _unknown(
            (
                f"mechanism {mid!r} has no safe trigger theorem; "
                "dominance cannot certify",
            ),
            "unsafe_mechanism",
        )

    mechanism = SUPPORTED_MECHANISMS.get(mid)
    if mechanism is None:
        return _unknown(
            (f"unsupported mechanism {mid!r}; refuse dominance certification",),
            "unsupported_mechanism",
        )

    if not rows:
        return _unknown(("empty locked corpus; refuse dominance certification",), "empty_corpus")

    if not scan_complete:
        return _unknown(
            (
                "incomplete corpus scan: refuse dominance / skip certification",
            ),
            "incomplete_scan",
        )

    unknown_ids = [
        row.input_id
        for row in rows
        if not trigger_evidence_complete(row.trigger)
        or row.trigger is TriggerEvidence.UNKNOWN
    ]
    if unknown_ids:
        return _unknown(
            (
                "unknown/incomplete trigger or equivalence evidence on "
                + ",".join(unknown_ids),
            ),
            "incomplete_equivalence_evidence",
        )

    present_ids = [
        row.input_id for row in rows if row.trigger is TriggerEvidence.PRESENT
    ]
    if present_ids:
        return _unknown(
            (
                "necessary trigger present on "
                + ",".join(present_ids)
                + "; theorem-backed dominance requires trigger absence",
            ),
            "trigger_present_not_certifiable_as_no_effect",
        )

    try:
        no_effect = emit_no_effect_certificate(
            mechanism=mechanism,
            cost_model=model,
            corpus_id=scope.corpus_id,
            corpus=rows,
            theorem_ref=theorem_ref,
            claim_class=claim_class,
        )
    except CertificateError as exc:
        return _unknown(
            (f"no-effect certificate refused: {exc}",),
            "no_effect_certificate_refused",
        )

    try:
        dominance = emit_dominance_certificate(
            mechanism=mechanism,
            cost_model=model,
            corpus_id=scope.corpus_id,
            corpus=rows,
            theorem_ref=theorem_ref,
            claim_class=claim_class,
        )
    except CertificateError as exc:
        # No-effect holds but cost blocks dominance → cheaper (or mixed).
        return _seal_evidence(
            disposition="no_effect_cheaper",
            scope=scope,
            scan_complete=True,
            theorem_ref=no_effect.theorem_ref,
            claim_class=claim_class,
            rows=rows,
            reasons=(
                "no-effect on locked corpus but treatment is cheaper on at least "
                f"one row under {model.tag}; not dominated ({exc})",
            ),
            certificate=None,
            no_effect=no_effect,
            empirical_remainder="cost_improvement_blocks_dominance",
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            knob_signature_sha256=knob_signature_sha256,
        )

    return _seal_evidence(
        disposition="skip_dominated",
        scope=scope,
        scan_complete=True,
        theorem_ref=dominance.no_effect.theorem_ref,
        claim_class=claim_class,
        rows=rows,
        reasons=(
            f"treatment output-equivalent and no cheaper than baseline on "
            f"locked corpus {scope.corpus_id!r}@{scope.corpus_version} / "
            f"model {scope.model_id!r} / {model.tag}; skip re-run",
        ),
        certificate=dominance,
        no_effect=dominance.no_effect,
        empirical_remainder=None,
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        manifest_sha256=manifest_sha256,
        knob_signature_sha256=knob_signature_sha256,
    )


def evidence_integrity_ok(payload: Mapping[str, Any]) -> bool:
    """True iff replay digest matches and schema is known."""

    if payload.get("schema") != EVIDENCE_SCHEMA:
        return False
    body = {k: v for k, v in payload.items() if k != "content_sha256"}
    return content_sha(body) == str(payload.get("content_sha256") or "")


def replay_dominance_decision(payload: Mapping[str, Any]) -> DispositionKind:
    """Reconstruct the sealed disposition; tamper / unknown schema → unknown."""

    if not evidence_integrity_ok(payload):
        return "unknown"
    disposition = payload.get("disposition")
    if disposition in {"skip_dominated", "no_effect_cheaper", "unknown"}:
        return disposition  # type: ignore[return-value]
    return "unknown"


def scope_matches_evidence(
    payload: Mapping[str, Any],
    *,
    baseline_id: str,
    treatment_id: str,
    corpus_id: str,
    corpus_version: str,
    model_id: str,
    cost_model: NamedCostModel | str,
    mechanism_id: str,
) -> bool:
    """Identity check — any mutated field fails closed."""

    if not evidence_integrity_ok(payload):
        return False
    scope = payload.get("scope") or {}
    cost_tag = (
        cost_model.tag if isinstance(cost_model, NamedCostModel) else str(cost_model)
    )
    expected = {
        "baseline_id": str(baseline_id),
        "treatment_id": str(treatment_id),
        "corpus_id": str(corpus_id),
        "corpus_version": str(corpus_version),
        "model_id": str(model_id),
        "cost_model": cost_tag,
        "mechanism_id": str(mechanism_id),
    }
    return dict(scope) == expected


def admit_mechanism_dominance(
    *,
    mechanism_id: str,
    baseline_id: str,
    treatment_id: str,
    corpus_id: str,
    corpus_version: str,
    model_id: str,
    observations: Sequence[Observation | Mapping[str, Any]],
    scan_complete: bool,
    cost_model: NamedCostModel | None = None,
    theorem_ref: str = THEOREM_NECESSARY,
    claim_class: str = "fixture",
    campaign_id: str | None = None,
    experiment_id: str | None = None,
    manifest_sha256: str | None = None,
    knob_signature_sha256: str | None = None,
) -> MechanismDominanceAdmissionV1:
    evidence = emit_dominance_evidence(
        mechanism_id=mechanism_id,
        baseline_id=baseline_id,
        treatment_id=treatment_id,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        model_id=model_id,
        observations=observations,
        scan_complete=scan_complete,
        cost_model=cost_model,
        theorem_ref=theorem_ref,
        claim_class=claim_class,
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        manifest_sha256=manifest_sha256,
        knob_signature_sha256=knob_signature_sha256,
    )
    body = {
        "schema": ADMISSION_SCHEMA,
        "disposition": evidence.disposition,
        "evidence": evidence.to_dict(),
    }
    return MechanismDominanceAdmissionV1(
        schema=ADMISSION_SCHEMA,
        disposition=evidence.disposition,
        evidence=evidence,
        content_sha256=content_sha(body),
    )


def record_dominated_skip(
    evidence: CorpusLocalDominanceEvidenceV1 | Mapping[str, Any],
    ledger: Any,
    *,
    knobs: Mapping[str, Any] | None = None,
    knob_signature_sha256: str | None = None,
    claim_class: str | None = None,
) -> Any | None:
    """Append to ``ExhaustedKnobLedger`` only for ``skip_dominated``.

    Reuses hillclimb history so equivalent knob signatures under the same
    corpus/model/cost identity are not repeatedly re-run.
    """

    payload = evidence.to_dict() if hasattr(evidence, "to_dict") else dict(evidence)
    if payload.get("disposition") != "skip_dominated":
        return None
    if not evidence_integrity_ok(payload):
        raise ValueError("dominance evidence failed integrity check")

    from slm_training.autoresearch.hillclimb import knob_signature_sha256 as _knob_sha

    scope = payload["scope"]
    sig = knob_signature_sha256 or payload.get("knob_signature_sha256")
    if not sig:
        if knobs is None:
            knobs = {
                "mechanism_id": scope["mechanism_id"],
                "treatment_id": scope["treatment_id"],
                "baseline_id": scope["baseline_id"],
            }
        sig = _knob_sha(dict(knobs))
    data_eval = content_sha(
        {
            "baseline_id": scope["baseline_id"],
            "treatment_id": scope["treatment_id"],
            "corpus_id": scope["corpus_id"],
            "corpus_version": scope["corpus_version"],
            "model_id": scope["model_id"],
            "cost_model": scope["cost_model"],
            "mechanism_id": scope["mechanism_id"],
        }
    )
    return ledger.record_null(
        knob_signature_sha256=str(sig),
        data_eval_identity=data_eval,
        claim_class=str(claim_class or payload.get("claim_class") or "fixture"),
        reason=DOMINANCE_SKIP_REASON,
        note=f"evidence_sha256={payload['content_sha256']}",
    )


def feed_dominance_to_disposition(
    evidence: CorpusLocalDominanceEvidenceV1 | Mapping[str, Any],
    *,
    evidence_cutoff: str = "integ-08/corpus-local-dominance",
) -> dict[str, Any]:
    """Project sealed dominance evidence into ``MechanismDispositionRecordV1``."""

    payload = evidence.to_dict() if hasattr(evidence, "to_dict") else dict(evidence)
    if not evidence_integrity_ok(payload):
        raise ValueError("dominance evidence failed integrity check")
    disposition_kind = str(payload.get("disposition") or "unknown")
    if disposition_kind == "skip_dominated":
        disposition = Disposition.REJECT
        evidence_class = "rejected"
        default_state = "off"
        rationale = (
            "INTEG-08: corpus-local dominance — output-equivalent and no cheaper; "
            "reject re-spend under equivalent knob signature on this corpus/model"
        )
    elif disposition_kind == "no_effect_cheaper":
        disposition = Disposition.RETAIN_DIAGNOSTIC
        evidence_class = "fixture"
        default_state = "off"
        rationale = (
            "INTEG-08: no-effect but cheaper under declared cost model — not "
            "dominated; retain as diagnostic cost-win signal (not a skip)"
        )
    else:
        disposition = Disposition.INCONCLUSIVE
        evidence_class = "fixture"
        default_state = "n/a"
        rationale = (
            "INTEG-08: incomplete/unknown equivalence or cost evidence; "
            "no dominance certification"
        )

    record = build_mechanism_disposition_record(
        mechanism_id=str(payload["scope"]["mechanism_id"]),
        owner_module=(
            "src/slm_training/autoresearch/preflight/mechanism_dominance.py"
        ),
        owner_version="integ-08/v1",
        default_state=default_state,
        authority_class="oracle-diagnostic",
        evidence_cutoff=evidence_cutoff,
        evidence_class=evidence_class,
        disposition=disposition,
        rationale=rationale,
        activation_evidence=content_sha(payload),
        applicable_packs=("autoresearch/preflight",),
        matched_controls=True,
        semantic_effect="none_output_equivalent",
        cost_cold_effect=str((payload.get("aggregate") or {}).get("relation") or ""),
        safety_invariants=(
            "corpus_local_only",
            "no_automatic_generalization",
            "unknown_never_certifies",
        ),
        issues=(ISSUE, "SLM-531"),
    )
    return {
        "schema": FEEDBACK_SCHEMA,
        "dominance_disposition": disposition_kind,
        "disposition_record": record.to_dict(),
        "evidence_sha256": payload["content_sha256"],
        "scope": dict(payload["scope"]),
    }


def _candidate_observations(candidate: Mapping[str, Any]) -> list[Any] | None:
    raw = candidate.get("observations")
    if raw is None:
        raw = candidate.get("locked_corpus")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise TypeError("observations/locked_corpus must be a sequence")
    return list(raw)


def _run(candidate: dict[str, Any]) -> PreflightVerdict:
    mechanism_id = candidate.get("mechanism_id")
    if not mechanism_id:
        return PreflightVerdict(
            check_id=CHECK_ID,
            verdict="pass",
            reasons=["no mechanism_id on candidate; mechanism dominance check idle"],
            data={},
        )
    # Dominance admission is opt-in: require explicit baseline/treatment/model
    # identity so INTEG-02 no-effect skips are not auto-upgraded.
    required = ("baseline_id", "treatment_id", "model_id")
    missing = [key for key in required if not candidate.get(key)]
    if missing:
        return PreflightVerdict(
            check_id=CHECK_ID,
            verdict="pass",
            reasons=[
                "mechanism_id present but dominance identity incomplete "
                f"(missing {','.join(missing)}); idle (not unknown-certify)"
            ],
            data={"mechanism_id": str(mechanism_id), "disposition": "unknown"},
        )

    observations = _candidate_observations(candidate)
    if observations is None:
        return PreflightVerdict(
            check_id=CHECK_ID,
            verdict="pass",
            reasons=[
                "dominance identity present but no locked corpus observations; "
                "refuse certify without evidence"
            ],
            data={"mechanism_id": str(mechanism_id), "disposition": "unknown"},
        )

    scan_raw = candidate.get("scan_complete")
    if scan_raw is None:
        scan_complete = False
        missing_scan_flag = True
    else:
        scan_complete = bool(scan_raw)
        missing_scan_flag = False

    cost_model = None
    if candidate.get("cost_model"):
        cost_model = NamedCostModel(
            id=str(candidate["cost_model"]),
            version=int(candidate.get("cost_model_version") or 1),
        )

    admission = admit_mechanism_dominance(
        mechanism_id=str(mechanism_id),
        baseline_id=str(candidate["baseline_id"]),
        treatment_id=str(candidate["treatment_id"]),
        corpus_id=str(
            candidate.get("corpus_id") or candidate.get("locked_corpus_id") or "unspecified"
        ),
        corpus_version=str(candidate.get("corpus_version") or "unspecified"),
        model_id=str(candidate["model_id"]),
        observations=observations,
        scan_complete=scan_complete,
        cost_model=cost_model,
        theorem_ref=str(candidate.get("theorem_ref") or THEOREM_NECESSARY),
        claim_class=str(candidate.get("claim_class") or "fixture"),
        campaign_id=(
            str(candidate["campaign_id"]) if candidate.get("campaign_id") else None
        ),
        experiment_id=(
            str(candidate["experiment_id"]) if candidate.get("experiment_id") else None
        ),
        manifest_sha256=(
            str(candidate["manifest_sha256"])
            if candidate.get("manifest_sha256")
            else None
        ),
        knob_signature_sha256=(
            str(candidate["knob_signature_sha256"])
            if candidate.get("knob_signature_sha256")
            else None
        ),
    )
    reasons = list(admission.evidence.reasons)
    if missing_scan_flag:
        reasons.insert(
            0,
            "scan_complete omitted on candidate; treated as incomplete (unknown)",
        )
    data = admission.to_dict()
    if admission.disposition == "skip_dominated":
        return PreflightVerdict(
            check_id=CHECK_ID,
            verdict="block",
            reasons=reasons,
            data=data,
        )
    return PreflightVerdict(
        check_id=CHECK_ID,
        verdict="pass",
        reasons=reasons,
        data=data,
    )


class _MechanismDominanceCheck:
    check_id = CHECK_ID

    def run(self, candidate: dict[str, Any]) -> PreflightVerdict:
        try:
            return _run(dict(candidate or {}))
        except Exception as exc:  # noqa: BLE001 — plugins never raise
            return PreflightVerdict(
                check_id=CHECK_ID,
                verdict="warn",
                reasons=[
                    "mechanism_dominance check errored and fails open: "
                    f"{type(exc).__name__}: {exc}"
                ],
                data={"error": f"{type(exc).__name__}: {exc}"},
            )


CHECK = _MechanismDominanceCheck()

__all__ = [
    "ADMISSION_SCHEMA",
    "CHECK",
    "CHECK_ID",
    "DOMINANCE_SKIP_REASON",
    "DOMINANCE_SCHEMA",
    "EVIDENCE_SCHEMA",
    "FEEDBACK_SCHEMA",
    "SUPPORTED_MECHANISMS",
    "AggregateRelation",
    "CorpusLocalDominanceEvidenceV1",
    "DispositionKind",
    "DominanceScopeIdentityV1",
    "MechanismDominanceAdmissionV1",
    "admit_mechanism_dominance",
    "emit_dominance_evidence",
    "evidence_integrity_ok",
    "feed_dominance_to_disposition",
    "record_dominated_skip",
    "replay_dominance_decision",
    "scope_matches_evidence",
]
