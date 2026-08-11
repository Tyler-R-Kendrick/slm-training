"""INTEG-02 / SLM-555: theorem-backed trigger/no-effect admission preflight.

Optional fail-closed adapter on the existing autoresearch preflight seam.
Reuses KERN-08 ``formal.mechanism_trigger`` certificates — never invents a
second no-effect authority. Skips an expensive treatment only when every
locked-corpus row has **complete** trigger-absence evidence under a safe
necessary-trigger theorem and the corpus scan itself is complete.

Unknown evidence, incomplete scans, present triggers, and mechanisms without
a safe theorem always fall back to ``run`` (never skip). Zero *observed*
activation on an incomplete scan is not no-effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from slm_training.formal.mechanism_trigger import (
    CACHE_REUSE,
    CERTIFICATE_SCHEMA,
    CLOSURE_REMOVAL,
    FORCED_SPAN,
    NEURAL_FORWARD_ONLY,
    SINGLETON_BYPASS,
    THEOREM_NECESSARY,
    UNCONSTRAINED_RERANK,
    CertificateError,
    Mechanism,
    NamedCostModel,
    NoEffectCertificateV1,
    Observation,
    TriggerEvidence,
    emit_no_effect_certificate,
    trigger_evidence_complete,
)
from slm_training.harness_core.lineage.records import content_sha

try:
    from slm_training.autoresearch.preflight import PreflightVerdict
except ImportError:  # pragma: no cover - package owner always present in-tree
    from typing import Literal as _Lit

    from pydantic import BaseModel, Field

    class PreflightVerdict(BaseModel):  # type: ignore[no-redef]
        check_id: str
        verdict: _Lit["pass", "warn", "block"]
        reasons: list[str]
        data: dict = Field(default_factory=dict)

CHECK_ID = "mechanism_no_effect"
ADMISSION_SCHEMA = "mechanism_no_effect_admission/v1"
DispositionKind = Literal["skip_no_effect", "run"]

SUPPORTED_MECHANISMS: dict[str, Mechanism] = {
    SINGLETON_BYPASS.id: SINGLETON_BYPASS,
    CLOSURE_REMOVAL.id: CLOSURE_REMOVAL,
    CACHE_REUSE.id: CACHE_REUSE,
    FORCED_SPAN.id: FORCED_SPAN,
}

_UNSAFE_MECHANISMS: dict[str, Mechanism] = {
    UNCONSTRAINED_RERANK.id: UNCONSTRAINED_RERANK,
}


@dataclass(frozen=True)
class MechanismNoEffectAdmissionV1:
    """Replayable skip/run disposition for a mechanism treatment on a corpus."""

    schema: str
    disposition: DispositionKind
    mechanism_id: str
    corpus_id: str
    scan_complete: bool
    theorem_ref: str | None
    reasons: tuple[str, ...]
    campaign_id: str | None
    experiment_id: str | None
    manifest_sha256: str | None
    claim_class: str
    certificate: dict[str, Any] | None
    evidence_lineage: dict[str, Any]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "disposition": self.disposition,
            "mechanism_id": self.mechanism_id,
            "corpus_id": self.corpus_id,
            "scan_complete": self.scan_complete,
            "theorem_ref": self.theorem_ref,
            "reasons": list(self.reasons),
            "campaign_id": self.campaign_id,
            "experiment_id": self.experiment_id,
            "manifest_sha256": self.manifest_sha256,
            "claim_class": self.claim_class,
            "certificate": self.certificate,
            "evidence_lineage": dict(self.evidence_lineage),
            "content_sha256": self.content_sha256,
        }


def _as_observation(row: Observation | Mapping[str, Any]) -> Observation:
    if isinstance(row, Observation):
        return row
    return Observation.from_dict(row)


def _lineage(
    *,
    mechanism: Mechanism,
    corpus_id: str,
    rows: tuple[Observation, ...],
    scan_complete: bool,
    campaign_id: str | None,
    experiment_id: str | None,
    manifest_sha256: str | None,
    certificate: NoEffectCertificateV1 | None,
) -> dict[str, Any]:
    payload = {
        "mechanism": mechanism.to_dict(),
        "corpus_id": corpus_id,
        "scan_complete": scan_complete,
        "observations": [row.to_dict() for row in rows],
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
        "manifest_sha256": manifest_sha256,
        "certificate_schema": CERTIFICATE_SCHEMA,
        "certificate_digest": (
            content_sha(certificate.to_dict()) if certificate is not None else None
        ),
        "kernel": "slm_training.formal.mechanism_trigger",
        "issue": "SLM-555",
        "integ": "INTEG-02",
    }
    return {
        "observation_sha256": content_sha([row.to_dict() for row in rows]),
        "binding_sha256": content_sha(payload),
        "certificate_sha256": payload["certificate_digest"],
        "kernel": payload["kernel"],
    }


def _seal(
    *,
    disposition: DispositionKind,
    mechanism_id: str,
    corpus_id: str,
    scan_complete: bool,
    theorem_ref: str | None,
    reasons: tuple[str, ...],
    campaign_id: str | None,
    experiment_id: str | None,
    manifest_sha256: str | None,
    claim_class: str,
    certificate: NoEffectCertificateV1 | None,
    evidence_lineage: dict[str, Any],
) -> MechanismNoEffectAdmissionV1:
    body = {
        "schema": ADMISSION_SCHEMA,
        "disposition": disposition,
        "mechanism_id": mechanism_id,
        "corpus_id": corpus_id,
        "scan_complete": scan_complete,
        "theorem_ref": theorem_ref,
        "reasons": list(reasons),
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
        "manifest_sha256": manifest_sha256,
        "claim_class": claim_class,
        "certificate": certificate.to_dict() if certificate is not None else None,
        "evidence_lineage": evidence_lineage,
    }
    digest = content_sha(body)
    return MechanismNoEffectAdmissionV1(
        schema=ADMISSION_SCHEMA,
        disposition=disposition,
        mechanism_id=mechanism_id,
        corpus_id=corpus_id,
        scan_complete=scan_complete,
        theorem_ref=theorem_ref,
        reasons=reasons,
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        manifest_sha256=manifest_sha256,
        claim_class=claim_class,
        certificate=body["certificate"],
        evidence_lineage=evidence_lineage,
        content_sha256=digest,
    )


def admit_mechanism_treatment(
    *,
    mechanism_id: str,
    corpus_id: str,
    observations: Sequence[Observation | Mapping[str, Any]],
    scan_complete: bool,
    campaign_id: str | None = None,
    experiment_id: str | None = None,
    manifest_sha256: str | None = None,
    cost_model: NamedCostModel | None = None,
    theorem_ref: str = THEOREM_NECESSARY,
    claim_class: str = "fixture",
) -> MechanismNoEffectAdmissionV1:
    """Admit a mechanism treatment against a locked corpus.

    Returns ``skip_no_effect`` only when a KERN-08 no-effect certificate
    discharges on a **complete** scan with every trigger proved absent.
    Every other case returns ``run``.
    """

    mid = str(mechanism_id)
    cid = str(corpus_id)
    model = cost_model or NEURAL_FORWARD_ONLY
    rows = tuple(_as_observation(item) for item in observations)

    unsafe = _UNSAFE_MECHANISMS.get(mid)
    if unsafe is not None:
        lineage = _lineage(
            mechanism=unsafe,
            corpus_id=cid,
            rows=rows,
            scan_complete=bool(scan_complete),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            certificate=None,
        )
        return _seal(
            disposition="run",
            mechanism_id=mid,
            corpus_id=cid,
            scan_complete=bool(scan_complete),
            theorem_ref=None,
            reasons=(
                f"mechanism {mid!r} has no safe trigger theorem; "
                "run normal experimentation",
            ),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            claim_class=claim_class,
            certificate=None,
            evidence_lineage=lineage,
        )

    mechanism = SUPPORTED_MECHANISMS.get(mid)
    if mechanism is None:
        lineage = _lineage(
            mechanism=Mechanism(id=mid, has_safe_trigger_theorem=False),
            corpus_id=cid,
            rows=rows,
            scan_complete=bool(scan_complete),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            certificate=None,
        )
        return _seal(
            disposition="run",
            mechanism_id=mid,
            corpus_id=cid,
            scan_complete=bool(scan_complete),
            theorem_ref=None,
            reasons=(f"unsupported mechanism {mid!r}; run rather than skip",),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            claim_class=claim_class,
            certificate=None,
            evidence_lineage=lineage,
        )

    if not rows:
        lineage = _lineage(
            mechanism=mechanism,
            corpus_id=cid,
            rows=rows,
            scan_complete=bool(scan_complete),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            certificate=None,
        )
        return _seal(
            disposition="run",
            mechanism_id=mid,
            corpus_id=cid,
            scan_complete=bool(scan_complete),
            theorem_ref=theorem_ref,
            reasons=("empty locked corpus; refuse no-effect skip",),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            claim_class=claim_class,
            certificate=None,
            evidence_lineage=lineage,
        )

    # Incomplete scan: never infer no-effect from zero observed activation.
    if not scan_complete:
        lineage = _lineage(
            mechanism=mechanism,
            corpus_id=cid,
            rows=rows,
            scan_complete=False,
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            certificate=None,
        )
        return _seal(
            disposition="run",
            mechanism_id=mid,
            corpus_id=cid,
            scan_complete=False,
            theorem_ref=theorem_ref,
            reasons=(
                "incomplete corpus scan: zero observed activation is not "
                "no-effect; run",
            ),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            claim_class=claim_class,
            certificate=None,
            evidence_lineage=lineage,
        )

    present_ids = [
        row.input_id for row in rows if row.trigger is TriggerEvidence.PRESENT
    ]
    unknown_ids = [
        row.input_id
        for row in rows
        if not trigger_evidence_complete(row.trigger)
        or row.trigger is TriggerEvidence.UNKNOWN
    ]
    if present_ids or unknown_ids:
        reasons: list[str] = []
        if present_ids:
            reasons.append(
                "necessary trigger present on "
                + ",".join(present_ids)
                + "; run normal experimentation"
            )
        if unknown_ids:
            reasons.append(
                "unknown/incomplete trigger evidence on "
                + ",".join(unknown_ids)
                + "; fall back to run"
            )
        lineage = _lineage(
            mechanism=mechanism,
            corpus_id=cid,
            rows=rows,
            scan_complete=True,
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            certificate=None,
        )
        return _seal(
            disposition="run",
            mechanism_id=mid,
            corpus_id=cid,
            scan_complete=True,
            theorem_ref=theorem_ref,
            reasons=tuple(reasons),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            claim_class=claim_class,
            certificate=None,
            evidence_lineage=lineage,
        )

    try:
        certificate = emit_no_effect_certificate(
            mechanism=mechanism,
            cost_model=model,
            corpus_id=cid,
            corpus=rows,
            theorem_ref=theorem_ref,
            claim_class=claim_class,
        )
    except CertificateError as exc:
        lineage = _lineage(
            mechanism=mechanism,
            corpus_id=cid,
            rows=rows,
            scan_complete=True,
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            certificate=None,
        )
        return _seal(
            disposition="run",
            mechanism_id=mid,
            corpus_id=cid,
            scan_complete=True,
            theorem_ref=theorem_ref,
            reasons=(f"no-effect certificate refused: {exc}",),
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            claim_class=claim_class,
            certificate=None,
            evidence_lineage=lineage,
        )

    lineage = _lineage(
        mechanism=mechanism,
        corpus_id=cid,
        rows=rows,
        scan_complete=True,
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        manifest_sha256=manifest_sha256,
        certificate=certificate,
    )
    return _seal(
        disposition="skip_no_effect",
        mechanism_id=mid,
        corpus_id=cid,
        scan_complete=True,
        theorem_ref=certificate.theorem_ref,
        reasons=(
            f"all triggers proved absent on complete locked corpus {cid!r}; "
            "skip treatment (theorem-backed no-effect)",
        ),
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        manifest_sha256=manifest_sha256,
        claim_class=claim_class,
        certificate=certificate,
        evidence_lineage=lineage,
    )


def admit_for_campaign(
    manifest: Any,
    *,
    mechanism_id: str,
    corpus_id: str,
    observations: Sequence[Observation | Mapping[str, Any]],
    scan_complete: bool,
    manifest_sha256: str | None = None,
    cost_model: NamedCostModel | None = None,
    theorem_ref: str = THEOREM_NECESSARY,
) -> MechanismNoEffectAdmissionV1:
    """Bind admission to an ``ExperimentCampaignV1`` identity (HARN-10 seam)."""

    campaign_id = getattr(manifest, "campaign_id", None)
    experiment_id = getattr(manifest, "experiment_id", None)
    claim_class = getattr(manifest, "claim_class", None) or "fixture"
    return admit_mechanism_treatment(
        mechanism_id=mechanism_id,
        corpus_id=corpus_id,
        observations=observations,
        scan_complete=scan_complete,
        campaign_id=str(campaign_id) if campaign_id else None,
        experiment_id=str(experiment_id) if experiment_id else None,
        manifest_sha256=manifest_sha256,
        cost_model=cost_model,
        theorem_ref=theorem_ref,
        claim_class=str(claim_class),
    )


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
    """Preflight CHECK entry: optional; missing mechanism evidence → pass."""

    mechanism_id = candidate.get("mechanism_id")
    if not mechanism_id:
        return PreflightVerdict(
            check_id=CHECK_ID,
            verdict="pass",
            reasons=["no mechanism_id on candidate; mechanism no-effect check idle"],
            data={},
        )
    observations = _candidate_observations(candidate)
    if observations is None:
        return PreflightVerdict(
            check_id=CHECK_ID,
            verdict="pass",
            reasons=[
                "mechanism_id present but no locked corpus observations; "
                "refuse skip without evidence (run)"
            ],
            data={"mechanism_id": str(mechanism_id), "disposition": "run"},
        )

    scan_raw = candidate.get("scan_complete")
    if scan_raw is None:
        scan_complete = False
        missing_scan_flag = True
    else:
        scan_complete = bool(scan_raw)
        missing_scan_flag = False

    corpus_id = str(
        candidate.get("corpus_id") or candidate.get("locked_corpus_id") or ""
    )
    if not corpus_id:
        corpus_id = "unspecified"

    cost_model = None
    if candidate.get("cost_model"):
        cost_model = NamedCostModel(
            id=str(candidate["cost_model"]),
            version=int(candidate.get("cost_model_version") or 1),
        )

    admission = admit_mechanism_treatment(
        mechanism_id=str(mechanism_id),
        corpus_id=corpus_id,
        observations=observations,
        scan_complete=scan_complete,
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
        cost_model=cost_model,
        theorem_ref=str(candidate.get("theorem_ref") or THEOREM_NECESSARY),
        claim_class=str(candidate.get("claim_class") or "fixture"),
    )
    reasons = list(admission.reasons)
    if missing_scan_flag:
        reasons.insert(
            0,
            "scan_complete omitted on candidate; treated as incomplete (run, not skip)",
        )
    data = admission.to_dict()
    if admission.disposition == "skip_no_effect":
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


class _MechanismNoEffectCheck:
    check_id = CHECK_ID

    def run(self, candidate: dict[str, Any]) -> PreflightVerdict:
        try:
            return _run(dict(candidate or {}))
        except Exception as exc:  # noqa: BLE001 — plugins never raise
            return PreflightVerdict(
                check_id=CHECK_ID,
                verdict="warn",
                reasons=[
                    "mechanism_no_effect check errored and fails open: "
                    f"{type(exc).__name__}: {exc}"
                ],
                data={"error": f"{type(exc).__name__}: {exc}"},
            )


CHECK = _MechanismNoEffectCheck()

__all__ = [
    "ADMISSION_SCHEMA",
    "CHECK",
    "CHECK_ID",
    "SUPPORTED_MECHANISMS",
    "DispositionKind",
    "MechanismNoEffectAdmissionV1",
    "admit_for_campaign",
    "admit_mechanism_treatment",
]
