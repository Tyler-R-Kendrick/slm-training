"""Exact-vs-estimated arity certificates and provenance-aware reports (CAP0-04).

Redesigned onto main's canonical arity API. The pure certificate schema
(``EvidenceKind``, ``ConstraintFrame``, ``ExactEvidence`` / ``EstimatedEvidence``,
``ArityResult``, ``ArityProvenance``, ``ArityCertificate``,
``ArityCertificateBundle``, ``certificate_digest``) is reused verbatim from the
CAP0-04 design. What changed is the *backing*:

* a certificate is backed by the canonical
  :class:`~slm_training.dsl.analysis.arity.report.ExactArityReport` (produced by
  ``analyze``) via a small :class:`ReportView` bridge, instead of the retired
  stub ``ArityReport`` and its report-attached ``CodingMetadata``; and
* :func:`exact_certificate_from_report` consumes an explicit CAP0-03 coding
  witness (a :class:`~slm_training.dsl.analysis.arity.coding.CodeVerification`
  over a named construction) from the standalone
  :mod:`slm_training.dsl.analysis.arity.coding` API, instead of reading a
  ``coding_metadata`` attribute off the report.

Torch-free. Builds on the CAP0-02 report
(``docs/design/cap0-02-arity-analyzer-20260718.md``) and the standalone CAP0-03
coding API (``docs/design/cap0-03-coding-precision-20260718.md``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping

from slm_training.dsl.analysis.arity.coding import CodeVerification
from slm_training.dsl.analysis.arity.report import AnalysisBounds, ExactArityReport


class EvidenceKind(str, Enum):
    """How an arity claim was established."""

    EXACT_LOCAL = "exact_local"
    EXACT_EXTERNAL = "exact_external"
    ESTIMATED = "estimated"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class ConstraintFrame:
    """Constraints and versions under which a claim applies."""

    grammar_hash: str
    parser_version: str
    codec_version: str
    state_signature_version: str
    generation_order: str
    ast_bounds: Mapping[str, int | None]
    scope_bounds: Mapping[str, int | None]
    template_classes: tuple[str, ...]
    latent_role: str | None = None
    dimensions: int | None = None
    noise_model: str | None = None
    packing_assumption: str | None = None


@dataclass(frozen=True)
class ExactEvidence:
    """Evidence for a claim established by exact means."""

    evidence_kind: Literal[EvidenceKind.EXACT_LOCAL, EvidenceKind.EXACT_EXTERNAL]
    theorem_or_algorithm: str
    constraints: ConstraintFrame
    complete: bool
    witness_or_proof_hash: str | None
    work_counters: Mapping[str, int]
    source_uri: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_kind not in (EvidenceKind.EXACT_LOCAL, EvidenceKind.EXACT_EXTERNAL):
            raise ValueError("ExactEvidence evidence_kind must be exact_local or exact_external")
        if self.evidence_kind == EvidenceKind.EXACT_LOCAL and not self.complete:
            raise ValueError("EXACT_LOCAL evidence must be complete")
        if self.evidence_kind == EvidenceKind.EXACT_LOCAL and not self.witness_or_proof_hash:
            raise ValueError("EXACT_LOCAL evidence requires a witness_or_proof_hash")
        if self.evidence_kind == EvidenceKind.EXACT_EXTERNAL and not self.source_uri:
            raise ValueError("EXACT_EXTERNAL evidence requires a source_uri")


@dataclass(frozen=True)
class EstimatedEvidence:
    """Evidence for a claim established by sampling or measurement."""

    evidence_kind: Literal[EvidenceKind.ESTIMATED]
    constraints: ConstraintFrame
    dataset_ids: tuple[str, ...]
    trace_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    sample_count: int
    sampling_design: str
    coverage: Mapping[str, float | int]
    estimator: str
    confidence_interval: tuple[float, float] | None = None
    tail_metric: Mapping[str, float] | None = None
    seed_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.evidence_kind != EvidenceKind.ESTIMATED:
            raise ValueError("EstimatedEvidence evidence_kind must be estimated")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")


@dataclass(frozen=True)
class ArityResult:
    """One metric/value pair with evidence and a conservative status."""

    metric_name: str
    value: Any
    units: str | None
    evidence: ExactEvidence | EstimatedEvidence
    status: Literal["supported", "infeasible", "unknown", "diagnostic"]


@dataclass(frozen=True)
class ArityProvenance:
    """Provenance for a certificate bundle."""

    generated_at: str
    source_commit: str | None = None
    analyzer_version: str = "cap0-04-v1"
    frame_version: str = "cap0-02-v1"
    run_id: str | None = None
    trace_id: str | None = None
    input_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArityCertificate:
    """Self-contained certificate that binds an arity report to exact/estimated claims."""

    certificate_id: str
    report_digest: str
    frame_id: str
    provenance: ArityProvenance
    results: tuple[ArityResult, ...]
    version: str = "cap0-04-certificate-v1"

    def __post_init__(self) -> None:
        if not self.certificate_id:
            raise ValueError("certificate_id must be non-empty")
        if not self.report_digest:
            raise ValueError("report_digest must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""

        def _convert(value: object) -> object:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _convert(v) for k, v in value.items()}
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, (ConstraintFrame, ArityProvenance)):
                return _convert(asdict(value))
            if isinstance(value, ExactEvidence):
                return {
                    "evidence_kind": value.evidence_kind.value,
                    "theorem_or_algorithm": value.theorem_or_algorithm,
                    "constraints": _convert(value.constraints),
                    "complete": value.complete,
                    "witness_or_proof_hash": value.witness_or_proof_hash,
                    "work_counters": _convert(value.work_counters),
                    "source_uri": value.source_uri,
                }
            if isinstance(value, EstimatedEvidence):
                return {
                    "evidence_kind": value.evidence_kind.value,
                    "constraints": _convert(value.constraints),
                    "dataset_ids": _convert(value.dataset_ids),
                    "trace_ids": _convert(value.trace_ids),
                    "checkpoint_ids": _convert(value.checkpoint_ids),
                    "sample_count": value.sample_count,
                    "sampling_design": value.sampling_design,
                    "coverage": _convert(value.coverage),
                    "estimator": value.estimator,
                    "confidence_interval": _convert(value.confidence_interval),
                    "tail_metric": _convert(value.tail_metric),
                    "seed_ids": _convert(value.seed_ids),
                }
            if isinstance(value, ArityResult):
                return {
                    "metric_name": value.metric_name,
                    "value": _convert(value.value),
                    "units": value.units,
                    "evidence": _convert(value.evidence),
                    "status": value.status,
                }
            return repr(value)

        base = asdict(self)
        return {k: _convert(v) for k, v in base.items()}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def certificate_digest(certificate: ArityCertificate) -> str:
    """Deterministic digest over the certificate's semantic content."""
    payload = certificate.to_json(indent=None)
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:32]


@dataclass(frozen=True)
class ReportView:
    """Certificate-facing bridge over a canonical :class:`ExactArityReport`.

    Derives the four quantities a CAP0-04 certificate needs without editing the
    canonical ``report.py``:

    * ``frame_id`` — a stable identity from the committed fixture name and the
      ``K**d`` capacity dimension (e.g. ``"bounded-expr/4"``);
    * ``total_states`` — the report's ``raw_state_count``;
    * ``minimized_states`` — the report's ``minimized_state_count``;
    * ``digest`` — a stable content hash over the report's existing deterministic
      :meth:`ExactArityReport.to_json`.

    :meth:`to_dict` delegates to the wrapped report so a bundle serialises the
    canonical report unchanged.
    """

    report: ExactArityReport

    @property
    def frame_id(self) -> str:
        dimensions = self.report.capacity.get("d")
        if dimensions is None:
            return self.report.fixture
        return f"{self.report.fixture}/{dimensions}"

    @property
    def total_states(self) -> int:
        return self.report.raw_state_count

    @property
    def minimized_states(self) -> int:
        return self.report.minimized_state_count

    @property
    def bounds(self) -> AnalysisBounds:
        return self.report.bounds

    @property
    def digest(self) -> str:
        payload = self.report.to_json()
        return hashlib.sha256(
            payload.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return self.report.to_dict()


def report_view(report: ExactArityReport) -> ReportView:
    """Adapt a canonical arity report to the certificate-facing bridge view."""
    return ReportView(report)


@dataclass(frozen=True)
class ArityCertificateBundle:
    """Bundle that links an arity report with its certificate."""

    report: ReportView
    certificate: ArityCertificate
    bundle_digest: str = field(init=False)

    def __post_init__(self) -> None:
        canonical = {
            "report": self.report.to_dict(),
            "certificate": self.certificate.to_dict(),
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:32]
        object.__setattr__(self, "bundle_digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_digest": self.bundle_digest,
            "report": self.report.to_dict(),
            "certificate": self.certificate.to_dict(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _witness_hash(construction: str, verification: CodeVerification) -> str:
    """Deterministic proof-hash fingerprint of a CAP0-03 coding witness."""
    payload = json.dumps(
        {
            "construction": construction,
            "q": verification.q,
            "n": verification.n,
            "size": verification.size,
            "minimum_distance": verification.minimum_distance,
            "required_size": verification.required_size,
            "required_distance": verification.required_distance,
            "ok": verification.ok,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"sha256:{digest}"


def exact_certificate_from_report(
    report: ExactArityReport,
    *,
    verification: CodeVerification,
    construction: str,
    generated_at: str,
    source_commit: str | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    input_hashes: tuple[str, ...] = (),
) -> ArityCertificateBundle:
    """Build an exact-local certificate bundle from a canonical arity report.

    The exact-vs-estimated wiring path for CAP0-04, redesigned onto main's API.
    It binds a canonical
    :class:`~slm_training.dsl.analysis.arity.report.ExactArityReport` (from
    ``analyze``) to a *separately constructed* CAP0-03 coding witness — a
    :class:`~slm_training.dsl.analysis.arity.coding.CodeVerification` returned by
    :func:`~slm_training.dsl.analysis.arity.coding.verify_code` over a named
    ``construction`` (e.g. ``build_mds_7_4_2_3``) — rather than a report-attached
    ``CodingMetadata``.

    The verified construction is recorded as ``EXACT_LOCAL`` evidence for the
    report's minimized-state count: the exhaustive verification is complete, so
    the evidence is complete, and ``status`` is ``supported`` when the code meets
    the required size and distance and ``infeasible`` when the exact check
    disproves it. The constraint frame, dimensions, and versions are read from
    the canonical report; nothing is estimated or sampled on this path.
    """
    view = report_view(report)
    capacity_dim = report.capacity.get("d")
    dimensions = None if capacity_dim is None else int(capacity_dim)

    constraints = ConstraintFrame(
        grammar_hash=report.grammar_hash,
        parser_version=str(report.parser_version),
        codec_version=str(report.codec_version),
        state_signature_version=str(report.signature_version),
        generation_order="preorder",
        ast_bounds={
            "max_ast_nodes": report.bounds.max_ast_nodes,
            "max_ast_depth": report.bounds.max_ast_depth,
        },
        scope_bounds={"max_live_bindings": report.bounds.max_live_bindings},
        template_classes=report.bounds.template_classes,
        latent_role=None,
        dimensions=dimensions,
        noise_model=None,
        packing_assumption=None,
    )
    evidence = ExactEvidence(
        evidence_kind=EvidenceKind.EXACT_LOCAL,
        theorem_or_algorithm=construction,
        constraints=constraints,
        complete=True,
        witness_or_proof_hash=_witness_hash(construction, verification),
        work_counters={
            "states": view.minimized_states,
            "code_size": verification.size,
            "minimum_distance": verification.minimum_distance,
            "required_size": verification.required_size,
            "required_distance": verification.required_distance,
        },
        source_uri=None,
    )
    status: Literal["supported", "infeasible", "unknown", "diagnostic"] = (
        "supported" if verification.ok else "infeasible"
    )
    result = ArityResult(
        metric_name="minimized_state_count",
        value=view.minimized_states,
        units="states",
        evidence=evidence,
        status=status,
    )
    provenance = ArityProvenance(
        generated_at=generated_at,
        source_commit=source_commit,
        run_id=run_id,
        trace_id=trace_id,
        input_hashes=input_hashes,
    )
    certificate = ArityCertificate(
        certificate_id=f"cert-{view.digest}",
        report_digest=view.digest,
        frame_id=view.frame_id,
        provenance=provenance,
        results=(result,),
    )
    return ArityCertificateBundle(report=view, certificate=certificate)
