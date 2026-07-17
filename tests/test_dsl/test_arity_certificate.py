"""Regression tests for CAP0-04 exact-vs-estimated arity certificates."""

from __future__ import annotations

import json

import pytest

from slm_training.dsl.analysis.arity import (
    AnalysisBounds,
    ArityAnalyzer,
    ArityCertificate,
    ArityProvenance,
    ArityResult,
    ConstraintFrame,
    EstimatedEvidence,
    EvidenceKind,
    ExactEvidence,
    certificate_digest,
    exact_certificate_from_report,
)
from slm_training.dsl.analysis.arity.render import one_line_summary, to_csv, to_markdown


@pytest.fixture
def toy_report():
    bounds = AnalysisBounds(max_ast_nodes=4)
    analyzer = ArityAnalyzer("toy-layout", bounds)
    return analyzer.analyze(
        ['root = row(title, action)\ntitle = text(":hero.title")\naction = button(":cta.label")'],
        include_coding_metadata=True,
    )


def _exact_evidence() -> ExactEvidence:
    return ExactEvidence(
        evidence_kind=EvidenceKind.EXACT_LOCAL,
        theorem_or_algorithm="mds_7_4_2_3",
        constraints=ConstraintFrame(
            grammar_hash="grammar-sha",
            parser_version="1.0",
            codec_version="1.0",
            state_signature_version="cap0-02-v1",
            generation_order="preorder",
            ast_bounds={"max_ast_nodes": 6},
            scope_bounds={"max_live_bindings": 2},
            template_classes=(),
            latent_role="state",
            dimensions=4,
            noise_model="none",
            packing_assumption=None,
        ),
        complete=True,
        witness_or_proof_hash="sha256:abc",
        work_counters={"states": 41},
    )


def _estimated_evidence() -> EstimatedEvidence:
    return EstimatedEvidence(
        evidence_kind=EvidenceKind.ESTIMATED,
        constraints=ConstraintFrame(
            grammar_hash="grammar-sha",
            parser_version="1.0",
            codec_version="1.0",
            state_signature_version="cap0-02-v1",
            generation_order="preorder",
            ast_bounds={"max_ast_nodes": 6},
            scope_bounds={"max_live_bindings": 2},
            template_classes=(),
        ),
        dataset_ids=("ds-1",),
        trace_ids=("trace-1",),
        checkpoint_ids=("ckpt-1",),
        sample_count=100,
        sampling_design="stratified-by-state",
        coverage={"states": 0.95},
        estimator="empirical_mean",
        confidence_interval=(0.12, 0.34),
    )


def test_exact_evidence_requires_witness():
    with pytest.raises(ValueError, match="witness_or_proof_hash"):
        ExactEvidence(
            evidence_kind=EvidenceKind.EXACT_LOCAL,
            theorem_or_algorithm="mds",
            constraints=ConstraintFrame(
                grammar_hash="",
                parser_version="",
                codec_version="",
                state_signature_version="",
                generation_order="",
                ast_bounds={},
                scope_bounds={},
                template_classes=(),
            ),
            complete=True,
            witness_or_proof_hash=None,
            work_counters={},
        )


def test_exact_external_requires_source_uri():
    with pytest.raises(ValueError, match="source_uri"):
        ExactEvidence(
            evidence_kind=EvidenceKind.EXACT_EXTERNAL,
            theorem_or_algorithm="code_table",
            constraints=ConstraintFrame(
                grammar_hash="",
                parser_version="",
                codec_version="",
                state_signature_version="",
                generation_order="",
                ast_bounds={},
                scope_bounds={},
                template_classes=(),
            ),
            complete=True,
            witness_or_proof_hash=None,
            work_counters={},
            source_uri=None,
        )


def test_estimated_evidence_requires_positive_sample_count():
    with pytest.raises(ValueError, match="sample_count"):
        EstimatedEvidence(
            evidence_kind=EvidenceKind.ESTIMATED,
            constraints=ConstraintFrame(
                grammar_hash="",
                parser_version="",
                codec_version="",
                state_signature_version="",
                generation_order="",
                ast_bounds={},
                scope_bounds={},
                template_classes=(),
            ),
            dataset_ids=(),
            trace_ids=(),
            checkpoint_ids=(),
            sample_count=0,
            sampling_design="none",
            coverage={},
            estimator="none",
        )


def test_certificate_json_round_trip():
    cert = ArityCertificate(
        certificate_id="cert-001",
        report_digest="digest-001",
        frame_id="toy-layout/4",
        provenance=ArityProvenance(generated_at="2026-07-17T00:00:00Z"),
        results=(
            ArityResult(
                metric_name="minimized_state_count",
                value=41,
                units="states",
                evidence=_exact_evidence(),
                status="supported",
            ),
        ),
    )
    data = cert.to_dict()
    assert data["certificate_id"] == "cert-001"
    assert data["results"][0]["evidence"]["evidence_kind"] == "exact_local"
    assert json.loads(cert.to_json())["frame_id"] == "toy-layout/4"


def test_certificate_digest_stable():
    cert = ArityCertificate(
        certificate_id="cert-001",
        report_digest="digest-001",
        frame_id="toy-layout/4",
        provenance=ArityProvenance(generated_at="2026-07-17T00:00:00Z"),
        results=(
            ArityResult(
                metric_name="minimized_state_count",
                value=41,
                units="states",
                evidence=_exact_evidence(),
                status="supported",
            ),
        ),
    )
    d1 = certificate_digest(cert)
    d2 = certificate_digest(cert)
    assert d1 == d2
    assert len(d1) == 32


def test_bundle_digest_varies_with_provenance(toy_report):
    bundle1 = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z"
    )
    bundle2 = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z"
    )
    bundle3 = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:01Z"
    )
    assert bundle1.bundle_digest == bundle2.bundle_digest
    assert bundle1.bundle_digest != bundle3.bundle_digest
    assert bundle1.bundle_digest != toy_report.digest


def test_exact_certificate_from_report_with_construction(toy_report):
    bundle = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z"
    )
    cert = bundle.certificate
    assert cert.results[0].evidence.evidence_kind in (
        EvidenceKind.EXACT_LOCAL,
        EvidenceKind.INCOMPLETE,
    )


def test_render_markdown_contains_summary(toy_report):
    bundle = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z"
    )
    md = to_markdown(bundle)
    assert "CAP0-04 Arity Certificate" in md
    assert bundle.certificate.certificate_id in md


def test_render_csv_contains_header_and_row(toy_report):
    bundle = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z"
    )
    csv_text = to_csv(bundle)
    assert "certificate_id" in csv_text
    assert bundle.certificate.certificate_id in csv_text


def test_one_line_summary(toy_report):
    bundle = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z"
    )
    line = one_line_summary(bundle)
    assert line.startswith("cap0-04")
    assert bundle.bundle_digest in line
