"""Regression tests for CAP0-04 exact-vs-estimated arity certificates.

Redesigned onto main's canonical arity API. The pure certificate-schema tests
(evidence validation, JSON round-trip, digest stability) are unchanged. The
report-backed tests build the report with main's exact pipeline
(``analyze`` -> ``ExactArityReport``) and a *separately constructed* CAP0-03
coding witness (``verify_code`` over ``build_mds_7_4_2_3``), replacing the
retired ``ArityAnalyzer`` + report-attached ``CodingMetadata``. Only *how* the
report and witness are built changed; each assertion's intent — exact-local vs
estimated evidence, construction provenance, digest stability, renderer output —
is preserved.
"""

from __future__ import annotations

import json

import pytest

from slm_training.dsl.analysis.arity import (
    AnalysisBounds,
    ArityCertificate,
    ArityProvenance,
    ArityResult,
    ConstraintFrame,
    EstimatedEvidence,
    EvidenceKind,
    ExactEvidence,
    analyze,
    build_mds_7_4_2_3,
    certificate_digest,
    exact_certificate_from_report,
    report_view,
    verify_code,
)
from slm_training.dsl.analysis.arity.render import one_line_summary, to_csv, to_markdown

# The committed ``bounded-expr`` fixture certifies minimized_state_count=28 (see
# tests/test_dsl/test_arity_analysis.py). The MDS [4,2,3]_7 construction has 49
# codewords at minimum distance 3, so it robustly codes that 28-state target.
FIXTURE_MINIMIZED_STATES = 28


def _toy_bounds() -> AnalysisBounds:
    return AnalysisBounds(
        max_ast_nodes=6,
        max_live_bindings=2,
        template_classes=("N",),
        result_types=("number",),
    )


@pytest.fixture(scope="module")
def toy_report():
    # Main's exact pipeline replaces the retired ArityAnalyzer("toy-layout", ...).
    return analyze(fixture="bounded-expr", bounds=_toy_bounds(), dimensions=4)


@pytest.fixture
def toy_witness(toy_report):
    """A separately constructed CAP0-03 coding witness for the report's target.

    Verifies the MDS [4,2,3]_7 construction against the report's minimized-state
    count, standing in for the retired report-attached ``CodingMetadata`` (which
    carried ``construction``/``feasible``/``state_count``).
    """
    verification = verify_code(
        build_mds_7_4_2_3(),
        q=7,
        n=4,
        required_size=toy_report.minimized_state_count,
        required_distance=3,
    )
    return {"verification": verification, "construction": "mds_7_4_2_3"}


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


def test_estimated_evidence_is_distinct_from_exact():
    # Exact-vs-estimated: both evidence kinds coexist in the schema and serialise
    # with their own evidence_kind tag.
    exact = _exact_evidence()
    estimated = _estimated_evidence()
    assert exact.evidence_kind == EvidenceKind.EXACT_LOCAL
    assert estimated.evidence_kind == EvidenceKind.ESTIMATED
    assert estimated.estimator == "empirical_mean"


def test_bundle_digest_varies_with_provenance(toy_report, toy_witness):
    bundle1 = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z", **toy_witness
    )
    bundle2 = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z", **toy_witness
    )
    bundle3 = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:01Z", **toy_witness
    )
    assert bundle1.bundle_digest == bundle2.bundle_digest
    assert bundle1.bundle_digest != bundle3.bundle_digest
    # The bundle digest is not the report's own content digest (bridge view).
    assert bundle1.bundle_digest != report_view(toy_report).digest


def test_exact_certificate_from_report_with_construction(toy_report, toy_witness):
    bundle = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z", **toy_witness
    )
    cert = bundle.certificate
    assert cert.results[0].evidence.evidence_kind in (
        EvidenceKind.EXACT_LOCAL,
        EvidenceKind.INCOMPLETE,
    )
    # The verified MDS witness records exact-local, supported evidence and names
    # the construction that produced it (construction provenance).
    evidence = cert.results[0].evidence
    assert evidence.evidence_kind == EvidenceKind.EXACT_LOCAL
    assert cert.results[0].status == "supported"
    assert evidence.theorem_or_algorithm == "mds_7_4_2_3"
    assert evidence.complete is True
    assert evidence.witness_or_proof_hash
    assert cert.results[0].value == FIXTURE_MINIMIZED_STATES
    # frame_id derives from the committed fixture name + capacity dimension.
    assert cert.frame_id == "bounded-expr/4"


def test_render_markdown_contains_summary(toy_report, toy_witness):
    bundle = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z", **toy_witness
    )
    md = to_markdown(bundle)
    assert "CAP0-04 Arity Certificate" in md
    assert bundle.certificate.certificate_id in md


def test_render_csv_contains_header_and_row(toy_report, toy_witness):
    bundle = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z", **toy_witness
    )
    csv_text = to_csv(bundle)
    assert "certificate_id" in csv_text
    assert bundle.certificate.certificate_id in csv_text


def test_one_line_summary(toy_report, toy_witness):
    bundle = exact_certificate_from_report(
        toy_report, generated_at="2026-07-17T00:00:00Z", **toy_witness
    )
    line = one_line_summary(bundle)
    assert line.startswith("cap0-04")
    assert bundle.bundle_digest in line


def test_cli_emits_certificate_without_disturbing_report(tmp_path):
    # The optional CAP0-04 CLI flags emit a certificate bundle + renders while
    # leaving the canonical CAP0-02 report outputs byte-identical (guards the
    # #354/#327 report contract under the new flags).
    from scripts.analyze_grammar_arity import main

    out = tmp_path / "scratch.json"
    durable = tmp_path / "durable.json"
    cert = tmp_path / "cert.json"
    md = tmp_path / "cert.md"
    csv_path = tmp_path / "cert.csv"
    code = main(
        [
            "--fixture", "bounded-expr",
            "--max-ast-nodes", "6",
            "--max-live-bindings", "2",
            "--dimensions", "4",
            "--out", str(out),
            "--durable-out", str(durable),
            "--certificate-out", str(cert),
            "--out-md", str(md),
            "--out-csv", str(csv_path),
        ]
    )
    assert code == 0
    assert out.exists() and durable.exists()
    assert out.read_text(encoding="utf-8") == durable.read_text(encoding="utf-8")

    bundle = json.loads(cert.read_text(encoding="utf-8"))
    assert set(bundle) == {"bundle_digest", "certificate", "report"}
    result0 = bundle["certificate"]["results"][0]
    assert result0["evidence"]["evidence_kind"] == "exact_local"
    assert result0["status"] == "supported"
    assert bundle["certificate"]["frame_id"] == "bounded-expr/4"
    assert bundle["report"]["minimized_state_count"] == FIXTURE_MINIMIZED_STATES
    assert "CAP0-04 Arity Certificate" in md.read_text(encoding="utf-8")
    assert "certificate_id" in csv_path.read_text(encoding="utf-8")
