"""Deterministic Markdown, CSV, and one-line renderers for CAP0-04 certificates."""

from __future__ import annotations

import csv
import io
from typing import Any

from slm_training.dsl.analysis.arity.certificate import (
    ArityCertificateBundle,
    ExactEvidence,
)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def one_line_summary(bundle: ArityCertificateBundle) -> str:
    """Compact machine-readable summary of a certificate bundle."""
    cert = bundle.certificate
    report = bundle.report
    exact_count = sum(
        1 for r in cert.results if isinstance(r.evidence, ExactEvidence)
    )
    estimated_count = len(cert.results) - exact_count
    return (
        f"cap0-04 {cert.frame_id} states={report.minimized_states} "
        f"exact={exact_count} estimated={estimated_count} "
        f"cert={cert.certificate_id} digest={bundle.bundle_digest}"
    )


def to_markdown(bundle: ArityCertificateBundle) -> str:
    """Render a concise Markdown certificate report."""
    cert = bundle.certificate
    report = bundle.report
    lines: list[str] = []
    lines.append(f"# CAP0-04 Arity Certificate: `{cert.frame_id}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Certificate ID:** `{cert.certificate_id}`")
    lines.append(f"- **Report digest:** `{cert.report_digest}`")
    lines.append(f"- **Bundle digest:** `{bundle.bundle_digest}`")
    lines.append(f"- **Total states:** {report.total_states}")
    lines.append(f"- **Minimized states:** {report.minimized_states}")
    lines.append(f"- **Generated at:** {cert.provenance.generated_at}")
    lines.append(f"- **Analyzer version:** {cert.provenance.analyzer_version}")
    lines.append("")
    lines.append("## Claims")
    lines.append("")
    lines.append("| metric | value | units | status | evidence kind | source |")
    lines.append("|---|---|---|---|---|---|")
    for result in cert.results:
        evidence = result.evidence
        source: str
        if isinstance(evidence, ExactEvidence):
            source = evidence.source_uri or evidence.witness_or_proof_hash or "local"
        else:
            source = f"datasets={_fmt(evidence.dataset_ids)} traces={_fmt(evidence.trace_ids)}"
        lines.append(
            f"| {result.metric_name} | {_fmt(result.value)} | {_fmt(result.units)} | "
            f"{result.status} | {evidence.evidence_kind.value} | {source} |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **Source commit:** {cert.provenance.source_commit or 'unknown'}")
    lines.append(f"- **Run ID:** {cert.provenance.run_id or 'none'}")
    lines.append(f"- **Trace ID:** {cert.provenance.trace_id or 'none'}")
    lines.append(f"- **Input hashes:** {_fmt(cert.provenance.input_hashes) or 'none'}")
    lines.append("")
    return "\n".join(lines)


def to_csv(bundle: ArityCertificateBundle) -> str:
    """Render certificate results as a CSV string (header + one row per result)."""
    cert = bundle.certificate
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "certificate_id",
            "frame_id",
            "metric_name",
            "value",
            "units",
            "status",
            "evidence_kind",
            "source",
            "bundle_digest",
        ],
    )
    writer.writeheader()
    for result in cert.results:
        evidence = result.evidence
        if isinstance(evidence, ExactEvidence):
            source = evidence.source_uri or evidence.witness_or_proof_hash or "local"
        else:
            source = f"datasets={_fmt(evidence.dataset_ids)}"
        writer.writerow(
            {
                "certificate_id": cert.certificate_id,
                "frame_id": cert.frame_id,
                "metric_name": result.metric_name,
                "value": _fmt(result.value),
                "units": result.units or "",
                "status": result.status,
                "evidence_kind": evidence.evidence_kind.value,
                "source": source,
                "bundle_digest": bundle.bundle_digest,
            }
        )
    return output.getvalue()
