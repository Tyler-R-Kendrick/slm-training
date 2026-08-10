"""RSP-009 (SLM-490): cross-experiment EXP-SR disposition report.

Aggregates committed EXP-SR-1..EXP-SR-12 evidence under ``docs/design/`` and
publishes a machine-auditable disposition using
:class:`~slm_training.harnesses.experiments.mechanism_disposition_report.MechanismDispositionReportV1`
(SGS-009 / SLM-456) — no parallel disposition engine.

Follows the SLM-160/SPV4-02 narrative precedent for cross-mechanism closeout
while reusing the SGS-009 record schema and fail-closed validation rules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.exp_sr_catalogue import EXP_SR_FAMILY_SPECS
from slm_training.harnesses.experiments.mechanism_disposition_report import (
    REPORT_SCHEMA,
    Disposition,
    MechanismDispositionRecordV1,
    MechanismDispositionReportV1,
    build_disposition_report,
    build_mechanism_disposition_record,
    disposition_report_integrity_ok,
)
from slm_training.versioning import UNKNOWN, build_version_stamp, git_commit

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "REPORT_WRAPPER_SCHEMA",
    "Rsp009CrossExperimentDispositionV1",
    "build_exp_sr_disposition_records",
    "load_evidence_summary",
    "render_markdown",
    "run_rsp009_disposition_audit",
]

MATRIX_VERSION = "rsp009-v1"
MATRIX_SET = "slm490_rsp009_disposition"
EXPERIMENT_ID = "slm490-rsp-009-disposition"
REPORT_WRAPPER_SCHEMA = "rsp009_cross_experiment_disposition/v1"

_ADOPTED = frozenset({Disposition.ADOPT_PRIMARY, Disposition.ADOPT_OPTIONAL})
_REJECTED_OR_BLOCKED = frozenset({Disposition.REJECT, Disposition.BLOCKED})

# Evidence docs cited by SLM-490 / sibling EXP-SR issues (repo-relative).
_EVIDENCE_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "exp-sr-1": ("docs/design/iter-slm476-sie-002-oracle-localization-20260810.json",),
    "exp-sr-2": ("docs/design/iter-slm482-sie-004-predictor-campaign-20260810.json",),
    "exp-sr-3": (
        "docs/design/iter-slm478-sie-005-blinded-packet-20260810.json",
        "docs/design/iter-slm483-sie-006-calibration-campaign-20260810.json",
    ),
    "exp-sr-4": ("docs/design/iter-slm488-rsp-001-cegis-repair-20260810.json",),
    "exp-sr-5": ("docs/design/iter-slm484-sie-008-voc-controller-20260810.json",),
    "exp-sr-6": ("docs/design/iter-slm489-rsp-002-bounded-search-20260810.json",),
    "exp-sr-7": ("docs/design/iter-slm485-rsp-003-static-summary-20260810.json",),
    "exp-sr-8": ("docs/design/exp-sr-8-egraph-equivalence-experiment.json",),
    "exp-sr-9": ("docs/design/exp-sr-9-macro-library-experiment.json",),
    "exp-sr-10": ("docs/design/iter-slm481-rsp-006-quality-diversity-20260810.json",),
    "exp-sr-11": ("docs/design/iter-slm486-rsp-007-pysr-srbench-20260810.json",),
    "exp-sr-12": ("docs/design/iter-slm487-rsp-008-portability-20260810.json",),
}

_ISSUE_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "exp-sr-1": ("SLM-476", "SIE-002"),
    "exp-sr-2": ("SLM-482", "SIE-004"),
    "exp-sr-3": ("SLM-478", "SLM-483", "SIE-005", "SIE-006"),
    "exp-sr-4": ("SLM-488", "RSP-001"),
    "exp-sr-5": ("SLM-484", "SIE-008"),
    "exp-sr-6": ("SLM-489", "RSP-002"),
    "exp-sr-7": ("SLM-485", "RSP-003"),
    "exp-sr-8": ("EXP-SR-8",),
    "exp-sr-9": ("EXP-SR-9",),
    "exp-sr-10": ("SLM-481", "RSP-006"),
    "exp-sr-11": ("SLM-486", "RSP-007"),
    "exp-sr-12": ("SLM-487", "RSP-008"),
}

_OWNER_BY_FAMILY: dict[str, str] = {
    "exp-sr-1": "src/slm_training/harnesses/experiments/sie002_oracle_localization.py",
    "exp-sr-2": "src/slm_training/harnesses/experiments/sie004_predictor_campaign.py",
    "exp-sr-3": "src/slm_training/harnesses/experiments/sie006_calibration_campaign.py",
    "exp-sr-4": "src/slm_training/harnesses/experiments/rsp001_cegis_repair.py",
    "exp-sr-5": "src/slm_training/harnesses/experiments/sie008_voc_controller.py",
    "exp-sr-6": "src/slm_training/harnesses/experiments/rsp002_bounded_search.py",
    "exp-sr-7": "src/slm_training/harnesses/experiments/rsp003_static_summary.py",
    "exp-sr-8": "src/slm_training/harnesses/experiments/symbolic_egraph.py",
    "exp-sr-9": "src/slm_training/harnesses/experiments/symbolic_macro_library.py",
    "exp-sr-10": "src/slm_training/harnesses/experiments/rsp006_quality_diversity.py",
    "exp-sr-11": "src/slm_training/harnesses/experiments/rsp007_pysr_srbench.py",
    "exp-sr-12": "src/slm_training/harnesses/experiments/rsp008_second_pack_portability.py",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_version_stamp() -> dict[str, Any]:
    try:
        return build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm490_rsp009_disposition",
        )
    except KeyError:
        base = build_version_stamp("harness.experiments")
        base["components"]["harness.experiments.slm490_rsp009_disposition"] = UNKNOWN
        return base


def load_evidence_summary(
    path: str | Path, repo_root: Path | None = None
) -> dict[str, Any]:
    """Load one evidence JSON and return a small summary dict for disposition logic."""
    if repo_root is None:
        repo_root = _repo_root()
    target = repo_root / path
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"present": False, "path": str(path)}
    summary: dict[str, Any] = {
        "present": True,
        "path": str(path),
        "claim_class": data.get("claim_class"),
        "status": data.get("status"),
        "promotion": data.get("promotion"),
        "recommendation": data.get("recommendation"),
        "authorized": data.get("authorized"),
        "falsifier": data.get("falsifier"),
        "killed_by_catalogue_gate": data.get("killed_by_catalogue_gate"),
        "evidence": data.get("evidence"),
        "complete": data.get("complete"),
        "adoption_decision": data.get("adoption_decision"),
        "certified": data.get("certified"),
        "analysis": data.get("analysis"),
    }
    return summary


def _claim_to_evidence_class(summary: dict[str, Any]) -> str:
    if summary.get("evidence") == "external-blocked":
        return "blocked"
    claim = str(summary.get("claim_class") or "fixture")
    if claim == "diagnostic":
        return "scratch"
    if claim in {"fixture", "wiring"}:
        return "fixture"
    if claim in {"measured", "promotable", "ship"}:
        return claim
    return "fixture"


def _default_state_for(disposition: Disposition) -> str:
    if disposition == Disposition.BLOCKED:
        return "blocked"
    if disposition in _ADOPTED:
        return "on"
    return "off"


def _family_spec(family_id: str):
    for spec in EXP_SR_FAMILY_SPECS:
        if spec.family_id == family_id:
            return spec
    raise KeyError(family_id)


def _resolve_disposition(
    *,
    family_id: str,
    pack_scope: str,
    summaries: tuple[dict[str, Any], ...],
) -> tuple[Disposition, str, tuple[str, ...]]:
    """Return disposition, rationale, and follow-on issue hints for one scoped row."""
    missing = [s["path"] for s in summaries if not s.get("present")]
    if missing:
        return (
            Disposition.INCONCLUSIVE,
            f"Evidence missing at audit time: {', '.join(missing)}",
            (),
        )

    primary = summaries[0]
    spec = _family_spec(family_id)

    if family_id == "exp-sr-1":
        return (
            Disposition.RETAIN_DIAGNOSTIC,
            "Fixture oracle localization authorized roles/bindings for later predictor "
            "work; claim_class=fixture — retain as OpenUI diagnostic, never default-on.",
            ("SLM-482",),
        )

    if family_id == "exp-sr-2":
        falsifier = primary.get("falsifier") or {}
        if falsifier.get("falsifier_holds"):
            return (
                Disposition.REJECT,
                "Catalogue falsifier holds: predictor F1 does not beat the frequency/"
                "no-prediction baseline on the held-out slice (fixture evidence).",
                ("SIE-003",),
            )
        return (
            Disposition.INCONCLUSIVE,
            "Predictor campaign evidence present but falsifier outcome unclear.",
            (),
        )

    if family_id == "exp-sr-3":
        calib = summaries[-1] if len(summaries) > 1 else primary
        if calib.get("killed_by_catalogue_gate") or not calib.get("authorized"):
            return (
                Disposition.REVISE_AND_RETEST,
                "SIE-005 packet is fixture-only; SIE-006 diagnostic used synthetic/mock "
                "human labels and failed the catalogue kappa floor — real blinded human "
                "calibration required before any promotion claim.",
                ("SLM-483", "VCE-010"),
            )
        return (
            Disposition.INCONCLUSIVE,
            "Construct-validity evidence incomplete for EXP-SR-3.",
            ("SLM-483",),
        )

    if family_id == "exp-sr-4":
        rec = str(primary.get("recommendation") or "")
        if "adopt" in rec:
            return (
                Disposition.RETAIN_DIAGNOSTIC,
                "Witness CEGIS cleared matched controls on the fixture corpus "
                f"(recommendation={rec!r}), but claim_class=fixture caps disposition "
                "at retain_diagnostic per SGS-009 — no adopt_primary/adopt_optional.",
                ("RSP-001",),
            )
        return (
            Disposition.RETAIN_DIAGNOSTIC,
            "CEGIS repair harness wired; fixture evidence only.",
            (),
        )

    if family_id == "exp-sr-5":
        rec = primary.get("recommendation")
        if isinstance(rec, dict) and rec.get("disposition") == "reject":
            return (
                Disposition.REJECT,
                "Symbolic VoC controller failed the catalogue falsifier versus the "
                "hand-threshold control on fixture decode traces.",
                ("SIE-008",),
            )
        return (
            Disposition.REJECT,
            "VoC controller evidence recommends reject/off on fixture traces.",
            (),
        )

    if family_id == "exp-sr-6":
        rec = str(primary.get("recommendation") or "")
        if "adopt" in rec:
            return (
                Disposition.RETAIN_DIAGNOSTIC,
                "Neural-guided ordering beat left-to-right on the bounded fixture "
                f"(recommendation={rec!r}), but claim_class=fixture — retain default-off.",
                ("RSP-002",),
            )
        return (
            Disposition.RETAIN_DIAGNOSTIC,
            "Bounded-search harness wired with fixture-only evidence.",
            (),
        )

    if family_id == "exp-sr-7":
        rec = str(primary.get("recommendation") or "")
        if rec == "inconclusive_fixture":
            return (
                Disposition.RETAIN_DIAGNOSTIC,
                "Packed semantic summary achieved exact domain parity on the differential "
                "fixture (certified=true) but cold/warm cost evidence is fixture-scale "
                "only — retain OpenUI diagnostic, not production default.",
                ("RSP-003", "PCT-008"),
            )
        return (
            Disposition.INCONCLUSIVE,
            "Static-summary evidence present without a clear fixture recommendation.",
            (),
        )

    if family_id == "exp-sr-8":
        decision = str(primary.get("adoption_decision") or "")
        return (
            Disposition.RETAIN_DIAGNOSTIC,
            "E-graph regions match canonical-hash dedup on the fixture with certified "
            f"rewrites ({decision}); no Pareto win — isolated symbolic-pack experiment only.",
            ("EXP-SR-8",),
        )

    if family_id == "exp-sr-9":
        return (
            Disposition.RETAIN_DIAGNOSTIC,
            "Certified macro library fixture shows size reduction with zero semantics "
            "regressions; claim_class=fixture — diagnostic for symbolic-pack search only.",
            ("EXP-SR-9",),
        )

    if family_id == "exp-sr-10":
        rec = str(primary.get("recommendation") or "")
        if "reject" in rec:
            return (
                Disposition.REJECT,
                "Quality-diversity coverage did not beat matched-compute rejection "
                "sampling on the OpenUI ProgramSpec fixture grid.",
                ("RSP-006",),
            )
        return (
            Disposition.INCONCLUSIVE,
            "Quality-diversity fixture evidence without a clear reject signal.",
            (),
        )

    if family_id == "exp-sr-11":
        if primary.get("evidence") == "external-blocked" or not primary.get("complete"):
            return (
                Disposition.BLOCKED,
                "PySR/SRBench matched arm is external-blocked (no live Julia/PySR env); "
                "gap is not scored as a benchmark loss — blocked, not reject.",
                ("RSP-007", "SRP-011"),
            )
        return (
            Disposition.INCONCLUSIVE,
            "PySR/SRBench evidence incomplete without external-blocked labeling.",
            ("RSP-007",),
        )

    if family_id == "exp-sr-12":
        analysis = (primary.get("analysis") or {}) if isinstance(primary.get("analysis"), dict) else {}
        if pack_scope == "openui":
            return (
                Disposition.RETAIN_DIAGNOSTIC,
                "OpenUI pack passes shared-seam readiness checks in the RSP-008 diagnostic; "
                "reference control only — not a promotion of second-pack portability.",
                ("RSP-008",),
            )
        if pack_scope == "symbolic_regression":
            if analysis.get("falsifier_holds") or not primary.get("certified"):
                return (
                    Disposition.REVISE_AND_RETEST,
                    "Symbolic-regression pack still requires pack-specific seam forks "
                    f"({analysis.get('fork_ids', [])}); zero-fork certification falsified.",
                    ("RSP-008", "SLM-487"),
                )
        return (
            Disposition.REVISE_AND_RETEST,
            f"Second-pack portability not certified for scope={pack_scope!r}.",
            ("RSP-008",),
        )

    return (
        Disposition.INCONCLUSIVE,
        f"No preregistered disposition rule for {family_id} ({pack_scope}).",
        (),
    )


def _packs_for(family_id: str, pack_scope: str) -> tuple[str, ...]:
    if pack_scope == "cross_pack":
        return ("openui", "symbolic_regression")
    if pack_scope == "pack_agnostic":
        return ("openui", "symbolic_regression")
    return (pack_scope,)


def _disposition_specs() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for spec in EXP_SR_FAMILY_SPECS:
        fid = spec.family_id
        if fid == "exp-sr-12":
            rows.append(
                {
                    "family_id": fid,
                    "mechanism_id": f"{fid}:openui",
                    "pack_scope": "openui",
                }
            )
            rows.append(
                {
                    "family_id": fid,
                    "mechanism_id": f"{fid}:symbolic_regression",
                    "pack_scope": "symbolic_regression",
                }
            )
            continue
        pack_scope = {
            "exp-sr-1": "openui",
            "exp-sr-2": "openui",
            "exp-sr-3": "openui",
            "exp-sr-7": "openui",
            "exp-sr-8": "symbolic_regression",
            "exp-sr-9": "symbolic_regression",
            "exp-sr-10": "openui",
            "exp-sr-11": "symbolic_regression",
        }.get(fid, "pack_agnostic")
        rows.append(
            {
                "family_id": fid,
                "mechanism_id": fid,
                "pack_scope": pack_scope,
            }
        )
    return tuple(rows)


def build_exp_sr_disposition_records(
    repo_root: Path | None = None,
) -> tuple[MechanismDispositionRecordV1, ...]:
    """Build SGS-009 disposition records for every EXP-SR family (pack-scoped where required)."""
    if repo_root is None:
        repo_root = _repo_root()

    records: list[MechanismDispositionRecordV1] = []
    for row in _disposition_specs():
        family_id = row["family_id"]
        mechanism_id = row["mechanism_id"]
        pack_scope = row["pack_scope"]
        evidence_paths = _EVIDENCE_BY_FAMILY[family_id]
        summaries = tuple(load_evidence_summary(p, repo_root) for p in evidence_paths)
        disposition, rationale, issue_hints = _resolve_disposition(
            family_id=family_id,
            pack_scope=pack_scope,
            summaries=summaries,
        )
        evidence_class = _claim_to_evidence_class(summaries[0])
        if disposition == Disposition.BLOCKED:
            evidence_class = "blocked"
        spec = _family_spec(family_id)
        issues = tuple(dict.fromkeys(_ISSUE_BY_FAMILY[family_id] + issue_hints))
        records.append(
            build_mechanism_disposition_record(
                mechanism_id=mechanism_id,
                owner_module=_OWNER_BY_FAMILY[family_id],
                owner_version=MATRIX_VERSION,
                default_state=_default_state_for(disposition),
                authority_class="evaluation-only",
                evidence_cutoff=git_commit(),
                evidence_class=evidence_class,
                disposition=disposition,
                rationale=rationale,
                activation_evidence="",
                applicable_packs=_packs_for(family_id, pack_scope),
                matched_controls=True,
                semantic_effect=spec.decision,
                safety_invariants=("I6", "no_production_gate_bypass"),
                external_dependencies=issues,
                issues=issues,
            )
        )
    return tuple(records)


def _follow_up_gaps() -> tuple[str, ...]:
    return (
        "Real blinded human/provider calibration for EXP-SR-3 (SIE-006) — current "
        "evidence is mock/synthetic only.",
        "Isolated PySR/Julia SRBench environment for EXP-SR-11 (RSP-007) — external arm "
        "is external-blocked, not reject.",
        "Eliminate shared-seam pack forks for symbolic_regression before EXP-SR-12 can "
        "certify second-pack portability.",
        "Measured (non-fixture) witness CEGIS and bounded-search confirmation on live "
        "decode/search loops (EXP-SR-4/6).",
        "Production cold/warm confirmation for packed semantic summary beyond the "
        "fixture differential corpus (EXP-SR-7).",
        "Re-open EXP-SR-2 only if a learned predictor beats the frequency baseline on "
        "a real held-out prompt slice.",
    )


@dataclass(frozen=True)
class Rsp009CrossExperimentDispositionV1:
    """RSP-009 wrapper around the SGS-009 mechanism disposition report."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    linear_issue: str
    run_id: str
    status: str
    claim_class: str
    evidence_cutoff_commit: str
    generated_at: str
    mechanism_disposition_report: MechanismDispositionReportV1
    catalogue_families: tuple[str, ...]
    cross_pack_summary: str
    follow_up_gaps: tuple[str, ...]
    champion_pointer_ids: tuple[str, ...]
    rejected_or_blocked_ids: tuple[str, ...]
    version_stamp: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        report_dict = self.mechanism_disposition_report.to_dict()
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "linear_issue": self.linear_issue,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "evidence_cutoff_commit": self.evidence_cutoff_commit,
            "generated_at": self.generated_at,
            "mechanism_disposition_report_schema": REPORT_SCHEMA,
            "mechanism_disposition_report": report_dict,
            "catalogue_families": list(self.catalogue_families),
            "cross_pack_summary": self.cross_pack_summary,
            "follow_up_gaps": list(self.follow_up_gaps),
            "champion_pointer_ids": list(self.champion_pointer_ids),
            "rejected_or_blocked_ids": list(self.rejected_or_blocked_ids),
            "version_stamp": self.version_stamp,
        }


def run_rsp009_disposition_audit(
    *,
    run_id: str = "slm490_rsp009_disposition",
    status: str = "fixture",
    repo_root: Path | None = None,
) -> Rsp009CrossExperimentDispositionV1:
    """Build the cross-experiment disposition report; fails closed on integrity."""
    if repo_root is None:
        repo_root = _repo_root()

    records = build_exp_sr_disposition_records(repo_root=repo_root)
    report = build_disposition_report(records)
    if not disposition_report_integrity_ok(report):
        raise ValueError("mechanism disposition report integrity check failed")

    adopted = [r for r in records if r.disposition in _ADOPTED]
    if adopted:
        raise ValueError(
            "champion/default adoption forbidden: "
            + ", ".join(r.mechanism_id for r in adopted)
        )

    rejected_or_blocked = tuple(
        r.mechanism_id for r in records if r.disposition in _REJECTED_OR_BLOCKED
    )
    cross_pack_summary = (
        "RSP-009 audited all twelve EXP-SR catalogue families (exp-sr-1..12) using "
        "committed sibling evidence. OpenUI-scoped rows (localization, predictor, "
        "calibration, static summary, QD) are separated from symbolic-pack rows "
        "(e-graph, macro library, PySR) and from the dual-scope EXP-SR-12 portability "
        "certification. No row satisfies adopt_primary or adopt_optional under SGS-009 "
        "fail-closed rules on fixture/scratch/blocked evidence — champion pointers stay empty."
    )

    return Rsp009CrossExperimentDispositionV1(
        schema=REPORT_WRAPPER_SCHEMA,
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        linear_issue="SLM-490",
        run_id=run_id,
        status=status,
        claim_class="wiring",
        evidence_cutoff_commit=git_commit(),
        generated_at=_now(),
        mechanism_disposition_report=report,
        catalogue_families=tuple(spec.family_id for spec in EXP_SR_FAMILY_SPECS),
        cross_pack_summary=cross_pack_summary,
        follow_up_gaps=_follow_up_gaps(),
        champion_pointer_ids=(),
        rejected_or_blocked_ids=rejected_or_blocked,
        version_stamp=_build_version_stamp(),
    )


def render_markdown(report: Rsp009CrossExperimentDispositionV1) -> str:
    """Render the durable markdown closeout for RSP-009."""
    lines = [
        f"# SLM-490 (RSP-009): Cross-experiment EXP-SR disposition ({report.run_id})",
        "",
        f"**Schema:** `{report.schema}` / `{REPORT_SCHEMA}`",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        "",
        f"**Status:** {report.status} — disposition audit only; no ship-gate bypass.",
        "",
        f"**Evidence cutoff commit:** `{report.evidence_cutoff_commit}`",
        "",
        f"**Generated at:** {report.generated_at}",
        "",
        "## Executive finding",
        "",
        report.cross_pack_summary,
        "",
        "## Disposition table (EXP-SR-1..12)",
        "",
        "| Family | Pack scope | Disposition | Default | Evidence class | Rationale |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in report.mechanism_disposition_report.records:
        packs = ", ".join(record.applicable_packs) or "—"
        rationale = record.rationale.replace("|", "\\|")
        lines.append(
            f"| {record.mechanism_id} | {packs} | {record.disposition.value} | "
            f"{record.default_state} | {record.evidence_class} | {rationale} |"
        )

    lines.extend(
        [
            "",
            "## Rejected or blocked",
            "",
        ]
    )
    if report.rejected_or_blocked_ids:
        for mid in report.rejected_or_blocked_ids:
            lines.append(f"- `{mid}`")
    else:
        lines.append("- _None._")

    lines.extend(["", "## Follow-up gaps", ""])
    for gap in report.follow_up_gaps:
        lines.append(f"- {gap}")

    lines.extend(
        [
            "",
            "## Champion / default pointers",
            "",
            "_Empty — no adopt_primary or adopt_optional rows; rejected/blocked cannot enter "
            "champion/default pointers._",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_rsp009_disposition --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
