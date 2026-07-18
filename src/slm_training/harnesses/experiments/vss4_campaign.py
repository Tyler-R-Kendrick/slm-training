"""VSS4-03 (SLM-76): matched verified-scope-solver campaign harness.

Executes the CPU-runnable phases of the VSS4 campaign and marks every frontier
phase/row as blocked with an explicit, reproducible reason. Does not manufacture
model-backed results when required checkpoints or benchmark families are absent.

The report is deterministic where possible: sorted JSON, stable run_id hashed from
the artifact-lock identity + phase configs (not wall-clock timings), and explicit
blocked-reason strings for every deferred frontier item.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.verified_solver_matrix import (
    MATRIX_VERSION,
    run_fixture_matrix,
)
from slm_training.harnesses.solver_bench import run_reference_suite

__all__ = [
    "ArtifactLock",
    "PhaseResult",
    "CampaignReport",
    "run_vss4_campaign",
    "render_markdown",
]

CAMPAIGN_ID = "vss4-03"
HONESTY_NOTE = (
    "CPU/fixture wiring campaign. Phases 0, 1, and 5 run the committed VSS4-01 "
    "benchmark and VSS4-02 fixture matrix (R0/R1) on CPU. Phases 2-4 and 6 and "
    "matrix rows R2-R6 are blocked because the required frontier artifacts "
    "(trainable checkpoints, capsule/surface benchmark families, energy/surface "
    "training CLIs) are not present in this repository state. No model, quality, "
    "or ship claim is made."
)


# --------------------------------------------------------------------------- #
# Artifact lock
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactLock:
    """Immutable snapshot of the environment and required artifacts."""

    source_commit: str
    source_dirty: bool
    python_version: str
    torch_version: str | None
    cuda_available: bool
    device: str
    schema_versions: dict[str, str]
    required_artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_commit": self.source_commit,
            "source_dirty": self.source_dirty,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "device": self.device,
            "schema_versions": dict(self.schema_versions),
            "required_artifacts": dict(self.required_artifacts),
        }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[4]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=Path(__file__).resolve().parents[4]
        ).strip()
        return bool(out)
    except (OSError, subprocess.CalledProcessError):
        return False


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _device() -> str:
    try:
        from slm_training.runtime.accel import best_device

        return str(best_device())
    except Exception:
        return "cpu"


def _artifact_lock() -> ArtifactLock:
    required = {
        "vss4_01_benchmark": {
            "status": "embedded",
            "path": "src/slm_training/harnesses/solver_bench.py",
            "note": "committed closed word-tree fixture (family A)",
        },
        "vss4_02_matrix": {
            "status": "embedded",
            "path": "src/slm_training/harnesses/experiments/verified_solver_matrix.py",
            "version": MATRIX_VERSION,
            "note": "committed R0-R6 schema + CPU fixture runner",
        },
        "twotower_ranker_checkpoint": {
            "status": "missing",
            "reason": "no checkpoint referenced for topology-solver model ranker (R2/R3)",
        },
        "capsule_benchmark_family_c": {
            "status": "missing",
            "reason": "capsule-aware benchmark family C not committed (R3/R4/R5)",
        },
        "cost_to_go_energy_checkpoint": {
            "status": "missing",
            "reason": "energy-ranker training CLI/integration not wired (R4)",
        },
        "surface_benchmark_family_e": {
            "status": "missing",
            "reason": "surface-realization benchmark family E not committed (R5/R6)",
        },
        "surface_ar_checkpoint": {
            "status": "missing",
            "reason": "surface-AR training CLI/integration not wired (R6)",
        },
    }
    return ArtifactLock(
        source_commit=_git_commit(),
        source_dirty=_git_dirty(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        torch_version=_torch_version(),
        cuda_available=_cuda_available(),
        device=_device(),
        schema_versions={
            "vss4_01_benchmark": "v1",
            "vss4_02_matrix": MATRIX_VERSION,
            "solver_supervision": "1",
        },
        required_artifacts=required,
    )


# --------------------------------------------------------------------------- #
# Phase results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhaseResult:
    phase: str
    status: str  # ran | blocked | smoke_only
    blocked_reason: str | None
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "blocked_reason": self.blocked_reason,
            "evidence": dict(self.evidence),
        }


def _phase_blocked(phase: str, reason: str) -> PhaseResult:
    return PhaseResult(phase=phase, status="blocked", blocked_reason=reason, evidence={})


def _run_phase_0() -> PhaseResult:
    lock = _artifact_lock()
    return PhaseResult(
        phase="phase_0_artifact_lock",
        status="ran",
        blocked_reason=None,
        evidence={"lock": lock.to_dict()},
    )


def _run_phase_1() -> PhaseResult:
    suite = run_reference_suite()
    return PhaseResult(
        phase="phase_1_correctness_reference",
        status="ran",
        blocked_reason=None,
        evidence={
            "benchmark_id": "vss4-01/verified_scope_solver/v1",
            "manifest_digest": suite.manifest_digest,
            "passed": suite.passed,
            "report": suite.to_dict(),
        },
    )


def _run_phase_2() -> PhaseResult:
    # The VSS4-01 fixture is a closed benchmark, not on-policy solver decode traces
    # from a train split. Real on-policy collection requires a trainable checkpoint
    # running solver-guided decode over a declared train split.
    return _phase_blocked(
        "phase_2_on_policy_supervision",
        "requires on-policy solver decode traces from a train split and a trainable "
        "checkpoint/config; the VSS4-01 fixture is a closed benchmark, not an "
        "on-policy trace corpus",
    )


def _run_phase_3() -> PhaseResult:
    return _phase_blocked(
        "phase_3_energy_training",
        "requires cost_to_go_energy_checkpoint and an energy-ranker training CLI; "
        "CandidateEnergyScorer exists but no end-to-end training path is wired",
    )


def _run_phase_4() -> PhaseResult:
    return _phase_blocked(
        "phase_4_surface_training",
        "requires surface_ar_checkpoint and a surface-AR realizer training CLI; "
        "SurfaceAutoregressor exists but no end-to-end training path is wired",
    )


def _run_phase_5() -> PhaseResult:
    report = run_fixture_matrix()
    return PhaseResult(
        phase="phase_5_matched_matrix",
        status="ran",
        blocked_reason=None,
        evidence={
            "matrix_set": report.matrix_set,
            "version": report.version,
            "mode": report.mode,
            "run_id": report.run_id,
            "passed": report.passed,
            "gate_failure_count": len(report.gate_failures),
            "rows_ran": [
                r.row_id for r in report.rows if r.capability_status == "run"
            ],
            "rows_blocked": [
                r.row_id
                for r in report.rows
                if r.capability_status in ("blocked", "not_run")
            ],
            "rows": [r.to_dict() for r in report.rows],
            "gates": [g.to_dict() for g in report.gate_results],
        },
    )


def _run_phase_6() -> PhaseResult:
    return _phase_blocked(
        "phase_6_adversarial_ood",
        "requires frontier checkpoints and solver-relevant adversarial/OOD suites "
        "beyond the VSS4-01 closed fixture; no CPU-runnable solver adversarial "
        "suite is committed",
    )


# --------------------------------------------------------------------------- #
# Campaign report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CampaignReport:
    campaign_id: str
    run_id: str
    timestamp: str
    artifact_lock: ArtifactLock
    phases: tuple[PhaseResult, ...]
    blocked_phases: tuple[str, ...]
    honesty_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "artifact_lock": self.artifact_lock.to_dict(),
            "phases": [p.to_dict() for p in self.phases],
            "blocked_phases": list(self.blocked_phases),
            "honesty_note": self.honesty_note,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _run_id(lock: ArtifactLock, phases: tuple[PhaseResult, ...]) -> str:
    """Deterministic id over the campaign config + artifact identity.

    Excludes wall-clock timings and measured timestamps so identical environments
    produce identical run_ids.
    """
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "source_commit": lock.source_commit,
        "source_dirty": lock.source_dirty,
        "schema_versions": lock.schema_versions,
        "required_artifacts": {
            k: {kk: vv for kk, vv in v.items() if kk != "note"}
            for k, v in lock.required_artifacts.items()
        },
        "phases": [
            {
                "phase": p.phase,
                "status": p.status,
                "blocked_reason": p.blocked_reason,
            }
            for p in phases
        ],
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_vss4_campaign() -> CampaignReport:
    phases = (
        _run_phase_0(),
        _run_phase_1(),
        _run_phase_2(),
        _run_phase_3(),
        _run_phase_4(),
        _run_phase_5(),
        _run_phase_6(),
    )
    lock = _artifact_lock()
    blocked = tuple(p.phase for p in phases if p.status == "blocked")
    return CampaignReport(
        campaign_id=CAMPAIGN_ID,
        run_id=_run_id(lock, phases),
        timestamp=_utc_now(),
        artifact_lock=lock,
        phases=phases,
        blocked_phases=blocked,
        honesty_note=HONESTY_NOTE,
    )


def describe_vss4_campaign() -> CampaignReport:
    """Resolve every phase status + artifact lock without running benchmarks."""
    lock = _artifact_lock()
    phases = (
        PhaseResult(
            phase="phase_0_artifact_lock",
            status="ran",
            blocked_reason=None,
            evidence={"lock": lock.to_dict()},
        ),
        _phase_blocked(
            "phase_1_correctness_reference",
            "--describe mode: benchmark not executed",
        ),
        _run_phase_2(),
        _run_phase_3(),
        _run_phase_4(),
        _phase_blocked(
            "phase_5_matched_matrix",
            "--describe mode: fixture matrix not executed",
        ),
        _run_phase_6(),
    )
    blocked = tuple(p.phase for p in phases if p.status == "blocked")
    return CampaignReport(
        campaign_id=CAMPAIGN_ID,
        run_id=_run_id(lock, phases),
        timestamp=_utc_now(),
        artifact_lock=lock,
        phases=phases,
        blocked_phases=blocked,
        honesty_note=HONESTY_NOTE,
    )


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def render_markdown(report: CampaignReport) -> str:
    lines = [
        f"# VSS4-03 verified-scope-solver campaign ({report.campaign_id})",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*Source commit:* `{report.artifact_lock.source_commit}`"
        f"{' (+dirty)' if report.artifact_lock.source_dirty else ''}",
        "",
        "## Honest caveat",
        "",
        report.honesty_note,
        "",
        "## Artifact lock",
        "",
        f"- Python: `{report.artifact_lock.python_version}`",
        f"- Torch: `{report.artifact_lock.torch_version or 'not installed'}`",
        f"- CUDA available: {report.artifact_lock.cuda_available}",
        f"- Device: `{report.artifact_lock.device}`",
        "",
        "### Required artifacts",
        "",
        "| artifact | status | detail |",
        "| --- | --- | --- |",
    ]
    for name, info in sorted(report.artifact_lock.required_artifacts.items()):
        status = info.get("status", "unknown")
        detail = info.get("reason") or info.get("note") or ""
        lines.append(f"| {name} | {status} | {detail} |")

    lines += ["", "## Phases", ""]
    lines.append("| phase | status | blocked reason |")
    lines.append("| --- | --- | --- |")
    for p in report.phases:
        reason = p.blocked_reason or ""
        lines.append(f"| {p.phase} | {p.status} | {reason} |")

    lines += ["", "### Phase 1 — correctness reference", ""]
    p1 = next((p for p in report.phases if p.phase == "phase_1_correctness_reference"), None)
    if p1 and p1.status == "ran":
        ev = p1.evidence
        lines.append(f"- Benchmark: `{ev.get('benchmark_id')}`")
        lines.append(f"- Passed: {ev.get('passed')}")
        lines.append(f"- Manifest digest: `{ev.get('manifest_digest')}`")
    else:
        lines.append(f"Blocked: {p1.blocked_reason if p1 else 'not present'}")

    lines += ["", "### Phase 5 — matched matrix", ""]
    p5 = next((p for p in report.phases if p.phase == "phase_5_matched_matrix"), None)
    if p5 and p5.status == "ran":
        ev = p5.evidence
        lines.append(f"- Matrix set: `{ev.get('matrix_set')}` ({ev.get('version')})")
        lines.append(f"- Rows ran: {', '.join(ev.get('rows_ran', []))}")
        lines.append(f"- Rows blocked: {', '.join(ev.get('rows_blocked', []))}")
        lines.append(f"- Hard gates passed: {ev.get('passed')}")
        lines.append(f"- Gate failures: {ev.get('gate_failure_count')}")
    else:
        lines.append(f"Blocked: {p5.blocked_reason if p5 else 'not present'}")

    if report.blocked_phases:
        lines += ["", "## Blocked frontier scope", ""]
        for bp in report.blocked_phases:
            p = next((x for x in report.phases if x.phase == bp), None)
            if p:
                lines.append(f"- `{bp}`: {p.blocked_reason}")

    lines.append("")
    return "\n".join(lines)
