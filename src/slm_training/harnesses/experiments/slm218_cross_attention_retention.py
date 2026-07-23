"""SLM-218 cross-attention and parent/child subspace retrospective.

The numerical owners are checkpoint-agnostic and deterministic. The committed
study fails closed when a complete provenance-resolvable checkpoint family is
not available; synthetic controls validate geometry but cannot support H1/H2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from slm_training.harnesses.experiments.semantic_floor_gate import (
    DEFAULT_GATE_PATH as SEMANTIC_FLOOR_GATE_PATH,
    load_semantic_floor_gate,
)
from slm_training.versioning import build_version_stamp, git_commit

__all__ = [
    "CheckpointCompatibilityV1",
    "SubspaceRetentionV1",
    "activation_subspace_alignment",
    "analyze_retention",
    "assert_checkpoint_compatible",
    "build_family_coverage",
    "principal_angles",
    "qk_bilinear_summary",
    "restriction_energy",
    "run_retrospective",
]

MATRIX_SET = "slm218_cross_attention_retention"
MATRIX_VERSION = "ncs1-02-v1"
PREREGISTERED_K = (4, 8, 16, 32)
PREREGISTERED_ENERGY = (0.5, 0.8, 0.9)

CONTEXT_SOURCES = (
    "docs/design/iter-e135-hf-context-control-20260715.json",
    "docs/design/iter-e136-hf-context-32step-20260715.json",
    "docs/design/iter-e138-hf-seed1-8step-20260715.json",
    "docs/design/iter-e139-hf-seed2-8step-20260715.json",
    "docs/design/iter-e176-broad-corpus-20260716.json",
)
RETENTION_SOURCES = (
    "docs/design/iter-e501-e396-e500-warm-start-20260719.json",
    "docs/design/iter-e502-initialization-prior-retention-20260719.json",
    "docs/design/iter-e503-initialized-weight-retention-20260719.json",
    "docs/design/iter-e504-parent-corpus-replay-20260719.json",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _without_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_volatile(child)
            for key, child in value.items()
            if key not in {"stamped_at", "timestamp"}
        }
    if isinstance(value, (list, tuple)):
        return [_without_volatile(child) for child in value]
    return value


def _right_subspace(weight: torch.Tensor, k: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("subspace metrics require two-dimensional matrices")
    _, _, vh = torch.linalg.svd(weight.detach().cpu().double(), full_matrices=False)
    return vh[: min(k, vh.shape[0])].T


def principal_angles(left: torch.Tensor, right: torch.Tensor) -> tuple[float, ...]:
    """Principal angles in radians for two column-orthonormal bases."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("subspaces must be [dimension,k] with equal dimensions")
    if not left.shape[1] or not right.shape[1]:
        return ()
    singular = torch.linalg.svdvals(left.double().T @ right.double()).clamp(0, 1)
    return tuple(float(value) for value in torch.acos(singular))


@dataclass(frozen=True)
class CheckpointCompatibilityV1:
    architecture: str
    tokenizer_sha: str
    config_hash: str
    module_registry_version: str


def assert_checkpoint_compatible(
    parent: CheckpointCompatibilityV1,
    child: CheckpointCompatibilityV1,
) -> None:
    if parent != child:
        fields = [
            name
            for name in parent.__dataclass_fields__
            if getattr(parent, name) != getattr(child, name)
        ]
        raise ValueError(f"checkpoint compatibility mismatch: {', '.join(fields)}")


@dataclass(frozen=True)
class SubspaceRetentionV1:
    k: int
    principal_angles_radians: tuple[float, ...]
    projection_overlap: float
    retained_parent_subspace_energy: float
    update_energy_inside_parent_subspace: float
    update_energy_outside_parent_subspace: float
    rms_drift: float
    parent_singular_values: tuple[float, ...]
    child_singular_values: tuple[float, ...]
    schema: str = "SubspaceRetentionV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_retention(
    parent: torch.Tensor,
    child: torch.Tensor,
    *,
    k: int,
) -> SubspaceRetentionV1:
    if parent.shape != child.shape or parent.ndim != 2:
        raise ValueError("parent and child matrices must have the same 2D shape")
    parent_values = torch.linalg.svdvals(parent.detach().cpu().double())
    child_values = torch.linalg.svdvals(child.detach().cpu().double())
    parent_basis = _right_subspace(parent, k)
    child_basis = _right_subspace(child, k)
    effective_k = min(parent_basis.shape[1], child_basis.shape[1])
    angles = principal_angles(parent_basis[:, :effective_k], child_basis[:, :effective_k])
    overlap = (
        float(torch.linalg.matrix_norm(parent_basis.T @ child_basis).square())
        / effective_k
        if effective_k
        else 0.0
    )
    child_energy = float(child.double().square().sum())
    retained = (
        float((child.double() @ parent_basis).square().sum()) / child_energy
        if child_energy
        else 0.0
    )
    update = child.double() - parent.double()
    projection = parent_basis @ parent_basis.T
    inside = update @ projection
    outside = update - inside
    update_energy = float(update.square().sum())
    return SubspaceRetentionV1(
        k=effective_k,
        principal_angles_radians=angles,
        projection_overlap=overlap,
        retained_parent_subspace_energy=retained,
        update_energy_inside_parent_subspace=(
            float(inside.square().sum()) / update_energy if update_energy else 0.0
        ),
        update_energy_outside_parent_subspace=(
            float(outside.square().sum()) / update_energy if update_energy else 0.0
        ),
        rms_drift=float(torch.sqrt(torch.mean(update.square()))),
        parent_singular_values=tuple(float(value) for value in parent_values),
        child_singular_values=tuple(float(value) for value in child_values),
    )


def activation_subspace_alignment(
    covariance: torch.Tensor,
    weight: torch.Tensor,
    *,
    k: int,
) -> dict[str, Any]:
    """Align context activation eigenvectors with a K/V right-singular subspace."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("activation covariance must be square")
    if covariance.shape[0] != weight.shape[1]:
        raise ValueError("covariance dimension must match K/V in_features")
    _, eigenvectors = torch.linalg.eigh(covariance.detach().cpu().double())
    activation_basis = eigenvectors[:, -min(k, covariance.shape[0]) :]
    weight_basis = _right_subspace(weight, k)
    effective_k = min(activation_basis.shape[1], weight_basis.shape[1])
    angles = principal_angles(
        activation_basis[:, -effective_k:], weight_basis[:, :effective_k]
    )
    return {
        "k": effective_k,
        "principal_angles_radians": list(angles),
        "projection_overlap": (
            float(
                torch.linalg.matrix_norm(
                    activation_basis[:, -effective_k:].T
                    @ weight_basis[:, :effective_k]
                ).square()
            )
            / effective_k
            if effective_k
            else 0.0
        ),
    }


def qk_bilinear_summary(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
) -> dict[str, Any]:
    """Pairwise input-side bilinear map ``Wq.T @ Wk``."""
    if query_weight.ndim != 2 or key_weight.ndim != 2:
        raise ValueError("Q/K weights must be matrices")
    if query_weight.shape[0] != key_weight.shape[0]:
        raise ValueError("Q/K output dimensions must match for dot-product attention")
    bilinear = query_weight.detach().cpu().double().T @ key_weight.detach().cpu().double()
    singular = torch.linalg.svdvals(bilinear)
    return {
        "orientation": "input_query_by_input_context = Wq.T @ Wk",
        "shape": list(bilinear.shape),
        "frobenius_norm": float(torch.linalg.matrix_norm(bilinear)),
        "spectral_norm": float(singular.max()) if singular.numel() else 0.0,
        "effective_rank": int(torch.linalg.matrix_rank(bilinear)),
    }


def restriction_energy(jacobian: torch.Tensor, activation_basis: torch.Tensor) -> float:
    """Compute ``||J V||_F² / ||J||_F²`` on activation-side directions."""
    if jacobian.ndim != 2 or activation_basis.ndim != 2:
        raise ValueError("J and V must be matrices")
    if jacobian.shape[1] != activation_basis.shape[0]:
        raise ValueError("J input dimension must match activation-side V")
    denominator = float(jacobian.double().square().sum())
    if not denominator:
        return 0.0
    return float((jacobian.double() @ activation_basis.double()).square().sum()) / denominator


def _nested(payload: dict[str, Any], *path: str) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _source_coverage(repo_root: Path, source: str, family: str) -> dict[str, Any]:
    path = repo_root / source
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_paths: list[str] = []
    remote_uris: list[str] = []
    training_path = _nested(payload, "training", "checkpoint")
    if isinstance(training_path, str):
        checkpoint_paths.append(training_path)
    parent_uri = _nested(payload, "parent_checkpoint", "remote_uri")
    if isinstance(parent_uri, str):
        remote_uris.append(parent_uri)
    for row in payload.get("matched_runs", []):
        if isinstance(row, dict):
            candidate = row.get("checkpoint")
            if isinstance(candidate, str):
                checkpoint_paths.append(candidate)
            uri = row.get("remote_uri")
            if isinstance(uri, str):
                remote_uris.append(uri)
    local_resolved = [
        candidate
        for candidate in checkpoint_paths
        if (repo_root / candidate).is_file()
    ]
    child_count = len(payload.get("matched_runs", []))
    complete = bool(local_resolved) and (
        family == "context" or len(local_resolved) >= child_count + 1
    )
    return {
        "family": family,
        "source": source,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "declared_checkpoint_paths": checkpoint_paths,
        "resolved_local_checkpoints": local_resolved,
        "remote_parent_uris": remote_uris,
        "declared_children": child_count,
        "complete_family": complete,
        "exclusion_reason": (
            None
            if complete
            else "unresolved local-history checkpoints prevent matrix comparison"
        ),
    }


def build_family_coverage(repo_root: Path) -> dict[str, Any]:
    rows = [
        *(_source_coverage(repo_root, source, "context") for source in CONTEXT_SOURCES),
        *(
            _source_coverage(repo_root, source, "retention")
            for source in RETENTION_SOURCES
        ),
    ]
    return {
        "schema": "CrossAttentionRetentionFamilyManifestV1",
        "sources": rows,
        "complete_context_families": sum(
            row["complete_family"] and row["family"] == "context" for row in rows
        ),
        "complete_retention_families": sum(
            row["complete_family"] and row["family"] == "retention" for row in rows
        ),
        "manifest_hash": _sha(rows),
    }


@dataclass(frozen=True)
class CrossAttentionRetentionReportV1:
    run_id: str
    family_manifest: dict[str, Any]
    synthetic_controls: dict[str, Any]
    h1_verdict: str
    h2_verdict: str
    overall_verdict: str
    rationale: tuple[str, ...]
    ranked_candidates: tuple[dict[str, Any], ...]
    semantic_floor_hash: str
    semantic_floor_verdict: str
    source_commit: str
    version_stamp: dict[str, Any]
    schema: str = "CrossAttentionRetentionReportV1"
    claim_class: str = "diagnostic"
    honesty_mode: str = "zero_training_retrospective"

    @property
    def report_hash(self) -> str:
        return _sha(_without_volatile(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if include_hash:
            payload["report_hash"] = self.report_hash
        return payload


def _synthetic_controls() -> dict[str, Any]:
    parent = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    scaled = parent * 2
    rotation = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rotated = parent @ rotation
    orthogonal = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    covariance = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0]))
    query = torch.eye(4)
    key = torch.diag(torch.tensor([1.0, 0.8, 0.6, 0.4]))
    jacobian = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    basis = _right_subspace(parent, 2)
    return {
        "k_values": list(PREREGISTERED_K),
        "energy_thresholds": list(PREREGISTERED_ENERGY),
        "identical": analyze_retention(parent, parent, k=2).to_dict(),
        "scaled": analyze_retention(parent, scaled, k=2).to_dict(),
        "isospectral_rotated": analyze_retention(parent, rotated, k=2).to_dict(),
        "different_ordering": analyze_retention(parent, orthogonal, k=2).to_dict(),
        "context_alignment": activation_subspace_alignment(
            covariance, key, k=2
        ),
        "qk_bilinear": qk_bilinear_summary(query, key),
        "restriction_energy": restriction_energy(jacobian, basis),
    }


def run_retrospective(repo_root: Path) -> CrossAttentionRetentionReportV1:
    floor = load_semantic_floor_gate(repo_root / SEMANTIC_FLOOR_GATE_PATH)
    coverage = build_family_coverage(repo_root)
    has_context = coverage["complete_context_families"] > 0
    has_retention = coverage["complete_retention_families"] > 0
    return CrossAttentionRetentionReportV1(
        run_id="slm218-cross-attention-retention-20260723",
        family_manifest=coverage,
        synthetic_controls=_synthetic_controls(),
        h1_verdict="inconclusive" if not has_context else "not_evaluated",
        h2_verdict="inconclusive" if not has_retention else "not_evaluated",
        overall_verdict="inconclusive",
        rationale=(
            "all declared context-family checkpoints are unresolved local history",
            "retention families retain a durable parent reference but rejected child checkpoints are local-only and absent",
            "synthetic controls validate geometry but cannot rank historical outcomes",
            f"SemanticFloorGateV1 is {floor.verdict}; semantic interpretation is blocked",
        ),
        ranked_candidates=(),
        semantic_floor_hash=floor.gate_hash,
        semantic_floor_verdict=floor.verdict,
        source_commit=git_commit() or "UNKNOWN",
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm217_functional_spectra",
            "harness.experiments.slm218_cross_attention_retention",
        ),
    )


def render_markdown(report: CrossAttentionRetentionReportV1) -> str:
    manifest = report.family_manifest
    lines = [
        "# SLM-218: cross-attention and subspace-retention retrospective",
        "",
        f"**Overall / H1 / H2:** `{report.overall_verdict}` / "
        f"`{report.h1_verdict}` / `{report.h2_verdict}`",
        "",
        f"**Report hash:** `{report.report_hash}`",
        "",
        f"**Family manifest:** `{manifest['manifest_hash']}`",
        "",
        f"**Semantic floor:** `{report.semantic_floor_hash}` "
        f"(`{report.semantic_floor_verdict}`)",
        "",
        "## Coverage",
        "",
        "| Family | Source | Declared children | Resolved local checkpoints | Complete | Exclusion |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in manifest["sources"]:
        lines.append(
            f"| `{row['family']}` | `{row['source']}` | "
            f"{row['declared_children']} | {len(row['resolved_local_checkpoints'])} "
            f"| `{str(row['complete_family']).lower()}` | "
            f"{row['exclusion_reason'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Verdict rationale",
            "",
            *[f"- {reason}" for reason in report.rationale],
            "",
            "No cross-attention role is ranked and no retention target is nominated. "
            "The compact synthetic controls validate principal-angle, overlap, "
            "inside/outside update energy, Q/K orientation, context alignment, and "
            "activation-side restriction-energy formulas only.",
            "",
            "No new training, checkpoint, semantic evaluation, causal intervention, "
            "optimizer change, promotion, or ship decision was performed.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python "
            "-m scripts.run_cross_attention_retention --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
