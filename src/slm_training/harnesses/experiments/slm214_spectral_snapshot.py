"""SLM-214 (NCS0-01): SpectralSnapshotV1 with per-shape randomized-ESD null calibration.

CPU-only fixture/wiring harness. Computes native PyTorch spectral statistics for
weight matrices, classifies them by semantic role, deduplicates tied storage,
and calibrates every fitted observable against same-shape null matrices.

No model is trained, no GPU is required, and no ship-gate claim is made.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "SpectralSnapshotV1",
    "SpectralSnapshotReport",
    "build_toy_model",
    "make_low_rank_matrix",
    "make_pareto_tail_matrix",
    "make_spiked_matrix",
    "null_cache_key",
    "render_markdown",
    "run_spectral_snapshot_fixture",
    "sample_null_summary",
    "spectral_trap_statistics",
]

MATRIX_VERSION = "ncs0-01-v1"
MATRIX_SET = "slm214_spectral_snapshot"
EXPERIMENT_ID = "slm214-spectral-snapshot"

_HYPOTHESIS = (
    "After correcting for exact (rows, cols, dtype, initializer) null matrices, "
    "synthetic controls (random, Pareto-tail, spiked, low-rank) separate cleanly "
    "and real model matrices can be tagged with calibrated z-scores and "
    "randomized-ESD distances."
)

_FALSIFIER = (
    "Native Hill/MLE alpha estimates are unstable across estimator seeds, or "
    "calibrated null scores cannot distinguish the synthetic controls, or tied "
    "storage aliases produce duplicate snapshots."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.",
    "Statistics are native PyTorch SVD; optional WeightWatcher parity is not exercised.",
    "Null calibration uses a small default draw count so the harness stays CPU-only; "
    "publication-quality cells should increase --null-draws and chunk via --max-matrices.",
    "The toy fixture model is not a trained TwoTower; role classification is validated "
    "against the canonical path conventions used by the adapter target map.",
    "Matrices with fewer than 8 singular values are marked ineligible for fitted alpha.",
)

_ROLE_PATTERNS: list[tuple[str, str]] = [
    # Attention Q/K/V/O (self and cross)
    (r"(^|\.)self_attn\.q_proj\b", "self_attn_q"),
    (r"(^|\.)self_attn\.k_proj\b", "self_attn_k"),
    (r"(^|\.)self_attn\.v_proj\b", "self_attn_v"),
    (r"(^|\.)self_attn\.out_proj\b", "self_attn_out"),
    (r"(^|\.)cross_attn\.q_proj\b", "cross_attn_q"),
    (r"(^|\.)cross_attn\.k_proj\b", "cross_attn_k"),
    (r"(^|\.)cross_attn\.v_proj\b", "cross_attn_v"),
    (r"(^|\.)cross_attn\.out_proj\b", "cross_attn_out"),
    # FFN
    (r"(^|\.)mlp\.[02]\b", "mlp"),
    (r"(^|\.)linear1\b", "mlp_in"),
    (r"(^|\.)linear2\b", "mlp_out"),
    (r"(^|\.)ffn\.", "mlp"),
    (r"(^|\.)feed_forward\.", "mlp"),
    # Embeddings / output heads
    (r"(^|\.)tok\b", "token_embedding"),
    (r"(^|\.)token_embed\b", "token_embedding"),
    (r"(^|\.)lm_head\b", "lm_head"),
    (r"(^|\.)action_head\b", "action_head"),
    # Recursive latent projections
    (r"(^|\.)ctx_proj\b", "ctx_proj"),
    (r"(^|\.)z_latent\b", "z_latent"),
    # Low-rank adapters
    (r"(^|\.)lora_A\b", "lora_A"),
    (r"(^|\.)lora_B\b", "lora_B"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _storage_ptr(param: torch.Tensor) -> int:
    try:
        return param.untyped_storage().data_ptr()
    except Exception:  # noqa: BLE001
        return id(param)


def _dtype_family(dtype: torch.dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def null_cache_key(
    rows: int,
    cols: int,
    dtype: torch.dtype,
    initializer: str,
    estimator_version: str = "svd-native-v1",
    policy: str = "shape-initializer",
) -> str:
    """Canonical deterministic key for a null-cache entry."""
    parts = (rows, cols, _dtype_family(dtype), initializer, estimator_version, policy)
    return _sha256(_canonical_json(parts))


@dataclass(frozen=True)
class SpectralSnapshotV1:
    """Per-storage-identity spectral snapshot with mandatory null calibration."""

    snapshot_version: str = "SpectralSnapshotV1"
    estimator_version: str = "svd-native-v1"
    backend_version: str = "pytorch-native-v1"
    matrix_id: str = ""
    canonical_path: str = ""
    semantic_role: str = "unknown"
    storage_identity: str = ""
    tied_aliases: tuple[str, ...] = ()
    shape: tuple[int, int] = (0, 0)
    aspect_ratio: float = 0.0
    dtype: str = ""
    device: str = ""
    trainable: bool = True
    eligibility: str = "eligible"
    ineligibility_reason: str = ""
    singular_values: tuple[float, ...] = ()
    lambda_max: float = 0.0
    frobenius_norm: float = 0.0
    spectral_norm: float = 0.0
    stable_rank: float = 0.0
    effective_rank: float = 0.0
    spectral_entropy: float = 0.0
    hill_alpha: float | None = None
    hill_xmin: float | None = None
    hill_tail_count: int = 0
    null_key: str = ""
    null_draws: int = 0
    null_mean_alpha: float | None = None
    null_sd_alpha: float | None = None
    alpha_z: float | None = None
    randomized_esd_distance: float | None = None
    tie_output_embedding: bool | None = None
    warnings: tuple[str, ...] = ()
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "estimator_version": self.estimator_version,
            "backend_version": self.backend_version,
            "matrix_id": self.matrix_id,
            "canonical_path": self.canonical_path,
            "semantic_role": self.semantic_role,
            "storage_identity": self.storage_identity,
            "tied_aliases": list(self.tied_aliases),
            "shape": list(self.shape),
            "aspect_ratio": self.aspect_ratio,
            "dtype": self.dtype,
            "device": self.device,
            "trainable": self.trainable,
            "eligibility": self.eligibility,
            "ineligibility_reason": self.ineligibility_reason,
            "singular_values": list(self.singular_values),
            "lambda_max": self.lambda_max,
            "frobenius_norm": self.frobenius_norm,
            "spectral_norm": self.spectral_norm,
            "stable_rank": self.stable_rank,
            "effective_rank": self.effective_rank,
            "spectral_entropy": self.spectral_entropy,
            "hill_alpha": self.hill_alpha,
            "hill_xmin": self.hill_xmin,
            "hill_tail_count": self.hill_tail_count,
            "null_key": self.null_key,
            "null_draws": self.null_draws,
            "null_mean_alpha": self.null_mean_alpha,
            "null_sd_alpha": self.null_sd_alpha,
            "alpha_z": self.alpha_z,
            "randomized_esd_distance": self.randomized_esd_distance,
            "tie_output_embedding": self.tie_output_embedding,
            "warnings": list(self.warnings),
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpectralSnapshotV1":
        return cls(
            snapshot_version=str(data.get("snapshot_version", "SpectralSnapshotV1")),
            estimator_version=str(data.get("estimator_version", "svd-native-v1")),
            backend_version=str(data.get("backend_version", "pytorch-native-v1")),
            matrix_id=str(data.get("matrix_id", "")),
            canonical_path=str(data.get("canonical_path", "")),
            semantic_role=str(data.get("semantic_role", "unknown")),
            storage_identity=str(data.get("storage_identity", "")),
            tied_aliases=tuple(data.get("tied_aliases", ())),
            shape=tuple(int(x) for x in data.get("shape", (0, 0))),
            aspect_ratio=float(data.get("aspect_ratio", 0.0)),
            dtype=str(data.get("dtype", "")),
            device=str(data.get("device", "")),
            trainable=bool(data.get("trainable", True)),
            eligibility=str(data.get("eligibility", "eligible")),
            ineligibility_reason=str(data.get("ineligibility_reason", "")),
            singular_values=tuple(float(x) for x in data.get("singular_values", ())),
            lambda_max=float(data.get("lambda_max", 0.0)),
            frobenius_norm=float(data.get("frobenius_norm", 0.0)),
            spectral_norm=float(data.get("spectral_norm", 0.0)),
            stable_rank=float(data.get("stable_rank", 0.0)),
            effective_rank=float(data.get("effective_rank", 0.0)),
            spectral_entropy=float(data.get("spectral_entropy", 0.0)),
            hill_alpha=data.get("hill_alpha"),
            hill_xmin=data.get("hill_xmin"),
            hill_tail_count=int(data.get("hill_tail_count", 0)),
            null_key=str(data.get("null_key", "")),
            null_draws=int(data.get("null_draws", 0)),
            null_mean_alpha=data.get("null_mean_alpha"),
            null_sd_alpha=data.get("null_sd_alpha"),
            alpha_z=data.get("alpha_z"),
            randomized_esd_distance=data.get("randomized_esd_distance"),
            tie_output_embedding=data.get("tie_output_embedding"),
            warnings=tuple(data.get("warnings", ())),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
        )


def _classify_role(path: str) -> str:
    """Classify a canonical module path into a semantic role."""
    for pattern, role in _ROLE_PATTERNS:
        if re.search(pattern, path):
            return role
    if re.search(r"(^|\.)embed", path):
        return "embedding"
    if re.search(r"(_head|\.head)\b", path):
        return "aux_head"
    return "unknown"


def _make_null_matrix(
    rows: int,
    cols: int,
    initializer: str,
    dtype: torch.dtype,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw one null matrix with the requested initializer family."""
    if initializer == "xavier_uniform":
        bound = math.sqrt(6.0 / (rows + cols))
        return torch.empty((rows, cols), dtype=dtype, device=device).uniform_(
            -bound,
            bound,
            generator=generator,
        )
    if initializer == "kaiming_uniform":
        # nn.Linear.reset_parameters uses kaiming_uniform_(a=sqrt(5)), whose
        # bound simplifies to 1/sqrt(fan_in).
        bound = 1.0 / math.sqrt(cols)
        return torch.empty((rows, cols), dtype=dtype, device=device).uniform_(
            -bound,
            bound,
            generator=generator,
        )
    # Default Gaussian/null policy.
    return torch.randn(rows, cols, dtype=dtype, device=device, generator=generator)


def _null_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _seed_from_key(null_key: str) -> int:
    """Deterministic 64-bit seed derived from the null-cache key."""
    return int(_sha256(null_key)[:16], 16)


def _hill_alpha(singular_values: torch.Tensor, tail_fraction: float = 0.5) -> tuple[float, float, int]:
    """Hill estimator over the largest tail_fraction singular values.

    Returns (alpha, xmin, tail_count). Alpha≈shape-dependent for pure random matrices.
    """
    s = singular_values[singular_values > 0]
    if s.numel() < 8:
        raise ValueError("insufficient singular values for Hill estimator")
    k = max(4, int(math.ceil(tail_fraction * s.numel())))
    tail = s[:k]
    xmin = float(tail[-1])
    logs = torch.log(tail / xmin)
    alpha = float(k / logs.sum())
    return alpha, xmin, k


def _svd_stats(singular_values: torch.Tensor) -> dict[str, float]:
    s = singular_values
    frob = float(torch.sqrt((s * s).sum()))
    spec = float(s[0]) if s.numel() else 0.0
    lam_max = float(s[0] * s[0]) if s.numel() else 0.0
    stable = (frob * frob) / (spec * spec) if spec > 0 else 0.0
    probs = (s * s) / (frob * frob + 1e-30)
    entropy = float(-(probs * torch.log(probs + 1e-30)).sum())
    # Effective rank (von-Neumann style): entropy exponentiated.
    eff_rank = float(torch.exp(torch.tensor(entropy))) if entropy > 0 else 1.0
    return {
        "lambda_max": lam_max,
        "frobenius_norm": frob,
        "spectral_norm": spec,
        "stable_rank": stable,
        "effective_rank": eff_rank,
        "spectral_entropy": entropy,
    }


def spectral_trap_statistics(
    matrix: torch.Tensor,
    *,
    null_draws: int = 24,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return scale-invariant outlier statistics using the canonical SVD owner.

    SLM-219 consumes this small projection instead of maintaining a second
    spectral implementation beside :class:`SpectralSnapshotV1`.
    """
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError(
            "trap metrics require a two-dimensional matrix with rank dimension >= 2"
        )
    if null_draws < 3:
        raise ValueError("null_draws must be at least 3")

    observed = torch.linalg.svdvals(matrix.detach().cpu().double())
    if observed.numel() < 2 or float(observed[1]) <= 0:
        raise ValueError(
            "trap metrics require nonzero first and second singular values"
        )
    stats = _svd_stats(observed)
    energy = observed.square()
    outlier = float(energy[0] / energy.sum())

    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = float(matrix.detach().cpu().double().std(unbiased=False)) or 1.0
    null_outliers: list[float] = []
    for _ in range(null_draws):
        null = (
            torch.randn(
                matrix.shape,
                generator=generator,
                dtype=torch.float64,
            )
            * scale
        )
        singular = torch.linalg.svdvals(null)
        null_energy = singular.square()
        null_outliers.append(float(null_energy[0] / null_energy.sum()))
    null_values = torch.tensor(null_outliers, dtype=torch.float64)
    null_mean = float(null_values.mean())
    null_sd = float(null_values.std(unbiased=True))
    return {
        "top_gap_ratio": float(observed[0] / observed[1]),
        "outlier_energy_fraction": outlier,
        "stable_rank": stats["stable_rank"],
        "effective_rank": stats["effective_rank"],
        "spectral_entropy": stats["spectral_entropy"] / math.log(len(observed)),
        "trap_z": (outlier - null_mean) / null_sd if null_sd > 0 else 0.0,
        "null_draws": null_draws,
        "null_mean_outlier_energy": null_mean,
        "null_sd_outlier_energy": null_sd,
    }


def sample_null_summary(
    rows: int,
    cols: int,
    dtype: torch.dtype,
    initializer: str,
    draws: int = 50,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Draw `draws` null matrices and return summary statistics."""
    device = device or torch.device("cpu")
    key = null_cache_key(rows, cols, dtype, initializer)
    gen = _null_generator(_seed_from_key(key))
    alphas: list[float] = []
    sv_lists: list[torch.Tensor] = []
    for _ in range(draws):
        w = _make_null_matrix(rows, cols, initializer, dtype, gen, device)
        s = torch.linalg.svdvals(w)
        sv_lists.append(s)
        try:
            alpha, _, _ = _hill_alpha(s)
            alphas.append(alpha)
        except Exception:  # noqa: BLE001
            pass
    # Mean singular values across draws (pad with zeros if shapes differ? same shape).
    max_len = max(s.numel() for s in sv_lists)
    padded = torch.stack([torch.nn.functional.pad(s, (0, max_len - s.numel())) for s in sv_lists])
    mean_sv = padded.mean(dim=0)
    return {
        "null_key": key,
        "initializer": initializer,
        "draws": draws,
        "mean_alpha": float(sum(alphas) / len(alphas)) if alphas else None,
        "sd_alpha": float((torch.tensor(alphas).std()).item()) if len(alphas) > 1 else None,
        "mean_singular_values": tuple(float(x) for x in mean_sv.tolist()),
    }


def _snapshot_one(
    path: str,
    param: torch.Tensor,
    aliases: list[str],
    null_draws: int,
    initializer_guess: str,
    device: torch.device,
    tie_output_embedding: bool | None = None,
) -> SpectralSnapshotV1:
    """Build one SpectralSnapshotV1 row."""
    import time

    t0 = time.perf_counter()
    rows, cols = param.shape
    shape = (rows, cols)
    aspect = max(rows, cols) / min(rows, cols)
    role = _classify_role(path)
    ptr = _storage_ptr(param)
    storage_identity = f"ptr:{ptr:#x}"

    warnings: list[str] = []
    eligibility = "eligible"
    ineligibility_reason = ""

    if min(rows, cols) < 8:
        eligibility = "ineligible"
        ineligibility_reason = "matrix too small for fitted alpha (<8 singular values)"

    w = param.detach().to(device=device, dtype=param.dtype)
    s = torch.linalg.svdvals(w)
    stats = _svd_stats(s)

    hill_alpha = None
    hill_xmin = None
    hill_tail_count = 0
    null_key = ""
    null_mean_alpha = None
    null_sd_alpha = None
    alpha_z = None
    randomized_esd_distance = None

    if eligibility == "eligible":
        null_summary = sample_null_summary(rows, cols, param.dtype, initializer_guess, draws=null_draws, device=device)
        null_key = null_summary["null_key"]
        null_mean_alpha = null_summary["mean_alpha"]
        null_sd_alpha = null_summary["sd_alpha"]
        try:
            hill_alpha, hill_xmin, hill_tail_count = _hill_alpha(s)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Hill estimator failed: {exc}")
            hill_alpha = None
        if hill_alpha is not None and null_mean_alpha is not None and (null_sd_alpha or 0) > 0:
            alpha_z = (hill_alpha - null_mean_alpha) / null_sd_alpha
        # Randomized-ESD distance: L2 between observed sorted SV and mean null SV, normalized by Frobenius norm.
        mean_sv = torch.tensor(null_summary["mean_singular_values"], dtype=s.dtype, device=s.device)
        n = min(s.numel(), mean_sv.numel())
        dist = float(torch.sqrt(((s[:n] - mean_sv[:n]) ** 2).sum()))
        randomized_esd_distance = dist / (stats["frobenius_norm"] + 1e-30)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return SpectralSnapshotV1(
        matrix_id=path,
        canonical_path=path,
        semantic_role=role,
        storage_identity=storage_identity,
        tied_aliases=tuple(sorted(aliases)),
        shape=shape,
        aspect_ratio=aspect,
        dtype=_dtype_family(param.dtype),
        device=str(param.device),
        trainable=getattr(param, "requires_grad", True),
        eligibility=eligibility,
        ineligibility_reason=ineligibility_reason,
        singular_values=tuple(float(x) for x in s.tolist()),
        lambda_max=stats["lambda_max"],
        frobenius_norm=stats["frobenius_norm"],
        spectral_norm=stats["spectral_norm"],
        stable_rank=stats["stable_rank"],
        effective_rank=stats["effective_rank"],
        spectral_entropy=stats["spectral_entropy"],
        hill_alpha=hill_alpha,
        hill_xmin=hill_xmin,
        hill_tail_count=hill_tail_count,
        null_key=null_key,
        null_draws=null_draws if eligibility == "eligible" else 0,
        null_mean_alpha=null_mean_alpha,
        null_sd_alpha=null_sd_alpha,
        alpha_z=alpha_z,
        randomized_esd_distance=randomized_esd_distance,
        tie_output_embedding=tie_output_embedding,
        warnings=tuple(warnings),
        elapsed_ms=elapsed_ms,
    )


def _tied_groups(model: nn.Module) -> dict[int, list[tuple[str, torch.Tensor]]]:
    """Bucket parameters by untyped storage pointer."""
    groups: dict[int, list[tuple[str, torch.Tensor]]] = {}
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        ptr = _storage_ptr(param)
        groups.setdefault(ptr, []).append((name, param))
    return groups


def _select_representative_name(names: list[str]) -> str:
    """Pick a canonical alias for a tied storage group."""
    # Prefer the shortest, lexicographically first name.
    return sorted(names, key=lambda n: (len(n), n))[0]


def run_spectral_snapshot_fixture(
    model: nn.Module | None = None,
    *,
    null_draws: int = 50,
    roles: list[str] | None = None,
    max_matrices: int | None = None,
    device: str = "cpu",
    initializer_guess: str = "gaussian",
    run_id: str | None = None,
) -> SpectralSnapshotReport:
    """Run the SLM-214 fixture over a model.

    If ``model`` is None, a tiny toy model is used so the harness can run
    without any checkpoint.
    """
    import time

    t0 = time.perf_counter()
    if model is None:
        model = build_toy_model()
    device_obj = torch.device(device)
    groups = _tied_groups(model)

    model_tie_output_embedding = getattr(
        getattr(model, "config", None), "tie_output_embedding", None
    )
    snapshots: list[SpectralSnapshotV1] = []
    for ptr, items in groups.items():
        names = [n for n, _ in items]
        rep_name = _select_representative_name(names)
        rep_param = next(p for n, p in items if n == rep_name)
        if roles is not None and _classify_role(rep_name) not in roles:
            continue
        snapshot = _snapshot_one(
            rep_name,
            rep_param,
            names,
            null_draws=null_draws,
            initializer_guess=initializer_guess,
            device=device_obj,
            tie_output_embedding=model_tie_output_embedding,
        )
        snapshots.append(snapshot)
        if max_matrices is not None and len(snapshots) >= max_matrices:
            break

    total_ms = (time.perf_counter() - t0) * 1000.0
    disposition, rationale = _resolve_disposition(snapshots)
    return SpectralSnapshotReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        snapshots=tuple(snapshots),
        n_matrices=len(snapshots),
        n_eligible=sum(1 for s in snapshots if s.eligibility == "eligible"),
        n_ineligible=sum(1 for s in snapshots if s.eligibility == "ineligible"),
        n_randomized_null=len(snapshots),
        total_elapsed_ms=total_ms,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm214_spectral_snapshot",
        ),
    )


def _resolve_disposition(snapshots: list[SpectralSnapshotV1]) -> tuple[str, str]:
    if not snapshots:
        return "inconclusive", "No matrices were inspected."
    eligible = [s for s in snapshots if s.eligibility == "eligible"]
    if not eligible:
        return "inconclusive", "All matrices were ineligible for fitted alpha."
    n_with_alpha = sum(1 for s in eligible if s.hill_alpha is not None)
    if n_with_alpha < len(eligible):
        return "partial", "Some eligible matrices failed to produce a Hill alpha."
    return "fixture_ok", "All eligible matrices produced native spectral statistics with null calibration."


@dataclass(frozen=True)
class SpectralSnapshotReport:
    """Full fixture report for SLM-214."""

    schema: str = "SpectralSnapshotReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm214-spectral-snapshot"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    snapshots: tuple[SpectralSnapshotV1, ...] = ()
    n_matrices: int = 0
    n_eligible: int = 0
    n_ineligible: int = 0
    n_randomized_null: int = 0
    total_elapsed_ms: float = 0.0
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "n_matrices": self.n_matrices,
            "n_eligible": self.n_eligible,
            "n_ineligible": self.n_ineligible,
            "n_randomized_null": self.n_randomized_null,
            "total_elapsed_ms": self.total_elapsed_ms,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpectralSnapshotReport":
        return cls(
            schema=str(data.get("schema", "SpectralSnapshotReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            snapshots=tuple(SpectralSnapshotV1.from_dict(s) for s in data.get("snapshots", ())),
            n_matrices=int(data.get("n_matrices", 0)),
            n_eligible=int(data.get("n_eligible", 0)),
            n_ineligible=int(data.get("n_ineligible", 0)),
            n_randomized_null=int(data.get("n_randomized_null", 0)),
            total_elapsed_ms=float(data.get("total_elapsed_ms", 0.0)),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


# -----------------------------------------------------------------------------
# Synthetic controls
# -----------------------------------------------------------------------------


def make_pareto_tail_matrix(
    rows: int,
    cols: int,
    *,
    alpha_true: float = 2.5,
    rank: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a matrix whose top singular values follow a Pareto tail.

    The tail exponent is controlled by ``alpha_true``. Returns a full-rank
    matrix built from a random orthogonal factor and a synthetic spectrum.
    """
    device = torch.device("cpu")
    q = torch.linalg.qr(torch.randn(rows, rows, dtype=dtype, device=device))[0][:, :rank]
    r = torch.linalg.qr(torch.randn(cols, cols, dtype=dtype, device=device))[0][:, :rank]
    # Pareto tail: s_i ~ (i+1)^(-1/alpha_true)
    s = torch.tensor(
        [((i + 1) ** (-1.0 / alpha_true)) for i in range(rank)],
        dtype=dtype,
        device=device,
    )
    return q @ torch.diag(s) @ r.T


def make_spiked_matrix(
    rows: int,
    cols: int,
    *,
    spike_count: int = 4,
    bulk_std: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Matrix with a few strong spikes on a Gaussian bulk."""
    device = torch.device("cpu")
    bulk = torch.randn(rows, cols, dtype=dtype, device=device) * bulk_std / math.sqrt(max(rows, cols))
    u = torch.linalg.qr(torch.randn(rows, spike_count, dtype=dtype, device=device))[0]
    v = torch.linalg.qr(torch.randn(cols, spike_count, dtype=dtype, device=device))[0]
    spikes = torch.linspace(10.0, 5.0, spike_count, dtype=dtype, device=device)
    return bulk + u @ torch.diag(spikes) @ v.T


def make_low_rank_matrix(
    rows: int,
    cols: int,
    *,
    rank: int = 3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact rank-``rank`` matrix."""
    device = torch.device("cpu")
    a = torch.randn(rows, rank, dtype=dtype, device=device)
    b = torch.randn(rank, cols, dtype=dtype, device=device)
    return a @ b


# -----------------------------------------------------------------------------
# Toy fixture model
# -----------------------------------------------------------------------------


def build_toy_model(d_model: int = 32, vocab: int = 16) -> nn.Module:
    """Tiny TwoTower-like fixture for wiring tests."""

    class ToyTwoTowerLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embed = nn.Embedding(vocab, d_model)
            self.norm = nn.LayerNorm(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=2,
                dim_feedforward=4 * d_model,
                batch_first=True,
                dtype=torch.float32,
            )
            self.denoiser = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.lm_head = nn.Linear(d_model, vocab, bias=False)
            self.lm_head.weight = self.token_embed.weight  # tied
            self.action_head = nn.Linear(d_model, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
            return self.lm_head(self.token_embed.weight.sum(dim=0, keepdim=True))

    return ToyTwoTowerLike()


# -----------------------------------------------------------------------------
# Markdown rendering
# -----------------------------------------------------------------------------


def render_markdown(report: SpectralSnapshotReport) -> str:
    lines = [
        f"# SLM-214 (NCS0-01): SpectralSnapshotV1 fixture ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Disposition:** {report.disposition} — {report.disposition_rationale}",
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Summary",
        "",
        f"- Matrices inspected: {report.n_matrices}",
        f"- Eligible for alpha: {report.n_eligible}",
        f"- Ineligible: {report.n_ineligible}",
        f"- Total elapsed: {report.total_elapsed_ms:.1f} ms",
        "",
        "## Per-matrix snapshots",
        "",
        "| matrix | role | shape | eligible | hill α | null α (mean±sd) | α z | rand-ESD dist |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in report.snapshots:
        alpha = f"{s.hill_alpha:.3f}" if s.hill_alpha is not None else "—"
        null_alpha = (
            f"{s.null_mean_alpha:.3f}±{s.null_sd_alpha:.3f}"
            if s.null_mean_alpha is not None and s.null_sd_alpha is not None
            else "—"
        )
        az = f"{s.alpha_z:.3f}" if s.alpha_z is not None else "—"
        dist = f"{s.randomized_esd_distance:.4f}" if s.randomized_esd_distance is not None else "—"
        lines.append(
            f"| {s.matrix_id} | {s.semantic_role} | {s.shape[0]}×{s.shape[1]} | {s.eligibility} | "
            f"{alpha} | {null_alpha} | {az} | {dist} |"
        )
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. No checkpoint, GPU train, or ship gate is claimed.",
        "",
    ]
    return "\n".join(lines)
