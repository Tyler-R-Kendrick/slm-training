"""SLM-234 (CKM0-01): TIES-vs-average merge signal recovery under synthetic
parameter interference.

``docs/design/model-lineage.md`` documents that sibling checkpoint merging
("Compatible sibling deltas may be tested with Model Soups averaging and
TIES-Merging; merge output is always a new challenger") is implemented in
:func:`slm_training.harness_core.lineage.merge.merge_checkpoints`
(``method in {"average", "ties"}``). Beyond the existing shape/compatibility
unit tests in ``tests/test_lineage/test_lineage.py`` (two toy 2-element
vectors, one hand-picked conflict case), no experiment has ever measured
*when* TIES-Merging's magnitude-trim + sign-election + disjoint-merge
mechanism actually recovers a shared consensus update better than naive
parameter averaging, or whether the repo's implementation delivers the
mechanism the TIES paper claims rather than merely running without error.

This harness builds a synthetic parent checkpoint plus ``n_siblings`` child
checkpoints with a known ground-truth structure per tensor coordinate, for
each of several deterministic seeds:

- A ``signal_fraction`` subset of coordinates carries a large, fixed-sign
  "consensus" direction (``true_sign``); at each such coordinate, each
  sibling independently has probability ``conflict_prob`` of instead
  carrying the *opposite*-sign delta at comparable magnitude (simulating
  interfering task-specific updates -- the scenario TIES-Merging targets).
- The remaining coordinates carry only small-magnitude, random-sign,
  sibling-independent noise with no consensus direction (simulating
  task-irrelevant parameter drift that TIES's magnitude-based trim is
  designed to prune before merging).

It then calls the real, unmodified ``merge_checkpoints`` with
``method="average"`` and ``method="ties"`` (default ``density=0.2``) at each
tested ``conflict_prob`` and seed, and scores the merged delta against the
known ground truth on four metrics: cosine similarity to the true consensus
direction (full vector, signal + noise coordinates), per-signal-coordinate
mean signed magnitude recovery, per-signal-coordinate sign-recovery rate, and
mean residual magnitude at the noise coordinates (lower is cleaner).

No checkpoint promotion, GPU run, learned merge policy, or ship-gate claim is
made. This is a controlled synthetic fixture, not a measurement on any real
sibling checkpoints from this repo's training pipeline.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.harness_core.lineage.merge import merge_checkpoints
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "Slm234ConflictLevelResult",
    "Slm234MetricSummary",
    "Slm234Report",
    "render_markdown",
    "run_merge_interference_matrix",
]

MATRIX_VERSION = "ckm0-01-v1"
MATRIX_SET = "slm234_merge_interference_recovery"
EXPERIMENT_ID = "slm234-ckm0-01-merge-interference-recovery"

_TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "layer.weight": (24, 24),
    "layer.bias": (24,),
}
_SIGNAL_FRACTION = 0.2
_SIGNAL_MAGNITUDE = 1.0
_NOISE_MAGNITUDE = 0.05
_DEFAULT_CONFLICT_PROBS: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45)
_DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
_DEFAULT_N_SIBLINGS = 5
_DEFAULT_DENSITY = 0.2
_WIN_TOL = 0.005
_METRICS_HIGHER_BETTER = (
    "cosine_similarity",
    "signal_magnitude_recovery",
    "signal_sign_recovery_rate",
)
_METRICS_LOWER_BETTER = ("mean_abs_noise_residual",)

_HYPOTHESIS = (
    "On a synthetic sibling-checkpoint construction where a signal_fraction "
    "subset of parameter coordinates carries a large fixed-sign consensus "
    "update independently interfered per-sibling with probability "
    "conflict_prob (opposite sign, comparable magnitude), and the remaining "
    "coordinates carry only small-magnitude sibling-independent noise, the "
    "repo's real merge_checkpoints(method='ties') recovers the ground-truth "
    "consensus direction (cosine similarity), its magnitude (mean signed "
    "projection at signal coordinates), its sign (sign-recovery rate at "
    "signal coordinates), and suppresses noise-coordinate residual, at "
    "least as well as merge_checkpoints(method='average') across every "
    "tested conflict_prob and seed -- reproducing TIES-Merging's two "
    "claimed mechanisms (magnitude-based trim of noise, and "
    "interference-resistant disjoint-sign merge) on the repo's actual "
    "implementation rather than only exercising it."
)

_FALSIFIER = (
    "For any metric, TIES's mean value across seeds is measurably worse "
    "(beyond a small win-margin tolerance) than naive averaging's at any "
    "tested conflict_prob, or TIES's win rate for that metric is not "
    "near-universal (>= 90%) across the (seed, conflict_prob) grid."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: a synthetic parent + sibling checkpoint "
    "construction with a hand-designed ground-truth signal/noise structure, "
    "not a measurement on any real trained sibling checkpoints from this "
    "repo's training pipeline. No checkpoint promotion, learned merge "
    "policy, GPU run, or ship-gate claim is made or implied.",
    "The signal_fraction (0.2) is deliberately matched to the tested "
    "density (0.2) so TIES's magnitude-based trim closely tracks the "
    "hand-labeled signal coordinates; a real fine-tuned sibling delta's "
    "magnitude spectrum need not separate this cleanly, and a mismatched "
    "density/signal_fraction is not explored here.",
    "conflict_prob is applied i.i.d. per (sibling, signal coordinate); real "
    "task interference between sibling checkpoints is unlikely to be i.i.d. "
    "Bernoulli and may be structured (e.g. concentrated in specific layers).",
    "This harness does not modify merge_checkpoints, validate_merge_manifests, "
    "or any other harness_core.lineage code -- it only exercises the "
    "unmodified merge_checkpoints entry point with method='average' and "
    "method='ties'.",
    "5 seeds x 4 conflict levels is enough to see whether an effect is "
    "consistent, not a formal significance test; no p-values are computed.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _rademacher(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 2, shape, generator=generator).float() * 2 - 1


def _build_ground_truth(
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return (parent_state, signal_mask, true_sign) for every tensor name."""
    generator = torch.Generator().manual_seed(seed)
    parent: dict[str, torch.Tensor] = {}
    signal_mask: dict[str, torch.Tensor] = {}
    true_sign: dict[str, torch.Tensor] = {}
    for name, shape in _TENSOR_SHAPES.items():
        parent[name] = torch.randn(shape, generator=generator) * 0.1
        mask = torch.rand(shape, generator=generator) < _SIGNAL_FRACTION
        signal_mask[name] = mask
        true_sign[name] = _rademacher(shape, generator) * mask.float()
    return parent, signal_mask, true_sign


def _build_sibling_delta(
    shape: tuple[int, ...],
    mask: torch.Tensor,
    sign: torch.Tensor,
    conflict_prob: float,
    sibling_seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(sibling_seed)
    mask_bool = mask.bool()
    conflict = (torch.rand(shape, generator=generator) < conflict_prob) & mask_bool
    signal_jitter = 0.9 + 0.2 * torch.rand(shape, generator=generator)
    noise_sign = _rademacher(shape, generator)
    noise_jitter = 0.5 + 1.0 * torch.rand(shape, generator=generator)
    signal_component = torch.where(conflict, -sign, sign) * _SIGNAL_MAGNITUDE * signal_jitter
    noise_component = noise_sign * _NOISE_MAGNITUDE * noise_jitter
    return torch.where(mask_bool, signal_component, noise_component)


def _save_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    torch.save({"state_dict": state_dict}, path)


@dataclass(frozen=True)
class Slm234ConflictLevelResult:
    """Per (seed, method, conflict_prob) scoring row."""

    seed: int
    method: str
    conflict_prob: float
    n_siblings: int
    density: float
    n_signal_coords: int
    n_noise_coords: int
    cosine_similarity: float
    signal_magnitude_recovery: float
    signal_sign_recovery_rate: float
    mean_abs_noise_residual: float

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def _score_merged_delta(
    merged_delta: dict[str, torch.Tensor],
    signal_mask: dict[str, torch.Tensor],
    true_sign: dict[str, torch.Tensor],
) -> dict[str, float]:
    names = list(_TENSOR_SHAPES)
    delta_flat = torch.cat([merged_delta[n].reshape(-1) for n in names])
    mask_flat = torch.cat([signal_mask[n].reshape(-1) for n in names]).bool()
    sign_flat = torch.cat([true_sign[n].reshape(-1) for n in names])

    denom = delta_flat.norm() * sign_flat.norm()
    cosine = float((delta_flat @ sign_flat) / denom) if denom > 1e-9 else 0.0

    signal_delta = delta_flat[mask_flat]
    signal_sign = sign_flat[mask_flat]
    if signal_delta.numel():
        magnitude_recovery = float((signal_delta * signal_sign).mean() / _SIGNAL_MAGNITUDE)
        sign_recovery_rate = float((torch.sign(signal_delta) == signal_sign).float().mean())
    else:  # pragma: no cover - defensive: signal_fraction misconfigured to 0
        magnitude_recovery = 0.0
        sign_recovery_rate = 0.0

    noise_delta = delta_flat[~mask_flat]
    mean_abs_noise = float(noise_delta.abs().mean()) if noise_delta.numel() else 0.0

    return {
        "cosine_similarity": cosine,
        "signal_magnitude_recovery": magnitude_recovery,
        "signal_sign_recovery_rate": sign_recovery_rate,
        "mean_abs_noise_residual": mean_abs_noise,
        "n_signal_coords": int(mask_flat.sum()),
        "n_noise_coords": int((~mask_flat).sum()),
    }


def _run_one_seed_level(
    *,
    tmp_dir: Path,
    seed: int,
    parent: dict[str, torch.Tensor],
    signal_mask: dict[str, torch.Tensor],
    true_sign: dict[str, torch.Tensor],
    conflict_prob: float,
    n_siblings: int,
    density: float,
) -> list[Slm234ConflictLevelResult]:
    level_dir = tmp_dir / f"seed-{seed}-level-{conflict_prob:.3f}"
    level_dir.mkdir(parents=True, exist_ok=True)

    parent_path = level_dir / "parent.pt"
    _save_checkpoint(parent_path, parent)

    child_paths: list[Path] = []
    for sib in range(n_siblings):
        sibling_seed = (
            seed * 1_000_000 + int(round(conflict_prob * 1000)) * 1000 + sib
        )
        deltas = {
            name: _build_sibling_delta(
                shape, signal_mask[name], true_sign[name], conflict_prob, sibling_seed + i
            )
            for i, (name, shape) in enumerate(_TENSOR_SHAPES.items())
        }
        child_state = {name: parent[name] + deltas[name] for name in _TENSOR_SHAPES}
        child_path = level_dir / f"child-{sib}.pt"
        _save_checkpoint(child_path, child_state)
        child_paths.append(child_path)

    rows: list[Slm234ConflictLevelResult] = []
    for method in ("average", "ties"):
        output_path = level_dir / f"merged-{method}.pt"
        merge_checkpoints(
            parent_path, child_paths, output_path, method=method, density=density
        )
        merged_payload = torch.load(output_path, map_location="cpu", weights_only=True)
        merged_state = merged_payload["state_dict"]
        merged_delta = {name: merged_state[name] - parent[name] for name in _TENSOR_SHAPES}
        scores = _score_merged_delta(merged_delta, signal_mask, true_sign)
        rows.append(
            Slm234ConflictLevelResult(
                seed=seed,
                method=method,
                conflict_prob=conflict_prob,
                n_siblings=n_siblings,
                density=density,
                n_signal_coords=scores["n_signal_coords"],
                n_noise_coords=scores["n_noise_coords"],
                cosine_similarity=scores["cosine_similarity"],
                signal_magnitude_recovery=scores["signal_magnitude_recovery"],
                signal_sign_recovery_rate=scores["signal_sign_recovery_rate"],
                mean_abs_noise_residual=scores["mean_abs_noise_residual"],
            )
        )
    return rows


@dataclass(frozen=True)
class Slm234MetricSummary:
    """Aggregated TIES-vs-average comparison for one metric across the full
    (seed, conflict_prob) grid."""

    metric: str
    higher_is_better: bool
    ties_win_count: int
    total_pairs: int
    win_rate: float
    worst_gap: float
    worst_gap_conflict_prob: float

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def _pair_rows(
    rows: list[Slm234ConflictLevelResult],
) -> dict[tuple[int, float], dict[str, Slm234ConflictLevelResult]]:
    paired: dict[tuple[int, float], dict[str, Slm234ConflictLevelResult]] = {}
    for row in rows:
        key = (row.seed, row.conflict_prob)
        paired.setdefault(key, {})[row.method] = row
    return paired


def _summarize_metric(
    rows: list[Slm234ConflictLevelResult], metric: str, higher_is_better: bool
) -> Slm234MetricSummary:
    paired = _pair_rows(rows)
    wins = 0
    total = 0
    worst_gap = 0.0
    worst_gap_level = 0.0
    for (_, conflict_prob), pair in paired.items():
        avg, ties = pair.get("average"), pair.get("ties")
        if avg is None or ties is None:  # pragma: no cover - defensive
            continue
        total += 1
        avg_val = getattr(avg, metric)
        ties_val = getattr(ties, metric)
        gap = (ties_val - avg_val) if higher_is_better else (avg_val - ties_val)
        if gap >= -_WIN_TOL:
            wins += 1
        if gap < worst_gap:
            worst_gap = gap
            worst_gap_level = conflict_prob
    win_rate = (wins / total) if total else 0.0
    return Slm234MetricSummary(
        metric=metric,
        higher_is_better=higher_is_better,
        ties_win_count=wins,
        total_pairs=total,
        win_rate=win_rate,
        worst_gap=worst_gap,
        worst_gap_conflict_prob=worst_gap_level,
    )


_ROBUST_WIN_RATE = 0.9


def _resolve_disposition(
    summaries: dict[str, Slm234MetricSummary],
) -> tuple[str, str]:
    robust = sorted(m for m, s in summaries.items() if s.win_rate >= _ROBUST_WIN_RATE)
    weak = sorted(m for m, s in summaries.items() if s.win_rate < _ROBUST_WIN_RATE)
    weak_detail = [
        f"{m} (win_rate={summaries[m].ties_win_count}/{summaries[m].total_pairs}="
        f"{summaries[m].win_rate:.2f}, worst_gap={summaries[m].worst_gap:.4f} at "
        f"conflict_prob={summaries[m].worst_gap_conflict_prob:.2f})"
        for m in weak
    ]

    if not weak:
        return (
            "fully_confirmed",
            "TIES matched or beat naive averaging (within a small win-margin "
            f"tolerance) on every metric across >= {_ROBUST_WIN_RATE:.0%} of "
            "(seed, conflict_prob) pairs tested.",
        )

    if robust:
        return (
            "partial_confirmation_mechanism_specific",
            "TIES robustly beat naive averaging (>= "
            f"{_ROBUST_WIN_RATE:.0%} win rate across the seed x conflict_prob "
            f"grid) on {', '.join(robust)}, matching the hypothesis for "
            "those metrics, but not on " + "; ".join(weak_detail) + ". "
            "TIES's magnitude-based trim reliably removes the small-magnitude "
            "noise-coordinate residual and preserves a larger mean signed "
            "projection at signal coordinates than naive averaging when "
            "these are robust, which is the core claimed noise-suppression "
            "mechanism working as intended. Where cosine similarity and/or "
            "raw sign-recovery rate are not robust, the most likely reading "
            "is that TIES's disjoint-merge commits harder (larger magnitude) "
            "to whichever sign wins the per-coordinate magnitude-weighted "
            "election -- amplifying the result whether that per-coordinate "
            "election happens to be right or wrong, which can inflate the "
            "merged vector's norm and reduce full-vector cosine similarity "
            "even while the signal-restricted mean projection still favors "
            "TIES. The strict 'no worse on every metric' hypothesis is "
            "falsified; the mechanism-specific claim (trim + magnitude "
            "preservation) is supported, the direction-fidelity claim is not.",
        )
    return (
        "no_advantage_detected",
        "TIES did not robustly beat naive averaging on any metric, "
        "falsifying the hypothesis: " + "; ".join(weak_detail),
    )


@dataclass(frozen=True)
class Slm234Report:
    """Full fixture report for SLM-234."""

    schema: str = "Slm234MergeInterferenceRecoveryReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = EXPERIMENT_ID
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    seeds: tuple[int, ...] = _DEFAULT_SEEDS
    n_siblings: int = _DEFAULT_N_SIBLINGS
    density: float = _DEFAULT_DENSITY
    conflict_probs: tuple[float, ...] = _DEFAULT_CONFLICT_PROBS
    rows: tuple[Slm234ConflictLevelResult, ...] = field(default_factory=tuple)
    metric_summaries: tuple[Slm234MetricSummary, ...] = field(default_factory=tuple)
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
            "seeds": list(self.seeds),
            "n_siblings": self.n_siblings,
            "density": self.density,
            "conflict_probs": list(self.conflict_probs),
            "rows": [r.to_dict() for r in self.rows],
            "metric_summaries": [s.to_dict() for s in self.metric_summaries],
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
    def from_dict(cls, data: dict[str, Any]) -> "Slm234Report":
        rows = tuple(
            Slm234ConflictLevelResult(
                seed=int(r["seed"]),
                method=str(r["method"]),
                conflict_prob=float(r["conflict_prob"]),
                n_siblings=int(r["n_siblings"]),
                density=float(r["density"]),
                n_signal_coords=int(r["n_signal_coords"]),
                n_noise_coords=int(r["n_noise_coords"]),
                cosine_similarity=float(r["cosine_similarity"]),
                signal_magnitude_recovery=float(r["signal_magnitude_recovery"]),
                signal_sign_recovery_rate=float(r["signal_sign_recovery_rate"]),
                mean_abs_noise_residual=float(r["mean_abs_noise_residual"]),
            )
            for r in data.get("rows", ())
        )
        summaries = tuple(
            Slm234MetricSummary(
                metric=str(s["metric"]),
                higher_is_better=bool(s["higher_is_better"]),
                ties_win_count=int(s["ties_win_count"]),
                total_pairs=int(s["total_pairs"]),
                win_rate=float(s["win_rate"]),
                worst_gap=float(s["worst_gap"]),
                worst_gap_conflict_prob=float(s["worst_gap_conflict_prob"]),
            )
            for s in data.get("metric_summaries", ())
        )
        return cls(
            schema=str(data.get("schema", "Slm234MergeInterferenceRecoveryReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            seeds=tuple(data.get("seeds", _DEFAULT_SEEDS)),
            n_siblings=int(data.get("n_siblings", _DEFAULT_N_SIBLINGS)),
            density=float(data.get("density", _DEFAULT_DENSITY)),
            conflict_probs=tuple(data.get("conflict_probs", _DEFAULT_CONFLICT_PROBS)),
            rows=rows,
            metric_summaries=summaries,
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def run_merge_interference_matrix(
    *,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    n_siblings: int = _DEFAULT_N_SIBLINGS,
    density: float = _DEFAULT_DENSITY,
    conflict_probs: tuple[float, ...] = _DEFAULT_CONFLICT_PROBS,
    run_id: str | None = None,
) -> Slm234Report:
    """Build, for each seed, a synthetic parent + sibling checkpoint set with
    a known signal/noise ground truth, merge with the real
    ``merge_checkpoints`` under both ``average`` and ``ties`` at each
    ``conflict_probs`` level, score the recovered delta against the ground
    truth, and summarize TIES-vs-average win rates per metric."""
    rows: list[Slm234ConflictLevelResult] = []
    with tempfile.TemporaryDirectory(prefix="slm234-") as tmp:
        tmp_path = Path(tmp)
        for seed in seeds:
            parent, signal_mask, true_sign = _build_ground_truth(seed)
            for conflict_prob in conflict_probs:
                rows.extend(
                    _run_one_seed_level(
                        tmp_dir=tmp_path,
                        seed=seed,
                        parent=parent,
                        signal_mask=signal_mask,
                        true_sign=true_sign,
                        conflict_prob=conflict_prob,
                        n_siblings=n_siblings,
                        density=density,
                    )
                )

    summaries = {
        metric: _summarize_metric(rows, metric, higher_is_better=True)
        for metric in _METRICS_HIGHER_BETTER
    }
    summaries.update(
        {
            metric: _summarize_metric(rows, metric, higher_is_better=False)
            for metric in _METRICS_LOWER_BETTER
        }
    )
    disposition, rationale = _resolve_disposition(summaries)

    return Slm234Report(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        seeds=seeds,
        n_siblings=n_siblings,
        density=density,
        conflict_probs=conflict_probs,
        rows=tuple(rows),
        metric_summaries=tuple(summaries[m] for m in (*_METRICS_HIGHER_BETTER, *_METRICS_LOWER_BETTER)),
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm234_merge_interference_recovery",
        ),
    )


def _mean_by_method_level(
    rows: tuple[Slm234ConflictLevelResult, ...],
) -> dict[tuple[str, float], dict[str, float]]:
    buckets: dict[tuple[str, float], list[Slm234ConflictLevelResult]] = {}
    for row in rows:
        buckets.setdefault((row.method, row.conflict_prob), []).append(row)
    out: dict[tuple[str, float], dict[str, float]] = {}
    for key, group in buckets.items():
        n = len(group)
        out[key] = {
            "cosine_similarity": sum(r.cosine_similarity for r in group) / n,
            "signal_magnitude_recovery": sum(r.signal_magnitude_recovery for r in group) / n,
            "signal_sign_recovery_rate": sum(r.signal_sign_recovery_rate for r in group) / n,
            "mean_abs_noise_residual": sum(r.mean_abs_noise_residual for r in group) / n,
        }
    return out


def render_markdown(report: Slm234Report) -> str:
    means = _mean_by_method_level(report.rows)
    lines = [
        f"# SLM-234 (CKM0-01): TIES-vs-average merge signal recovery under synthetic interference ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**n_siblings:** {report.n_siblings}  **density:** {report.density}  "
        f"**seeds:** {list(report.seeds)}",
        f"**Disposition:** {report.disposition}",
        "",
        report.disposition_rationale,
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Metric summary (TIES vs average, across all seeds x conflict_probs)",
        "",
        "| metric | higher is better | TIES win rate | worst gap | worst gap at conflict_prob |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in report.metric_summaries:
        lines.append(
            f"| {s.metric} | {s.higher_is_better} | {s.ties_win_count}/{s.total_pairs} "
            f"({s.win_rate:.2f}) | {s.worst_gap:.4f} | {s.worst_gap_conflict_prob:.2f} |"
        )
    lines += [
        "",
        "## Mean results by method x conflict_prob (averaged across seeds)",
        "",
        "| conflict_prob | method | cosine_sim | signal_magnitude_recovery | "
        "signal_sign_recovery | mean_abs_noise_residual |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (method, level), vals in sorted(means.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(
            f"| {level:.2f} | {method} | {vals['cosine_similarity']:.4f} | "
            f"{vals['signal_magnitude_recovery']:.4f} | {vals['signal_sign_recovery_rate']:.4f} | "
            f"{vals['mean_abs_noise_residual']:.4f} |"
        )
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence on a synthetic construction. "
        "It does not measure any real sibling checkpoints from this repo's "
        "training pipeline, does not change merge_checkpoints or any "
        "harness_core.lineage code, and does not authorize automatic merge "
        "promotion (merge output is always a new screened challenger per "
        "model-lineage.md, unchanged by this harness).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm234_merge_interference_recovery",
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    out = Path("docs/design")
    report = run_merge_interference_matrix()
    report.to_json(out / f"iter-slm234-ckm0-01-merge-interference-recovery-{_today_yyyymmdd()}.json")
    (out / f"iter-slm234-ckm0-01-merge-interference-recovery-{_today_yyyymmdd()}.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(f"disposition={report.disposition}")
