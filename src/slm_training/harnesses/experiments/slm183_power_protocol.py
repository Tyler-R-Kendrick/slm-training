"""SLM-183: powered, cluster-aware confirmation suite and seed-variance protocol.

Wiring/fixture-only harness.  Generates synthetic binary outcomes with known
seed + target variance, exercises the statistical utilities in
``slm_training.evals.power_protocol``, and produces an honest fixture report.
No model is trained and no GPU is required.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass, field
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from slm_training.autoresearch.experiment_campaign import (
    ExperimentCampaignV1,
)
from slm_training.evals.power_protocol import (
    bootstrap_paired_ci,
    classify_power,
    cluster_bootstrap_ci,
    exact_binomial_interval,
    exact_paired_binary_test,
    intraclass_correlation,
    holm_bonferroni,
    mde_simulation,
    wilson_interval,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ConfirmationSuiteManifest",
    "PowerProtocolReport",
    "analyze_existing_iter",
    "build_default_manifest",
    "build_experiment_campaign",
    "render_markdown",
    "run_variance_fixture",
]

MATRIX_VERSION = "pqr-v1"
MATRIX_SET = "slm183_power_protocol"
EXPERIMENT_ID = "slm183-power-protocol"

_DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class ConfirmationSuiteManifest:
    """Design-time manifest for a powered confirmation suite."""

    suite_role: str
    generator_version: str
    seeds: tuple[int, ...]
    example_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    target_cluster_id: str
    primary_endpoint: str
    primary_contrast: str
    mde: float
    alpha: float
    power: float
    multiplicity_family: str
    version_pins: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["seeds"] = list(self.seeds)
        data["example_ids"] = list(self.example_ids)
        data["content_hashes"] = list(self.content_hashes)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationSuiteManifest":
        return cls(
            suite_role=str(data.get("suite_role", "confirmatory")),
            generator_version=str(data.get("generator_version", "")),
            seeds=tuple(int(s) for s in data.get("seeds", _DEFAULT_SEEDS)),
            example_ids=tuple(str(e) for e in data.get("example_ids", [])),
            content_hashes=tuple(str(h) for h in data.get("content_hashes", [])),
            target_cluster_id=str(data.get("target_cluster_id", "")),
            primary_endpoint=str(data.get("primary_endpoint", "")),
            primary_contrast=str(data.get("primary_contrast", "")),
            mde=float(data.get("mde", 0.0)),
            alpha=float(data.get("alpha", 0.05)),
            power=float(data.get("power", 0.8)),
            multiplicity_family=str(data.get("multiplicity_family", "")),
            version_pins=dict(data.get("version_pins", {})),
        )


@dataclass(frozen=True)
class PowerProtocolReport:
    """Full fixture report for SLM-183."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    manifest: ConfirmationSuiteManifest
    cells: list[dict[str, Any]]
    scenarios: list[dict[str, Any]]
    mde_curve: list[dict[str, float]]
    seed_variance_components: dict[str, float]
    icc: dict[str, float]
    seed_contrast: dict[str, Any]
    conclusions: list[dict[str, Any]]
    dependency_caveats: list[str]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "manifest": self.manifest.to_dict(),
            "cells": list(self.cells),
            "scenarios": list(self.scenarios),
            "mde_curve": list(self.mde_curve),
            "seed_variance_components": dict(self.seed_variance_components),
            "icc": dict(self.icc),
            "seed_contrast": dict(self.seed_contrast),
            "conclusions": list(self.conclusions),
            "dependency_caveats": list(self.dependency_caveats),
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PowerProtocolReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm183_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "A powered confirmation protocol can separate seed variance from "
                "target variance and estimate the minimum detectable effect size.",
            ),
            falsifier=data.get(
                "falsifier",
                "The protocol collapses seed and target variance into a single "
                "pooled estimate and cannot produce a calibrated MDE curve.",
            ),
            manifest=ConfirmationSuiteManifest.from_dict(data.get("manifest", {})),
            cells=list(data.get("cells", [])),
            scenarios=list(data.get("scenarios", [])),
            mde_curve=list(data.get("mde_curve", [])),
            seed_variance_components=dict(data.get("seed_variance_components", {})),
            icc=dict(data.get("icc", {})),
            seed_contrast=dict(data.get("seed_contrast", {})),
            conclusions=list(data.get("conclusions", [])),
            dependency_caveats=list(data.get("dependency_caveats", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _extract_binary_outcome(record: dict[str, Any]) -> int | None:
    """Best-effort binary outcome extraction from an eval record."""
    for key in ("pass", "success", "correct", "binary_outcome"):
        if key in record:
            val = record[key]
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float)):
                return 1 if val else 0
    for key in ("target_score", "fidelity", "parse", "struct"):
        if key in record:
            try:
                return 1 if float(record[key]) >= 1.0 else 0
            except (TypeError, ValueError):
                continue
    return None


def analyze_existing_iter(json_path: str | Path) -> dict[str, Any]:
    """Read an existing iter JSON and extract per-record binary outcomes.

    Computes Wilson intervals and seed variance when multiple seeds are present.
    Accepts common nested shapes (``records``, ``rows``, ``examples``) and
    degrades gracefully when seed information is missing.
    """
    path = Path(json_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    data: dict[str, Any] = raw if isinstance(raw, dict) else {}

    records: list[dict[str, Any]] = []
    if isinstance(raw, list):
        records = raw
    else:
        for key in ("records", "rows", "examples", "items"):
            if isinstance(data.get(key), list):
                records = data[key]
                break

    outcomes: list[int] = []
    seeds: list[int] = []
    example_ids: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        outcome = _extract_binary_outcome(record)
        if outcome is None:
            continue
        outcomes.append(outcome)
        seeds.append(int(record.get("seed", 0)))
        example_ids.append(str(record.get("example_id", record.get("id", ""))))

    total = len(outcomes)
    successes = sum(outcomes)
    wilson = wilson_interval(successes, total)
    exact = exact_binomial_interval(successes, total)

    by_seed: dict[int, list[int]] = {}
    for seed, outcome in zip(seeds, outcomes):
        by_seed.setdefault(seed, []).append(outcome)

    seed_rates: dict[str, dict[str, Any]] = {}
    for seed, vals in sorted(by_seed.items()):
        s = sum(vals)
        n = len(vals)
        seed_rates[str(seed)] = {
            "successes": s,
            "n": n,
            "rate": s / n if n else 0.0,
            "wilson": wilson_interval(s, n),
        }

    seed_variance = 0.0
    if len(by_seed) > 1:
        rates = [sum(vals) / len(vals) for vals in by_seed.values()]
        seed_variance = float(np.var(rates, ddof=1)) if len(rates) > 1 else 0.0

    return {
        "source": str(path),
        "n_records": total,
        "n_successes": successes,
        "success_rate": successes / total if total else 0.0,
        "wilson_interval": wilson,
        "exact_interval": exact,
        "by_seed": seed_rates,
        "seed_variance": seed_variance,
        "example_ids": example_ids,
    }


def build_default_manifest(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
) -> ConfirmationSuiteManifest:
    """Build the default fixture manifest for SLM-183."""
    example_ids = tuple(f"ex_{i:03d}" for i in range(50))
    content_hashes = tuple(f"hash_{i:03d}" for i in range(50))
    return ConfirmationSuiteManifest(
        suite_role="confirmatory_seed_variance_fixture",
        generator_version="slm183-v1",
        seeds=seeds,
        example_ids=example_ids,
        content_hashes=content_hashes,
        target_cluster_id="target",
        primary_endpoint="binary_success_rate",
        primary_contrast="seed_variance_vs_target_variance",
        mde=0.08,
        alpha=0.05,
        power=0.8,
        multiplicity_family="primary_only",
        version_pins={
            "evals.power_protocol": "v1",
            "harness.experiments.slm183_power_protocol": "v1",
        },
    )


def build_experiment_campaign(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
) -> ExperimentCampaignV1:
    """Bridge the legacy fixture manifest to canonical campaign governance."""
    endpoint = "paired_binary_success_delta"

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return ExperimentCampaignV1(
        campaign_id=EXPERIMENT_ID,
        experiment_id=EXPERIMENT_ID,
        hypothesis=(
            "A paired candidate changes the locked primary endpoint relative "
            "to the unchanged control."
        ),
        decision="Keep this fixture wiring-only; never promote from synthetic outcomes.",
        endpoints=(
            {
                "endpoint_id": endpoint,
                "metric": endpoint,
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 0.08,
            },
        ),
        arms=(
            {
                "arm_id": "synthetic_control",
                "role": "control",
                "config_sha256": digest("slm183:synthetic-control"),
            },
            {
                "arm_id": "synthetic_candidate",
                "role": "candidate",
                "config_sha256": digest("slm183:synthetic-candidate"),
            },
        ),
        seeds=seeds,
        budget={
            "max_experiments": 1,
            "max_gpu_hours": 0.0,
            "max_wall_minutes": 2.0,
        },
        stopping_rules=("Stop after every declared fixture seed is reported.",),
        controls=(
            {
                "control_id": "synthetic_control",
                "description": "The unchanged fixture control must remain neutral.",
                "kind": "negative",
            },
        ),
        negative_controls=("synthetic_control",),
        multiplicity_families=(
            {
                "family_id": "primary",
                "hypothesis_ids": (endpoint,),
                "alpha": 0.05,
                "method": "holm",
            },
        ),
        promotion_gates=(
            {
                "gate_id": "fixture_endpoint_reachable",
                "endpoint_id": endpoint,
                "operator": "ge",
                "threshold": 0.08,
            },
        ),
        rollback_gates=(
            {
                "gate_id": "fixture_endpoint_regression",
                "endpoint_id": endpoint,
                "operator": "le",
                "threshold": -0.08,
            },
        ),
        artifact_requirements=(
            {"kind": "version_stamp", "minimum_count": 1},
        ),
        claim_class="wiring",
        source_commit="0" * 40,
        source_dirty=True,
        author="SLM-183 fixture bridge",
        created_at="2026-07-23T00:00:00Z",
    )


def run_variance_fixture(
    n_targets: int = 50,
    paths_per_target: int = 3,
    n_seeds: int = 5,
    *,
    run_id: str = "slm183-power-protocol",
    output_dir: Path | None = None,
    seed: int = 0,
    seeds: tuple[int, ...] | None = None,
) -> PowerProtocolReport:
    """Generate synthetic outcomes with known seed + target variance and report."""
    n_targets = max(1, int(n_targets))
    paths_per_target = max(1, int(paths_per_target))
    n_seeds = max(1, int(n_seeds))
    executed_seeds = seeds if seeds is not None else tuple(range(n_seeds))
    if not executed_seeds:
        raise ValueError("seeds must not be empty")
    if len(executed_seeds) != len(set(executed_seeds)):
        raise ValueError("seeds must be unique")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in executed_seeds):
        raise TypeError("seeds must contain only integer identifiers")
    n_seeds = len(executed_seeds)
    rng = np.random.default_rng(seed)

    base_rate = 0.70
    sigma_target = 0.30
    sigma_seed = 0.15
    true_mde = 0.08
    base_logit = math.log(base_rate / (1.0 - base_rate))

    # Generate mixed-effects binary outcomes.
    cells: list[dict[str, Any]] = []
    all_values: list[float] = []
    all_cluster_ids: list[str] = []
    per_target_means: dict[int, list[float]] = {}
    per_seed_means: dict[int, list[float]] = {}
    paired_control: list[int] = []
    paired_candidate: list[int] = []

    for t in range(n_targets):
        target_effect = float(rng.normal(0.0, sigma_target))
        for s in executed_seeds:
            seed_effect = float(rng.normal(0.0, sigma_seed))
            base = base_logit + target_effect + seed_effect
            draws = rng.random(paths_per_target)
            arm_outcomes: dict[str, list[float]] = {}
            for arm_id, effect in (
                ("synthetic_control", 0.0),
                ("synthetic_candidate", true_mde),
            ):
                probability = 1.0 / (1.0 + math.exp(-(base + effect)))
                outcomes = (draws < probability).astype(float).tolist()
                arm_outcomes[arm_id] = outcomes
                mean_outcome = sum(outcomes) / len(outcomes)
                cell_id = f"target{t:03d}_seed{s}_{arm_id}"
                cells.append(
                    {
                        "cell_id": cell_id,
                        "target_id": t,
                        "seed": s,
                        "arm_id": arm_id,
                        "n": paths_per_target,
                        "successes": int(sum(outcomes)),
                        "mean": float(mean_outcome),
                        "wilson": wilson_interval(
                            int(sum(outcomes)), paths_per_target
                        ),
                        "exact": exact_binomial_interval(
                            int(sum(outcomes)), paths_per_target
                        ),
                    }
                )
            control_outcomes = arm_outcomes["synthetic_control"]
            candidate_outcomes = arm_outcomes["synthetic_candidate"]
            paired_control.extend(int(value) for value in control_outcomes)
            paired_candidate.extend(int(value) for value in candidate_outcomes)
            control_mean = sum(control_outcomes) / len(control_outcomes)
            all_values.extend(control_outcomes)
            all_cluster_ids.extend([f"target_{t}"] * len(control_outcomes))
            per_target_means.setdefault(t, []).append(control_mean)
            per_seed_means.setdefault(s, []).append(control_mean)

    # Aggregate variance components.
    target_means = [np.mean(v) for v in per_target_means.values()]
    seed_means = [np.mean(v) for v in per_seed_means.values()]
    target_variance = float(np.var(target_means, ddof=1)) if len(target_means) > 1 else 0.0
    seed_variance = float(np.var(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0

    # MDE simulation.
    mde_result = mde_simulation(
        base_rate=base_rate,
        sigma_seed=sigma_seed,
        sigma_target=sigma_target,
        n_targets=n_targets,
        paths_per_target=paths_per_target,
        n_seeds=n_seeds,
        alpha=0.05,
        power=0.8,
        n_simulations=100,
        effect_sizes=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
        seed=seed,
    )

    # Cluster bootstrap CI for overall success rate.
    def _mean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        return float(np.mean(arr)) if arr.size else 0.0

    cluster_ci = cluster_bootstrap_ci(
        all_values, all_cluster_ids, _mean, seed=seed, resamples=500
    )

    # ICC.
    icc_result = intraclass_correlation(all_values, all_cluster_ids)

    # Paired candidate/control analysis uses the same target/seed/path draws.
    left = [
        cell["mean"] for cell in cells if cell["arm_id"] == "synthetic_candidate"
    ]
    right = [
        cell["mean"] for cell in cells if cell["arm_id"] == "synthetic_control"
    ]
    seed_contrast = bootstrap_paired_ci(
        left, right, lambda a, b: float(np.mean(a) - np.mean(b)), seed=seed
    )

    paired_test = exact_paired_binary_test(paired_control, paired_candidate)
    holm = holm_bonferroni(
        (("paired_binary_success_delta", paired_test["p_value"]),),
        alpha=0.05,
    )
    n_rejected = sum(1 for entry in holm if entry["rejected"])

    conclusions = [
        {
            "name": "seed_variance_detected",
            "value": seed_variance > target_variance * 0.01,
            "classification": classify_power(
                seed_variance > 0.0, true_mde, seed_variance
            ),
        },
        {
            "name": "mde_achievable_at_08",
            "value": (mde_result.get("mde") or float("inf")) <= true_mde * 1.5,
            "classification": classify_power(
                (mde_result.get("mde") or float("inf")) <= true_mde * 1.5,
                true_mde,
                float(mde_result.get("mde") or 0.0),
            ),
        },
        {
            "name": "cluster_aware_ci_finite",
            "value": math.isfinite(cluster_ci["low"]) and math.isfinite(cluster_ci["high"]),
            "classification": "decidable",
        },
        {
            "name": "holm_rejections",
            "value": n_rejected,
            "classification": "decidable",
        },
        {
            "name": "paired_binary_success_delta",
            "value": paired_test["effect"],
            "classification": classify_power(
                paired_test["p_value"] <= 0.05,
                true_mde,
                abs(paired_test["effect"]),
            ),
        },
    ]

    report = PowerProtocolReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=(
            "A powered confirmation protocol can separate seed variance from "
            "target variance and estimate the minimum detectable effect size."
        ),
        falsifier=(
            "The protocol collapses seed and target variance into a single "
            "pooled estimate and cannot produce a calibrated MDE curve."
        ),
        manifest=build_default_manifest(seeds=tuple(range(n_seeds))),
        cells=cells,
        scenarios=[
            {
                "scenario_id": "mixed_effects_binary",
                "n_targets": n_targets,
                "paths_per_target": paths_per_target,
                "n_seeds": n_seeds,
                "base_rate": base_rate,
                "sigma_target": sigma_target,
                "sigma_seed": sigma_seed,
            }
        ],
        mde_curve=list(mde_result.get("curve", [])),
        seed_variance_components={
            "target_variance": target_variance,
            "seed_variance": seed_variance,
            "pooled_variance": target_variance + seed_variance,
        },
        icc=icc_result,
        seed_contrast=seed_contrast,
        conclusions=conclusions,
        dependency_caveats=[
            "Synthetic outcomes are generated from a mixed-effects logit model; "
            "real eval records may have different correlation structures.",
            "MDE simulation uses a normal approximation (no scipy dependency); "
            "small-n results are exploratory only.",
            "No model is trained; this is wiring evidence for the protocol only.",
            "Cluster bootstrap and ICC assume a single target_cluster_id level; "
            "nested clusters are not modeled here.",
        ],
        version_stamp=build_version_stamp(
            "harness.autoresearch.experiment_campaign",
            "harness.experiments",
            "harness.experiments.slm183_power_protocol",
            "evals.power_protocol",
        ),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm183_power_protocol_report.json")

    return report


def render_markdown(report: PowerProtocolReport) -> str:
    """Render an honest fixture-caveat markdown summary."""
    lines = [
        f"# SLM-183 (PQR): powered cluster-aware confirmation protocol ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable weights "
        "were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Scenarios",
        "",
        "| scenario_id | n_targets | paths_per_target | n_seeds | base_rate | sigma_target | sigma_seed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in report.scenarios:
        lines.append(
            f"| {scenario['scenario_id']} | {scenario['n_targets']} | "
            f"{scenario['paths_per_target']} | {scenario['n_seeds']} | "
            f"{scenario['base_rate']:.2f} | {scenario['sigma_target']:.2f} | "
            f"{scenario['sigma_seed']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Sample cells",
            "",
            "| cell_id | target | seed | arm | n | successes | mean | wilson_low | wilson_high |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cell in report.cells[:8]:
        wilson = cell["wilson"]
        lines.append(
            f"| {cell['cell_id']} | {cell['target_id']} | {cell['seed']} | "
            f"{cell['arm_id']} | {cell['n']} | {cell['successes']} | {cell['mean']:.3f} | "
            f"{wilson['low']:.3f} | {wilson['high']:.3f} |"
        )
    if len(report.cells) > 8:
        lines.append(
            f"| ... | | | | | | | | | ({len(report.cells) - 8} more cells) |"
        )

    lines.extend(
        [
            "",
            "## Seed variance components",
            "",
            f"- target variance: **{report.seed_variance_components['target_variance']:.4f}**",
            f"- seed variance: **{report.seed_variance_components['seed_variance']:.4f}**",
            f"- pooled variance: **{report.seed_variance_components['pooled_variance']:.4f}**",
            "",
            "## ICC (one-way)",
            "",
            f"- ICC: **{report.icc.get('icc', float('nan')):.4f}**",
            f"- clusters: **{report.icc.get('n_clusters', 0)}**",
            "",
            "## MDE curve",
            "",
            "| effect_size | power |",
            "| --- | --- |",
        ]
    )
    for pt in report.mde_curve:
        lines.append(f"| {pt['effect_size']:.3f} | {pt['power']:.3f} |")

    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "| conclusion | value | classification |",
            "| --- | --- | --- |",
        ]
    )
    for conclusion in report.conclusions:
        lines.append(
            f"| {conclusion['name']} | {conclusion['value']} | "
            f"{conclusion['classification']} |"
        )

    lines.extend(
        [
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The power-protocol "
            "utilities, cluster bootstrap, ICC, and MDE simulation are wired and "
            "exercised on synthetic data, but no real eval records or trained model "
            "were used. The protocol remains ``retain_diagnostic`` / "
            "``blocked_pending_real_eval`` until it is run on actual suite results.",
            "",
            "## Honest caveats",
            "",
        ]
    )
    for caveat in report.dependency_caveats:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_flow_power_protocol --mode plan-only",
            "python -m scripts.run_flow_power_protocol --mode fixture",
            "python -m scripts.run_flow_power_protocol --mode analyze-existing --iter-json <path>",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
