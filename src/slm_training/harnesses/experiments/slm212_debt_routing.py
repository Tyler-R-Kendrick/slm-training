"""SLM-212 (SDE5-05): constraint-debt routing over decode paths wiring/fixture harness.

Compares fixed MaskGIT, fixed constrained LTR, fixed ASAp, a static debt router, a
calibrated debt router, a signal-permuted control, and an oracle ceiling on a purely
synthetic signal/outcome corpus.  No model, checkpoint, or GPU is used.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.debt_router import (
    ROUTES,
    CalibratedDebtRouter,
    DebtRoutingPolicy,
    OracleRouter,
    build_calibrator_artifact,
    decide_route,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ARM_NAMES",
    "DebtRoutingExample",
    "DebtRoutingArmResult",
    "DebtRoutingMatrixManifest",
    "build_synthetic_routing_examples",
    "build_matrix_manifest",
    "run_fixture_matrix",
    "render_markdown",
    "validate_manifest",
]

MATRIX_VERSION = "sde5-05-v1"
MATRIX_SET = "slm212_debt_routing"
EXPERIMENT_ID = "slm212-debt-routing"

ARM_NAMES = (
    "fixed_maskgit",
    "fixed_ltr",
    "fixed_asap",
    "static_debt_router",
    "calibrated_debt_router",
    "signal_permuted_router",
    "oracle_router_ceiling",
)

_BEST_ROUTE_SIGNAL_MEAN = {
    "maskgit": 0.4,
    "ltr": 3.0,
    "asap": 1.2,
}

_ROUTE_VERIFIER_COST = {
    "maskgit": 1.0,
    "asap": 1.5,
    "ltr": 3.0,
}

_DECISION_KINDS = (
    "constraint_shadow",
    "component_choice",
    "argument_value",
    "slot_binding",
    "root_closure",
    "array_insert",
)

_HYPOTHESIS = (
    "A deterministic constraint-debt router can choose the cheaper MaskGIT path when "
    "legal-mass debt is low and the stricter constrained-LTR path when debt is high, "
    "improving the quality/latency frontier over fixed policies at matched verifier "
    "budgets."
)

_FALSIFIER = (
    "The static or calibrated router does not match or exceed the better fixed policy "
    "after budget matching; the signal-permuted control achieves the same regret; debt "
    "is not calibrated enough to choose the winning decode path; or gains require "
    "unequal verifier/forward budgets."
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _clamp(value: float, low: float = 0.0, high: float = float("inf")) -> float:
    return max(low, min(value, high))


@dataclass(frozen=True)
class DebtRoutingExample:
    """One synthetic decode state with a signal, true best arm, and outcomes."""

    example_id: str
    signal_value: float
    true_best_route: str
    outcome_scores: dict[str, float]
    decision_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "signal_value": self.signal_value,
            "true_best_route": self.true_best_route,
            "outcome_scores": dict(self.outcome_scores),
            "decision_kind": self.decision_kind,
        }


@dataclass(frozen=True)
class DebtRoutingArmResult:
    """Aggregate result for one routing arm."""

    arm_name: str
    route_counts: dict[str, int]
    route_by_kind: dict[str, dict[str, int]]
    accuracy: float
    mean_outcome: float
    mean_regret: float
    total_verifier_cost: float
    budget_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "route_counts": dict(self.route_counts),
            "route_by_kind": {k: dict(v) for k, v in self.route_by_kind.items()},
            "accuracy": self.accuracy,
            "mean_outcome": self.mean_outcome,
            "mean_regret": self.mean_regret,
            "total_verifier_cost": self.total_verifier_cost,
            "budget_mode": self.budget_mode,
        }


@dataclass(frozen=True)
class DebtRoutingMatrixManifest:
    """Full fixture manifest for SLM-212."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    arms: tuple[DebtRoutingArmResult, ...]
    n_examples: int
    signal_name: str
    threshold_high: float
    threshold_low: float | None
    hysteresis: int
    budget_mode: str
    lineage: dict[str, Any]
    version_stamp: dict[str, Any]
    timestamp: str
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = ()

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
            "arms": [arm.to_dict() for arm in self.arms],
            "n_examples": self.n_examples,
            "signal_name": self.signal_name,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            "hysteresis": self.hysteresis,
            "budget_mode": self.budget_mode,
            "lineage": dict(self.lineage),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebtRoutingMatrixManifest":
        fields = set(cls.__dataclass_fields__)
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        return cls(
            schema=str(data.get("schema", "DebtRoutingMatrixManifest")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", f"{EXPERIMENT_ID}-fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            arms=tuple(
                DebtRoutingArmResult(
                    arm_name=str(a["arm_name"]),
                    route_counts=dict(a.get("route_counts", {})),
                    route_by_kind={
                        k: dict(v) for k, v in a.get("route_by_kind", {}).items()
                    },
                    accuracy=float(a["accuracy"]),
                    mean_outcome=float(a["mean_outcome"]),
                    mean_regret=float(a["mean_regret"]),
                    total_verifier_cost=float(a["total_verifier_cost"]),
                    budget_mode=str(a["budget_mode"]),
                )
                for a in data.get("arms", ())
            ),
            n_examples=int(data["n_examples"]),
            signal_name=str(data["signal_name"]),
            threshold_high=float(data["threshold_high"]),
            threshold_low=(
                float(v) if (v := data.get("threshold_low")) is not None else None
            ),
            hysteresis=int(data["hysteresis"]),
            budget_mode=str(data["budget_mode"]),
            lineage=dict(data.get("lineage", {})),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", ())),
        )


def build_synthetic_routing_examples(
    n_examples: int = 200,
    seed: int = 0,
    signal_name: str = "D_legal",
) -> tuple[DebtRoutingExample, ...]:
    """Generate deterministic synthetic signal/outcome rows."""
    rng = random.Random(seed)
    examples: list[DebtRoutingExample] = []
    for i in range(n_examples):
        decision_kind = rng.choice(_DECISION_KINDS)
        true_best = rng.choices(
            list(_BEST_ROUTE_SIGNAL_MEAN.keys()),
            weights=[0.45, 0.35, 0.20],
        )[0]
        mean = _BEST_ROUTE_SIGNAL_MEAN[true_best]
        signal_value = _clamp(mean + rng.gauss(0.0, 0.6), low=0.0)

        # Outcome scores: true-best is highest, others trail by a noisy margin.
        scores: dict[str, float] = {}
        for route in ROUTES:
            if route == true_best:
                scores[route] = _clamp(rng.uniform(0.70, 0.95))
            else:
                scores[route] = _clamp(
                    scores.get(true_best, 0.8) - rng.uniform(0.10, 0.35)
                )
        # Re-normalize so the true-best is strictly highest.
        best_score = max(scores.values())
        scores[true_best] = max(scores[true_best], best_score + 0.01)

        example_id = f"ex_{i:04d}_{decision_kind}_{true_best}"
        examples.append(
            DebtRoutingExample(
                example_id=example_id,
                signal_value=signal_value,
                true_best_route=true_best,
                outcome_scores=scores,
                decision_kind=decision_kind,
            )
        )
    return tuple(examples)


def _compute_arm_result(
    arm_name: str,
    chosen_routes: tuple[str, ...],
    examples: tuple[DebtRoutingExample, ...],
    budget_mode: str,
) -> DebtRoutingArmResult:
    """Aggregate accuracy, regret, and cost for one arm."""
    correct = 0
    total_outcome = 0.0
    total_regret = 0.0
    total_cost = 0.0
    route_counts: Counter[str] = Counter()
    route_by_kind: dict[str, Counter[str]] = {}

    for example, route in zip(examples, chosen_routes):
        route_counts[route] += 1
        route_by_kind.setdefault(example.decision_kind, Counter())[route] += 1
        if route == example.true_best_route:
            correct += 1
        total_outcome += example.outcome_scores.get(route, 0.0)
        best_score = max(example.outcome_scores.values())
        total_regret += best_score - example.outcome_scores.get(route, 0.0)
        total_cost += _ROUTE_VERIFIER_COST.get(route, 1.0)

    n = max(1, len(examples))
    return DebtRoutingArmResult(
        arm_name=arm_name,
        route_counts=dict(route_counts),
        route_by_kind={k: dict(v) for k, v in route_by_kind.items()},
        accuracy=correct / n,
        mean_outcome=total_outcome / n,
        mean_regret=total_regret / n,
        total_verifier_cost=total_cost,
        budget_mode=budget_mode,
    )


def _run_static_router(
    examples: tuple[DebtRoutingExample, ...],
    policy: DebtRoutingPolicy,
) -> tuple[str, ...]:
    """Return chosen routes using the static debt router with hysteresis state."""
    chosen: list[str] = []
    state: dict[str, Any] = {}
    for i, example in enumerate(examples):
        route, state = decide_route(
            example.signal_value, policy, previous_route=state.get("current_route"), step=i, state=state
        )
        chosen.append(route)
    return tuple(chosen)


def _run_calibrated_router(
    examples: tuple[DebtRoutingExample, ...],
    router: CalibratedDebtRouter,
) -> tuple[str, ...]:
    chosen: list[str] = []
    state: dict[str, Any] = {}
    for i, example in enumerate(examples):
        route, state = router.decide(
            example.signal_value, previous_route=state.get("current_route"), step=i, state=state
        )
        chosen.append(route)
    return tuple(chosen)


def _build_calibrated_router(
    signal_name: str,
    threshold_high: float,
    threshold_low: float | None,
    hysteresis: int,
    budget_mode: str,
    tmp_path: Path,
) -> CalibratedDebtRouter:
    """Persist and load a calibrator artifact so the hash path is exercised."""
    artifact = build_calibrator_artifact(
        signal=signal_name,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        hysteresis=hysteresis,
        fallback_policy="fixed_maskgit",
        budget_mode=budget_mode,
        calibration_split_digest="synthetic_calibration_split",
    )
    calibrator_path = tmp_path / "slm212_calibrator.json"
    calibrator_path.write_text(json.dumps(artifact), encoding="utf-8")
    config = ModelBuildConfig(
        train_dir=tmp_path / "train",
        constraint_debt_routing_mode="debt_router",
        constraint_debt_routing_signal=signal_name,
        constraint_debt_routing_threshold_high=threshold_high,
        constraint_debt_routing_threshold_low=threshold_low,
        constraint_debt_routing_hysteresis=hysteresis,
        constraint_debt_routing_budget_mode=budget_mode,
        constraint_debt_routing_calibrator_path=calibrator_path,
    )
    return CalibratedDebtRouter.from_config(config)


def _resolve_disposition(
    arms: tuple[DebtRoutingArmResult, ...],
) -> tuple[str, str]:
    """Return (disposition, rationale) from the arm results."""
    by_name = {arm.arm_name: arm for arm in arms}
    static = by_name.get("static_debt_router")
    calibrated = by_name.get("calibrated_debt_router")
    oracle = by_name.get("oracle_router_ceiling")
    fixed_best = max(
        (by_name.get(n) for n in ("fixed_maskgit", "fixed_ltr", "fixed_asap") if by_name.get(n)),
        key=lambda a: (a.mean_outcome, a.accuracy),
        default=None,
    )
    permuted = by_name.get("signal_permuted_router")

    if static is None or fixed_best is None or oracle is None:
        return ("inconclusive", "Missing required arms for disposition.")

    tolerance = 1e-6
    if static.mean_regret <= fixed_best.mean_regret + tolerance:
        if permuted is not None and static.mean_regret < permuted.mean_regret - 0.02:
            return (
                "signal_predictive",
                "Static debt router matches or beats the best fixed policy and improves "
                "over the signal-permuted control, suggesting the signal carries real "
                "routing information in this synthetic fixture.",
            )
        return (
            "modest_signal_lift",
            "Static debt router matches the best fixed policy but the lift over the "
            "signal-permuted control is small or absent.",
        )

    if calibrated is not None and calibrated.mean_regret <= static.mean_regret + tolerance:
        return (
            "calibration_helps",
            "Calibrated router improves over the static threshold, but the overall router "
            "still underperforms the best fixed policy in this fixture.",
        )

    return (
        "inconclusive",
        "Debt routing does not improve over the best fixed policy in this synthetic "
        "fixture; the limitation may be signal calibration, route ceiling, or "
        "decode-path equivalence.",
    )


def build_matrix_manifest(
    examples: tuple[DebtRoutingExample, ...] | None = None,
    *,
    signal_name: str = "D_legal",
    threshold_high: float = 2.0,
    threshold_low: float | None = None,
    hysteresis: int = 1,
    budget_mode: str = "equal_verifier_budget",
    run_id: str = "slm212-debt-routing-fixture",
    tmp_path: Path | None = None,
) -> DebtRoutingMatrixManifest:
    """Run every preregistered arm and build the manifest."""
    examples = examples or build_synthetic_routing_examples(signal_name=signal_name)
    threshold_low = threshold_low if threshold_low is not None else threshold_high

    static_policy = DebtRoutingPolicy(
        mode="debt_router",
        signal=signal_name,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        hysteresis=hysteresis,
        budget_mode=budget_mode,
    )
    static_routes = _run_static_router(examples, static_policy)

    tmp_path = tmp_path or Path("outputs/runs/slm212-debt-routing-tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)
    calibrated_router = _build_calibrated_router(
        signal_name,
        threshold_high,
        threshold_low,
        hysteresis,
        budget_mode,
        tmp_path,
    )
    calibrated_routes = _run_calibrated_router(examples, calibrated_router)

    # Signal-permuted control: same route frequency, random assignment.
    rng = random.Random(0)
    permuted_routes = tuple(rng.sample(list(static_routes), k=len(static_routes)))

    oracle = OracleRouter(
        {ex.example_id: ex.outcome_scores for ex in examples}
    )
    oracle_routes = tuple(oracle.decide(ex.example_id)[0] for ex in examples)

    arm_routes = {
        "fixed_maskgit": tuple("maskgit" for _ in examples),
        "fixed_ltr": tuple("ltr" for _ in examples),
        "fixed_asap": tuple("asap" for _ in examples),
        "static_debt_router": static_routes,
        "calibrated_debt_router": calibrated_routes,
        "signal_permuted_router": permuted_routes,
        "oracle_router_ceiling": oracle_routes,
    }

    arms = tuple(
        _compute_arm_result(name, arm_routes[name], examples, budget_mode)
        for name in ARM_NAMES
    )

    disposition, rationale = _resolve_disposition(arms)

    signal_digest = _digest([ex.signal_value for ex in examples])
    outcome_digest = _digest([ex.outcome_scores for ex in examples])

    return DebtRoutingMatrixManifest(
        schema="DebtRoutingMatrixManifest",
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=_HYPOTHESIS,
        falsifier=_FALSIFIER,
        arms=arms,
        n_examples=len(examples),
        signal_name=signal_name,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        hysteresis=hysteresis,
        budget_mode=budget_mode,
        lineage={
            "synthetic_example_count": len(examples),
            "signal_digest": signal_digest,
            "outcome_digest": outcome_digest,
            "calibrator_hash": calibrated_router.policy.calibrator_hash,
            "calibration_error": calibrated_router.calibration_error,
        },
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm212_debt_routing",
            "harness.model_build.eval",
            "matrix.slm212_debt_routing",
        ),
        timestamp=_now(),
        disposition=disposition,
        disposition_rationale=rationale,
        honest_caveats=(
            "Synthetic fixture: signals and outcomes are randomly generated and only weakly "
            "correlated with the true-best route; real constraint-debt telemetry will differ.",
            "No model, checkpoint, GPU, or verifier labels were used; this is wiring evidence only.",
            "The oracle router uses synthetic outcome scores, not serving-time signals, and is "
            "a diagnostic ceiling only.",
            "Budget accounting is a synthetic verifier-cost proxy, not measured wall time or "
            "forward passes on a real checkpoint.",
            "No ship-gate claim is made; the route ceiling and signal calibration must be "
            "re-evaluated with real decode paths and AgentV evaluation.",
        ),
    )


def _digest(value: object) -> str:
    import hashlib

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def run_fixture_matrix(
    output_dir: Path | None = None,
    *,
    n_examples: int = 200,
    signal_name: str = "D_legal",
    threshold_high: float = 2.0,
    threshold_low: float | None = None,
    hysteresis: int = 1,
    budget_mode: str = "equal_verifier_budget",
    seed: int = 0,
    write_design_docs: bool = True,
    design_json: Path | None = None,
    design_md: Path | None = None,
) -> DebtRoutingMatrixManifest:
    """Run the SLM-212 debt-routing fixture campaign."""
    start = time.perf_counter()
    examples = build_synthetic_routing_examples(
        n_examples=n_examples, seed=seed, signal_name=signal_name
    )
    tmp_path = (output_dir or Path("outputs/runs/slm212-debt-routing")) / "calibration"
    tmp_path.mkdir(parents=True, exist_ok=True)

    manifest = build_matrix_manifest(
        examples,
        signal_name=signal_name,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        hysteresis=hysteresis,
        budget_mode=budget_mode,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        tmp_path=tmp_path,
    )

    elapsed = time.perf_counter() - start
    lineage = dict(manifest.lineage)
    lineage["wall_seconds"] = _clamp(elapsed, low=0.001, high=10.0)
    manifest = DebtRoutingMatrixManifest(
        schema=manifest.schema,
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        experiment_id=manifest.experiment_id,
        run_id=manifest.run_id,
        status=manifest.status,
        claim_class=manifest.claim_class,
        hypothesis=manifest.hypothesis,
        falsifier=manifest.falsifier,
        arms=manifest.arms,
        n_examples=manifest.n_examples,
        signal_name=manifest.signal_name,
        threshold_high=manifest.threshold_high,
        threshold_low=manifest.threshold_low,
        hysteresis=manifest.hysteresis,
        budget_mode=manifest.budget_mode,
        lineage=lineage,
        version_stamp=manifest.version_stamp,
        timestamp=manifest.timestamp,
        disposition=manifest.disposition,
        disposition_rationale=manifest.disposition_rationale,
        honest_caveats=manifest.honest_caveats,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_json(output_dir / "slm212_debt_routing_report.json")

        if write_design_docs:
            if design_json is None or design_md is None:
                root = _project_root()
                design_json = root / f"docs/design/iter-slm212-debt-routing-{_today_yyyymmdd()}.json"
                design_md = root / f"docs/design/iter-slm212-debt-routing-{_today_yyyymmdd()}.md"
            design_json.parent.mkdir(parents=True, exist_ok=True)
            design_md.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_json(design_json)
            design_md.write_text(render_markdown(manifest), encoding="utf-8")

    return manifest


def render_markdown(manifest: DebtRoutingMatrixManifest) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-212 (SDE5-05): constraint-debt routing fixture ({manifest.run_id})",
        "",
        f"Matrix set: `{manifest.matrix_set}`",
        "",
        f"Version: `{manifest.matrix_version}`",
        "",
        f"Status: **{manifest.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no "
        "ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        manifest.falsifier,
        "",
        "## Arms",
        "",
        "| arm_name | accuracy | mean_outcome | mean_regret | total_verifier_cost | route_counts |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in manifest.arms:
        counts = arm.route_counts
        lines.append(
            f"| {arm.arm_name} | {arm.accuracy:.4f} | {arm.mean_outcome:.4f} | "
            f"{arm.mean_regret:.4f} | {arm.total_verifier_cost:.1f} | {counts} |"
        )

    lines.extend(
        [
            "",
            "## Signal and thresholds",
            "",
            f"- Signal: `{manifest.signal_name}`",
            f"- High threshold: {manifest.threshold_high}",
            f"- Low threshold: {manifest.threshold_low}",
            f"- Hysteresis: {manifest.hysteresis}",
            f"- Budget mode: `{manifest.budget_mode}`",
            "",
            "## Disposition",
            "",
            f"**{manifest.disposition}**",
            "",
            manifest.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The router policy, "
            "calibrator fallback, hysteresis, and budget accounting are exercised over "
            "deterministic synthetic states, but no real model or decode path was run. "
            "The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` "
            "until trained-model constraint-debt telemetry and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
        ]
    )
    for caveat in manifest.honest_caveats:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm212_debt_routing_fixture --mode plan-only",
            "python -m scripts.run_slm212_debt_routing_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def validate_manifest(manifest: DebtRoutingMatrixManifest) -> list[str]:
    """Validate the debt-routing matrix manifest."""
    errors: list[str] = []
    seen: set[str] = set()
    for arm in manifest.arms:
        if arm.arm_name in seen:
            errors.append(f"duplicate arm: {arm.arm_name}")
        seen.add(arm.arm_name)
        if arm.arm_name not in ARM_NAMES:
            errors.append(f"unknown arm: {arm.arm_name!r}")
        if not all(route in ROUTES for route in arm.route_counts):
            errors.append(f"{arm.arm_name}: invalid route in route_counts")
        if arm.total_verifier_cost < 0:
            errors.append(f"{arm.arm_name}: negative verifier cost")
    if manifest.n_examples <= 0:
        errors.append("n_examples must be positive")
    if manifest.hysteresis < 1:
        errors.append("hysteresis must be at least 1")
    return errors
