"""SLM-191 (FFE2-03): termination-policy protocol and exact-fixture matrix.

Wiring/fixture harness that defines a shared TerminationPolicy protocol and
ablate six reference arms on the exact finite-state CTMC domains from SLM-190.
No trained model, GPU, or ship-gate claim is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from slm_training.flow.reference import (
    ExactEnumerator,
    FlowTargetRowV1,
    GeneratorBuilder,
    GillespieSampler,
    build_uniform_rate_fn,
)
from slm_training.flow.termination import (
    STOP,
    AbsorbingHazardPolicy,
    ExplicitStopPolicy,
    FixedKPlusSelectorPolicy,
    FixedKPolicy,
    HybridMinProgressPolicy,
    OracleLengthPolicy,
    TerminationPolicy,
    brier_score,
    expected_calibration_error,
    sample_with_termination,
    total_variation,
)
from slm_training.harnesses.experiments.slm190_exact_flow import (
    build_canonical_edit_adapter,
    build_choice_sequence_adapter,
    build_toy_layout_adapter,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "ARM_NAMES",
    "TerminationTargetRowV1",
    "TerminationCase",
    "TerminationArmSummary",
    "TerminationManifestV1",
    "build_default_adapters",
    "run_termination_matrix",
    "render_markdown",
    "validate_manifest",
]

MATRIX_VERSION = "ffe2-03-v1"
MATRIX_SET = "slm191_termination_matrix"
EXPERIMENT_ID = "slm191-termination-matrix"

ARM_NAMES = (
    "explicit_stop",
    "absorbing_hazard",
    "fixed_k",
    "fixed_k_plus_selector",
    "hybrid_min_progress",
    "oracle_length",
)

_HYPOTHESIS = (
    "Termination semantics materially change the empirical endpoint distribution, "
    "edit-count distribution, and premature/late-stop rates on exact CTMC fixtures; "
    "a shared TerminationPolicy protocol lets direct-policy and flow samplers be "
    "compared on the same scalar signals."
)

_FALSIFIER = (
    "All six termination arms produce identical endpoint distributions and edit-count "
    "distributions on every exact domain, or the oracle-length arm is not the strongest "
    "baseline, or explicit-stop/absorbing-hazard arms are uncalibrated (Brier/ECE far "
    "above chance) on the known target signal."
)

_HONEST_CAVEATS = (
    "Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.",
    "STOP score, absorption probability, and selector probability are synthetic signals "
    "derived from the exact CTMC (hazard, absorption probabilities, edit-distance to target); "
    "production samplers must replace them with model heads.",
    "Only the canonical_edit_graph domain has a known target program; oracle_length and "
    "selector-based arms are intentionally weaker on toy_layout and choice_sequence.",
    "Exact edit-count and holding-time distributions are empirical samples, not closed-form "
    "CTMC jump-time distributions.",
    "Domains are intentionally tiny (<= a few hundred states) so the matrix stays CPU-only.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _clamp(value: float, low: float = 0.0, high: float = float("inf")) -> float:
    return max(low, min(value, high))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


@dataclass(frozen=True)
class TerminationTargetRowV1:
    """Exact target row extended with terminal/absorption and edit-count statistics."""

    schema: str = "TerminationTargetRowV1"
    row_id: str = ""
    source_fingerprint: str = ""
    target_fingerprint: str = ""
    time: float = 0.0
    state_fingerprint: str = ""
    exact_live_candidates: tuple[str, ...] = ()
    target_rates: dict[str, float] = field(default_factory=dict)
    total_hazard: float = 0.0
    next_state_fingerprints: tuple[str, ...] = ()
    endpoint_class: str = ""
    planner_version: str = "ffe2-03-v1"
    coupling_version: str = "exact_ctmc_v1"
    certificate_ids: tuple[str, ...] = ()
    terminal_fingerprints: tuple[str, ...] = ()
    absorption_mass: dict[str, float] = field(default_factory=dict)
    exact_edit_count_distribution: dict[str, float] = field(default_factory=dict)
    exact_holding_time_mean: float = 0.0
    exact_holding_time_p50: float = 0.0
    exact_holding_time_p95: float = 0.0
    oracle_edit_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "source_fingerprint": self.source_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "time": self.time,
            "state_fingerprint": self.state_fingerprint,
            "exact_live_candidates": list(self.exact_live_candidates),
            "target_rates": dict(self.target_rates),
            "total_hazard": self.total_hazard,
            "next_state_fingerprints": list(self.next_state_fingerprints),
            "endpoint_class": self.endpoint_class,
            "planner_version": self.planner_version,
            "coupling_version": self.coupling_version,
            "certificate_ids": list(self.certificate_ids),
            "terminal_fingerprints": list(self.terminal_fingerprints),
            "absorption_mass": dict(self.absorption_mass),
            "exact_edit_count_distribution": dict(self.exact_edit_count_distribution),
            "exact_holding_time_mean": self.exact_holding_time_mean,
            "exact_holding_time_p50": self.exact_holding_time_p50,
            "exact_holding_time_p95": self.exact_holding_time_p95,
            "oracle_edit_count": self.oracle_edit_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerminationTargetRowV1":
        return cls(
            schema=str(data.get("schema", "TerminationTargetRowV1")),
            row_id=str(data.get("row_id", "")),
            source_fingerprint=str(data.get("source_fingerprint", "")),
            target_fingerprint=str(data.get("target_fingerprint", "")),
            time=float(data.get("time", 0.0)),
            state_fingerprint=str(data.get("state_fingerprint", "")),
            exact_live_candidates=tuple(data.get("exact_live_candidates", ())),
            target_rates=dict(data.get("target_rates", {})),
            total_hazard=float(data.get("total_hazard", 0.0)),
            next_state_fingerprints=tuple(data.get("next_state_fingerprints", ())),
            endpoint_class=str(data.get("endpoint_class", "")),
            planner_version=str(data.get("planner_version", "ffe2-03-v1")),
            coupling_version=str(data.get("coupling_version", "exact_ctmc_v1")),
            certificate_ids=tuple(data.get("certificate_ids", ())),
            terminal_fingerprints=tuple(data.get("terminal_fingerprints", ())),
            absorption_mass=dict(data.get("absorption_mass", {})),
            exact_edit_count_distribution=dict(data.get("exact_edit_count_distribution", {})),
            exact_holding_time_mean=float(data.get("exact_holding_time_mean", 0.0)),
            exact_holding_time_p50=float(data.get("exact_holding_time_p50", 0.0)),
            exact_holding_time_p95=float(data.get("exact_holding_time_p95", 0.0)),
            oracle_edit_count=data.get("oracle_edit_count"),
        )

    def base_row(self) -> FlowTargetRowV1:
        return FlowTargetRowV1(
            row_id=self.row_id,
            source_fingerprint=self.source_fingerprint,
            target_fingerprint=self.target_fingerprint,
            time=self.time,
            state_fingerprint=self.state_fingerprint,
            exact_live_candidates=self.exact_live_candidates,
            target_rates=self.target_rates,
            total_hazard=self.total_hazard,
            next_state_fingerprints=self.next_state_fingerprints,
            endpoint_class=self.endpoint_class,
            planner_version=self.planner_version,
            coupling_version=self.coupling_version,
            certificate_ids=self.certificate_ids,
        )


@dataclass(frozen=True)
class TerminationCase:
    """One sampled termination path under one arm."""

    case_id: str
    domain_id: str
    arm: str
    source_fingerprint: str
    target_fingerprint: str
    terminal_fingerprint: str
    stop_reason: str
    edit_count: int
    holding_time: float
    reached_target: bool
    valid_states_visited: int
    premature: bool
    late_stop: bool
    abstained: bool
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain_id": self.domain_id,
            "arm": self.arm,
            "source_fingerprint": self.source_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "terminal_fingerprint": self.terminal_fingerprint,
            "stop_reason": self.stop_reason,
            "edit_count": self.edit_count,
            "holding_time": self.holding_time,
            "reached_target": self.reached_target,
            "valid_states_visited": self.valid_states_visited,
            "premature": self.premature,
            "late_stop": self.late_stop,
            "abstained": self.abstained,
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerminationCase":
        return cls(
            case_id=str(data["case_id"]),
            domain_id=str(data["domain_id"]),
            arm=str(data["arm"]),
            source_fingerprint=str(data["source_fingerprint"]),
            target_fingerprint=str(data["target_fingerprint"]),
            terminal_fingerprint=str(data["terminal_fingerprint"]),
            stop_reason=str(data["stop_reason"]),
            edit_count=int(data["edit_count"]),
            holding_time=float(data["holding_time"]),
            reached_target=bool(data["reached_target"]),
            valid_states_visited=int(data["valid_states_visited"]),
            premature=bool(data["premature"]),
            late_stop=bool(data["late_stop"]),
            abstained=bool(data["abstained"]),
            wall_seconds=float(data["wall_seconds"]),
        )


@dataclass(frozen=True)
class TerminationArmSummary:
    """Aggregate statistics for one termination arm."""

    arm_name: str
    n_samples: int
    stop_rate: float
    absorb_rate: float
    mean_edit_count: float
    p95_edit_count: float
    mean_holding_time: float
    brier_stop: float
    ece_stop: float
    premature_rate: float
    late_stop_rate: float
    endpoint_tv_vs_exact: float
    abstention_rate: float
    mean_visited_valid_states: float
    best_prefix_regret_mean: float
    reached_target_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "n_samples": self.n_samples,
            "stop_rate": self.stop_rate,
            "absorb_rate": self.absorb_rate,
            "mean_edit_count": self.mean_edit_count,
            "p95_edit_count": self.p95_edit_count,
            "mean_holding_time": self.mean_holding_time,
            "brier_stop": self.brier_stop,
            "ece_stop": self.ece_stop,
            "premature_rate": self.premature_rate,
            "late_stop_rate": self.late_stop_rate,
            "endpoint_tv_vs_exact": self.endpoint_tv_vs_exact,
            "abstention_rate": self.abstention_rate,
            "mean_visited_valid_states": self.mean_visited_valid_states,
            "best_prefix_regret_mean": self.best_prefix_regret_mean,
            "reached_target_rate": self.reached_target_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerminationArmSummary":
        return cls(
            arm_name=str(data["arm_name"]),
            n_samples=int(data["n_samples"]),
            stop_rate=float(data["stop_rate"]),
            absorb_rate=float(data["absorb_rate"]),
            mean_edit_count=float(data["mean_edit_count"]),
            p95_edit_count=float(data["p95_edit_count"]),
            mean_holding_time=float(data["mean_holding_time"]),
            brier_stop=float(data["brier_stop"]),
            ece_stop=float(data["ece_stop"]),
            premature_rate=float(data["premature_rate"]),
            late_stop_rate=float(data["late_stop_rate"]),
            endpoint_tv_vs_exact=float(data["endpoint_tv_vs_exact"]),
            abstention_rate=float(data["abstention_rate"]),
            mean_visited_valid_states=float(data["mean_visited_valid_states"]),
            best_prefix_regret_mean=float(data["best_prefix_regret_mean"]),
            reached_target_rate=float(data["reached_target_rate"]),
        )


@dataclass(frozen=True)
class TerminationManifestV1:
    """Full fixture manifest for SLM-191."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    arms: tuple[TerminationArmSummary, ...]
    cases: tuple[TerminationCase, ...]
    target_rows: tuple[TerminationTargetRowV1, ...]
    n_cases: int
    n_arms: int
    k_value: int
    n_samples_per_arm: int
    disposition: str
    disposition_rationale: str
    honest_caveats: tuple[str, ...]
    version_stamp: dict[str, Any]
    timestamp: str

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
            "arms": [a.to_dict() for a in self.arms],
            "cases": [c.to_dict() for c in self.cases],
            "target_rows": [r.to_dict() for r in self.target_rows],
            "n_cases": self.n_cases,
            "n_arms": self.n_arms,
            "k_value": self.k_value,
            "n_samples_per_arm": self.n_samples_per_arm,
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
    def from_dict(cls, data: dict[str, Any]) -> "TerminationManifestV1":
        return cls(
            schema=str(data.get("schema", "TerminationManifestV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", f"{EXPERIMENT_ID}-fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            arms=tuple(TerminationArmSummary.from_dict(a) for a in data.get("arms", ())),
            cases=tuple(TerminationCase.from_dict(c) for c in data.get("cases", ())),
            target_rows=tuple(
                TerminationTargetRowV1.from_dict(r) for r in data.get("target_rows", ())
            ),
            n_cases=int(data.get("n_cases", 0)),
            n_arms=int(data.get("n_arms", 0)),
            k_value=int(data.get("k_value", 4)),
            n_samples_per_arm=int(data.get("n_samples_per_arm", 0)),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def build_default_adapters() -> list[Any]:
    """Return the three SLM-190 fixture adapters."""
    return [
        build_toy_layout_adapter(),
        build_choice_sequence_adapter(),
        build_canonical_edit_adapter(),
    ]


def _target_program_text(adapter: Any, graph: Any, target_fingerprint: str) -> str | None:
    """Recover a target program text when available."""
    if hasattr(adapter, "target_canonical"):
        return adapter.target_canonical
    idx = graph.state_index.get(target_fingerprint)
    if idx is not None and 0 <= idx < len(graph.states):
        state = graph.states[idx]
        if hasattr(state.value, "program_text"):
            return state.value.program_text
    return None


def _edit_distance(adapter: Any, state: Any, target_text: str | None) -> int | None:
    """Best-effort edit distance from a state to a target program."""
    if target_text is None:
        return None
    try:
        if adapter.domain_id == "canonical_edit_graph":
            from slm_training.harnesses.experiments.slm188_edit_algebra import (
                plan_edit_sequence,
            )

            edits, _ = plan_edit_sequence(state.value.program_text, target_text)
            return len(edits)
        if adapter.domain_id == "toy_layout":
            from slm_training.data.edits import diff_programs

            diff = diff_programs(state.value.program_text, target_text)
            return int(diff.ast_operation_count)
        if adapter.domain_id == "choice_sequence":
            return abs(len(state.value.emitted) - len(target_text))
    except Exception:  # noqa: BLE001
        pass
    return None


def _oracle_edit_count(adapter: Any, source: Any, target_text: str | None) -> int | None:
    """Minimum edit count from source to target, when known."""
    distance = _edit_distance(adapter, source, target_text)
    if distance is None or distance < 0:
        return None
    return distance


def _absorption_probs(generator: Any, terminal_fps: set[str]) -> dict[str, float]:
    """Exact probability of eventually hitting a terminal state from each state."""
    terminal_indices = {generator.state_index[fp] for fp in terminal_fps if fp in generator.state_index}
    n = generator.n_states
    nonterminal = [i for i in range(n) if i not in terminal_indices]
    if not nonterminal:
        return {fp: 1.0 for fp in terminal_fps}
    idx_map = {i: pos for pos, i in enumerate(nonterminal)}
    Q_nt = generator.Q[np.ix_(nonterminal, nonterminal)]
    b = np.zeros(len(nonterminal))
    for i in nonterminal:
        for j in terminal_indices:
            b[idx_map[i]] += generator.Q[i, j]
    try:
        u = np.linalg.solve(Q_nt, -b)
    except Exception:  # noqa: BLE001
        u = np.zeros(len(nonterminal))
    probs: dict[str, float] = {}
    for fp, idx in generator.state_index.items():
        if idx in terminal_indices:
            probs[fp] = 1.0
        elif idx in idx_map:
            probs[fp] = float(_clamp(u[idx_map[idx]], 0.0, 1.0))
        else:
            probs[fp] = 0.0
    return probs


def _exact_reference_samples(
    sampler: GillespieSampler,
    source: Any,
    terminal_check: Any,
    rng: random.Random,
    n_samples: int,
) -> tuple[dict[int, float], float, float, float]:
    """Return empirical edit-count distribution and holding-time quantiles."""
    edit_counts: list[int] = []
    holding_times: list[float] = []
    for _ in range(n_samples):
        traj = sampler.sample(source, rng, terminal_check=terminal_check)
        edit_counts.append(len(traj.actions))
        holding_times.append(traj.total_time)
    total = len(edit_counts)
    dist: dict[int, float] = {}
    for k in edit_counts:
        dist[k] = dist.get(k, 0.0) + 1.0 / total
    return (
        dist,
        float(sum(holding_times) / max(1, len(holding_times))),
        _percentile(holding_times, 0.5),
        _percentile(holding_times, 0.95),
    )


def _exact_endpoint_mass(generator: Any, source: Any, terminal_fps: set[str]) -> dict[str, float]:
    """Exact terminal distribution at long time."""
    from slm_training.flow.reference.generator import apply_matrix_exp_row

    source_idx = generator.state_index[source.fingerprint]
    p0 = np.zeros(generator.n_states)
    p0[source_idx] = 1.0
    terminal_indices = {
        generator.state_index[fp]
        for fp in terminal_fps
        if fp in generator.state_index
    }
    pT = apply_matrix_exp_row(generator.Q, p0, 30.0)
    mass: dict[str, float] = {}
    for idx in terminal_indices:
        prob = float(pT[idx])
        if prob > 1e-9:
            mass[generator.index_state[idx].fingerprint] = prob
    return mass


def _build_generator(
    adapter: Any,
    horizon: float,
    rate_fn_name: str = "bridge_target_rate",
) -> tuple[Any, Any, Any, set[str], str | None, int | None]:
    """Enumerate, build a generator, and return reference objects."""
    enumerator = ExactEnumerator(adapter, max_states=adapter.max_states)
    graph = enumerator.enumerate()
    source = graph.initial_states[0]
    target = graph.terminal_states[0] if graph.terminal_states else source
    target_fp = target.fingerprint
    target_text = _target_program_text(adapter, graph, target_fp)
    oracle = _oracle_edit_count(adapter, source, target_text)

    if rate_fn_name == "distance_rate":
        rate_fn = _build_distance_rate_fn(adapter, source, target, graph)
    elif rate_fn_name == "bridge_target_rate":
        terminal_classes = {adapter.terminal_class(s) for s in graph.terminal_states}
        target_dist = {cls: 1.0 for cls in terminal_classes}
        from slm_training.flow.reference.generator import build_bridge_rate_fn

        rate_fn = build_bridge_rate_fn(
            target_dist,
            lambda s: adapter.terminal_class(s),
            base_rate_fn=build_uniform_rate_fn(1.0),
            temperature=1.0,
        )
    else:
        rate_fn = build_uniform_rate_fn(1.0)

    builder = GeneratorBuilder(graph)
    generator = builder.build_dense(rate_fn)
    terminal_fps = {s.fingerprint for s in graph.terminal_states}
    return graph, generator, source, terminal_fps, target_text, oracle


def _build_distance_rate_fn(adapter: Any, source: Any, target: Any, graph: Any) -> Any:
    """Distance-weighted rate biased toward the target state."""
    if adapter.domain_id == "canonical_edit_graph":
        from slm_training.harnesses.experiments.slm188_edit_algebra import plan_edit_sequence

        def distance_fn(s: Any, t: Any) -> float:
            try:
                edits, _ = plan_edit_sequence(s.value.program_text, t.value.program_text)
                return float(len(edits))
            except Exception:  # noqa: BLE001
                return 1.0
    elif adapter.domain_id == "toy_layout":
        from slm_training.data.edits import diff_programs

        def distance_fn(s: Any, t: Any) -> float:
            try:
                return float(diff_programs(s.value.program_text, t.value.program_text).ast_operation_count)
            except Exception:  # noqa: BLE001
                return 1.0
    else:
        def distance_fn(s: Any, t: Any) -> float:
            return 1.0

    from slm_training.flow.reference.generator import build_distance_rate_fn

    return build_distance_rate_fn(lambda s, _: distance_fn(s, target), temperature=1.0)


def _policy_instances(k: int, oracle: int | None) -> list[TerminationPolicy]:
    return [
        ExplicitStopPolicy(stop_threshold=0.5, max_steps=20),
        AbsorbingHazardPolicy(hazard_threshold=1e-6, absorb_threshold=0.9, max_steps=20),
        FixedKPolicy(k=k, max_steps=20),
        FixedKPlusSelectorPolicy(k=k, selector_threshold=0.5, max_steps=20),
        HybridMinProgressPolicy(min_k=max(1, k // 2), stop_threshold=0.5, selector_threshold=0.5, max_steps=20),
        OracleLengthPolicy(oracle_edit_count=oracle, max_steps=20),
    ]


def _sample_arm(
    adapter: Any,
    generator: Any,
    source: Any,
    terminal_fps: set[str],
    absorption_probs: dict[str, float],
    target_text: str | None,
    target_fp: str,
    oracle: int | None,
    policy: TerminationPolicy,
    n_samples: int,
    rng: random.Random,
    horizon: float = 1.0,
) -> tuple[list[TerminationCase], dict[str, Any]]:
    terminal_check = lambda s: s.fingerprint in terminal_fps  # noqa: E731

    def stop_score_fn(state: Any) -> float:
        h = generator.hazard(state)
        return 1.0 / (1.0 + h)

    def absorption_prob_fn(state: Any) -> float:
        return absorption_probs.get(state.fingerprint, 0.0)

    def selector_prob_fn(state: Any) -> float | None:
        distance = _edit_distance(adapter, state, target_text)
        if distance is None:
            return 0.5
        scale = max(1, oracle if oracle is not None else distance)
        return math.exp(-distance / scale)

    cases: list[TerminationCase] = []
    terminal_counts: dict[str, int] = {}
    stop_confidences: list[float] = []
    stop_outcomes: list[int] = []
    absorb_confidences: list[float] = []
    absorb_outcomes: list[int] = []
    edit_counts: list[int] = []
    holding_times: list[float] = []
    visited_counts: list[int] = []
    regrets: list[float] = []
    premature = 0
    late = 0
    abstentions = 0

    for sample_idx in range(n_samples):
        traj, meta = sample_with_termination(
            generator,
            source,
            policy,
            terminal_check,
            rng,
            max_wall_time=horizon * 10.0,
            oracle_edit_count=oracle,
            stop_score_fn=stop_score_fn,
            absorption_prob_fn=absorption_prob_fn,
            selector_prob_fn=selector_prob_fn,
        )
        edit_count = len(traj.actions)
        reached = traj.terminal_fingerprint == target_fp
        terminal_counts[traj.terminal_fingerprint] = terminal_counts.get(traj.terminal_fingerprint, 0) + 1
        cases.append(
            TerminationCase(
                case_id=f"{adapter.domain_id}__{policy.name}__{sample_idx}",
                domain_id=adapter.domain_id,
                arm=policy.name,
                source_fingerprint=source.fingerprint,
                target_fingerprint=target_fp,
                terminal_fingerprint=traj.terminal_fingerprint,
                stop_reason=str(meta.get("stop_reason", "")),
                edit_count=edit_count,
                holding_time=traj.total_time,
                reached_target=reached,
                valid_states_visited=int(meta.get("visited_valid_states", 1)),
                premature=oracle is not None and edit_count < oracle and not reached,
                late_stop=oracle is not None and edit_count > oracle and not reached,
                abstained=bool(meta.get("abstained", False)),
                wall_seconds=0.0,
            )
        )
        edit_counts.append(edit_count)
        holding_times.append(traj.total_time)
        visited_counts.append(int(meta.get("visited_valid_states", 1)))
        if bool(meta.get("abstained", False)):
            abstentions += 1
        if oracle is not None:
            if edit_count < oracle and not reached:
                premature += 1
            if edit_count > oracle and not reached:
                late += 1

        # Regret: minimum edit distance to target across visited states.
        best = float("inf")
        for fp in traj.states:
            state = generator.index_state.get(generator.state_index.get(fp, -1))
            if state is None:
                continue
            dist = _edit_distance(adapter, state, target_text)
            if dist is not None and dist < best:
                best = dist
        regrets.append(0.0 if reached else (best if best != float("inf") else 1.0))

        # Calibration samples from the final STOP / ABSORB decision.
        for d in meta.get("decisions", []):
            if d["reason"] in ("STOP_EDIT", "HYBRID_END") and d["action"] == STOP:
                stop_confidences.append(d["confidence"])
                stop_outcomes.append(1 if reached else 0)
            if d["reason"] in ("TOTAL_HAZARD", "ABSORB") and d["action"] == STOP:
                absorb_confidences.append(d["confidence"])
                absorb_outcomes.append(1 if reached else 0)
    total = len(cases)
    empirical: dict[str, float] = {}
    if total:
        empirical = {fp: c / total for fp, c in terminal_counts.items()}
    exact_mass = _exact_endpoint_mass(generator, source, terminal_fps)
    tv = total_variation(exact_mass, empirical)

    summary = {
        "arm_name": policy.name,
        "n_samples": total,
        "stop_rate": sum(1 for c in cases if c.stop_reason not in ("", "abstained", "terminal_state")) / max(1, total),
        "absorb_rate": sum(1 for c in cases if c.stop_reason in ("TOTAL_HAZARD", "ABSORB")) / max(1, total),
        "mean_edit_count": sum(edit_counts) / max(1, total),
        "p95_edit_count": _percentile([float(x) for x in edit_counts], 0.95),
        "mean_holding_time": sum(holding_times) / max(1, total),
        "brier_stop": brier_score(stop_confidences, stop_outcomes),
        "ece_stop": expected_calibration_error(stop_confidences, stop_outcomes),
        "premature_rate": premature / max(1, total),
        "late_stop_rate": late / max(1, total),
        "endpoint_tv_vs_exact": tv,
        "abstention_rate": abstentions / max(1, total),
        "mean_visited_valid_states": sum(visited_counts) / max(1, total),
        "best_prefix_regret_mean": sum(regrets) / max(1, total),
        "reached_target_rate": sum(1 for c in cases if c.reached_target) / max(1, total),
    }
    return cases, summary


def _build_target_row(
    graph: Any,
    generator: Any,
    source: Any,
    adapter: Any,
    time: float,
    rng: random.Random,
    target_text: str | None,
    oracle: int | None,
) -> TerminationTargetRowV1:
    terminal_fps = {s.fingerprint for s in graph.terminal_states}
    terminal_check = lambda s: s.fingerprint in terminal_fps  # noqa: E731
    source_idx = generator.state_index[source.fingerprint]
    succ = generator.legal_successors(source_idx)
    exact_mass = _exact_endpoint_mass(generator, source, terminal_fps)
    sampler = GillespieSampler(generator, max_steps=1_000, max_time=time * 10.0)
    edit_dist, h_mean, h_p50, h_p95 = _exact_reference_samples(
        sampler, source, terminal_check, rng, n_samples=500
    )
    target_fp = max(exact_mass, key=exact_mass.get, default="")
    return TerminationTargetRowV1(
        row_id=f"{adapter.domain_id}__{source.fingerprint[:16]}__t{time}",
        source_fingerprint=source.fingerprint,
        target_fingerprint=target_fp,
        time=time,
        state_fingerprint=source.fingerprint,
        exact_live_candidates=tuple(generator.index_state[j].fingerprint for j, _, _ in succ),
        target_rates={generator.index_state[j].fingerprint: r for j, _, r in succ},
        total_hazard=generator.hazard(source_idx),
        next_state_fingerprints=tuple(generator.index_state[j].fingerprint for j, _, _ in succ),
        endpoint_class=adapter.terminal_class(source),
        certificate_ids=tuple(
            f"cert:{source.fingerprint[:12]}->{generator.index_state[j].fingerprint[:12]}"
            for j, _, _ in succ
        ),
        terminal_fingerprints=tuple(sorted(terminal_fps)),
        absorption_mass=exact_mass,
        exact_edit_count_distribution={str(k): v for k, v in edit_dist.items()},
        exact_holding_time_mean=h_mean,
        exact_holding_time_p50=h_p50,
        exact_holding_time_p95=h_p95,
        oracle_edit_count=oracle,
    )


def _resolve_disposition(
    arms: tuple[TerminationArmSummary, ...],
    cases: tuple[TerminationCase, ...],
) -> tuple[str, str]:
    if not arms or not cases:
        return ("inconclusive", "No arms or cases were generated.")

    # At least one arm should differ from another in mean edit count or endpoint TV.
    mean_edits = {a.arm_name: a.mean_edit_count for a in arms}
    tvs = {a.arm_name: a.endpoint_tv_vs_exact for a in arms}
    if len(set(mean_edits.values())) <= 1 and len(set(tvs.values())) <= 1:
        return (
            "inconclusive",
            "All arms produced identical mean edit counts and endpoint TVs; "
            "termination semantics did not differentiate.",
        )

    return (
        "supports_termination_diversity",
        "Arms produce differentiated edit counts and/or endpoint distributions on the exact CTMC fixture.",
    )


def run_termination_matrix(
    output_dir: Path | None = None,
    *,
    adapters: list[Any] | None = None,
    k_value: int = 4,
    n_samples_per_arm: int = 100,
    horizon: float = 1.0,
    rate_fn_name: str = "uniform_rate",
    seed: int = 0,
    write_design_docs: bool = True,
    design_json: Path | None = None,
    design_md: Path | None = None,
) -> TerminationManifestV1:
    """Run the SLM-191 termination-policy fixture matrix."""
    start = time.perf_counter()
    rng = random.Random(seed)
    adapters = adapters if adapters is not None else build_default_adapters()

    target_rows: list[TerminationTargetRowV1] = []
    all_cases: list[TerminationCase] = []
    summaries: list[TerminationArmSummary] = []

    for adapter in adapters:
        graph, generator, source, terminal_fps, target_text, oracle = _build_generator(adapter, horizon, rate_fn_name)
        absorption_probs = _absorption_probs(generator, terminal_fps)
        exact_mass = _exact_endpoint_mass(generator, source, terminal_fps)
        target_fp = max(exact_mass, key=exact_mass.get, default="")
        target_rows.append(
            _build_target_row(graph, generator, source, adapter, horizon, rng, target_text, oracle)
        )

        policies = _policy_instances(k_value, oracle)
        for policy in policies:
            cases, summary = _sample_arm(
                adapter,
                generator,
                source,
                terminal_fps,
                absorption_probs,
                target_text,
                target_fp,
                oracle,
                policy,
                n_samples_per_arm,
                rng,
                horizon=horizon,
            )
            all_cases.extend(cases)
            summaries.append(TerminationArmSummary(**summary))

    disposition, rationale = _resolve_disposition(tuple(summaries), tuple(all_cases))

    version_stamp = build_version_stamp(
        "harness.experiments",
        "harness.experiments.slm191_termination_matrix",
        "flow.termination",
        "flow.reference",
        "harness.experiments.slm188_edit_algebra",
        "harness.experiments.slm190_exact_flow",
    )

    manifest = TerminationManifestV1(
        schema="TerminationManifestV1",
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        status="fixture",
        claim_class="wiring",
        hypothesis=_HYPOTHESIS,
        falsifier=_FALSIFIER,
        arms=tuple(summaries),
        cases=tuple(all_cases),
        target_rows=tuple(target_rows),
        n_cases=len(all_cases),
        n_arms=len(summaries),
        k_value=k_value,
        n_samples_per_arm=n_samples_per_arm,
        disposition=disposition,
        disposition_rationale=rationale,
        honest_caveats=_HONEST_CAVEATS,
        version_stamp=version_stamp,
        timestamp=_now(),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_json(output_dir / "slm191_termination_matrix_report.json")
        if write_design_docs:
            root = Path(__file__).resolve().parents[4]
            if design_json is None or design_md is None:
                design_json = root / f"docs/design/iter-slm191-termination-matrix-{_today_yyyymmdd()}.json"
                design_md = root / f"docs/design/iter-slm191-termination-matrix-{_today_yyyymmdd()}.md"
            design_json.parent.mkdir(parents=True, exist_ok=True)
            design_md.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_json(design_json)
            design_md.write_text(render_markdown(manifest), encoding="utf-8")

    elapsed = time.perf_counter() - start
    lineage_extra = {"wall_seconds": _clamp(elapsed, low=0.001, high=10.0)}
    stamp = dict(manifest.version_stamp)
    stamp["lineage"] = lineage_extra
    manifest = TerminationManifestV1(
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
        cases=manifest.cases,
        target_rows=manifest.target_rows,
        n_cases=manifest.n_cases,
        n_arms=manifest.n_arms,
        k_value=manifest.k_value,
        n_samples_per_arm=manifest.n_samples_per_arm,
        disposition=manifest.disposition,
        disposition_rationale=manifest.disposition_rationale,
        honest_caveats=manifest.honest_caveats,
        version_stamp=stamp,
        timestamp=manifest.timestamp,
    )
    return manifest


def render_markdown(manifest: TerminationManifestV1) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-191 (FFE2-03): termination-policy fixture matrix ({manifest.run_id})",
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
        f"k_value: {manifest.k_value}",
        f"n_samples_per_arm: {manifest.n_samples_per_arm}",
        "",
        "| arm_name | n_samples | stop_rate | absorb_rate | mean_edit_count | p95_edit_count | brier_stop | ece_stop | endpoint_tv_vs_exact | reached_target_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for a in manifest.arms:
        lines.append(
            f"| {a.arm_name} | {a.n_samples} | {a.stop_rate:.3f} | {a.absorb_rate:.3f} | "
            f"{a.mean_edit_count:.2f} | {a.p95_edit_count:.1f} | {a.brier_stop:.3f} | "
            f"{a.ece_stop:.3f} | {a.endpoint_tv_vs_exact:.3f} | {a.reached_target_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Target rows",
            "",
            f"Total target rows: {len(manifest.target_rows)}",
            "",
            "| domain | n_terminals | exact_edit_count_support | oracle_edit_count |",
            "| --- | --- | --- | --- |",
        ]
    )
    for r in manifest.target_rows:
        n_terminals = len(r.terminal_fingerprints)
        support = len(r.exact_edit_count_distribution)
        oracle = r.oracle_edit_count if r.oracle_edit_count is not None else "n/a"
        lines.append(f"| {r.row_id.split('__')[0]} | {n_terminals} | {support} | {oracle} |")
    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{manifest.disposition}**",
            "",
            manifest.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The TerminationPolicy protocol, "
            "six reference arms, and calibration instrumentation are exercised over deterministic "
            "exact CTMC domains with synthetic model signals. Production samplers must replace "
            "the synthetic signals with learned STOP, total-hazard, absorption, and selector heads "
            "and re-run on real checkpoints before any ship claim.",
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
            "python -m scripts.run_termination_matrix --describe",
            "python -m scripts.run_termination_matrix --exact-fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def validate_manifest(manifest: TerminationManifestV1) -> list[str]:
    """Validate the termination fixture manifest."""
    errors: list[str] = []
    if manifest.matrix_set != MATRIX_SET:
        errors.append(f"matrix_set mismatch: {manifest.matrix_set}")
    if manifest.matrix_version != MATRIX_VERSION:
        errors.append(f"matrix_version mismatch: {manifest.matrix_version}")
    if manifest.n_cases != len(manifest.cases):
        errors.append("n_cases does not match len(cases)")
    if manifest.n_arms != len(manifest.arms):
        errors.append("n_arms does not match len(arms)")
    for arm in manifest.arms:
        if arm.arm_name not in ARM_NAMES:
            errors.append(f"unknown arm: {arm.arm_name!r}")
        if arm.n_samples < 0:
            errors.append(f"{arm.arm_name}: negative n_samples")
    case_ids = {c.case_id for c in manifest.cases}
    if len(case_ids) != len(manifest.cases):
        errors.append("duplicate case_id")
    for case in manifest.cases:
        if case.arm not in ARM_NAMES:
            errors.append(f"{case.case_id}: unknown arm {case.arm!r}")
        if case.wall_seconds < 0:
            errors.append(f"{case.case_id}: negative wall_seconds")
    return errors
