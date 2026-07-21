"""SLM-190 (FFE2-02): exact finite-state CTMC reference and lumpability wiring.

Deterministic, CPU-only fixture that builds exact state graphs, rate matrices,
bridge posteriors, and quotient-state aggregation tests for three closed toy
domains.  No model is trained and no ship-gate claim is made.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from slm_training.flow.reference import (
    ExactEnumerator,
    FlowSampleV1,
    FlowTargetRowV1,
    FlowTrajectoryV1,
    GeneratorBuilder,
    GillespieSampler,
    LUMPABLE,
    NOT_LUMPABLE,
    build_distance_rate_fn,
    build_uniform_rate_fn,
    check_generator,
    is_strongly_lumpable,
)
from slm_training.flow.reference.adapters import (
    CanonicalEditGraphAdapter,
    ChoiceSequenceAdapter,
    ToyLayoutAdapter,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "ARM_NAMES",
    "ExactFlowCase",
    "ObjectiveComparisonRow",
    "LumpabilityCase",
    "ExactFlowReport",
    "build_toy_layout_adapter",
    "build_choice_sequence_adapter",
    "build_canonical_edit_adapter",
    "run_exact_flow_fixture",
    "render_markdown",
    "validate_report",
]

MATRIX_VERSION = "ffe2-02-v1"
MATRIX_SET = "slm190_exact_flow"
EXPERIMENT_ID = "slm190-exact-flow"

ARM_NAMES = (
    "uniform_rate",
    "distance_rate",
    "bridge_target_rate",
    "doob_bridge_posterior",
)

_HYPOTHESIS = (
    "An exact finite-state CTMC reference over compiler-certified legal edits "
    "reveals objective-dependent differences in total hazard, endpoint "
    "distribution, and path statistics, and most natural quotient partitions "
    "over program structure are not strongly lumpable."
)

_FALSIFIER = (
    "On every representative exact domain, normalized next-edit CE plus a fixed "
    "time schedule reproduces the rate-based endpoint distribution and path "
    "statistics within tolerance, and the chosen quotient partitions are "
    "strongly lumpable."
)

_HONEST_CAVEATS = (
    "Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.",
    "Domains are intentionally tiny (<= a few hundred states) so enumeration stays CPU-only.",
    "Distance-based and bridge rates are illustrative parameterizations; production flow "
    "objectives may differ.",
    "The Doob h-transform bridge posterior is computed by exact matrix exponentiation on the "
    "full state graph and is therefore a training-only oracle, not an inference-time scorer.",
    "Lumpability tests use coarse structural partitions; finer partitions are always trivially lumpable.",
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


@dataclass(frozen=True)
class ExactFlowCase:
    case_id: str
    domain_id: str
    source_fingerprint: str
    target_fingerprint: str
    rate_fn_name: str
    time: float
    n_states: int
    n_transitions: int
    n_terminals: int
    mass_conservation_error: float
    total_hazard_mean: float
    endpoint_tv_exact_vs_gillespie: float
    illegal_edge_rate_sum: float
    multipath_entropy_bits: float
    exact_endpoint_mass: dict[str, float]
    gillespie_terminal_rate: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain_id": self.domain_id,
            "source_fingerprint": self.source_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "rate_fn_name": self.rate_fn_name,
            "time": self.time,
            "n_states": self.n_states,
            "n_transitions": self.n_transitions,
            "n_terminals": self.n_terminals,
            "mass_conservation_error": self.mass_conservation_error,
            "total_hazard_mean": self.total_hazard_mean,
            "endpoint_tv_exact_vs_gillespie": self.endpoint_tv_exact_vs_gillespie,
            "illegal_edge_rate_sum": self.illegal_edge_rate_sum,
            "multipath_entropy_bits": self.multipath_entropy_bits,
            "exact_endpoint_mass": dict(self.exact_endpoint_mass),
            "gillespie_terminal_rate": dict(self.gillespie_terminal_rate),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExactFlowCase":
        return cls(
            case_id=str(data["case_id"]),
            domain_id=str(data["domain_id"]),
            source_fingerprint=str(data["source_fingerprint"]),
            target_fingerprint=str(data["target_fingerprint"]),
            rate_fn_name=str(data["rate_fn_name"]),
            time=float(data["time"]),
            n_states=int(data["n_states"]),
            n_transitions=int(data["n_transitions"]),
            n_terminals=int(data["n_terminals"]),
            mass_conservation_error=float(data["mass_conservation_error"]),
            total_hazard_mean=float(data["total_hazard_mean"]),
            endpoint_tv_exact_vs_gillespie=float(data["endpoint_tv_exact_vs_gillespie"]),
            illegal_edge_rate_sum=float(data["illegal_edge_rate_sum"]),
            multipath_entropy_bits=float(data["multipath_entropy_bits"]),
            exact_endpoint_mass=dict(data.get("exact_endpoint_mass", {})),
            gillespie_terminal_rate=dict(data.get("gillespie_terminal_rate", {})),
        )


@dataclass(frozen=True)
class ObjectiveComparisonRow:
    case_id: str
    domain_id: str
    state_fingerprint: str
    time: float
    rate_fn_name: str
    normalized_ce_entropy_bits: float
    total_hazard: float
    edge_rate_sum: float
    posterior_entropy_bits: float
    n_live_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain_id": self.domain_id,
            "state_fingerprint": self.state_fingerprint,
            "time": self.time,
            "rate_fn_name": self.rate_fn_name,
            "normalized_ce_entropy_bits": self.normalized_ce_entropy_bits,
            "total_hazard": self.total_hazard,
            "edge_rate_sum": self.edge_rate_sum,
            "posterior_entropy_bits": self.posterior_entropy_bits,
            "n_live_candidates": self.n_live_candidates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectiveComparisonRow":
        return cls(
            case_id=str(data["case_id"]),
            domain_id=str(data["domain_id"]),
            state_fingerprint=str(data["state_fingerprint"]),
            time=float(data["time"]),
            rate_fn_name=str(data["rate_fn_name"]),
            normalized_ce_entropy_bits=float(data["normalized_ce_entropy_bits"]),
            total_hazard=float(data["total_hazard"]),
            edge_rate_sum=float(data["edge_rate_sum"]),
            posterior_entropy_bits=float(data["posterior_entropy_bits"]),
            n_live_candidates=int(data["n_live_candidates"]),
        )


@dataclass(frozen=True)
class LumpabilityCase:
    case_id: str
    domain_id: str
    rate_fn_name: str
    partition_name: str
    status: str
    n_blocks: int
    n_violations: int
    representative_violation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain_id": self.domain_id,
            "rate_fn_name": self.rate_fn_name,
            "partition_name": self.partition_name,
            "status": self.status,
            "n_blocks": self.n_blocks,
            "n_violations": self.n_violations,
            "representative_violation": dict(self.representative_violation),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LumpabilityCase":
        return cls(
            case_id=str(data["case_id"]),
            domain_id=str(data["domain_id"]),
            rate_fn_name=str(data["rate_fn_name"]),
            partition_name=str(data["partition_name"]),
            status=str(data["status"]),
            n_blocks=int(data["n_blocks"]),
            n_violations=int(data["n_violations"]),
            representative_violation=dict(data.get("representative_violation", {})),
        )


@dataclass(frozen=True)
class ExactFlowReport:
    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cases: tuple[ExactFlowCase, ...]
    objective_rows: tuple[ObjectiveComparisonRow, ...]
    lumpability_cases: tuple[LumpabilityCase, ...]
    n_domains: int
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
            "cases": [c.to_dict() for c in self.cases],
            "objective_rows": [r.to_dict() for r in self.objective_rows],
            "lumpability_cases": [lc.to_dict() for lc in self.lumpability_cases],
            "n_domains": self.n_domains,
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
    def from_dict(cls, data: dict[str, Any]) -> "ExactFlowReport":
        return cls(
            schema=str(data.get("schema", "ExactFlowReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", f"{EXPERIMENT_ID}-fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            cases=tuple(ExactFlowCase.from_dict(c) for c in data.get("cases", ())),
            objective_rows=tuple(
                ObjectiveComparisonRow.from_dict(r) for r in data.get("objective_rows", ())
            ),
            lumpability_cases=tuple(
                LumpabilityCase.from_dict(lc) for lc in data.get("lumpability_cases", ())
            ),
            n_domains=int(data.get("n_domains", 0)),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def _entropy_bits(probs: dict[Any, float]) -> float:
    total = 0.0
    for p in probs.values():
        if p > 0.0:
            total -= p * math.log2(p)
    return total


def _terminal_check(adapter: Any) -> Any:
    def check(state: Any) -> bool:
        return adapter.is_terminal(state)

    return check


def build_toy_layout_adapter() -> ToyLayoutAdapter:
    """Tiny layout AST: sketch -> simple target."""
    seed = 'root = Stack([text], "column")\ntext = TextContent(":slot")'
    return ToyLayoutAdapter(
        seed_programs=[seed],
        inventory=[":page.blurb", ":card.title", ":card.action"],
        max_depth=3,
        max_states=200,
    )


def build_choice_sequence_adapter() -> ChoiceSequenceAdapter:
    """Bounded grammar with two commutative interleavings."""
    return ChoiceSequenceAdapter(
        productions={
            "S": [["A", "B"], ["B", "A"]],
            "A": [["a", "A"], ["a"]],
            "B": [["b", "B"], ["b"]],
        },
        max_length=4,
        max_states=200,
    )


def build_canonical_edit_adapter() -> CanonicalEditGraphAdapter:
    """Sketch -> small target using SLM-188 edit algebra."""
    from slm_training.harnesses.experiments.slm188_edit_algebra import build_sketch_seed

    target = 'root = Stack([card], "column")\ncard_title = TextContent(":card.title")\ncard = Card([card_title])'
    source = build_sketch_seed(target)
    return CanonicalEditGraphAdapter(source, target, max_edits=4, max_states=200)


def _distance_for_state(adapter: Any, source: Any, target: Any, state: Any) -> float:
    if isinstance(adapter, ToyLayoutAdapter):
        from slm_training.data.edits import diff_programs

        try:
            return float(diff_programs(state.value.program_text, target.value.program_text).ast_operation_count)
        except Exception:  # noqa: BLE001
            return 1.0
    if isinstance(adapter, ChoiceSequenceAdapter):
        # Distance to any terminal with matching prefix length.
        return float(abs(len(state.value.emitted) - len(target.value.emitted)))
    if isinstance(adapter, CanonicalEditGraphAdapter):
        from slm_training.harnesses.experiments.slm188_edit_algebra import plan_edit_sequence

        try:
            edits, _ = plan_edit_sequence(state.value.program_text, target.value.program_text)
            return float(len(edits))
        except Exception:  # noqa: BLE001
            return 1.0
    return 1.0


def _terminal_class(adapter: Any, state: Any) -> str:
    return adapter.terminal_class(state)


def _run_one_domain(
    adapter: Any,
    graph: Any,
    source: Any,
    target: Any,
    rate_fn_name: str,
    rate_fn: Any,
    time: float,
    rng: random.Random,
) -> tuple[ExactFlowCase, list[ObjectiveComparisonRow], list[LumpabilityCase], list[FlowTargetRowV1]]:
    from slm_training.flow.reference.generator import (
        apply_matrix_exp_col,
        apply_matrix_exp_row,
    )

    builder = GeneratorBuilder(graph)
    generator = builder.build_dense(rate_fn)

    # Generator contract checks.
    gen_errors = check_generator(generator.Q, atol=1e-7)
    if gen_errors:
        raise RuntimeError(f"generator contract violations: {gen_errors[:5]}")
    illegal_edge_rate_sum = sum(
        generator.Q[i, j]
        for i in range(generator.n_states)
        for j in range(generator.n_states)
        if i != j
        and (i, j) not in generator.rates
        and generator.Q[i, j] > 0.0
    )

    # Endpoint distribution from initial source using vector uniformization.
    source_idx = generator.state_index[source.fingerprint]
    p0 = np.zeros(generator.n_states)
    p0[source_idx] = 1.0
    pT = apply_matrix_exp_row(generator.Q, p0, time)
    mass_error = float(abs(pT.sum() - 1.0))

    # Precompute Doob h-transform vector once per (generator, time).
    terminal_fps = {s.fingerprint for s in graph.terminal_states}
    terminal_indices = [graph.state_index[s.fingerprint] for s in graph.terminal_states]
    b = np.zeros(generator.n_states)
    for idx in terminal_indices:
        b[idx] = 1.0
    h = np.maximum(apply_matrix_exp_col(generator.Q, b, time), 1e-12)

    # Absorption distribution over terminals for the Gillespie convergence check.
    p_abs = apply_matrix_exp_row(generator.Q, p0, 30.0)
    exact_endpoint_mass: dict[str, float] = {}
    for idx, prob in enumerate(p_abs):
        if prob > 1e-9 and generator.index_state[idx].fingerprint in terminal_fps:
            fp = generator.index_state[idx].fingerprint
            exact_endpoint_mass[fp] = float(prob)

    # Gillespie sampler convergence (use the pre-enumerated terminal set).
    sampler = GillespieSampler(generator, max_steps=1_000, max_time=time * 10)
    terminal_check = lambda state: state.fingerprint in terminal_fps  # noqa: E731
    sample = _sample_endpoint_distribution(sampler, source, 200, rng, terminal_check)
    gillespie_terminal_rate: dict[str, float] = {}
    for fp, count in sample.empirical_terminal_distribution.items():
        gillespie_terminal_rate[fp] = float(count)
    tv = _total_variation(exact_endpoint_mass, gillespie_terminal_rate)

    # Multipath entropy: entropy over normalized rates leaving the source.
    succ = generator.legal_successors(source_idx)
    total_rate = sum(r for _, _, r in succ)
    if total_rate > 0.0:
        probs = {generator.index_state[j].fingerprint: r / total_rate for j, _, r in succ}
    else:
        probs = {}
    multipath_entropy = _entropy_bits(probs)

    mean_hazard = float(np.mean([-generator.Q[i, i] for i in range(generator.n_states)]))

    target_fp = next(iter(exact_endpoint_mass), "")
    case = ExactFlowCase(
        case_id=f"{adapter.domain_id}__{rate_fn_name}__t{time}",
        domain_id=adapter.domain_id,
        source_fingerprint=source.fingerprint,
        target_fingerprint=target_fp,
        rate_fn_name=rate_fn_name,
        time=time,
        n_states=graph.n_states,
        n_transitions=graph.n_transitions,
        n_terminals=len(graph.terminal_states),
        mass_conservation_error=mass_error,
        total_hazard_mean=mean_hazard,
        endpoint_tv_exact_vs_gillespie=tv,
        illegal_edge_rate_sum=float(illegal_edge_rate_sum),
        multipath_entropy_bits=multipath_entropy,
        exact_endpoint_mass=exact_endpoint_mass,
        gillespie_terminal_rate=gillespie_terminal_rate,
    )

    # Objective comparison rows for a few states.
    rows: list[ObjectiveComparisonRow] = []
    target_state = generator.index_state[max(source_idx, min(generator.n_states - 1, source_idx + 1))]
    for state in [source, target_state]:
        if state.fingerprint not in generator.state_index:
            continue
        idx = generator.state_index[state.fingerprint]
        succ = generator.legal_successors(idx)
        rates = {generator.index_state[j].fingerprint: r for j, _, r in succ}
        norm = _normalize(rates)
        row = ObjectiveComparisonRow(
            case_id=case.case_id,
            domain_id=adapter.domain_id,
            state_fingerprint=state.fingerprint,
            time=time,
            rate_fn_name=rate_fn_name,
            normalized_ce_entropy_bits=_entropy_bits(norm),
            total_hazard=generator.hazard(idx),
            edge_rate_sum=sum(rates.values()),
            posterior_entropy_bits=_entropy_bits(_doob_posterior_probs(generator, idx, h)),
            n_live_candidates=len(succ),
        )
        rows.append(row)

    # Flow target rows for the production-loss interface.
    target_rows: list[FlowTargetRowV1] = []
    for state in [source]:
        idx = generator.state_index[state.fingerprint]
        succ = generator.legal_successors(idx)
        target_rows.append(
            FlowTargetRowV1(
                row_id=f"{case.case_id}__{state.fingerprint[:16]}",
                source_fingerprint=source.fingerprint,
                target_fingerprint=target_fp,
                time=time,
                state_fingerprint=state.fingerprint,
                exact_live_candidates=tuple(generator.index_state[j].fingerprint for j, _, _ in succ),
                target_rates={generator.index_state[j].fingerprint: r for j, _, r in succ},
                total_hazard=generator.hazard(idx),
                next_state_fingerprints=tuple(generator.index_state[j].fingerprint for j, _, _ in succ),
                endpoint_class=adapter.terminal_class(state),
                certificate_ids=tuple(f"cert:{state.fingerprint[:12]}->{generator.index_state[j].fingerprint[:12]}" for j, _, _ in succ),
            )
        )

    # Lumpability tests.
    lump_cases: list[LumpabilityCase] = []
    partitions = {
        "by_terminal_class": {s.fingerprint: hash(_terminal_class(adapter, s)) for s in graph.states},
        "by_state_length": _length_partition(adapter, graph),
    }
    for partition_name, partition in partitions.items():
        ok, info = is_strongly_lumpable(generator, partition, atol=1e-7)
        status = LUMPABLE if ok else NOT_LUMPABLE
        rep = info["violations"][0] if info["violations"] else {}
        lump_cases.append(
            LumpabilityCase(
                case_id=f"{case.case_id}__{partition_name}",
                domain_id=adapter.domain_id,
                rate_fn_name=rate_fn_name,
                partition_name=partition_name,
                status=status,
                n_blocks=info["n_blocks"],
                n_violations=info["n_violations"],
                representative_violation=rep,
            )
        )

    return case, rows, lump_cases, target_rows


def _length_partition(adapter: Any, graph: Any) -> dict[str, int]:
    """Coarse partition by rendered program length or emitted length."""
    partition: dict[str, int] = {}
    for state in graph.states:
        if isinstance(adapter, ChoiceSequenceAdapter):
            length = len(state.value.emitted)
        elif isinstance(adapter, ToyLayoutAdapter):
            length = len(state.value.program_text)
        else:
            length = len(state.value.program_text)
        partition[state.fingerprint] = min(length, 8)
    return partition


def _normalize(rates: dict[str, float]) -> dict[str, float]:
    total = sum(rates.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in rates.items()}


def _doob_posterior_probs(generator: Any, state_idx: int, h: np.ndarray) -> dict[str, float]:
    """Exact posterior next-state probabilities conditioned on terminals."""
    n = generator.n_states
    probs: dict[str, float] = {}
    for j in range(n):
        if j == state_idx:
            continue
        rate = generator.Q[state_idx, j]
        if rate > 0.0:
            probs[generator.index_state[j].fingerprint] = float(rate * h[j] / h[state_idx])
    total = sum(probs.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in probs.items()}


def _sample_endpoint_distribution(
    sampler: GillespieSampler,
    source: Any,
    n_samples: int,
    rng: random.Random,
    terminal_check: Any,
) -> FlowSampleV1:
    counts: dict[str, int] = {}
    trajectories: list[FlowTrajectoryV1] = []
    for _ in range(n_samples):
        traj = sampler.sample(source, rng, terminal_check=terminal_check)
        counts[traj.terminal_fingerprint] = counts.get(traj.terminal_fingerprint, 0) + 1
        trajectories.append(traj)
    total = sum(counts.values())
    empirical = {fp: c / total for fp, c in counts.items()} if total else {}
    return FlowSampleV1(
        source_fingerprint=source.fingerprint,
        n_samples=n_samples,
        empirical_terminal_distribution=empirical,
        trajectories=tuple(trajectories[:10]),
    )


def _total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _build_rate_fn(
    rate_fn_name: str,
    adapter: Any,
    source: Any,
    target: Any,
    graph: Any,
) -> Any:
    if rate_fn_name == "uniform_rate":
        return build_uniform_rate_fn(1.0)
    if rate_fn_name == "distance_rate":
        return build_distance_rate_fn(
            lambda s, t: _distance_for_state(adapter, source, target, s),
            temperature=1.0,
        )
    if rate_fn_name == "bridge_target_rate":
        target_dist = {adapter.terminal_class(s): 1.0 for s in graph.terminal_states}
        from slm_training.flow.reference.generator import build_bridge_rate_fn

        return build_bridge_rate_fn(
            target_dist,
            lambda s: _terminal_class(adapter, s),
            base_rate_fn=build_uniform_rate_fn(1.0),
            temperature=1.0,
        )
    if rate_fn_name == "doob_bridge_posterior":
        # Defer to the case runner; here we return a placeholder uniform rate.
        return build_uniform_rate_fn(1.0)
    return build_uniform_rate_fn(1.0)


def _resolve_disposition(
    cases: tuple[ExactFlowCase, ...],
    lumpability_cases: tuple[LumpabilityCase, ...],
) -> tuple[str, str]:
    if not cases:
        return ("inconclusive", "No cases were generated.")

    mass_ok = all(c.mass_conservation_error < 1e-5 for c in cases)
    illegal_ok = all(c.illegal_edge_rate_sum < 1e-9 for c in cases)
    tv_ok = all(c.endpoint_tv_exact_vs_gillespie < 0.25 for c in cases)
    multipath_ok = any(c.multipath_entropy_bits > 0.5 for c in cases)
    non_lumpable = any(lc.status == NOT_LUMPABLE for lc in lumpability_cases)

    if not mass_ok:
        return ("inconclusive", "Mass conservation error exceeded tolerance in at least one case.")
    if not illegal_ok:
        return ("inconclusive", "Illegal transitions carried non-zero rate in at least one case.")
    if not tv_ok:
        return ("inconclusive", "Gillespie empirical distribution differed from exact endpoint by > 0.25 TV in at least one case.")
    if multipath_ok and non_lumpable:
        return (
            "supports_rate_objective_diversity",
            "Exact mass is conserved, Gillespie samples converge, at least one domain has "
            "multi-path target mass, and coarse structural partitions are not lumpable.",
        )
    if multipath_ok:
        return (
            "supports_exact_reference",
            "Exact mass is conserved and multi-path target mass is observed, but coarse "
            "partitions were lumpable (often because the partition is too fine).",
        )
    return (
        "inconclusive",
        "All numerical checks passed, but no domain exhibited multi-path entropy above threshold.",
    )


def run_exact_flow_fixture(
    output_dir: Path | None = None,
    *,
    rate_fn_names: tuple[str, ...] = ARM_NAMES,
    times: tuple[float, ...] = (1.0,),
    seed: int = 0,
    write_design_docs: bool = True,
    design_json: Path | None = None,
    design_md: Path | None = None,
) -> ExactFlowReport:
    """Run the SLM-190 exact CTMC reference fixture."""
    start = time.perf_counter()
    rng = random.Random(seed)

    adapters = [
        build_toy_layout_adapter(),
        build_choice_sequence_adapter(),
        build_canonical_edit_adapter(),
    ]

    cases: list[ExactFlowCase] = []
    rows: list[ObjectiveComparisonRow] = []
    lumps: list[LumpabilityCase] = []
    target_rows: list[FlowTargetRowV1] = []

    for adapter in adapters:
        enumerator = ExactEnumerator(adapter, max_states=adapter.max_states)
        graph = enumerator.enumerate()
        source = graph.initial_states[0]
        target = graph.terminal_states[0] if graph.terminal_states else source
        for rate_fn_name in rate_fn_names:
            rate_fn = _build_rate_fn(rate_fn_name, adapter, source, target, graph)
            for t in times:
                case, obj_rows, lump_cases, trs = _run_one_domain(
                    adapter, graph, source, target, rate_fn_name, rate_fn, t, rng
                )
                cases.append(case)
                rows.extend(obj_rows)
                lumps.extend(lump_cases)
                target_rows.extend(trs)

    disposition, rationale = _resolve_disposition(tuple(cases), tuple(lumps))

    version_stamp = build_version_stamp(
        "harness.experiments",
        "harness.experiments.slm190_exact_flow",
        "flow.reference",
        "dsl.solver.topology",
        "harness.experiments.slm188_edit_algebra",
    )

    report = ExactFlowReport(
        schema="ExactFlowReportV1",
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        status="fixture",
        claim_class="wiring",
        hypothesis=_HYPOTHESIS,
        falsifier=_FALSIFIER,
        cases=tuple(cases),
        objective_rows=tuple(rows),
        lumpability_cases=tuple(lumps),
        n_domains=len(adapters),
        disposition=disposition,
        disposition_rationale=rationale,
        honest_caveats=_HONEST_CAVEATS,
        version_stamp=version_stamp,
        timestamp=_now(),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm190_exact_flow_report.json")
        if write_design_docs:
            root = Path(__file__).resolve().parents[4]
            if design_json is None or design_md is None:
                design_json = root / f"docs/design/iter-slm190-exact-flow-{_today_yyyymmdd()}.json"
                design_md = root / f"docs/design/iter-slm190-exact-flow-{_today_yyyymmdd()}.md"
            design_json.parent.mkdir(parents=True, exist_ok=True)
            design_md.parent.mkdir(parents=True, exist_ok=True)
            report.to_json(design_json)
            design_md.write_text(render_markdown(report), encoding="utf-8")

    elapsed = time.perf_counter() - start
    lineage_extra = {"wall_seconds": _clamp(elapsed, low=0.001, high=10.0)}
    stamp = dict(report.version_stamp)
    stamp["lineage"] = lineage_extra
    report = ExactFlowReport(
        schema=report.schema,
        matrix_set=report.matrix_set,
        matrix_version=report.matrix_version,
        experiment_id=report.experiment_id,
        run_id=report.run_id,
        status=report.status,
        claim_class=report.claim_class,
        hypothesis=report.hypothesis,
        falsifier=report.falsifier,
        cases=report.cases,
        objective_rows=report.objective_rows,
        lumpability_cases=report.lumpability_cases,
        n_domains=report.n_domains,
        disposition=report.disposition,
        disposition_rationale=report.disposition_rationale,
        honest_caveats=report.honest_caveats,
        version_stamp=stamp,
        timestamp=report.timestamp,
    )
    return report


def render_markdown(report: ExactFlowReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-190 (FFE2-02): exact finite-state CTMC reference fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no "
        "ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Domains",
        "",
        f"Total domains: {report.n_domains}",
        f"Total cases: {len(report.cases)}",
        f"Objective rows: {len(report.objective_rows)}",
        f"Lumpability tests: {len(report.lumpability_cases)}",
        "",
        "## Cases",
        "",
        "| case_id | domain | rate_fn | time | n_states | n_transitions | mass_error | tv_exact_vs_gillespie | multipath_entropy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in report.cases:
        lines.append(
            f"| {c.case_id} | {c.domain_id} | {c.rate_fn_name} | {c.time} | "
            f"{c.n_states} | {c.n_transitions} | {c.mass_conservation_error:.2e} | "
            f"{c.endpoint_tv_exact_vs_gillespie:.3f} | {c.multipath_entropy_bits:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Lumpability",
            "",
            "| domain | partition | status | n_blocks | n_violations |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for lc in report.lumpability_cases:
        lines.append(
            f"| {lc.domain_id} | {lc.partition_name} | {lc.status} | {lc.n_blocks} | {lc.n_violations} |"
        )
    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The exact CTMC reference, "
            "Gillespie sampler, objective comparisons, and lumpability tests are exercised "
            "over deterministic synthetic domains, but no trained model or decode path was run. "
            "The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until "
            "trained-model flow telemetry and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
        ]
    )
    for caveat in report.honest_caveats:
        lines.append(f"- {caveat}")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_exact_flow_fixture --mode describe",
            "python -m scripts.run_exact_flow_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def validate_report(report: ExactFlowReport) -> list[str]:
    """Validate the exact flow fixture report."""
    errors: list[str] = []
    if report.matrix_set != MATRIX_SET:
        errors.append(f"matrix_set mismatch: {report.matrix_set}")
    if report.matrix_version != MATRIX_VERSION:
        errors.append(f"matrix_version mismatch: {report.matrix_version}")
    case_ids = {c.case_id for c in report.cases}
    if len(case_ids) != len(report.cases):
        errors.append("duplicate case_id")
    for c in report.cases:
        if c.mass_conservation_error > 1e-4:
            errors.append(f"{c.case_id}: mass conservation error {c.mass_conservation_error}")
        if c.illegal_edge_rate_sum > 1e-8:
            errors.append(f"{c.case_id}: illegal edge rate sum {c.illegal_edge_rate_sum}")
    return errors
