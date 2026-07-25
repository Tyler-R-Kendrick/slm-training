"""SLM-192 (FFE3-01): stage-accurate cost profiler for valid-edit bridge/training/decode/closure/verification.

Wiring/fixture harness that profiles the per-stage CPU cost of the OpenUI flow
pipeline (canonicalization, bridge planning, candidate enumeration, exact
closure, support oracle checks, and an optional direct scorer fixture).  No
trained model, GPU, or ship-gate claim is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.flow.bridge_planner import plan_bridge
from slm_training.dsl.canonicalize import canonical_fingerprint, canonicalize
from slm_training.dsl.solver.closure import (
    ReplayResult,
    exact_closure,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
)
from slm_training.dsl.solver.support import (
    SupportCertificate,
    SupportQuery,
    SupportResult,
    SupportVerdict,
)
from slm_training.evals.tree_edit_scaling import _enumerate_edits
from slm_training.harnesses.experiments.slm188_edit_algebra import build_sketch_seed
from slm_training.models.tree_edit_diffusion import (
    TreeEditSpace,
    parse_statements,
)
from slm_training.runtime.telemetry import CycleTelemetry
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "ARM_NAMES",
    "CostSpanRecord",
    "FlowCostProfileV1",
    "OnPolicyFeasibilityV1",
    "CostGateManifestV1",
    "FlowPipelineProfileCase",
    "FlowPipelineManifestV1",
    "render_markdown",
    "run_profile_flow_pipeline",
    "validate_manifest",
]

MATRIX_VERSION = "ffe3-01-v1"
MATRIX_SET = "slm192_profile_flow_pipeline"
EXPERIMENT_ID = "slm192-profile-flow-pipeline"

ARM_NAMES = (
    "no_model_canonical_baseline",
    "bridge_planner_canonical_greedy",
    "x22_edit_enumeration",
    "exact_closure_toy",
    "support_oracle_check",
    "direct_scorer_fixture",
)

HERO = '''root = Stack([hero], "column")
hero_title = TextContent(":hero.title")
hero_body = TextContent(":hero.body")
hero = Card([hero_title, hero_body])'''

_MAX_ON_POLICY_EPOCH_SECONDS = 1800.0

_HYPOTHESIS = (
    "Valid-edit bridge/training/decode/closure/verification stages have separable, "
    "measurable CPU cost profiles; the dominant bottleneck on toy fixtures is either "
    "candidate enumeration, exact closure, or verifier replay; and the combined "
    "per-target bridge+enumeration cost extrapolates to an on-policy epoch budget."
)

_FALSIFIER = (
    "The cold and warm profiles are identical (no caching benefit), or the top "
    "bottleneck is not enumeration/closure/verification, or the per-target cost "
    "extrapolates beyond the 30-minute on-policy epoch bound despite the tiny "
    "fixture domain."
)

_HONEST_CAVEATS = (
    "Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.",
    "Direct/flow decode arms are optional or skipped when torch is absent; no learned "
    "policy or value head is measured.",
    "The toy support provider is synthetic (payload['ok'] flag); real support queries "
    "require a verifier and problem expander.",
    "Real model decode cost (forward pass, beam scoring, sampler overhead) is not "
    "measured here; extrapolations are from CPU-only canonical operations.",
    "Only the HERO fixture is exercised; production targets will have different "
    "statement counts, domain sizes, and verifier profiles.",
)

_SPAN_NAMES = (
    "request_encoding",
    "source_seed",
    "state_materialization",
    "candidate_enumeration",
    "candidate_features",
    "exact_closure",
    "state_hashing",
    "model_encode",
    "candidate_projection",
    "transition_apply",
    "canonicalization",
    "serialization",
    "verifier",
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


def _cvar(values: list[float], alpha: float = 0.05) -> float:
    """Mean of the worst ``alpha`` fraction of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = max(1, int(math.ceil(alpha * len(sorted_vals))))
    worst = sorted_vals[-k:]
    return sum(worst) / len(worst)


@dataclass(frozen=True)
class CostSpanRecord:
    """Aggregated timing for one named span."""

    name: str
    total_ms: float
    count: int
    mean_ms: float
    max_ms: float
    pct_of_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_ms": self.total_ms,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "max_ms": self.max_ms,
            "pct_of_total": self.pct_of_total,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostSpanRecord":
        return cls(
            name=str(data["name"]),
            total_ms=float(data["total_ms"]),
            count=int(data["count"]),
            mean_ms=float(data["mean_ms"]),
            max_ms=float(data["max_ms"]),
            pct_of_total=float(data["pct_of_total"]),
        )


@dataclass(frozen=True)
class FlowCostProfileV1:
    """Cost profile for one arm under one thermal condition."""

    arm_name: str
    condition: str
    total_ms: float
    span_records: tuple[CostSpanRecord, ...]
    work_units: dict[str, float]
    n_repeats: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "condition": self.condition,
            "total_ms": self.total_ms,
            "span_records": [r.to_dict() for r in self.span_records],
            "work_units": dict(self.work_units),
            "n_repeats": self.n_repeats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowCostProfileV1":
        return cls(
            arm_name=str(data["arm_name"]),
            condition=str(data["condition"]),
            total_ms=float(data["total_ms"]),
            span_records=tuple(CostSpanRecord.from_dict(r) for r in data.get("span_records", ())),
            work_units={k: float(v) for k, v in (data.get("work_units") or {}).items()},
            n_repeats=int(data.get("n_repeats", 1)),
        )


@dataclass(frozen=True)
class OnPolicyFeasibilityV1:
    """On-policy cost extrapolation from measured per-target bridge+enum time."""

    strategy: str
    projected_seconds_per_target: float
    projected_seconds_for_108_targets: float
    extrapolated_dagger_round_seconds: float
    extrapolated_five_seeds_seconds: float
    extrapolated_confirmation_suite_seconds: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "projected_seconds_per_target": self.projected_seconds_per_target,
            "projected_seconds_for_108_targets": self.projected_seconds_for_108_targets,
            "extrapolated_dagger_round_seconds": self.extrapolated_dagger_round_seconds,
            "extrapolated_five_seeds_seconds": self.extrapolated_five_seeds_seconds,
            "extrapolated_confirmation_suite_seconds": self.extrapolated_confirmation_suite_seconds,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OnPolicyFeasibilityV1":
        return cls(
            strategy=str(data["strategy"]),
            projected_seconds_per_target=float(data["projected_seconds_per_target"]),
            projected_seconds_for_108_targets=float(data["projected_seconds_for_108_targets"]),
            extrapolated_dagger_round_seconds=float(data["extrapolated_dagger_round_seconds"]),
            extrapolated_five_seeds_seconds=float(data["extrapolated_five_seeds_seconds"]),
            extrapolated_confirmation_suite_seconds=float(data["extrapolated_confirmation_suite_seconds"]),
            rationale=str(data["rationale"]),
        )


@dataclass(frozen=True)
class CostGateManifestV1:
    """Cost gate and bottleneck summary."""

    max_on_policy_epoch_seconds: float
    allowed_strategy: str
    enumeration_bound: bool
    bottlenecks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_on_policy_epoch_seconds": self.max_on_policy_epoch_seconds,
            "allowed_strategy": self.allowed_strategy,
            "enumeration_bound": self.enumeration_bound,
            "bottlenecks": list(self.bottlenecks),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostGateManifestV1":
        return cls(
            max_on_policy_epoch_seconds=float(data["max_on_policy_epoch_seconds"]),
            allowed_strategy=str(data["allowed_strategy"]),
            enumeration_bound=bool(data["enumeration_bound"]),
            bottlenecks=list(data.get("bottlenecks", ())),
        )


@dataclass(frozen=True)
class FlowPipelineProfileCase:
    """One measured run of one arm/condition pair."""

    case_id: str
    arm_name: str
    condition: str
    source_fingerprint: str
    target_fingerprint: str
    wall_seconds: float
    work_units: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arm_name": self.arm_name,
            "condition": self.condition,
            "source_fingerprint": self.source_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "wall_seconds": self.wall_seconds,
            "work_units": dict(self.work_units),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowPipelineProfileCase":
        return cls(
            case_id=str(data["case_id"]),
            arm_name=str(data["arm_name"]),
            condition=str(data["condition"]),
            source_fingerprint=str(data["source_fingerprint"]),
            target_fingerprint=str(data["target_fingerprint"]),
            wall_seconds=float(data["wall_seconds"]),
            work_units={k: float(v) for k, v in (data.get("work_units") or {}).items()},
        )


@dataclass(frozen=True)
class FlowPipelineManifestV1:
    """Full fixture manifest for SLM-192."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    disposition: str
    disposition_rationale: str
    arms: tuple[FlowCostProfileV1, ...]
    cases: tuple[FlowPipelineProfileCase, ...]
    cost_gate: CostGateManifestV1
    on_policy: OnPolicyFeasibilityV1
    n_cases: int
    n_arms: int
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
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "arms": [a.to_dict() for a in self.arms],
            "cases": [c.to_dict() for c in self.cases],
            "cost_gate": self.cost_gate.to_dict(),
            "on_policy": self.on_policy.to_dict(),
            "n_cases": self.n_cases,
            "n_arms": self.n_arms,
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
    def from_dict(cls, data: dict[str, Any]) -> "FlowPipelineManifestV1":
        return cls(
            schema=str(data.get("schema", "FlowPipelineManifestV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", f"{EXPERIMENT_ID}-fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            disposition=str(data.get("disposition", "cost_profile_wired")),
            disposition_rationale=str(
                data.get(
                    "disposition_rationale",
                    "CPU-only fixture cost profile wired; no ship claim.",
                )
            ),
            arms=tuple(FlowCostProfileV1.from_dict(a) for a in data.get("arms", ())),
            cases=tuple(FlowPipelineProfileCase.from_dict(c) for c in data.get("cases", ())),
            cost_gate=CostGateManifestV1.from_dict(data.get("cost_gate", {})),
            on_policy=OnPolicyFeasibilityV1.from_dict(data.get("on_policy", {})),
            n_cases=int(data.get("n_cases", 0)),
            n_arms=int(data.get("n_arms", 0)),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


class _ToySupportProvider:
    """Synthetic support provider: ok=False values are UNSUPPORTED, else SUPPORTED."""

    def __init__(self, problem_id: str = "toy", pack_id: str = "openui") -> None:
        self.problem_id = problem_id
        self.pack_id = pack_id
        self.constraint_version = "toy-v1"
        self.bounds = SolverBounds(
            max_tokens=64,
            max_nodes=16,
            max_depth=4,
            max_backtracks=4,
            max_verifier_calls=8,
        )

    @property
    def backend_version(self) -> str:
        return "toy/ok-flag-v1"

    def _certificate(
        self, query: SupportQuery, verdict: SupportVerdict
    ) -> SupportCertificate:
        return SupportCertificate(
            schema_version=1,
            query=query,
            verdict=verdict,
            problem_id=self.problem_id,
            pack_id=self.pack_id,
            constraint_version=self.constraint_version,
            bounds=self.bounds,
            search_order="canonical-domain-value-v1",
            explored_state_fingerprints=(),
            coverage_observations=("complete",),
            verifier_profile="toy/ok-flag",
            witness_source="toy" if verdict is SupportVerdict.SUPPORTED else None,
            witness_digest=None,
            failure_counts=(),
            exhausted=verdict is SupportVerdict.UNSUPPORTED,
            stop_reason=None,
        )

    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult:
        ok = bool(query.candidate.payload.get("ok", False))
        verdict = SupportVerdict.SUPPORTED if ok else SupportVerdict.UNSUPPORTED
        from slm_training.dsl.solver.support import SearchCounters

        return SupportResult(
            verdict=verdict,
            certificate=self._certificate(query, verdict),
            counters=SearchCounters(verifier_calls=1),
        )

    def replay(
        self, certificate: SupportCertificate, *, state: FiniteDomainState
    ) -> ReplayResult:
        return ReplayResult(ok=True, verdict=certificate.verdict)


def _make_toy_state() -> FiniteDomainState:
    """Finite-domain state with one hole and four values."""
    hole_id = HoleId(namespace="toy", path=("root",), kind="slot")
    values = tuple(
        DomainValue.create("ok_flag", {"ok": i % 2 == 0, "idx": i})
        for i in range(4)
    )
    return FiniteDomainState(
        problem_id="toy",
        pack_id="openui",
        constraint_version="toy-v1",
        bounds=SolverBounds(
            max_tokens=64,
            max_nodes=16,
            max_depth=4,
            max_backtracks=4,
            max_verifier_calls=8,
        ),
        holes=(HoleDomain(hole_id, values),),
    )


def _span_records_from_telemetry(telemetry: CycleTelemetry) -> tuple[CostSpanRecord, ...]:
    summary = telemetry.summary()
    records: list[CostSpanRecord] = []
    for name, row in summary.get("spans", {}).items():
        records.append(
            CostSpanRecord(
                name=name,
                total_ms=float(row["total_ms"]),
                count=int(row["count"]),
                mean_ms=float(row["mean_ms"]),
                max_ms=float(row["max_ms"]),
                pct_of_total=float(row["pct"]),
            )
        )
    records.sort(key=lambda r: r.total_ms, reverse=True)
    return tuple(records)


def _run_arm(
    arm_name: str,
    target_program: str,
    source_program: str,
    target_fingerprint: str,
    source_fingerprint: str,
    n_repeats: int,
) -> tuple[FlowCostProfileV1, FlowCostProfileV1, FlowPipelineProfileCase, FlowPipelineProfileCase]:
    """Return (cold_profile, warm_profile, cold_case, warm_case) for one arm."""

    def run_once(telemetry: CycleTelemetry, condition: str) -> dict[str, Any]:
        work_units: dict[str, float] = {
            "projected_candidates": 0.0,
            "legal_candidates": 0.0,
            "edits": 0.0,
            "nodes": 0.0,
            "verifier_calls": 0.0,
            "support_queries": 0.0,
            "cache_hits": 0.0,
            "bytes_allocated": 0.0,
        }

        if arm_name == "no_model_canonical_baseline":
            with telemetry.span("request_encoding"):
                encoded = _canonical_json({"source": source_program, "target": target_program})
            with telemetry.span("source_seed"):
                seed_text = source_program
            with telemetry.span("canonicalization"):
                target_canonical = canonicalize(target_program, validate=True)
                source_canonical = canonicalize(seed_text, validate=True)
            with telemetry.span("state_hashing"):
                src_fp = canonical_fingerprint(source_canonical)
                tgt_fp = canonical_fingerprint(target_canonical)
            work_units["nodes"] = float(
                (len(parse_statements(source_canonical) or []))
                + (len(parse_statements(target_canonical) or []))
            )
            work_units["bytes_allocated"] = float(len(encoded.encode("utf-8")) + len(src_fp) + len(tgt_fp))

        elif arm_name == "bridge_planner_canonical_greedy":
            with telemetry.span("request_encoding"):
                _canonical_json({"source": source_program, "target": target_program})
            with telemetry.span("source_seed"):
                seed_text = source_program
            with telemetry.span("candidate_enumeration"):
                result = plan_bridge(
                    seed_text,
                    target_program,
                    arm="canonical_greedy",
                    source_seed_id="slm192",
                )
            with telemetry.span("transition_apply"):
                plan = result.plan
            with telemetry.span("canonicalization"):
                if plan is not None:
                    canonicalize(source_program, validate=False)
                    canonicalize(target_program, validate=False)
            with telemetry.span("serialization"):
                if plan is not None:
                    _canonical_json(plan.to_dict())
            with telemetry.span("verifier"):
                verifier_calls = (
                    plan.cost_vector.get("verifier", 0.0) if plan is not None else 0.0
                )
                work_units["verifier_calls"] = float(verifier_calls)
            work_units["projected_candidates"] = float(result.nodes_expanded)
            work_units["legal_candidates"] = float(result.max_frontier)
            work_units["edits"] = float(plan.path_length if plan is not None else 0)
            work_units["nodes"] = float(
                result.scaling_features.get("source_nodes", 0.0)
                + result.scaling_features.get("target_nodes", 0.0)
            )
            work_units["support_queries"] = float(result.cost_attribution.get("closure_query", 0.0))
            work_units["cache_hits"] = float(result.cost_attribution.get("cache_hits", 0.0))

        elif arm_name == "x22_edit_enumeration":
            with telemetry.span("request_encoding"):
                _canonical_json({"source": source_program, "target": target_program})
            with telemetry.span("source_seed"):
                seed_text = source_program
            with telemetry.span("state_materialization"):
                statements = parse_statements(seed_text) or []
                space = TreeEditSpace()
                inventory = [":slot"]
            with telemetry.span("candidate_enumeration"):
                candidates = _enumerate_edits(statements, inventory, space)
            with telemetry.span("candidate_features"):
                legal = [result for _, result in candidates if result is not None]
            with telemetry.span("serialization"):
                _canonical_json([{"action": e.action, "stmt": e.stmt} for e, _ in candidates])
            work_units["projected_candidates"] = float(len(candidates))
            work_units["legal_candidates"] = float(len(legal))
            work_units["edits"] = float(len(candidates))
            work_units["nodes"] = float(len(statements))
            work_units["verifier_calls"] = float(len(legal))

        elif arm_name == "exact_closure_toy":
            with telemetry.span("request_encoding"):
                _canonical_json({"arm": arm_name})
            with telemetry.span("state_materialization"):
                state = _make_toy_state()
                provider = _ToySupportProvider()
                cache: dict[str, SupportResult] = {}
            with telemetry.span("exact_closure"):
                closure_result = exact_closure(state, provider, cache=cache)
            with telemetry.span("state_hashing"):
                _ = closure_result.state.fingerprint
            with telemetry.span("serialization"):
                _canonical_json(closure_result.counters.to_dict())
            counters = closure_result.counters
            work_units["projected_candidates"] = float(
                sum(len(h.values) for h in state.holes)
            )
            work_units["legal_candidates"] = float(
                closure_result.state.summary()["total_candidate_count"]
            )
            work_units["edits"] = float(len(closure_result.deductions))
            work_units["nodes"] = float(len(state.holes))
            work_units["verifier_calls"] = float(counters.verifier_calls)
            work_units["support_queries"] = float(counters.support_queries)
            work_units["cache_hits"] = float(counters.cache_hits)

        elif arm_name == "support_oracle_check":
            with telemetry.span("request_encoding"):
                _canonical_json({"arm": arm_name})
            with telemetry.span("state_materialization"):
                state = _make_toy_state()
                provider = _ToySupportProvider()
                hole_id = state.holes[0].hole_id
            supported_count = 0
            for value in state.holes[0].values:
                with telemetry.span("verifier"):
                    query = SupportQuery(state_fingerprint=state.fingerprint, hole_id=hole_id, candidate=value)
                    result = provider.check(state, query)
                    if result.verdict is SupportVerdict.SUPPORTED:
                        supported_count += 1
            work_units["projected_candidates"] = float(len(state.holes[0].values))
            work_units["legal_candidates"] = float(supported_count)
            work_units["support_queries"] = float(len(state.holes[0].values))
            work_units["verifier_calls"] = float(len(state.holes[0].values))

        elif arm_name == "direct_scorer_fixture":
            try:
                from slm_training.models.legal_action_scorer import (
                    LegalActionScorer,
                    LegalActionScorerConfig,
                    make_fixture_decisions,
                )
            except Exception:  # pragma: no cover - torch may be absent
                work_units["projected_candidates"] = 0.0
                work_units["legal_candidates"] = 0.0
                work_units["bytes_allocated"] = 0.0
                return {"skipped": True, "work_units": work_units}

            with telemetry.span("request_encoding"):
                decisions = make_fixture_decisions(n=1, seed=0)
            with telemetry.span("model_encode"):
                scorer = LegalActionScorer(
                    config=LegalActionScorerConfig(variant="mlp", d_model=32, hidden_dim=32),
                    device="cpu",
                )
            decision = decisions[0]
            with telemetry.span("candidate_projection"):
                scores = scorer.score(
                    decision.context_features,
                    decision.state_features,
                    decision.legal_actions,
                    plan_features=decision.plan_features,
                    plan_action_features=decision.plan_action_features,
                    pack_id=decision.pack_id,
                )
            with telemetry.span("transition_apply"):
                scorer.decode(scores, list(scores.legal_actions))
            work_units["projected_candidates"] = float(len(decision.legal_actions))
            work_units["legal_candidates"] = float(scores.metadata.get("n_legal", len(decision.legal_actions)))
            work_units["bytes_allocated"] = float(
                sum(p.numel() for p in scorer.parameters()) * 4
            )

        return {"work_units": work_units}

    # Cold run.
    cold_telemetry = CycleTelemetry()
    cold_info = run_once(cold_telemetry, "cold")
    if cold_info.get("skipped"):
        cold_profile = FlowCostProfileV1(
            arm_name=arm_name,
            condition="cold",
            total_ms=0.0,
            span_records=(),
            work_units=cold_info["work_units"],
            n_repeats=1,
        )
        cold_case = FlowPipelineProfileCase(
            case_id=f"{arm_name}__cold",
            arm_name=arm_name,
            condition="cold",
            source_fingerprint=source_fingerprint,
            target_fingerprint=target_fingerprint,
            wall_seconds=0.0,
            work_units=cold_info["work_units"],
        )
        warm_profile = FlowCostProfileV1(
            arm_name=arm_name,
            condition="warm",
            total_ms=0.0,
            span_records=(),
            work_units=cold_info["work_units"],
            n_repeats=max(0, n_repeats - 1),
        )
        warm_case = FlowPipelineProfileCase(
            case_id=f"{arm_name}__warm",
            arm_name=arm_name,
            condition="warm",
            source_fingerprint=source_fingerprint,
            target_fingerprint=target_fingerprint,
            wall_seconds=0.0,
            work_units=cold_info["work_units"],
        )
        return cold_profile, warm_profile, cold_case, warm_case

    cold_total = cold_telemetry.summary()["total_ms"]
    cold_work = dict(cold_info["work_units"])
    cold_profile = FlowCostProfileV1(
        arm_name=arm_name,
        condition="cold",
        total_ms=cold_total,
        span_records=_span_records_from_telemetry(cold_telemetry),
        work_units=cold_work,
        n_repeats=1,
    )
    cold_case = FlowPipelineProfileCase(
        case_id=f"{arm_name}__cold",
        arm_name=arm_name,
        condition="cold",
        source_fingerprint=source_fingerprint,
        target_fingerprint=target_fingerprint,
        wall_seconds=cold_total / 1000.0,
        work_units=cold_work,
    )

    # Warm runs.
    warm_repeats = max(0, n_repeats - 1)
    warm_telemetry = CycleTelemetry()
    for _ in range(warm_repeats):
        run_once(warm_telemetry, "warm")
    warm_total = warm_telemetry.summary()["total_ms"]
    warm_profile = FlowCostProfileV1(
        arm_name=arm_name,
        condition="warm",
        total_ms=warm_total,
        span_records=_span_records_from_telemetry(warm_telemetry),
        work_units=cold_work,
        n_repeats=warm_repeats,
    )
    warm_case = FlowPipelineProfileCase(
        case_id=f"{arm_name}__warm",
        arm_name=arm_name,
        condition="warm",
        source_fingerprint=source_fingerprint,
        target_fingerprint=target_fingerprint,
        wall_seconds=warm_total / 1000.0,
        work_units=cold_work,
    )
    return cold_profile, warm_profile, cold_case, warm_case


def _compute_on_policy_feasibility(
    arms: tuple[FlowCostProfileV1, ...],
) -> OnPolicyFeasibilityV1:
    bridge_total = sum(
        a.total_ms for a in arms if a.arm_name == "bridge_planner_canonical_greedy" and a.condition == "warm"
    )
    enum_total = sum(
        a.total_ms for a in arms if a.arm_name == "x22_edit_enumeration" and a.condition == "warm"
    )
    per_target_sec = (bridge_total + enum_total) / 1000.0
    projected_108 = per_target_sec * 108.0
    five_seeds = projected_108 * 5.0
    dagger = projected_108 * 0.5
    confirmation = projected_108 * 0.1

    if five_seeds > _MAX_ON_POLICY_EPOCH_SECONDS:
        strategy = "offline_only"
        rationale = (
            f"Five-seeds extrapolation ({five_seeds:.1f}s) exceeds the "
            f"{_MAX_ON_POLICY_EPOCH_SECONDS:.0f}s on-policy epoch bound; "
            "fixture suggests offline generation or model-speedup before on-policy training."
        )
    else:
        strategy = "on_policy_viable"
        rationale = (
            f"Measured warm bridge+enum per target is {per_target_sec:.4f}s. "
            f"108 targets = {projected_108:.1f}s; five seeds = {five_seeds:.1f}s, "
            f"well under the {_MAX_ON_POLICY_EPOCH_SECONDS:.0f}s epoch bound on this fixture."
        )

    return OnPolicyFeasibilityV1(
        strategy=strategy,
        projected_seconds_per_target=per_target_sec,
        projected_seconds_for_108_targets=projected_108,
        extrapolated_dagger_round_seconds=dagger,
        extrapolated_five_seeds_seconds=five_seeds,
        extrapolated_confirmation_suite_seconds=confirmation,
        rationale=rationale,
    )


def _compute_cost_gate(
    arms: tuple[FlowCostProfileV1, ...],
) -> CostGateManifestV1:
    warm_total = sum(a.total_ms for a in arms if a.condition == "warm")
    enum_total = 0.0
    span_totals: dict[str, float] = {}
    for arm in arms:
        if arm.condition != "warm":
            continue
        for record in arm.span_records:
            span_totals[record.name] = span_totals.get(record.name, 0.0) + record.total_ms
        if arm.arm_name in {
            "x22_edit_enumeration",
            "exact_closure_toy",
            "support_oracle_check",
        }:
            for record in arm.span_records:
                if record.name in {"candidate_enumeration", "exact_closure", "verifier"}:
                    enum_total += record.total_ms

    enumeration_bound = (enum_total / max(warm_total, 1e-9)) > 0.5
    ranked = sorted(span_totals.items(), key=lambda x: x[1], reverse=True)
    bottlenecks = [name for name, _ in ranked[:3]]

    on_policy = _compute_on_policy_feasibility(arms)
    allowed = on_policy.strategy

    return CostGateManifestV1(
        max_on_policy_epoch_seconds=_MAX_ON_POLICY_EPOCH_SECONDS,
        allowed_strategy=allowed,
        enumeration_bound=enumeration_bound,
        bottlenecks=bottlenecks,
    )


def run_profile_flow_pipeline(
    output_dir: Path | None = None,
    *,
    n_repeats: int = 3,
    seed: int = 0,
    write_design_docs: bool = True,
    design_json: Path | None = None,
    design_md: Path | None = None,
) -> FlowPipelineManifestV1:
    """Run the SLM-192 stage-accurate cost-profile fixture matrix."""
    start = time.perf_counter()
    target_program = canonicalize(HERO, validate=True)
    source_program = build_sketch_seed(target_program)
    target_fingerprint = canonical_fingerprint(target_program)
    source_fingerprint = canonical_fingerprint(source_program)

    profiles: list[FlowCostProfileV1] = []
    cases: list[FlowPipelineProfileCase] = []
    for arm_name in ARM_NAMES:
        cold_profile, warm_profile, cold_case, warm_case = _run_arm(
            arm_name,
            target_program,
            source_program,
            target_fingerprint,
            source_fingerprint,
            n_repeats,
        )
        profiles.append(cold_profile)
        profiles.append(warm_profile)
        cases.append(cold_case)
        cases.append(warm_case)

    arms_tuple = tuple(profiles)
    cases_tuple = tuple(cases)
    cost_gate = _compute_cost_gate(arms_tuple)
    on_policy = _compute_on_policy_feasibility(arms_tuple)

    version_stamp = build_version_stamp(
        "harness.experiments",
        "harness.experiments.slm192_profile_flow_pipeline",
        "flow.termination",
        "flow.reference",
        "harness.experiments.slm188_edit_algebra",
        "harness.experiments.slm189_bridge_planner",
    )

    manifest = FlowPipelineManifestV1(
        schema="FlowPipelineManifestV1",
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        status="fixture",
        claim_class="wiring",
        hypothesis=_HYPOTHESIS,
        falsifier=_FALSIFIER,
        disposition="cost_profile_wired",
        disposition_rationale=(
            "CPU-only fixture cost profile wired for all declared arms; "
            "no model, GPU, checkpoint, or ship claim is made."
        ),
        arms=arms_tuple,
        cases=cases_tuple,
        cost_gate=cost_gate,
        on_policy=on_policy,
        n_cases=len(cases_tuple),
        n_arms=len(ARM_NAMES),
        honest_caveats=_HONEST_CAVEATS,
        version_stamp=version_stamp,
        timestamp=_now(),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_json(output_dir / "slm192_profile_flow_pipeline_report.json")
        if write_design_docs:
            root = Path(__file__).resolve().parents[4]
            if design_json is None or design_md is None:
                design_json = root / f"docs/design/iter-slm192-profile-flow-pipeline-{_today_yyyymmdd()}.json"
                design_md = root / f"docs/design/iter-slm192-profile-flow-pipeline-{_today_yyyymmdd()}.md"
            design_json.parent.mkdir(parents=True, exist_ok=True)
            design_md.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_json(design_json)
            design_md.write_text(render_markdown(manifest), encoding="utf-8")

    elapsed = time.perf_counter() - start
    lineage_extra = {"wall_seconds": _clamp(elapsed, low=0.001, high=10.0)}
    stamp = dict(manifest.version_stamp)
    stamp["lineage"] = lineage_extra
    manifest = FlowPipelineManifestV1(
        schema=manifest.schema,
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        experiment_id=manifest.experiment_id,
        run_id=manifest.run_id,
        status=manifest.status,
        claim_class=manifest.claim_class,
        hypothesis=manifest.hypothesis,
        falsifier=manifest.falsifier,
        disposition=manifest.disposition,
        disposition_rationale=manifest.disposition_rationale,
        arms=manifest.arms,
        cases=manifest.cases,
        cost_gate=manifest.cost_gate,
        on_policy=manifest.on_policy,
        n_cases=manifest.n_cases,
        n_arms=manifest.n_arms,
        honest_caveats=manifest.honest_caveats,
        version_stamp=stamp,
        timestamp=manifest.timestamp,
    )
    return manifest


def render_markdown(manifest: FlowPipelineManifestV1) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-192 (FFE3-01): stage-accurate flow-pipeline cost profile ({manifest.run_id})",
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
        "| arm_name | condition | total_ms | n_repeats | top_span | top_span_ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in manifest.arms:
        top = arm.span_records[0] if arm.span_records else None
        top_name = top.name if top else "n/a"
        top_ms = f"{top.total_ms:.3f}" if top else "n/a"
        lines.append(
            f"| {arm.arm_name} | {arm.condition} | {arm.total_ms:.3f} | {arm.n_repeats} | "
            f"{top_name} | {top_ms} |"
        )
    lines.extend(
        [
            "",
            "## Cost gate",
            "",
            f"Max on-policy epoch seconds: `{manifest.cost_gate.max_on_policy_epoch_seconds:.0f}`",
            "",
            f"Allowed strategy: **{manifest.cost_gate.allowed_strategy}**",
            "",
            f"Enumeration bound (enum/closure/verifier > 50% warm wall time): "
            f"**{manifest.cost_gate.enumeration_bound}**",
            "",
            "Bottlenecks:",
        ]
    )
    for bottleneck in manifest.cost_gate.bottlenecks:
        lines.append(f"- {bottleneck}")
    lines.extend(
        [
            "",
            "## On-policy feasibility",
            "",
            f"Strategy: **{manifest.on_policy.strategy}**",
            "",
            f"Projected seconds per target: `{manifest.on_policy.projected_seconds_per_target:.4f}`",
            "",
            f"Projected seconds for 108 targets: `{manifest.on_policy.projected_seconds_for_108_targets:.2f}`",
            "",
            f"Extrapolated five-seeds seconds: `{manifest.on_policy.extrapolated_five_seeds_seconds:.2f}`",
            "",
            f"Extrapolated DAgger round seconds: `{manifest.on_policy.extrapolated_dagger_round_seconds:.2f}`",
            "",
            f"Extrapolated confirmation suite seconds: `{manifest.on_policy.extrapolated_confirmation_suite_seconds:.2f}`",
            "",
            f"Rationale: {manifest.on_policy.rationale}",
            "",
            "## Disposition",
            "",
            f"**{manifest.disposition}**",
            "",
            manifest.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The cost spans are measured "
            "over deterministic CPU-only operations with synthetic model and verifier signals. "
            "Production on-policy training requires real model decode timing, GPU kernel "
            "profiles, and checkpoint benchmarks before any ship claim.",
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
            "python -m scripts.profile_flow_pipeline --describe",
            "python -m scripts.profile_flow_pipeline --fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def validate_manifest(manifest: FlowPipelineManifestV1) -> list[str]:
    """Validate the flow-pipeline cost-profile fixture manifest."""
    errors: list[str] = []
    if manifest.matrix_set != MATRIX_SET:
        errors.append(f"matrix_set mismatch: {manifest.matrix_set}")
    if manifest.matrix_version != MATRIX_VERSION:
        errors.append(f"matrix_version mismatch: {manifest.matrix_version}")
    if manifest.n_cases != len(manifest.cases):
        errors.append("n_cases does not match len(cases)")
    if manifest.n_arms != len(ARM_NAMES):
        errors.append("n_arms does not match len(ARM_NAMES)")
    arm_conditions = {(a.arm_name, a.condition) for a in manifest.arms}
    for arm_name in ARM_NAMES:
        for condition in ("cold", "warm"):
            if (arm_name, condition) not in arm_conditions:
                errors.append(f"missing arm/condition: {arm_name}/{condition}")
    case_ids = {c.case_id for c in manifest.cases}
    if len(case_ids) != len(manifest.cases):
        errors.append("duplicate case_id")
    for case in manifest.cases:
        if case.arm_name not in ARM_NAMES:
            errors.append(f"{case.case_id}: unknown arm {case.arm_name!r}")
        if case.wall_seconds < 0:
            errors.append(f"{case.case_id}: negative wall_seconds")
    return errors
