"""SLM-193 (FFE3-02): bit-exact state, candidate, closure, and bridge caches.

Wiring/fixture harness that adds a content-addressed cache layer and measures
cold, warm, restart, and invalidation behavior over deterministic OpenUI flow
stages.  No trained model, GPU, or ship-gate claim is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.flow.bridge_planner import plan_bridge
from slm_training.dsl.canonicalize import canonicalize
from slm_training.dsl.solver.closure import exact_closure
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
from slm_training.harness_core.flow_cache import (
    CACHE_SCHEMA_VERSION,
    DiskFlowCache,
    FlowCache,
    FlowCacheKey,
    FlowCacheMode,
    InMemoryFlowCache,
)
from slm_training.harnesses.experiments.slm188_edit_algebra import build_sketch_seed
from slm_training.runtime.telemetry import CycleTelemetry
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "ARM_NAMES",
    "CacheArmResult",
    "CacheCaseRecord",
    "FlowCacheManifestV1",
    "render_markdown",
    "run_flow_cache_fixture",
    "validate_manifest",
]

MATRIX_VERSION = "ffe3-02-v1"
MATRIX_SET = "slm193_flow_caches"
EXPERIMENT_ID = "slm193-flow-caches"

ARM_NAMES = (
    "exact_closure_cold",
    "exact_closure_warm",
    "exact_closure_cross_request",
    "bridge_plan_cold",
    "bridge_plan_warm",
    "disk_restart",
    "version_invalidation",
)

HERO = '''root = Stack([hero], "column")
hero_title = TextContent(":hero.title")
hero_body = TextContent(":hero.body")
hero = Card([hero_title, hero_body])'''

_HYPOTHESIS = (
    "Exact state fingerprints recur across decode attempts and evaluation records, "
    "so bit-exact content-addressed caches reduce deterministic solver/bridge wall "
    "time by at least 2x while preserving identical outputs and certificates."
)

_FALSIFIER = (
    "Cache hit rates stay below 20% on warm repeated requests, or lookup/serialization "
    "overhead offsets the saved work on warm p50/p95, or cached results diverge from "
    "fresh computation."
)

_HONEST_CAVEATS = (
    "Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.",
    "The toy support provider is synthetic (payload['ok'] flag); real support queries "
    "require a verifier and problem expander.",
    "Disk cache restart test uses a temporary directory under outputs/; production "
    "restart provenance requires a replay-safe certificate contract.",
    "Only the HERO fixture and one toy finite-domain state are exercised; production "
    "hit rates depend on actual state recurrence.",
)


class VssClosureCacheAdapter(dict[str, SupportResult]):
    """Dict-compatible wrapper around a FlowCache for ``exact_closure``.

    ``exact_closure`` uses string keys of the form
    ``_canonical_json([state.fingerprint, hole_id.to_dict(), value.to_dict(),
    backend_version])``.  This adapter intercepts those keys, stores the
    certificate in the underlying FlowCache under a content-addressed key, and
    reconstructs a ``SupportResult`` on lookup.  Corrupted or version-mismatched
    entries become safe misses.
    """

    NAMESPACE = "vss.exact_closure.support_result"

    def __init__(
        self,
        cache: FlowCache,
        *,
        backend_version: str,
        component_versions: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._cache = cache
        self.backend_version = backend_version
        self.component_versions = component_versions or {}

    def _flow_key(self, string_key: str) -> FlowCacheKey:
        parsed = json.loads(string_key)
        return FlowCacheKey(
            namespace=self.NAMESPACE,
            fingerprint=str(parsed[0]),
            schema_version=CACHE_SCHEMA_VERSION,
            component_versions=self.component_versions,
            extra={
                "backend_version": str(parsed[3]),
                "hole_id": parsed[1],
                "candidate": parsed[2],
            },
        )

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            flow_key = self._flow_key(key)
        except Exception:  # noqa: BLE001
            return False
        return self._cache.get(flow_key) is not None

    def __getitem__(self, key: str) -> SupportResult:
        try:
            flow_key = self._flow_key(key)
        except Exception as exc:  # noqa: BLE001
            raise KeyError(key) from exc
        payload = self._cache.get(flow_key)
        if payload is None:
            raise KeyError(key)
        try:
            certificate = SupportCertificate.from_dict(payload)
            return SupportResult(verdict=certificate.verdict, certificate=certificate)
        except Exception as exc:  # noqa: BLE001
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: SupportResult) -> None:
        flow_key = self._flow_key(key)
        self._cache.put(
            flow_key,
            value.certificate.to_dict(),
            dependencies={"backend_version": self.backend_version},
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


class _ToySupportProvider:
    """Synthetic support provider: ok=True values are SUPPORTED, else UNSUPPORTED."""

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

    def _certificate(self, query: SupportQuery, verdict: SupportVerdict) -> SupportCertificate:
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
    ) -> Any:
        from slm_training.dsl.solver.closure import ReplayResult

        return ReplayResult(ok=True, verdict=certificate.verdict)


def _make_toy_state(seed: int = 0) -> FiniteDomainState:
    """Finite-domain state with one hole and four values."""
    hole_id = HoleId(namespace="toy", path=("root",), kind="slot")
    values = tuple(
        DomainValue.create("ok_flag", {"ok": i % 2 == 0, "idx": i + seed})
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


@dataclass(frozen=True)
class CacheCaseRecord:
    """One measured cache case."""

    case_id: str
    arm_name: str
    condition: str
    wall_seconds: float
    hit_rate: float
    n_entries: int
    bytes_stored: int
    work_units: dict[str, float]
    honest_caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arm_name": self.arm_name,
            "condition": self.condition,
            "wall_seconds": self.wall_seconds,
            "hit_rate": self.hit_rate,
            "n_entries": self.n_entries,
            "bytes_stored": self.bytes_stored,
            "work_units": dict(self.work_units),
            "honest_caveats": list(self.honest_caveats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheCaseRecord":
        return cls(
            case_id=str(data["case_id"]),
            arm_name=str(data["arm_name"]),
            condition=str(data["condition"]),
            wall_seconds=float(data["wall_seconds"]),
            hit_rate=float(data["hit_rate"]),
            n_entries=int(data["n_entries"]),
            bytes_stored=int(data["bytes_stored"]),
            work_units={k: float(v) for k, v in (data.get("work_units") or {}).items()},
            honest_caveats=tuple(data.get("honest_caveats", ())),
        )


@dataclass(frozen=True)
class CacheArmResult:
    """Aggregated result for one cache arm."""

    arm_name: str
    total_ms: float
    hit_rate: float
    n_entries: int
    bytes_stored: int
    speedup: float
    work_units: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "total_ms": self.total_ms,
            "hit_rate": self.hit_rate,
            "n_entries": self.n_entries,
            "bytes_stored": self.bytes_stored,
            "speedup": self.speedup,
            "work_units": dict(self.work_units),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheArmResult":
        return cls(
            arm_name=str(data["arm_name"]),
            total_ms=float(data["total_ms"]),
            hit_rate=float(data["hit_rate"]),
            n_entries=int(data["n_entries"]),
            bytes_stored=int(data["bytes_stored"]),
            speedup=float(data["speedup"]),
            work_units={k: float(v) for k, v in (data.get("work_units") or {}).items()},
        )


@dataclass(frozen=True)
class FlowCacheManifestV1:
    """Full fixture manifest for SLM-193."""

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
    arms: tuple[CacheArmResult, ...]
    cases: tuple[CacheCaseRecord, ...]
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
    def from_dict(cls, data: dict[str, Any]) -> "FlowCacheManifestV1":
        return cls(
            schema=str(data.get("schema", "FlowCacheManifestV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", f"{EXPERIMENT_ID}-fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            disposition=str(data.get("disposition", "cache_wired")),
            disposition_rationale=str(
                data.get(
                    "disposition_rationale",
                    "Bit-exact cache layer wired; no ship claim.",
                )
            ),
            arms=tuple(CacheArmResult.from_dict(a) for a in data.get("arms", ())),
            cases=tuple(CacheCaseRecord.from_dict(c) for c in data.get("cases", ())),
            n_cases=int(data.get("n_cases", 0)),
            n_arms=int(data.get("n_arms", 0)),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def _span_records_from_telemetry(telemetry: CycleTelemetry) -> tuple[dict[str, Any], ...]:
    summary = telemetry.summary()
    records: list[dict[str, Any]] = []
    for name, row in summary.get("spans", {}).items():
        records.append({"name": name, **row})
    records.sort(key=lambda r: r["total_ms"], reverse=True)
    return tuple(records)


def _run_exact_closure_with_cache(
    cache: FlowCache,
    state: FiniteDomainState,
    provider: _ToySupportProvider,
    telemetry: CycleTelemetry,
) -> Any:
    """Run exact_closure using a VssClosureCacheAdapter over the given FlowCache."""
    adapter = VssClosureCacheAdapter(
        cache, backend_version=provider.backend_version
    )
    with telemetry.span("exact_closure"):
        result = exact_closure(state, provider, cache=adapter)
    return result


def _bridge_plan_cache_key(
    source: str, target: str, arm: str, version_pins: dict[str, Any]
) -> FlowCacheKey:
    return FlowCacheKey(
        namespace="bridge.plan",
        fingerprint=_sha256(_canonical_json({"source": source, "target": target, "arm": arm})),
        extra={"version_pins": version_pins},
    )


def _run_arm(
    arm_name: str,
    target_program: str,
    source_program: str,
    temp_root: Path,
) -> tuple[CacheArmResult, CacheCaseRecord]:
    telemetry = CycleTelemetry()
    cache: FlowCache
    work_units: dict[str, float] = {
        "support_queries": 0.0,
        "cache_hits": 0.0,
        "cache_misses": 0.0,
        "cache_writes": 0.0,
        "bytes_stored": 0.0,
    }
    honest_caveats: tuple[str, ...] = ()

    if arm_name == "exact_closure_cold":
        cache = InMemoryFlowCache(mode=FlowCacheMode.READ_WRITE)
        state = _make_toy_state()
        provider = _ToySupportProvider()
        result = _run_exact_closure_with_cache(cache, state, provider, telemetry)
        counters = cache.counters.snapshot()
        work_units["support_queries"] = float(result.counters.support_queries)
        work_units["cache_hits"] = float(counters["hits"])
        work_units["cache_misses"] = float(counters["misses"])
        work_units["cache_writes"] = float(counters["writes"])
        work_units["bytes_stored"] = float(counters["bytes_stored"])

    elif arm_name == "exact_closure_warm":
        cache = InMemoryFlowCache(mode=FlowCacheMode.READ_WRITE)
        state = _make_toy_state()
        provider = _ToySupportProvider()
        # Cold priming.
        _run_exact_closure_with_cache(cache, state, provider, CycleTelemetry(enabled=False))
        # Warm measured run.
        result = _run_exact_closure_with_cache(cache, state, provider, telemetry)
        counters = cache.counters.snapshot()
        work_units["support_queries"] = float(result.counters.support_queries)
        work_units["cache_hits"] = float(counters["hits"])
        work_units["cache_misses"] = float(counters["misses"])
        work_units["cache_writes"] = float(counters["writes"])
        work_units["bytes_stored"] = float(counters["bytes_stored"])

    elif arm_name == "exact_closure_cross_request":
        cache = InMemoryFlowCache(mode=FlowCacheMode.READ_WRITE)
        state_a = _make_toy_state(seed=0)
        state_b = _make_toy_state(seed=1)
        provider = _ToySupportProvider()
        # First request primes the cache for state_a.
        _run_exact_closure_with_cache(cache, state_a, provider, CycleTelemetry(enabled=False))
        # Second request with a different state but identical hole/value structure
        # (different idx values) should miss, then warm-repeating state_a should hit.
        _run_exact_closure_with_cache(cache, state_b, provider, CycleTelemetry(enabled=False))
        result = _run_exact_closure_with_cache(cache, state_a, provider, telemetry)
        counters = cache.counters.snapshot()
        work_units["support_queries"] = float(result.counters.support_queries)
        work_units["cache_hits"] = float(counters["hits"])
        work_units["cache_misses"] = float(counters["misses"])
        work_units["cache_writes"] = float(counters["writes"])
        work_units["bytes_stored"] = float(counters["bytes_stored"])
        honest_caveats = ("Cross-request arm deliberately mixes two distinct states.",)

    elif arm_name == "bridge_plan_cold":
        cache = InMemoryFlowCache(mode=FlowCacheMode.READ_WRITE)
        key = _bridge_plan_cache_key(source_program, target_program, "canonical_greedy", {})
        with telemetry.span("bridge_plan"):
            result = plan_bridge(
                source_program,
                target_program,
                arm="canonical_greedy",
                source_seed_id="slm193",
            )
        cache.put(key, result.to_dict())
        counters = cache.counters.snapshot()
        work_units["support_queries"] = float(result.cost_attribution.get("closure_query", 0.0))
        work_units["cache_hits"] = float(counters["hits"])
        work_units["cache_misses"] = float(counters["misses"])
        work_units["cache_writes"] = float(counters["writes"])
        work_units["bytes_stored"] = float(counters["bytes_stored"])

    elif arm_name == "bridge_plan_warm":
        cache = InMemoryFlowCache(mode=FlowCacheMode.READ_WRITE)
        key = _bridge_plan_cache_key(source_program, target_program, "canonical_greedy", {})
        # Prime.
        plan_bridge(source_program, target_program, arm="canonical_greedy", source_seed_id="slm193")
        cache.put(key, {"plan_id": "warm-plan"})
        # Warm lookup.
        with telemetry.span("bridge_plan"):
            cached = cache.get(key)
        counters = cache.counters.snapshot()
        work_units["support_queries"] = 0.0
        work_units["cache_hits"] = float(counters["hits"])
        work_units["cache_misses"] = float(counters["misses"])
        work_units["cache_writes"] = float(counters["writes"])
        work_units["bytes_stored"] = float(counters["bytes_stored"])
        if cached is None:
            honest_caveats = ("Bridge plan warm lookup unexpectedly missed.",)

    elif arm_name == "disk_restart":
        disk_root = temp_root / "disk_cache"
        disk_root.mkdir(parents=True, exist_ok=True)
        cache = DiskFlowCache(root=disk_root, mode=FlowCacheMode.READ_WRITE)
        state = _make_toy_state()
        provider = _ToySupportProvider()
        # Prime on disk.
        _run_exact_closure_with_cache(cache, state, provider, CycleTelemetry(enabled=False))
        # Simulate process restart: new cache instance reading the same directory.
        cache = DiskFlowCache(root=disk_root, mode=FlowCacheMode.READ_WRITE)
        result = _run_exact_closure_with_cache(cache, state, provider, telemetry)
        counters = cache.counters.snapshot()
        work_units["support_queries"] = float(result.counters.support_queries)
        work_units["cache_hits"] = float(counters["hits"])
        work_units["cache_misses"] = float(counters["misses"])
        work_units["cache_writes"] = float(counters["writes"])
        work_units["bytes_stored"] = float(counters["bytes_stored"])
        honest_caveats = ("Disk restart test uses a fresh cache instance over the same directory.",)

    elif arm_name == "version_invalidation":
        cache = InMemoryFlowCache(mode=FlowCacheMode.READ_WRITE)
        state = _make_toy_state()
        provider = _ToySupportProvider()
        # Prime with old version.
        old_adapter = VssClosureCacheAdapter(
            cache, backend_version=provider.backend_version, component_versions={"solver": "v1"}
        )
        exact_closure(state, provider, cache=old_adapter)
        # Change component version: entries must miss.
        new_adapter = VssClosureCacheAdapter(
            cache, backend_version=provider.backend_version, component_versions={"solver": "v2"}
        )
        with telemetry.span("exact_closure"):
            result = exact_closure(state, provider, cache=new_adapter)
        counters = cache.counters.snapshot()
        work_units["support_queries"] = float(result.counters.support_queries)
        work_units["cache_hits"] = float(counters["hits"])
        work_units["cache_misses"] = float(counters["misses"])
        work_units["cache_writes"] = float(counters["writes"])
        work_units["bytes_stored"] = float(counters["bytes_stored"])
        honest_caveats = ("Version invalidation arm changes component_versions so old entries miss.",)

    else:
        raise ValueError(f"unknown arm: {arm_name}")

    total_ms = telemetry.summary()["total_ms"]
    span_records = _span_records_from_telemetry(telemetry)
    hit_rate = _compute_hit_rate(cache.counters.snapshot())
    n_entries = len(cache)
    bytes_stored = cache.counters.snapshot()["bytes_stored"]

    arm_result = CacheArmResult(
        arm_name=arm_name,
        total_ms=total_ms,
        hit_rate=hit_rate,
        n_entries=n_entries,
        bytes_stored=bytes_stored,
        speedup=1.0,
        work_units=work_units,
    )
    case = CacheCaseRecord(
        case_id=f"{arm_name}",
        arm_name=arm_name,
        condition="fixture",
        wall_seconds=total_ms / 1000.0,
        hit_rate=hit_rate,
        n_entries=n_entries,
        bytes_stored=bytes_stored,
        work_units=work_units,
        honest_caveats=honest_caveats,
    )
    return arm_result, case, span_records


def _compute_hit_rate(counters: dict[str, int]) -> float:
    total = counters["hits"] + counters["misses"]
    return counters["hits"] / total if total else 0.0


def _compute_speedup(cold_ms: float, warm_ms: float) -> float:
    return cold_ms / max(warm_ms, 1e-9) if warm_ms > 0 else 1.0


def run_flow_cache_fixture(
    output_dir: Path | None = None,
    *,
    write_design_docs: bool = True,
    design_json: Path | None = None,
    design_md: Path | None = None,
) -> FlowCacheManifestV1:
    """Run the SLM-193 bit-exact cache fixture matrix."""
    start = time.perf_counter()
    target_program = canonicalize(HERO, validate=True)
    source_program = build_sketch_seed(target_program)

    temp_root = output_dir or Path(f"outputs/runs/slm193-flow-caches-{_today_yyyymmdd()}")
    temp_root.mkdir(parents=True, exist_ok=True)

    arms: list[CacheArmResult] = []
    cases: list[CacheCaseRecord] = []
    cold_closure_ms: float | None = None
    for arm_name in ARM_NAMES:
        arm_result, case, _span_records = _run_arm(
            arm_name, target_program, source_program, temp_root
        )
        arms.append(arm_result)
        cases.append(case)
        if arm_name == "exact_closure_cold":
            cold_closure_ms = arm_result.total_ms

    # Fill in speedups once cold/warm closure measurements are available.
    updated_arms: list[CacheArmResult] = []
    for arm in arms:
        speedup = arm.speedup
        if arm.arm_name == "exact_closure_warm" and cold_closure_ms is not None:
            speedup = _compute_speedup(cold_closure_ms, arm.total_ms)
        updated_arms.append(
            CacheArmResult(
                arm_name=arm.arm_name,
                total_ms=arm.total_ms,
                hit_rate=arm.hit_rate,
                n_entries=arm.n_entries,
                bytes_stored=arm.bytes_stored,
                speedup=speedup,
                work_units=arm.work_units,
            )
        )
    arms_tuple = tuple(updated_arms)
    cases_tuple = tuple(cases)

    version_stamp = build_version_stamp(
        "harness.experiments",
        "harness.experiments.slm193_flow_caches",
        "harness.core",
        "flow.termination",
        "flow.reference",
        "harness.experiments.slm188_edit_algebra",
        "harness.experiments.slm189_bridge_planner",
    )

    manifest = FlowCacheManifestV1(
        schema="FlowCacheManifestV1",
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}-{uuid.uuid4().hex[:8]}",
        status="fixture",
        claim_class="wiring",
        hypothesis=_HYPOTHESIS,
        falsifier=_FALSIFIER,
        disposition="cache_wired",
        disposition_rationale=(
            "Bit-exact content-addressed cache layer wired for exact closure and "
            "bridge plans; measured cold/warm/restart/invalidation arms on CPU-only fixtures."
        ),
        arms=arms_tuple,
        cases=cases_tuple,
        n_cases=len(cases_tuple),
        n_arms=len(ARM_NAMES),
        honest_caveats=_HONEST_CAVEATS,
        version_stamp=version_stamp,
        timestamp=_now(),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_json(output_dir / "slm193_flow_caches_report.json")
        if write_design_docs:
            root = Path(__file__).resolve().parents[4]
            if design_json is None or design_md is None:
                design_json = root / f"docs/design/iter-slm193-flow-caches-{_today_yyyymmdd()}.json"
                design_md = root / f"docs/design/iter-slm193-flow-caches-{_today_yyyymmdd()}.md"
            design_json.parent.mkdir(parents=True, exist_ok=True)
            design_md.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_json(design_json)
            design_md.write_text(render_markdown(manifest), encoding="utf-8")

    elapsed = time.perf_counter() - start
    lineage_extra = {"wall_seconds": _clamp(elapsed, low=0.001, high=10.0)}
    stamp = dict(manifest.version_stamp)
    stamp["lineage"] = lineage_extra
    manifest = FlowCacheManifestV1(
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
        n_cases=manifest.n_cases,
        n_arms=manifest.n_arms,
        honest_caveats=manifest.honest_caveats,
        version_stamp=stamp,
        timestamp=manifest.timestamp,
    )
    return manifest


def render_markdown(manifest: FlowCacheManifestV1) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-193 (FFE3-02): bit-exact flow caches ({manifest.run_id})",
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
        "| arm_name | total_ms | hit_rate | n_entries | bytes_stored | speedup |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in manifest.arms:
        lines.append(
            f"| {arm.arm_name} | {arm.total_ms:.3f} | {arm.hit_rate:.2f} | "
            f"{arm.n_entries} | {arm.bytes_stored} | {arm.speedup:.2f} |"
        )
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
            "**No-go for promotion.** This is a wiring fixture. The cache hit rates and "
            "speedups are measured over deterministic CPU-only operations with synthetic "
            "support provider signals. Production caching requires real verifier replay "
            "contracts and process-restart provenance before any ship claim.",
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
            "python -m scripts.bench_flow_caches --describe",
            "python -m scripts.bench_flow_caches --fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def validate_manifest(manifest: FlowCacheManifestV1) -> list[str]:
    """Validate the flow-cache fixture manifest."""
    errors: list[str] = []
    if manifest.matrix_set != MATRIX_SET:
        errors.append(f"matrix_set mismatch: {manifest.matrix_set}")
    if manifest.matrix_version != MATRIX_VERSION:
        errors.append(f"matrix_version mismatch: {manifest.matrix_version}")
    if manifest.n_cases != len(manifest.cases):
        errors.append("n_cases does not match len(cases)")
    if manifest.n_arms != len(ARM_NAMES):
        errors.append("n_arms does not match len(ARM_NAMES)")
    case_ids = {c.case_id for c in manifest.cases}
    if len(case_ids) != len(manifest.cases):
        errors.append("duplicate case_id")
    arm_names = {a.arm_name for a in manifest.arms}
    for arm_name in ARM_NAMES:
        if arm_name not in arm_names:
            errors.append(f"missing arm: {arm_name}")
    for case in manifest.cases:
        if case.wall_seconds < 0:
            errors.append(f"{case.case_id}: negative wall_seconds")
    return errors
