"""Tests for the bit-exact flow cache layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleDomain, HoleId, SolverBounds
from slm_training.dsl.solver.support import (
    SearchCounters,
    SupportCertificate,
    SupportQuery,
    SupportResult,
    SupportVerdict,
)
from slm_training.harness_core.flow_cache import (
    DiskFlowCache,
    FlowCacheKey,
    FlowCacheMode,
    InMemoryFlowCache,
)
from slm_training.harnesses.experiments.slm193_flow_caches import VssClosureCacheAdapter


def _sample_key(namespace: str = "test", fingerprint: str = "abc") -> FlowCacheKey:
    return FlowCacheKey(
        namespace=namespace,
        fingerprint=fingerprint,
        component_versions={"solver": "v1"},
        extra={"hole_id": "h1"},
    )


def _sample_state() -> FiniteDomainState:
    hole_id = HoleId(namespace="toy", path=("root",), kind="slot")
    values = tuple(DomainValue.create("ok_flag", {"ok": True, "idx": i}) for i in range(2))
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


class _ToySupportProvider:
    """Minimal synthetic support provider for cache parity tests."""

    def __init__(self) -> None:
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
            problem_id="toy",
            pack_id="openui",
            constraint_version="toy-v1",
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


def _sample_result() -> SupportResult:
    query = SupportQuery(
        state_fingerprint="a" * 64,
        hole_id=HoleId(namespace="toy", path=("root",), kind="slot"),
        candidate=DomainValue.create("ok_flag", {"ok": True}),
    )
    cert = SupportCertificate(
        schema_version=1,
        query=query,
        verdict=SupportVerdict.SUPPORTED,
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
        search_order="canonical-domain-value-v1",
        explored_state_fingerprints=(),
        coverage_observations=("complete",),
        verifier_profile="toy/ok-flag",
        witness_source="toy",
        witness_digest=None,
        failure_counts=(),
        exhausted=False,
        stop_reason=None,
    )
    return SupportResult(verdict=SupportVerdict.SUPPORTED, certificate=cert)


def test_in_memory_cache_hit_miss() -> None:
    cache = InMemoryFlowCache()
    key = _sample_key()
    assert cache.get(key) is None
    assert cache.counters.misses == 1
    cache.put(key, {"value": 42})
    assert cache.get(key) == {"value": 42}
    assert cache.counters.hits == 1


def test_in_memory_cache_mode_read_blocks_write() -> None:
    cache = InMemoryFlowCache(mode=FlowCacheMode.READ)
    with pytest.raises(RuntimeError):
        cache.put(_sample_key(), {"value": 1})


def test_in_memory_cache_mode_off_always_misses() -> None:
    cache = InMemoryFlowCache(mode=FlowCacheMode.READ_WRITE)
    key = _sample_key()
    cache.put(key, {"value": 1})
    off = InMemoryFlowCache(mode=FlowCacheMode.OFF)
    # Simulate transferring the same store to an OFF-mode cache: reads miss.
    off._store = cache._store.copy()
    assert off.get(key) is None
    with pytest.raises(RuntimeError):
        off.put(key, {"value": 2})


def test_in_memory_cache_lru_eviction() -> None:
    cache = InMemoryFlowCache(max_entries=2)
    cache.put(_sample_key(fingerprint="a"), {"value": 1})
    cache.put(_sample_key(fingerprint="b"), {"value": 2})
    cache.put(_sample_key(fingerprint="c"), {"value": 3})
    assert len(cache) == 2
    assert cache.get(_sample_key(fingerprint="a")) is None


def test_in_memory_cache_reset() -> None:
    cache = InMemoryFlowCache()
    cache.put(_sample_key(), {"value": 1})
    cache.reset()
    assert len(cache) == 0
    assert cache.counters.hits == 0


def test_in_memory_cache_corruption_is_miss() -> None:
    cache = InMemoryFlowCache()
    key = _sample_key()
    cache.put(key, {"value": 1})
    # Tamper with the internal store directly.
    payload, _checksum = cache._store[key.digest()]
    payload["value"] = 2
    assert cache.get(key) is None
    assert cache.counters.misses == 1  # corrupted entry treated as miss


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    cache = DiskFlowCache(root=tmp_path)
    key = _sample_key()
    cache.put(key, {"value": 42}, dependencies={"solver": "v1"})
    assert cache.get(key) == {"value": 42}
    assert len(cache) == 1


def test_disk_cache_restart_reads_existing(tmp_path: Path) -> None:
    cache = DiskFlowCache(root=tmp_path)
    key = _sample_key()
    cache.put(key, {"value": 42})
    restart = DiskFlowCache(root=tmp_path)
    assert restart.get(key) == {"value": 42}
    assert restart.counters.hits == 1


def test_disk_cache_corruption_is_miss(tmp_path: Path) -> None:
    cache = DiskFlowCache(root=tmp_path)
    key = _sample_key()
    cache.put(key, {"value": 42})
    path = list(tmp_path.rglob("*.json"))[0]
    data = json.loads(path.read_text())
    data["payload"]["value"] = 99
    path.write_text(json.dumps(data))
    assert cache.get(key) is None


def test_disk_cache_schema_version_mismatch_is_miss(tmp_path: Path) -> None:
    cache = DiskFlowCache(root=tmp_path, schema_version=1)
    key = _sample_key()
    cache.put(key, {"value": 42})
    cache2 = DiskFlowCache(root=tmp_path, schema_version=2)
    assert cache2.get(key) is None


def test_vss_adapter_exact_closure_parity() -> None:
    from slm_training.dsl.solver.closure import exact_closure

    state = _sample_state()
    provider = _ToySupportProvider()

    # Fresh computation.
    fresh = exact_closure(state, provider)

    # Cached computation.
    cache = InMemoryFlowCache()
    adapter = VssClosureCacheAdapter(cache, backend_version=provider.backend_version)
    cached = exact_closure(state, provider, cache=adapter)

    assert cached.state.fingerprint == fresh.state.fingerprint
    assert cached.counters.cache_hits == 0
    assert cached.counters.support_queries == fresh.counters.support_queries

    # Warm run should hit everything.
    cache2 = InMemoryFlowCache()
    adapter2 = VssClosureCacheAdapter(cache2, backend_version=provider.backend_version)
    exact_closure(state, provider, cache=adapter2)
    warm = exact_closure(state, provider, cache=adapter2)
    assert warm.state.fingerprint == fresh.state.fingerprint
    assert warm.counters.cache_hits == fresh.counters.support_queries
    assert warm.counters.support_queries == 0


def test_vss_adapter_version_invalidation() -> None:
    from slm_training.dsl.solver.closure import exact_closure

    state = _sample_state()
    provider = _ToySupportProvider()

    cache = InMemoryFlowCache()
    old_adapter = VssClosureCacheAdapter(
        cache, backend_version=provider.backend_version, component_versions={"solver": "v1"}
    )
    exact_closure(state, provider, cache=old_adapter)

    new_adapter = VssClosureCacheAdapter(
        cache, backend_version=provider.backend_version, component_versions={"solver": "v2"}
    )
    result = exact_closure(state, provider, cache=new_adapter)
    assert result.counters.cache_hits == 0
    assert result.counters.support_queries == len(state.holes[0].values)
