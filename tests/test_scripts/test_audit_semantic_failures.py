from __future__ import annotations

from scripts.audit_semantic_failures import census
from slm_training.harness_core.eval_cache import (
    EvalCache,
    EvalCacheConfig,
    EvalCacheMode,
)


def test_census_cache_and_accounting_are_deterministic(tmp_path) -> None:
    rows = [{"id": "x", "prediction": "root = Stack([])", "record": {"id": "x", "prompt": "Build a Button. Placeholders: :cta.label", "openui": "root = Button(\":cta.label\")", "split": "smoke", "source": "fixture"}}]
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path / "cache"))
    one = census(rows, cache=cache)
    cache_read = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ, root=tmp_path / "cache"))
    two = census(rows, cache=cache_read)
    assert one["first_failure_counts"] == two["first_failure_counts"]
    assert one["n_traces"] == 1
    assert one["traces"][0]["trace_fingerprint"] == two["traces"][0]["trace_fingerprint"]
    assert two["cache_hits"] == 1
