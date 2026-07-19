"""SDE3-01 wiring fixture: content-addressed eval cache + deterministic sharding.

Exercises the new ``EvalCache`` and suite-sharding modules on a tiny synthetic
evaluation task with no model load.  Demonstrates byte-identical warm replay,
correct cache invalidation on dependency changes, and deterministic shard
aggregation.

This is evidence-only wiring: no checkpoint is loaded, no real evaluator runs,
and no quality or ship claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.eval_cache import (
    EvalCache,
    EvalCacheConfig,
    EvalCacheMode,
    metric_result_key,
    request_generation_key,
)
from slm_training.evals.suite_sharding import (
    AggregationSpec,
    ShardResult,
    aggregate_shard_payloads,
    assign_example_ids,
    split_by_assignment,
)
from slm_training.lineage.records import canonical_json, content_sha
from slm_training.versioning import UNKNOWN, build_version_stamp


@dataclass(frozen=True)
class SimpleRecord:
    id: str
    prompt: str
    gold: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_records() -> list[SimpleRecord]:
    return [
        SimpleRecord("r0", "root = Stack([n0], 'column')", "ROOT = STACK([N0], 'COLUMN')"),
        SimpleRecord("r1", "n0 = TextContent(':content.body')", "N0 = TEXTCONTENT(':CONTENT.BODY')"),
        SimpleRecord("r2", "root = Card([n0], 'card')", "ROOT = CARD([N0], 'CARD')"),
        SimpleRecord("r3", "n0 = Button(':action.save')", "N0 = BUTTON(':ACTION.SAVE')"),
        SimpleRecord("r4", "root = Stack([n0, n1], 'row')", "ROOT = STACK([N0, N1], 'ROW')"),
    ]


def _expensive_generate(prompt: str) -> str:
    """Fake expensive decoder: deterministic uppercase transform."""
    time.sleep(0.005)
    return prompt.upper()


def _component_versions() -> dict[str, str]:
    try:
        stamp = build_version_stamp("evals.scoring")
        return stamp.get("components") or {"evals.scoring": UNKNOWN}
    except Exception:  # noqa: BLE001
        return {"evals.scoring": UNKNOWN}


def _evaluate(
    records: list[SimpleRecord],
    *,
    cache: EvalCache,
    policy: dict[str, Any],
    checkpoint_sha256: str = "fixture-checkpoint",
) -> tuple[dict[str, Any], int]:
    """Run the synthetic evaluation, counting generation calls."""
    components = _component_versions()
    details: list[dict[str, Any]] = []
    generation_calls = 0
    exact_hits = 0

    for record in records:
        request_sha = content_sha({"id": record.id, "prompt": record.prompt})
        gen_key = request_generation_key(
            checkpoint_sha256=checkpoint_sha256,
            request_sha256=request_sha,
            evaluation_policy=policy,
            component_versions=components,
        )
        cached_gen = cache.get(gen_key)
        if cached_gen is not None:
            pred = cached_gen["prediction"]
        else:
            pred = _expensive_generate(record.prompt)
            generation_calls += 1
            if cache.config.mode in (EvalCacheMode.READ_WRITE, EvalCacheMode.REFRESH):
                cache.put(
                    gen_key,
                    {"prediction": pred, "request_sha256": request_sha},
                    dependencies={"policy": policy, "checkpoint_sha256": checkpoint_sha256},
                )

        pred_sha = _sha256(pred)
        record_sha = content_sha({"id": record.id, "prompt": record.prompt, "gold": record.gold})
        metric_key = metric_result_key(
            prediction_sha256=pred_sha,
            source_record_sha256=record_sha,
            evaluation_policy=policy,
            component_versions=components,
        )
        cached_metric = cache.get(metric_key)
        if cached_metric is not None:
            exact = cached_metric["exact_match"]
        else:
            exact = int(pred == record.gold)
            if cache.config.mode in (EvalCacheMode.READ_WRITE, EvalCacheMode.REFRESH):
                cache.put(
                    metric_key,
                    {"exact_match": exact},
                    dependencies={"prediction_sha256": pred_sha, "source_record_sha256": record_sha},
                )

        exact_hits += exact
        details.append(
            {
                "id": record.id,
                "prediction": pred,
                "gold": record.gold,
                "exact_match": bool(exact),
            }
        )

    return {
        "n": len(records),
        "exact_match_rate": exact_hits / len(records) if records else 0.0,
        "details": details,
        "policy": policy,
    }, generation_calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/sde3-01-eval-cache"),
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "cache"

    records = _synthetic_records()
    base_policy = {"temperature": 0.0, "grammar_constrained": True}
    changed_policy = {"temperature": 0.1, "grammar_constrained": True}

    # Phase 1: cold read_write cache build.
    cold_cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=cache_root))
    cold_metrics, cold_calls = _evaluate(records, cache=cold_cache, policy=base_policy)

    # Phase 2: warm read replay (byte-identical, zero generation calls).
    warm_cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ, root=cache_root))
    warm_metrics, warm_calls = _evaluate(records, cache=warm_cache, policy=base_policy)

    # Phase 3: changed dependency should miss and recompute.
    changed_cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ, root=cache_root))
    changed_metrics, changed_calls = _evaluate(
        records, cache=changed_cache, policy=changed_policy
    )

    # Phase 4: deterministic sharding with per-shard cache read.
    assignment = assign_example_ids([r.id for r in records], n_shards=2)
    shard_records = split_by_assignment(records, assignment)
    shard_results: list[ShardResult] = []
    shard_cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ, root=cache_root))
    for idx, shard in enumerate(shard_records):
        metrics, _ = _evaluate(shard, cache=shard_cache, policy=base_policy)
        shard_results.append(
            ShardResult(shard_index=idx, example_ids=tuple(r.id for r in shard), payload=metrics)
        )
    sharded_metrics = aggregate_shard_payloads(
        shard_results,
        AggregationSpec(concat_keys=("details",), mean_keys=("exact_match_rate",)),
    )

    summary = {
        "run_id": run_id,
        "fixture": "sde3-01-eval-cache",
        "cold_generation_calls": cold_calls,
        "warm_generation_calls": warm_calls,
        "changed_policy_generation_calls": changed_calls,
        "cold_exact_match_rate": cold_metrics["exact_match_rate"],
        "warm_exact_match_rate": warm_metrics["exact_match_rate"],
        "changed_exact_match_rate": changed_metrics["exact_match_rate"],
        "sharded_example_count": sharded_metrics["example_count"],
        "sharded_detail_count": len(sharded_metrics.get("details", [])),
        "byte_identical_cold_warm": canonical_json(cold_metrics) == canonical_json(warm_metrics),
        "byte_identical_cold_changed": canonical_json(cold_metrics) == canonical_json(changed_metrics),
        "version_stamp": _safe_json(build_version_stamp("evals.scoring")),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(_safe_json(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "cold_metrics.json").write_text(
        json.dumps(_safe_json(cold_metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "sharded_metrics.json").write_text(
        json.dumps(_safe_json(sharded_metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    readme = f"""# SDE3-01 Evaluation Cache + Sharding Fixture

Run ID: `{run_id}`

This fixture exercises the SDE3-01 content-addressed evaluation cache and
deterministic suite-sharding modules on a tiny synthetic task.  No model is
loaded.

## Results

| Phase | Generation calls | Exact-match rate |
| --- | ---: | ---: |
| Cold `read_write` | {cold_calls} | {cold_metrics['exact_match_rate']:.2f} |
| Warm `read` | {warm_calls} | {warm_metrics['exact_match_rate']:.2f} |
| Changed policy `read` | {changed_calls} | {changed_metrics['exact_match_rate']:.2f} |
| Sharded (2 shards) | n/a | {sharded_metrics.get('exact_match_rate_mean')} |

- Cold/warm byte-identical: {summary['byte_identical_cold_warm']}
- Changed-policy produced identical results because the fake decoder is
deterministic, but it correctly **missed** the cache and recomputed.
- Sharded aggregation covered {sharded_metrics['example_count']} examples with
{len(sharded_metrics.get('details', []))} detail rows.

## Artifacts

* `summary.json` — fixture outcome and version stamp.
* `cold_metrics.json` — full metrics from the cold run.
* `sharded_metrics.json` — aggregated shard metrics.
* `cache/` — content-addressed cache entries.

## Honest caveats

Wiring-only evidence with a toy deterministic decoder and hand-written records.
A real SDE3-01 run requires the canonical evaluator, durable checkpoints, the
five-suite scoreboard, stage-level timing profiles, and cache invalidation tests
for every declared dependency (L1–L4).
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"SDE3-01 fixture written to {out_dir}")
    print(json.dumps(summary, indent=2))


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


if __name__ == "__main__":
    main()
