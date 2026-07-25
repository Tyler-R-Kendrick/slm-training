"""Census first semantic failures from immutable generation manifests or replay bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.semantic_failure import (
    TAXONOMY_VERSION,
    SemanticFailureTaxonomyV1,
    trace_semantic_failure,
)
from slm_training.evals.suite_sharding import assign_example_ids
from slm_training.harness_core.eval_cache import (
    EvalCache,
    EvalCacheConfig,
    EvalCacheMode,
    metric_result_key,
)
from slm_training.versioning import build_version_stamp


def _portable_paths(value: Any) -> Any:
    """Keep committed evidence independent of the checkout location."""
    if isinstance(value, dict):
        return {key: _portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_paths(item) for item in value]
    if isinstance(value, str) and "/outputs/" in value:
        return "outputs/" + value.split("/outputs/", 1)[1]
    return value


def _read_rows(paths: list[Path], replay_bundle: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if replay_bundle:
        bundle = json.loads(replay_bundle.read_text(encoding="utf-8"))
        records = bundle.get("records") or {}
        for group_index, group in enumerate(bundle.get("sets") or ()):
            suite = str(group.get("suite") or "")
            for detail in group.get("details") or ():
                record = records.get(f"{suite}:{detail.get('id')}")
                if isinstance(record, dict):
                    rows.append({
                        # Labels may recur for several captured eval envelopes; the
                        # immutable set position keeps sharding/caching identities unique.
                        "id": f"{group.get('label')}:{suite}:{detail.get('id')}:{group_index}", "record": record,
                        "prediction": detail.get("prediction"), "suite": suite,
                        "checkpoint_sha256": group.get("checkpoint_sha256"),
                        "generation_request": detail.get("generation_request"),
                        "labels": detail.get("semantic_labels") or {},
                    })
    return rows


def census(rows: list[dict[str, Any]], *, cache: EvalCache | None = None) -> dict[str, Any]:
    traces = []
    rejected = []
    cache_hits = 0
    for index, row in enumerate(rows):
        if not isinstance(row.get("record"), dict) or not isinstance(row.get("prediction"), str):
            rejected.append({"row": index, "reason": "record_or_prediction_missing"})
            continue
        record = ExampleRecord.from_dict(row["record"])
        prediction = row["prediction"]
        request = (
            GenerationRequest.from_dict(row["generation_request"])
            if isinstance(row.get("generation_request"), dict) else None
        )
        key = metric_result_key(
            prediction_sha256=hashlib.sha256(prediction.encode()).hexdigest(),
            source_record_sha256=hashlib.sha256(
                json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            evaluation_policy={
                "taxonomy_version": TAXONOMY_VERSION,
                "generation_id": str(row.get("id") or record.id),
            },
            component_versions={"harness.eval.semantic_failure_census": "v1"},
            metric_name="semantic_failure_v1",
        )
        cached = cache.get(key) if cache else None
        if cached:
            traces.append(cached)
            cache_hits += 1
            continue
        labels = {
            key: value.get("verdict") if isinstance(value, dict) else value
            for key, value in (row.get("labels") or {}).items()
            if key in {"human", "agentv"} and isinstance(value.get("verdict") if isinstance(value, dict) else value, bool)
        }
        trace = trace_semantic_failure(
            prediction, record, request=request, generation_id=str(row.get("id") or record.id),
            checkpoint_sha256=row.get("checkpoint_sha256"), suite=row.get("suite"), labels=labels,
        ).to_dict()
        if cache and cache.config.mode in {EvalCacheMode.READ_WRITE, EvalCacheMode.REFRESH}:
            cache.put(key, trace, dependencies={"taxonomy_version": TAXONOMY_VERSION})
        traces.append(trace)
    first = Counter(trace["first_failure_family"] for trace in traces if trace["first_failure_family"])
    all_families = Counter(family for trace in traces for family in trace["all_failure_families"])
    unknown = Counter(code for trace in traces for code in trace["unmapped_reason_codes"])
    checkpoint_hashes = sorted({trace["checkpoint_sha256"] for trace in traces if trace["checkpoint_sha256"]})
    return {
        "schema_version": "semantic_failure_census/v1",
        "taxonomy": SemanticFailureTaxonomyV1().to_dict(),
        "n_input": len(rows), "n_traces": len(traces), "n_rejected": len(rejected),
        "rejected": rejected, "cache_hits": cache_hits, "traces": traces,
        "first_failure_counts": dict(sorted(first.items())),
        "all_failure_counts": dict(sorted(all_families.items())),
        "unmapped_reason_codes": dict(sorted(unknown.items())),
        "unique_checkpoint_sha256s": checkpoint_hashes,
        "coverage_blockers": [
            "requires_at_least_100_examples" if len(traces) < 100 else None,
            "requires_at_least_10_checkpoint_hashes" if len(checkpoint_hashes) < 10 else None,
        ],
    }


def _combine(paths: list[Path]) -> dict[str, Any]:
    """Recombine shard payloads only when their manifest and trace IDs agree."""
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = payloads[0].get("sharding", {}).get("all_input_ids") if payloads else None
    if not isinstance(expected, list) or any(
        payload.get("sharding", {}).get("all_input_ids") != expected for payload in payloads
    ):
        raise ValueError("shard manifests do not agree")
    traces = [trace for payload in payloads for trace in payload.get("traces") or ()]
    seen = [trace["generation_id"] for trace in traces]
    if len(seen) != len(set(seen)):
        raise ValueError("shards contain duplicate generation IDs")
    if set(seen) != set(expected):
        raise ValueError("shards are incomplete or contain unexpected generation IDs")
    combined = census([], cache=None)
    combined.update({
        "n_input": len(expected), "n_traces": len(traces), "traces": sorted(traces, key=lambda item: item["generation_id"]),
        "n_rejected": 0, "rejected": [], "cache_hits": sum(int(payload.get("cache_hits", 0)) for payload in payloads),
        "sharding": {"combined_from": [str(path) for path in paths], "all_input_ids": expected},
    })
    combined["first_failure_counts"] = dict(sorted(Counter(
        trace["first_failure_family"] for trace in traces if trace["first_failure_family"]
    ).items()))
    combined["all_failure_counts"] = dict(sorted(Counter(
        family for trace in traces for family in trace["all_failure_families"]
    ).items()))
    combined["unmapped_reason_codes"] = dict(sorted(Counter(
        reason for trace in traces for reason in trace["unmapped_reason_codes"]
    ).items()))
    combined["unique_checkpoint_sha256s"] = sorted({
        trace["checkpoint_sha256"] for trace in traces if trace["checkpoint_sha256"]
    })
    combined["coverage_blockers"] = [
        "requires_at_least_100_examples" if len(traces) < 100 else None,
        "requires_at_least_10_checkpoint_hashes" if len(combined["unique_checkpoint_sha256s"]) < 10 else None,
    ]
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--generation-manifest", action="append", type=Path, default=[])
    parser.add_argument("--replay-bundle", type=Path)
    parser.add_argument("--eval-snapshot", help="immutable eval snapshot digest or label")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cache-mode", choices=[item.value for item in EvalCacheMode], default="off")
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/eval_cache"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--combine", action="append", type=Path, default=[])
    parser.add_argument("--include-regret-oracles", action="store_true")
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({"taxonomy": SemanticFailureTaxonomyV1().to_dict(), "human_rating_gate": False}, indent=2))
        return 0
    if not args.out or not args.run_dir or not (args.generation_manifest or args.replay_bundle or args.combine):
        parser.error("census requires --out, --run-dir, and a manifest, replay bundle, or --combine")
    if args.resume and args.out.exists():
        print(json.dumps({"status": "resumed_existing", "out": str(args.out)}))
        return 0
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode(args.cache_mode), root=args.cache_root))
    rows = _read_rows(args.generation_manifest, args.replay_bundle)
    if args.combine:
        payload = _combine(args.combine)
    else:
        input_ids = [str(row.get("id") or index) for index, row in enumerate(rows)]
        assignment = assign_example_ids(input_ids, args.shards, seed=TAXONOMY_VERSION)
        if not 0 <= args.shard_index < args.shards:
            parser.error("--shard-index must be within --shards")
        selected = set(assignment.ids_for(args.shard_index))
        payload = census([row for index, row in enumerate(rows) if str(row.get("id") or index) in selected], cache=cache)
        payload["sharding"] = {
            "n_shards": args.shards, "shard_index": args.shard_index,
            "all_input_ids": input_ids, "selected_ids": sorted(selected),
        }
    payload["eval_snapshot"] = args.eval_snapshot
    payload["include_regret_oracles"] = args.include_regret_oracles
    payload["coverage_blockers"] = [item for item in payload["coverage_blockers"] if item]
    payload["status"] = "complete" if not payload["coverage_blockers"] else "provenance_or_coverage_blocked"
    payload["version_stamp"] = build_version_stamp("harness.eval.semantic_failure_census")
    payload["agentv"] = _portable_paths(publish_agentv_evaluation(args.run_dir, name="semantic-failure-census", claim="deterministic_semantic_failure_census", cases=[{
        "id": "accounting", "criteria": "every complete generation has one deterministic trace", "pass": not payload["rejected"],
        "failures": [row["reason"] for row in payload["rejected"]], "result": {"n": payload["n_traces"]},
    }]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "n": payload["n_traces"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
