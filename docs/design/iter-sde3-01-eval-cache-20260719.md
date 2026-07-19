# SDE3-01 — Content-addressed evaluation cache and deterministic sharding (2026-07-19)

Fixture-grade implementation of the cache/sharding layer requested by SLM-175.
Machine-readable evidence:
[`iter-sde3-01-eval-cache-20260719.json`](iter-sde3-01-eval-cache-20260719.json).
Linear SLM-175.

## What ran

The new `slm_training.evals.eval_cache` and `slm_training.evals.suite_sharding`
modules were exercised by `scripts/run_sde3_01_eval_cache_fixture.py` on a tiny
synthetic evaluation task with a deterministic fake decoder.

### Cache behavior

| Phase | Mode | Generation calls | Result |
| --- | --- | ---: | --- |
| Cold | `read_write` | 5 | Computed + stored |
| Warm | `read` | 0 | Byte-identical replay |
| Changed policy | `read` | 5 | Correct cache miss |

### Sharding behavior

Five synthetic records were split deterministically into two shards, evaluated
independently against the warm cache, and aggregated.  The union contained all
five example IDs with no duplicates.

## Added artifacts

- `src/slm_training/evals/eval_cache.py` — `EvalCache`, `EvalCacheKey`, layer key
  builders (`suite_result_key`, `request_generation_key`, `metric_result_key`),
  schema-versioned entries, checksum validation, atomic writes.
- `src/slm_training/evals/suite_sharding.py` — deterministic shard assignment by
  example-ID hash, record splitting, and validated aggregation.
- `src/slm_training/harnesses/model_build/eval_runner.py` — optional suite-level
  content-addressed cache integration in `evaluate()` and `evaluate_suites()`.
- `src/slm_training/harnesses/model_build/config.py` — `eval_cache_mode`,
  `eval_cache_root`, `eval_shards` config fields.
- `scripts/evaluate_model.py` — `--eval-cache-mode`, `--eval-cache-root`,
  `--eval-shards` CLI flags.
- `scripts/run_sde3_01_eval_cache_fixture.py` — wiring fixture.
- `tests/test_evals/test_eval_cache.py` and
  `tests/test_evals/test_suite_sharding.py` — regression tests.
- `docs/design/iter-sde3-01-eval-cache-20260719.md` and `.json`.

## Verdict

`inconclusive` at production scale.

The reusable cache/shard abstraction is implemented and integrated at the
suite-result level in the canonical runner, but this iteration provides only
fixture-scale evidence.  A full benchmark against the five-suite scoreboard on a
durable checkpoint is required before choosing among `adopt_exact_cache`,
`adopt_sharding_only`, `revise_storage_or_keys`, or `no_material_gain`.

## Honesty and limits

- **Wiring evidence only, not a ship claim.** The fixture uses a toy decoder and
  five hand-written records; no checkpoint is loaded and no real model runs.
- **Suite-level cache only.** L1 (context encodings), L2 (generation attempts),
  L3 (per-metric), and L4 (judge) granular caches are implemented as key
  builders but are not yet wired into the live evaluation runner.  The current
  integration caches the full suite result.
- **`--eval-shards` is a wiring flag.** Deterministic shard assignment and
  aggregation helpers exist, but the canonical runner does not yet split a suite
  across parallel shard workers.
- Component versions bumped: `harness.model_build.eval` → `v2`,
  `evals.scoring` → `v3`.
