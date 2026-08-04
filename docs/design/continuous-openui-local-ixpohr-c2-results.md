# Continuous autotrain: 2026-08-04 (session ixpohr) cycle 2 — decode-timeout harness signal, not a model result

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `4e9f27d3` (this session's cycle-1 docs commit, on top of `main` tip `eba6db30`)

**Verdict:** measurement incomplete. Both `control` and `component-plan` (matched,
1,755,764 params, seed 100002) hit `decode_timeout_count=3/3` on the smoke
suite. This is a **harness signal**, not a model comparison — both arms fail
identically, so it is not attributable to the `component-plan` lever.

## Root cause (forensic evidence)

From `eval.json` / `scoreboard.json` for both arms:

| Field | Value |
| --- | --- |
| `smoke.n` | 3 |
| `decode_timeout_count` | 3 (all) |
| `evaluation_policy.decode_timeout_seconds` (requested, per record) | 8.0 |
| `effective_decode_timeout_seconds_min/max` | 24.0 |
| `decode_batch_size_max` | 3 |
| `generate_batch_size` (model config) | 16 |
| `evaluation_wall_seconds` | 51.815429 |
| `decode_stats.compiler_ms_sum` (partial) | 23,341.878 |
| `decode_stats.compiler_prefill_states` (partial) | 33 |
| `decode_stats.compiler_prefill_tokens` (partial) | 8,448 |

`24.0 = 8.0 (requested per-record timeout) x 3 (chunk_record_n)` — an exact
match, which pins the mechanism: the smoke suite's tiny `n=3` is smaller than
the model's baked `generate_batch_size=16`
([`twotower.py:360`](../../src/slm_training/models/twotower.py)), so
`eval_runner.py`'s chunking loop
([`eval_runner.py:2154-2158`](../../src/slm_training/harnesses/model_build/eval_runner.py))
groups **all 3 documents into one decode chunk** rather than 3
individually-timed chunks. The one combined chunk gets a combined timeout
(`requested_seconds x chunk_record_n`), and the constrained-decode compiler
search alone consumed `23.3s` of that `24s` budget — almost certainly on
largely just the first bundled document (`compiler_prefill_states=33`,
`compiler_prefill_tokens=8448`, `compiler_candidates=291`) — leaving no time
for generation or the other two documents. All three are marked `timed_out`
together even though the batch may contain only one genuinely compiler-heavy
document.

This defeats `eval_runner.py`'s own per-record adaptive redistribution
(`_effective_record_decode_timeout`,
[`eval_runner.py:1187-1208`](../../src/slm_training/harnesses/model_build/eval_runner.py)),
which is designed to reallocate unused wall time across remaining records —
but that logic only has an effect at chunk granularity, and here
`chunk_record_n == n == 3`, so there is only one chunk with nothing left to
redistribute across.

## Candidate minimal fix (not applied this cycle)

Let eval-time config override the model's baked `generate_batch_size` for
the decode chunking loop only (independent of the checkpoint's
`TwoTowerConfig`), then set it to `1` for the screening role so each smoke
document gets its own chunk:

1. `src/slm_training/harnesses/model_build/config.py` — add
   `ModelBuildConfig.generate_batch_size: int | None = None`.
2. `src/slm_training/harnesses/model_build/eval_runner.py:1513-1519` — prefer
   `getattr(config, "generate_batch_size", None)` over the plugin-config
   fallback when set.
3. `scripts/evaluate_model.py` — add a `--generate-batch-size` CLI flag wired
   to `ModelBuildConfig(generate_batch_size=...)`.
4. `src/slm_training/autoresearch/schemas.py` — register `generate_batch_size`
   on `ExperimentKnobs` (`int`, `ge=1`) and add it to `DEFAULT_ALLOWED_KNOBS`.
   `ExperimentKnobs` is a `StrictModel` with `extra="forbid"`, so adding this
   knob to the continuous driver's screening recipe **before** the schema is
   updated would raise a validation error on every campaign, not just this
   one — confirmed by inspection, not applied speculatively.
5. `scripts/run_autotrain_continuous.py`'s `_matrix()`/`knobs()` — set
   `generate_batch_size=1` only when `role == "screening"`.

The translation path from the continuous driver's `knobs` dict through
`scripts.autoresearch` execution into the actual `evaluate_model.py` CLI
invocation was not fully traced this cycle. A regression test proving a
`ModelBuildConfig(generate_batch_size=1)` override produces per-record (not
per-suite) decode chunks is required before landing.

This is deliberately **deferred to a dedicated `improve-openui-harnesses`
session** rather than a speculative same-cycle multi-file edit, matching this
loop's own precedent for cross-cutting harness changes (see
[`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md),
which deferred a similarly-scoped two-bug frozen-replay fix rather than
attempting it speculatively).

## SDLC Phase A

**Not positive** (`primary_metric_unavailable`, `fixture_insufficient_n_alone`,
`harness_failure` on both arms). No stack layer opened for this cycle.

## Next priorities

1. Dedicated `improve-openui-harnesses` session: implement the
   `generate_batch_size` override above, with a regression test, before
   replaying the identical frozen c2 arm (manifest
   `5b2ed5b92b6d16fbba411d6f65a6a93e1151519e7333d5532c0a0c515e6f8153`).
2. Do not retry the identical c2 recipe speculatively — the failure is
   deterministic and wall-bound, not a stochastic flake.
3. Once repaired, prefer a distinct successor hypothesis to the exhausted c1
   bounds-only arm per c1's own priority ranking
   ([`continuous-openui-local-ixpohr-c1-results.md`](continuous-openui-local-ixpohr-c1-results.md)).

## Checkpoint determinism cross-check

Both checkpoints are byte-identical to the prior session `j48f8u`'s c2
`component-plan` checkpoints
([`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)):
control SHA `6abf57d4...db3512b`, component-plan SHA
`20e573b1...f0f8a8e741`. Training is fully deterministic across sessions —
`j48f8u`'s eval on these exact checkpoints *completed* and measured a
structural-similarity win, while this session's eval on the same checkpoints
hit the decode-timeout wall. That confirms the failure is evaluation-time
and resource/timing-dependent (host load, CPU contention), not a
deterministic property of the checkpoint or the `component-plan` lever
itself — consistent with the chunking root cause above.

Machine evidence:
[`continuous-openui-local-ixpohr-c2-results.json`](continuous-openui-local-ixpohr-c2-results.json).
