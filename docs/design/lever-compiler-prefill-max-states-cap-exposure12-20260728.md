# Does capping `compiler_prefill_max_states` get the hero suite under the 30s wall? (NOT SHIP)

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (suite `n=3`,
1 rep each, byte-identical recipe to the three prior docs in this chain). **Not
ship. A lever sweep, not a fix.**

## Why this run

Iterations 1-2 of this autonomous loop twice-replicated that the `exposure12`
quality-champion smoke suite cannot finish inside `decode_timeout_seconds=30`
on this hardware/config, even after two real fixes:
[lexer/scanner cache](decode-compiler-tree-lexer-cache-hero-rerun-20260728.md)
(#1173) and the
[deadline-swallow fix](decode-compiler-tree-deadline-fix-applied-rerun-20260728.md)
(landed as `30d8dd9`). Both prior docs point at `compiler_ms`
(`build_completion_forest`'s Node-bridge grammar-validation round trip) as the
dominant cost, and the lexer-cache doc separately reported
`compiler_prefill_batches` growing 3→17 and 4→33 across its before/after
comparison. This session tests the natural next question that growth number
raises: is the neural prefill-batch width itself now a material contributor to
wall time, such that bounding it would let records finish sooner?

## Hypothesis

Capping `compiler_prefill_max_states` to 1 (forcing one ambiguous grammar
state per neural prefill batch — `scripts/run_perf_matrix.py`'s existing `C9`
arm) reduces `latency_ms_p50`/`p95` and/or changes `decode_outcome` away from
`runtime_timeout` for the 3 isolated smoke records, relative to the default
auto-batched arm (`max_states=0`, CPU default 4).

## Why no new decode machinery was written

`compiler_prefill_max_states` / `compiler_prefill_token_budget` already exist
as `ModelBuildConfig` fields (`twotower.py:487-488`) and are already swept as
a lever pair by `scripts/run_perf_matrix.py`'s `C9` (serial, `max_states=1`)
vs `C10` (auto, `max_states=0`) rows — no new decode-time code. The only gap
was plumbing: `scripts/evaluate_model.py` has no
`--compiler-prefill-max-states` flag, and the key was unregistered in
`src/slm_training/flags/levers.py`'s `LEVER_BY_KEY`
(`ruleset_from_mapping` rejects unregistered keys), so `--flags-json`
couldn't reach it on the isolated-record eval path used by this whole doc
chain. Fixed with a single `_num(...)` registration, default `0` — byte-
identical to `ModelBuildConfig`'s own default, so an empty ruleset is
unaffected — the same tiny, reversible, default-off pattern as every other
`LEVER_FLAGS` entry.

## Recipe

Byte-identical to all three prior docs: rebuild `lever_exposure12_v1` (107
records, 5 `abstraction_ladder`), train the identical scratch checkpoint
(`checkpoint_sha256=cadeb1d3...` — matches all three prior docs bit-for-bit),
then evaluate each of the 3 smoke records **in isolation**
(`--eval-limit 1 --eval-offset {0,1,2}`), once per arm.

```bash
# baseline arm (auto, unchanged default)
python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite smoke \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --device cpu \
  --checkpoint outputs/runs/exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --eval-limit 1 --eval-offset <0|1|2>

# capped arm (diagnostic override)
... same flags ... --flags-json '{"compiler_prefill_max_states": 1}'
```

`env -u NODE_OPTIONS` prefix used for build/train/eval, same sandbox caveat
as all three prior docs. `code_git_sha=30d8dd9` (iteration 2's landed fix),
`code_dirty=true` (this session's lever registration not yet committed at
eval time).

## Results: baseline (auto) vs capped (serial, `max_states=1`)

| record | decode_outcome (base → capped) | meaningful (base → capped) | latency_ms_p50 base | latency_ms_p50 capped | Δ |
| --- | --- | --- | ---: | ---: | ---: |
| smoke_hero_01 (offset 0) | runtime_timeout → runtime_timeout | 0.0 → 0.0 | 30000.64 | 30200.44 | +199.8ms |
| smoke_button_01 (offset 1) | runtime_timeout → runtime_timeout | 0.0 → 0.0 | 30000.55 | 30000.41 | −0.14ms |
| smoke_callout_01 (offset 2) | runtime_timeout → runtime_timeout | 0.0 → 0.0 | 30000.49 | 30100.39 | +99.9ms |

Both arms ran the byte-identical `cadeb1d3...` checkpoint — a genuine same-
model comparison. `latency_ms_p95` equals `latency_ms_p50` on every row
(`n=1`).

## Decision

**REJECT the hypothesis.** Capping `compiler_prefill_max_states` to 1 changes
neither `decode_outcome` (all 6 runs across both arms report
`runtime_timeout`) nor `latency_ms_p50` in any consistent or material
direction — deltas of −0.14ms / +99.9ms / +199.8ms against a 30000ms wall are
noise-scale, not a trend, and if anything the capped arm is marginally
*slower* on 2 of 3 records.

## Interpretation

This is exactly what reading the code predicts, and this run confirms it
empirically rather than by argument alone. `compiler_ms` — the timer
identified as ~97% of wall time in
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
— wraps only the `build_completion_forest` call in `twotower.py` (~line
10341). `compiler_prefill_max_states` instead bounds a *separate* denoiser-
forward batching loop inside `_select_compiler_path` (`twotower.py`
~9797-9847), timed under `trie_ms`, which both prior isolated-record
diagnostics already measured as negligible (67-72ms total, vs `compiler_ms`'s
~29200ms). Bounding a batch-width knob on a sub-1% cost cannot move a
97%-dominated wall-clock outcome. The `compiler_prefill_batches` 3→17 / 4→33
growth that motivated this session's hypothesis tracks the number of decode-
loop iterations × per-iteration prefill chunking — a correlated symptom of
more grammar-forest positions being explored, not a cost driver in its own
right.

## Side finding: `decode_stats` is entirely unavailable for timed-out records

This extends
[`decode-compiler-tree-deadline-fix-applied-rerun-20260728.md`](decode-compiler-tree-deadline-fix-applied-rerun-20260728.md)'s
`decode_stats_populated=false` observation: none of this session's 6 eval
runs (all `decode_outcome=runtime_timeout` post iteration 2's fix) populated
`metrics["decode_stats"]` at all — the key is absent from the suite JSON, not
merely zeroed. Root cause (read, not modified this session):
`eval_runner.py`'s per-chunk timeout handler (~line 1161) does
`stats = getattr(exc, "decode_stats", None)`, expecting the propagated
`TimeoutError` to carry a `.decode_stats` attribute — but nothing in the
codebase ever sets that attribute (`check_decode_deadline`'s bare
`raise TimeoutError(...)`, and the SIGALRM handler's `raise TimeoutError(...)`,
grep-confirmed zero `.decode_stats =` assignments anywhere in
`src/slm_training`). This means iteration 2's genuine win — honest
`runtime_timeout` classification — came at the cost of losing per-record
`compiler_ms`/`compiler_prefill_batches`/`trie_ms` telemetry for exactly the
records that most need root-cause attribution. **This session's REJECT
verdict rests only on `latency_ms_p50`/`p95` and `decode_outcome`** (the only
telemetry still available post-fix), a materially weaker evidence base than
iteration 1's pre-fix isolated-record diagnostic, which had full
`compiler_ms`/`forwards_count`/`trie_ms` breakdowns.

## Scope note

No claim about the `exposure12` champion's quality or promotion status.
`compiler_prefill_max_states`'s `ModelBuildConfig` default (`0`=auto) is
unchanged and the new lever registry entry's default matches it exactly, so
this change is inert for every default-configured caller (no `--flags-json`
or `OPENUI_FLAGS_JSON`/`PATH`).

## Non-goals

No performance/latency improvement claimed or found. No re-investigation of
`build_completion_forest`'s own per-call cost breakdown (still open from
`decode-compiler-tree-lexer-cache-hero-rerun-20260728.md`'s next-steps item
2). No fix to the `decode_stats`-on-timeout gap identified above — flagged,
not remediated this session. No promotion, checkpoint sync, or
`--ship-gates` claim. No multi-rep confirmation (n=1 rep per record per arm,
matching all three prior docs' own protocol).

## Next steps

1. Fix the `decode_stats`-on-timeout gap (attach partial `DecodeStats` to the
   propagated `TimeoutError`, or read `get_active_stats()` in the
   except-`TimeoutError` handler before it is cleared) so future latency-
   attribution sessions can compare `compiler_ms`/`compiler_prefill_batches`
   directly instead of only `latency_ms_p50`.
2. With that telemetry restored, directly instrument `build_completion_forest`'s
   own internal cost (Node-bridge round-trip vs candidate-set size vs cache
   hit/miss) — the actual unexplored lever for finishing inside the 30s wall,
   not compiler-tree batch width.
3. Any future wall-clock lever should still land as a default-off diagnostic
   arm (matching this session's and `run_perf_matrix.py`'s existing
   convention), never a default serving-behavior change, per AGENTS.md's
   decode invariants.

## Tests / checks

```bash
python -m pytest -q tests/test_flags/test_openfeature_experiments.py
# 10 passed in 0.15s
```

Covers the new lever registration (`test_compiler_prefill_max_states_lever_default_matches_config_default`,
`test_compiler_prefill_max_states_lever_overridable`) plus the full existing
`harness.flags` suite (unchanged behavior for every other lever).

`python -m scripts.verify_version_stamps --check` passes with the
`harness.flags` v1→v2 bump.

## Required artifacts

This JSON/Markdown pair
(`docs/design/lever-compiler-prefill-max-states-cap-exposure12-20260728.{json,md}`).

## Cleanup note

The `lever_exposure12_v1` train-data rebuild and the scratch checkpoint used
for this diagnostic are not committed (`outputs/` is gitignored; the
auto-published copy under
`src/slm_training/resources/data/train/lever_exposure12_v1` is removed after
the runs above since it is reproduction scaffolding, not new curated training
data — same convention as all three prior docs in this chain).

Captured: 2026-07-28T02:30:00Z
