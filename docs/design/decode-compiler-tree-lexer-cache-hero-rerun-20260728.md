# Compiler-tree lexer-cache fix: does it get the hero suite under the 30s wall alone? (NOT SHIP)

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (suite `n=3`,
1 rep each, byte-identical recipe to the prior finding). **Not ship. Not a
fix — a measurement rerun.**

## Why this run

[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173, just landed on `main`) fixed the Lark lexer/scanner rebuild cost
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
found, and named its own "Next steps" item 1 directly: *"Re-run the PR #1171
isolated per-record eval protocol ... to see whether `compiler_ms` drops
enough in aggregate for the `exposure12` quality-champion hero to finish
inside `decode_timeout_seconds=30` without the deadline fix from PR #1171."*
This doc is exactly that rerun, nothing more.

## Hypothesis

Lexer/scanner caching alone (#1173) reduces `compiler_ms` enough that the
`exposure12` quality-champion smoke suite's records finish inside
`decode_timeout_seconds=30`, without also needing
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)'s
separate deadline-swallow fix (`compiler_draft.py:2313`'s bare
`except Exception` masquerading a `TimeoutError` as an empty completion
forest).

## Recipe

Byte-identical to the deadline-swallow finding's own reproduction: rebuild
`lever_exposure12_v1` (107 records, 5 `abstraction_ladder` — matches), train
the identical scratch checkpoint (`last_loss=7.188689231872559`, matches the
historical exposure12 seed47 champion bit-for-bit), then evaluate each of the
3 smoke records **in isolation** (`--eval-limit 1 --eval-offset {0,1,2}`) on
`main` post-fix (`code_git_sha=7bb77c9`, `code_dirty=false`, which already
includes fix commit `20b658f`).

```bash
python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train

python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --context-backend scratch --steps 16 --batch-size 2 \
  --lr 1e-3 --structural-bias 1.5 --seed 47 \
  --run-id exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47 \
  --no-sync-checkpoints --device cpu

python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite smoke \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --device cpu \
  --checkpoint outputs/runs/exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --eval-limit 1 --eval-offset <0|1|2>
```

Same sandbox caveat as the lexer-cache-fix doc: `NODE_OPTIONS` is unset
(`env -u NODE_OPTIONS`) only for `npm ci`/build/eval commands, because this
sandbox's ambient `NODE_OPTIONS="--import tsx" --max-old-space-size=8192` is
malformed for plain `node`/`npm` invocations.

## Results: before (PR #1171 finding) vs after (this rerun, post-#1173)

| record | meaningful | outcome (before → after) | total_ms (before → after) | compiler_ms % (before → after) | `compiler_prefill_batches` (before → after) | compiler_ms per call (before → after) |
| --- | ---: | --- | --- | --- | --- | --- |
| smoke_hero_01 (offset 0) | 0.0 → 0.0 | fallback_output → fallback_output | 30006.5 → 30104.6 | 97.3 → 97.5 | 3 → 17 | 9736.6ms → 1727.4ms (**5.6x**) |
| smoke_button_01 (offset 1) | 0.0 → 0.0 | fallback_output → **runtime_timeout** | 30006.5 → 32900.6 (p50) | n/a (timeout path has no `decode_stats`) | n/a | n/a |
| smoke_callout_01 (offset 2) | 1.0 → 1.0 | fallback_output → fallback_output | 30010.2 → 30003.8 | 97.3 → 97.1 | 4 → 33 | 7296.6ms → 883.2ms (**8.3x**) |

## Decision

**REJECT — HYPOTHESIS_REFUTED.** The lexer-cache fix alone does **not** get
any of the 3 records under the 30s budget. `meaningful_program_rate` is
unchanged on hero (0.0) and callout (1.0); both still land on
`fallback_output` at ~30.0–30.1s wall time with `compiler_ms` still ~97% of
the total, identical in shape to the pre-fix baseline.

The fix is real and independently reconfirmed here, though: per-call
`compiler_ms` (`compiler_ms_sum / compiler_prefill_batches_sum`) dropped
**5.6x** on hero and **8.3x** on callout — larger than PR #1173's own
~2.0–2.1x isolated `build_completion_forest` micro-benchmark, plausibly
because the full eval path now executes far more calls per record
(hero 3→17, callout 4→33) so process-warm-up cost amortizes over a larger
sample. The decode loop spends that recovered time doing **more search
inside the same fixed wall budget**, not finishing sooner — total wall time
stays pinned at the `decode_timeout_seconds=30` ceiling either way, so the
net user-visible outcome does not change for 2 of 3 records.

This is the second independent finding to confirm
`decode-compiler-tree-deadline-swallow-finding.md`'s `proposed_fix_sketch`
item 1 (add `check_decode_deadline()` calls to the compiler-tree
per-position loop; stop the bare `except Exception` at
`compiler_draft.py:2313` from swallowing a real `TimeoutError`) is still the
operative blocker — exactly what PR #1173's own "Next steps" predicted
rather than assumed.

## Side observation (not claimed)

`smoke_button_01` classified as `fallback_output` before the fix and
**`runtime_timeout`** after, on an otherwise-identical checkpoint + flags
rerun.
[`lever-decode-nondeterminism-hashseed-ruled-out-measured-results.md`](lever-decode-nondeterminism-hashseed-ruled-out-measured-results.md)
already established that `decode_outcome` classification varies run-to-run
under real wall-clock scheduling even with everything else pinned (not a
seedable RNG effect). One n=1-rep before/after pair is not enough to
attribute this shift to the lexer-cache fix — flagged, not claimed.

## Non-goals

No code change this session (measurement-only). Does not apply
`proposed_fix_sketch` item 1 or item 2 from the deadline-swallow finding —
both remain open. No multi-rep confirmation of the exact 5.6x/8.3x per-call
speedup magnitudes (n=1 rep per record, matching the prior finding's own
protocol). No promotion, checkpoint, or `--ship-gates` claim.

## Tests / checks

No code touched: `python -m scripts.verify_version_stamps --check` is a
no-op for this change (measurement-only rerun, `components: {}`).

## Required artifacts

This JSON/Markdown pair
(`docs/design/decode-compiler-tree-lexer-cache-hero-rerun-20260728.{json,md}`).

Captured: 2026-07-28T01:40:51Z
