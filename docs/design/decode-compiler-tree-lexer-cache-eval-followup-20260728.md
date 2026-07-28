# Follow-up: re-measuring the compiler-tree eval after the lexer-cache fix

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (suite
`n=3`, 1 rep each) — the exact same standing as its two predecessor
diagnostics. **Not ship. Not a fix — a re-measurement.** This is still a
3-record smoke suite, not a full scoreboard, regardless of what the numbers
below show.

## Task

Execute PR #1173's own "Next steps" item 1 verbatim:

> Re-run the PR #1171 isolated per-record eval protocol ... to see whether
> `compiler_ms` drops enough in aggregate for the `exposure12`
> quality-champion hero to finish inside `decode_timeout_seconds=30` without
> the deadline fix from PR #1171.

Predecessor chain:

1. [`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
   (PR #1171, finding) — found all 3 smoke records hit the 30s wall with
   ~97-98% of wall time in `compiler_ms`, and that a `TimeoutError` inside
   `build_completion_forest` gets silently swallowed as an empty completion
   forest (a fake grammar dead-end) instead of a classified timeout.
2. [`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
   (PR #1172, finding) — diagnosed the `compiler_ms` cost as an uncached Lark
   lexer/scanner rebuild on every `_sync` call.
3. [`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
   (PR #1173, fix, merged) — cached the Lark lexer via `@lru_cache`, measured
   a 2.0-2.14x speedup on isolated `build_completion_forest` micro-benchmarks,
   but explicitly left the real per-record re-measurement (this doc) as an
   open next step.

## Reproduction

Fresh checkout, no committed checkpoint or train data (`outputs/` is
gitignored). Environment rebuilt the same way as the two prior sessions:

```bash
python3.12 -m venv .venv-diag
.venv-diag/bin/pip install --quiet -e .
.venv-diag/bin/pip install --quiet "pytest>=8.0,<9" "pytest-asyncio>=0.23,<2" \
  "ruff>=0.9,<0.16" "torch>=2.2,<2.6" --extra-index-url https://download.pytorch.org/whl/cpu
npm ci --silent                              # repo root: AgentV SDK, needed by evaluate_model's publish step
cd src/apps/openui_bridge && npm ci --silent # DSL bridge, needed for lang_core.parse
```

(`NODE_OPTIONS` unset for all `node`/`npm` invocations — same sandbox artifact
noted in PR #1172/#1173: the ambient `--import tsx --max-old-space-size=8192`
is invalid for a plain `node` call.)

```bash
python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train
# 107 records, 5 abstraction_ladder -- matches PR #1171 exactly

python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --context-backend scratch --steps 16 --batch-size 2 \
  --lr 1e-3 --structural-bias 1.5 --seed 47 \
  --run-id exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47 \
  --no-sync-checkpoints --device cpu
# completed well inside MAX_RUN_MINUTES=3

# For each --eval-offset 0, 1, 2 (isolated, one record per run):
python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite smoke \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --device cpu \
  --checkpoint outputs/runs/exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --eval-limit 1 --eval-offset <0|1|2>
# each run: ~36.2-36.5s wall (well inside MAX_RUN_MINUTES=3 per invocation)
```

Identical protocol to PR #1171: no new instrumentation, n=1 rep per record,
same `decode_stats` fields already in `src/slm_training/models/decode_stats.py`.

## Measured: before (PR #1171) vs. after (this session, post PR #1173)

### Before — pre-fix baseline (PR #1171)

| record (offset) | meaningful | outcome | total_ms | compiler_ms | compiler_ms % | forwards | tokens | denoiser_ms | dead_ends | certified_fallbacks |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_hero_01 (0) | 0.0 | fallback_output | 30006.5 | 29209.7 | 97.3 | 3 | 7 | 0.0 | 1 | 1 |
| smoke_button_01 (1) | 0.0 | fallback_output | 30006.5 | 29378.1 | 97.9 | 4 | 8 | 0.0 | 1 | 1 |
| smoke_callout_01 (2) | 1.0 | fallback_output | 30010.2 | 29186.4 | 97.3 | 4 | 8 | 0.0 | 1 | 1 |

### After — this session, post lexer-cache fix (PR #1173)

| offset | meaningful | outcome | total_ms | compiler_ms | compiler_ms % | forwards | tokens | denoiser_ms | dead_ends | certified_fallbacks |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | fallback_output | 30004.5 | 29257.8 | 97.5 | **17** | **25** | 0.0 | 1 | 1 |
| 1 | 0.0 | **runtime_timeout** | 30000.5* | n/a — `decode_stats` absent | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | 1.0 | fallback_output | 30004.3 | 29059.4 | 96.9 | **30** | **51** | 0.0 | 1 | 1 |

\* offset 1's `total_ms`/`compiler_ms` are unavailable because the run's
`metrics["decode_stats"]` is entirely absent for this record — see below.

## What changed and what didn't

- **`compiler_ms` did NOT drop in aggregate.** 97.3/97.9/97.3% before vs.
  97.5/n-a/96.9% after — still ~97% of the entire 30s budget spent inside
  `compiler_ms` on both records where `decode_stats` was captured this
  session. The lexer-cache fix does not reduce the *share* of wall time the
  compiler-tree path consumes.
- **`forwards_count`/`compiler_prefill_batches` increased substantially**:
  offset 0 went 3 → 17 (5.7x), offset 2 went 4 → 30 (7.5x). This is the
  visible fingerprint of the fix working as designed — each individual
  `build_completion_forest` call is cheaper, so more of them fit inside the
  same ~29s window before the deadline cuts in (the decoded prefix at the
  dead-end also got deeper: offset 0's dead-end moved from position 8
  (`"root = Card([b1"`) in the baseline to position 26 this session; offset
  2's moved from position 9 to position 52). But **more (faster) calls does
  not mean less total time** — the loop simply keeps calling
  `build_completion_forest` until the wall cuts it off, so `total_ms` stays
  pinned at ~30000ms regardless.
- **`decode_outcome` changed for exactly one of three records.** Offset 1
  shifted from `fallback_output` (baseline) to `runtime_timeout` this
  session. Unlike the baseline (which recorded `compiler_ms`/`dead_end_trace`
  for every record via the swallow path), this record's `metrics["decode_stats"]`
  key is **entirely absent** — meaning the deadline `TimeoutError` this time
  was *not* caught by `build_completion_forest`'s bare
  `except Exception` (PR #1171's diagnosed swallow) but instead propagated
  up into `eval_runner`'s own `except TimeoutError` handler
  (`src/slm_training/harnesses/model_build/eval_runner.py:1160-1167`), which
  has no `DecodeStats` attached to the exception in this run and so recorded
  zero telemetry for it. `latency_ms_p50` for that record was still
  `30000.52` — the wall still fired at the same ~30s mark; only *where* the
  interrupt landed changed. This is consistent with PR #1171's own
  interpretation that a swallow-vs-propagate outcome is a timing race
  depending on which call happens to be in flight when the SIGALRM re-fires,
  not a deterministic property of the code path. **n=1 for this record this
  session — not re-run to check reproducibility**, per the isolated
  per-record protocol and to avoid retrying a wall-clock-sensitive run in a
  loop.
- **The exposure12 hero does NOT finish inside the 30s wall.** All three
  records still land at ~30000ms `total_ms`/`latency_ms_p50`, identical in
  substance to the pre-fix baseline. The lexer-cache win from PR #1173 is
  real (confirmed again indirectly here via the 5.7x/7.5x jump in
  `forwards_count`/`compiler_prefill_batches`) but does not reach far enough
  to change the wall-clock outcome for this checkpoint/suite.

## Verdict: null result

**This is a null result, not a win, for the specific question PR #1173 left
open.** The lexer-cache fix measurably lets more (cheaper) compiler calls fit
inside the fixed 30s budget, but it does not reduce `compiler_ms`'s *share*
of that budget or change the wall-clock ship-relevant outcome
(`meaningful_program_rate` is still 0.0 on 2 of 3 records; all 3 still graze
or hit the timeout). Per the task's explicit instruction, this null result is
reported as-is, not spun as a win.

The practical implication: **the deadline-swallow fix sketched in PR #1171 is
still needed on its own merits** — the lexer-layer optimization alone does
not obsolete it. If anything, offset 1's shift from `fallback_output` to
`runtime_timeout` this session is a (single, unreplicated) data point that
the swallow can already fail to trigger some of the time even without that
fix applied, matching PR #1171's own footnote that 1 of 3 historical reps in
`lever-hard-decode-timeout-wall-measured-results.json` landed as a "clean"
`runtime_timeout` rather than a masked `fallback_output`.

## Version stamps

No harness code was touched this session (`engine.py`, `compiler_draft.py`,
`twotower.py` untouched, per the task's explicit measurement-only rule).

```
python -m scripts.verify_version_stamps --check
# version-stamps: ok (0 component(s) touched)
```

## Validation

- `python -m scripts.verify_version_stamps --check` — `ok (0 component(s) touched)`.
- `python -m scripts.repo_policy` — clean.
- `git diff --check` — clean (no whitespace errors).
- `ruff check` on touched files — no Python files touched this session (only
  these two new docs), so this is a no-op.

## Next steps

1. The deadline-swallow fix sketched in PR #1171 (`except TimeoutError: raise`
   before the bare `except Exception:` in `compiler_draft.py:2313`, plus
   `check_decode_deadline()` calls inside the compiler-tree per-position
   loops in `twotower.py`) is now **higher priority** relative to further
   lexer-layer micro-optimization: this session's measurement shows the
   lexer cache alone does not clear the 30s wall, so correctly classifying
   (or actually preventing) the timeout is the more direct lever on
   `decode_outcome` for this checkpoint/suite. That work belongs to
   `improve-openui-harnesses`, not a measurement-only iteration like this one.
2. PR #1173's own still-open item 2 (make `_incremental_sync` genuinely
   incremental at the lex layer, only lexing newly-appended text instead of
   re-lexing the whole prefix every call) is a second-order perf lever that
   may buy more calls-per-budget, but per this session's null result is
   unlikely by itself to clear the 30s wall either — it would need to be
   combined with the deadline fix (or a substantially larger constant-factor
   win) to change the wall-clock outcome.
3. If the deadline-swallow fix lands, re-run this exact isolated per-record
   protocol again to see whether `decode_outcome` correctly and consistently
   classifies as `runtime_timeout` (if genuinely out of budget) rather than
   the current mix of `fallback_output`/occasional `runtime_timeout`
   depending on timing.

## Cleanup note

The `lever_exposure12_v1` train-data rebuild and the scratch checkpoint used
for this session's reproduction are not committed (`outputs/` is gitignored;
the auto-published copy under
`src/slm_training/resources/data/train/lever_exposure12_v1` was removed after
the runs above, same convention as PR #1171/#1172's cleanup notes).
`.venv-diag` and both `node_modules/` trees (repo root and
`src/apps/openui_bridge`) are gitignored and were also removed.

Captured: 2026-07-28T08:38:59Z
