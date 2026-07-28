# Finding: lexer/scanner cache fix (PR #1173) does not change eval-harness decode outcome at smoke scale

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (suite `n=3`,
1 rep each), re-running an existing protocol. **Not ship. Not a fix — a
diagnostic re-measurement.** Companion JSON:
`docs/design/decode-compiler-tree-lexer-cache-fix-eval-recheck.json`
(`version_stamp/v1`, `0 component(s) touched`).

## Task

[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173, `dsl.grammar_capabilities` v3 / `model.twotower` v261) left this as
"next steps" item 1: *"Re-run the PR #1171 isolated per-record eval protocol
... to see whether `compiler_ms` drops enough in aggregate for the
`exposure12` quality-champion hero to finish inside
`decode_timeout_seconds=30` without the deadline fix from PR #1171."* This
session re-ran exactly that protocol —
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)'s
isolated per-record eval, on today's `HEAD` — to answer it.

## Reproduction

Fresh `.venv-diag` (Python 3.12) + `npm ci` in both the repo root and
`src/apps/openui_bridge` (checkpoint not on disk; `outputs/` is gitignored).

```bash
python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train
# 107 records — matches decode-compiler-tree-deadline-swallow-finding.md

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

## Measured: same wall-clock, same outcome, but more work happens underneath it

| record | meaningful | outcome | total_ms | compiler_ms | compiler_ms % | forwards | tokens_emitted |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| smoke_hero_01 (offset 0) | 0.0 | fallback_output | 30206.0 | 29419.7 | 97.4 | 8 | 13 |
| smoke_button_01 (offset 1) | 0.0 | fallback_output | 30008.8 | 28519.2 | 95.0 | 47 | 84 |
| smoke_callout_01 (offset 2) | 1.0 | fallback_output | 30105.9 | 29099.1 | 96.7 | 8 | 13 |

Pre-fix baseline (same records, same checkpoint recipe, code at PR #1171 —
from `decode-compiler-tree-deadline-swallow-finding.md`):

| record | total_ms | compiler_ms | compiler_ms % | forwards | tokens_emitted |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke_hero_01 | 30006.5 | 29209.7 | 97.3 | 3 | 7 |
| smoke_button_01 | 30006.5 | 29378.1 | 97.9 | 4 | 8 |
| smoke_callout_01 | 30010.2 | 29186.4 | 97.3 | 4 | 8 |

`decode_outcome_counts` on every record, before and after: `{"fallback_output":
1, "runtime_timeout": 0, ...}` — unchanged. `meaningful_program_rate` per
record is also unchanged (0.0, 0.0, 1.0).

## Interpretation

**Next-steps item 1 from the lexer-cache-fix doc is answered negatively:**
`compiler_ms` did not drop enough — on any of the three records — to finish
inside the 30s `decode_timeout_seconds` budget. All three still land on
`decode_outcome=fallback_output` at ~30.0–30.2s wall, with `compiler_ms`
still 95–97% of total: the identical qualitative pattern the pre-fix
diagnostic measured.

This also implicitly answers deadline-swallow-finding.md's own next-steps
item 3: the deadline-swallow bug is still reachable and still masks every
record as `fallback_output` instead of a correctly-classified
`runtime_timeout`. That is expected — PR #1173 only cached lexer/scanner
*construction*; it did not touch
`decode-compiler-tree-deadline-swallow-finding.md`'s separate
`proposed_fix_sketch` (`except TimeoutError: raise` in
`compiler_draft.py:2313`, `check_decode_deadline()` calls in the
compiler-tree per-position loop). The two findings' fixes are independent
and neither alone changes the eval harness's outcome classification.

### Secondary observation: the cache fix is doing real work, just not visible at this granularity

`compiler_prefill_batches` / `forwards_count` rose on every record (offset 0:
3→8, offset 2: 4→8, offset 1: 4→47 with `tokens_emitted` 8→84). More
`build_completion_forest` calls now fit inside the same 30s window because
each one no longer pays a full uncached Lark lexer rebuild — consistent with
the fix's own measured ~2–2.1x per-call win in
`decode-compiler-tree-lexer-cache-fix.md`. The eval harness's
outcome/latency metrics don't surface this gain because the hard wall still
caps `total_ms` at ~30s regardless of how much more work completes
underneath it; only a lower-level field like `compiler_prefill_batches`
exposes it. offset 1's much larger jump (47 batches / 84 tokens vs. the
other two records' ~8/13) was not isolated further this session — treat it
as a single-record data point, not a general 10x throughput claim.

## Scope note (what this does NOT establish)

- No train/eval run against a real (non-scratch, non-smoke-scale) checkpoint;
  no `--ship-gates` scoreboard; not a readiness claim.
- No code was changed this session — this is a diagnostic re-measurement
  only, matching an existing finding's own prescribed next step.
- Does not by itself prove or disprove whether the deadline-swallow fix
  (unapplied) would change the outcome; it only confirms the lexer-cache fix
  alone does not.

## Next steps

1. Apply `decode-compiler-tree-deadline-swallow-finding.md`'s
   `proposed_fix_sketch` (`except TimeoutError: raise`;
   `check_decode_deadline()` in the compiler-tree loop) via
   `improve-openui-harnesses` — this is the fix that would actually change
   `decode_outcome` classification, not the lexer cache alone.
2. Apply `decode-compiler-tree-lexer-cache-fix.md`'s proposed-fix-sketch item
   2 (`_incremental_sync` genuinely incremental at the lex layer) — may raise
   `compiler_prefill_batches` per 30s window further but will not by itself
   resolve the still-open deadline-swallow classification bug.
3. After both land, re-run this exact isolated per-record protocol; only
   then would a change in `decode_outcome` (`fallback_output` →
   `runtime_timeout`, or a real on-time completion) be expected.

## Reproducibility

```bash
python -m scripts.verify_version_stamps --check
# version-stamps: ok (vs HEAD; 12 changed file(s), 0 component(s) touched)
python -m scripts.repo_policy
# repo-policy: ok (tracked + untracked)
```

Result (this session, on `claude/great-dirac-mleqha` at `origin/main`
`7bb77c919`): both clean, 0 components touched (no `src/` files edited this
session).

## Cleanup note

`.venv-diag`, `node_modules/` (both gitignored), and the `outputs/` run
artifacts are not committed. The auto-published copy under
`src/slm_training/resources/data/train/lever_exposure12_v1` is removed after
this session's runs — reproduction scaffolding, not new curated training
data, same as the prior finding's cleanup note.

Captured: 2026-07-28T10:42:41.000000Z
