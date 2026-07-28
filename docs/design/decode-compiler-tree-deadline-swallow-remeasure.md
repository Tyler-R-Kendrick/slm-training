# Re-measure: compiler-tree deadline reclassification after the swallow fix

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (`n=3`, 1 rep
each, same as the finding it re-measures). **Not ship. Not a fix — a
re-measurement.** Same status tier as
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
(PR #1171) and
[`decode-compiler-tree-deadline-swallow-fix.md`](decode-compiler-tree-deadline-swallow-fix.md)
(PR #1189), which this doc closes out.

## Task

PR #1189's own "Next steps" item 1 (and the finding's proposed-fix-sketch item
3): re-run the finding's exact isolated per-record eval protocol
(`evaluate_model --eval-limit 1 --eval-offset {0,1,2}` on the `exposure12`
quality-champion recipe/seed, `decode-timeout-seconds=30`) now that both
PR #1189's deadline-swallow fix and PR #1173's lexer-cache fix are in the
tree, to see whether `decode_outcome` reclassifies from
`fallback_output`/`empty_completion_forest` to a correctly-classified
`runtime_timeout` or to a real on-time completion.

## Reproduction

Checkpoint wasn't on disk (`outputs/` is gitignored, fresh checkout) — rebuilt
with the finding's exact recipe/seed:

```bash
python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train
# 107 records, 5 abstraction_ladder -- matches the finding exactly

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

The repo's existing `.venv` (python3.12.3) already had the needed packages —
no rebuild required. Unlike PR #1189's unit-test-only session, this eval
session calls `evaluate_model`'s AgentV publish path, which failed closed
(`RuntimeError: AgentV SDK is unavailable; run npm ci`) until `NODE_OPTIONS=
npm ci` ran at repo root (267 packages, ~31s). `src/apps/openui_bridge`'s
`node_modules` was already present from a prior session and reused as-is.

`model.twotower` in each run's `version_stamp` reads `v262` — confirming
PR #1189's fix is actually the code under test, not a stale checkpoint.

## Measured: before vs after, all three records reclassify

| record | before `decode_outcome` | before meaningful | after `decode_outcome` | after `stop_reason` | after `fallback_used` | after meaningful | after `latency_ms` |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| smoke_hero_01 (offset 0) | `fallback_output` | 0.0 | `runtime_timeout` | `decode_timeout` | false | 0.0 | 30001.05 |
| smoke_button_01 (offset 1) | `fallback_output` | 0.0 | `runtime_timeout` | `decode_timeout` | false | 0.0 | 30130.77 |
| smoke_callout_01 (offset 2) | `fallback_output` | **1.0** | `runtime_timeout` | `decode_timeout` | false | **0.0** | 30533.34 |

Before numbers are from the finding's table (`compiler_ms` ~97% of `total_ms`
on all three, `total_ms` ~30.0-30.0s). `decode_outcome_counts` on every
post-fix run: `{runtime_timeout: 1, fallback_output: 0, ...}` — i.e. the
aggregate flips from 3/3 `fallback_output` to 3/3 `runtime_timeout`.

## Interpretation

The reclassification worked exactly as the fix intended: `TimeoutError` now
propagates out of `build_completion_forest` through the compiler-tree loop's
new `check_decode_deadline()` call, and `eval_runner` classifies it as
`runtime_timeout` with `fallback_used=false` instead of masquerading it as an
empty-forest dead-end that gets certified-fallback'd to `fallback_output`.

This lands on the finding's **"genuinely out of budget"** branch, not the
**"swallow was masking otherwise-fast decode"** branch: latency stayed pinned
at ~30.0-30.5s on all three records — essentially identical to the pre-fix
`total_ms` of ~30.0-30.0s. The fix corrected *classification*, not *speed*.
Neither this fix nor PR #1173's lexer-cache fix moved the wall-clock needle
for this checkpoint's compiler-tree decode.

**Second-order finding, worth flagging honestly:** `meaningful_program_rate`
is strictly *worse* after the fix (pooled 0.33 → 0.0) because
`smoke_callout_01` previously scored `meaningful_program_v1=1.0` under the
mislabeled `fallback_output` path — that "meaningful" completion was an
artifact of certified-fallback content produced after a swallowed timeout,
not a genuine on-time decode. The fix trades an inflated
`meaningful_program_rate` for an honest one. Per decode-invariants I3
(constrained decoding is the product; unconstrained/fallback output is never
certified as ship evidence), this is the *correct* trade, but it means this
checkpoint's true on-time compiler-tree completion rate on this suite is
honestly 0/3, not the previously-reported 1/3.

## Gate status

Full AgentV smoke ship-gates on each `n=1` offset: all 8 criteria fail on
every record (`smoke:insufficient_n` n=1 < 20, `smoke:meaningful_program_rate`
0 < 0.66, `smoke:decode_timeout_count` 1 != 0, plus 5 more). Expected and
correct for an `n=1` diagnostic slice — no gate was weakened to pass it, no
ship claim is made here.

## Scope note

- Diagnostic re-measurement only. No `--ship-gates` scoreboard claim, no
  checkpoint promotion, no MODEL_CARD update — checkpoint is scratch/16-step,
  `--no-sync-checkpoints`.
- No harness code changed this session: `python -m
  scripts.verify_version_stamps --check` → `ok (vs HEAD; 0 changed file(s), 0
  component(s) touched)` — confirms this is a pure re-measurement, 0 version
  bumps required.
- `outputs/` artifacts from this session (train data, checkpoint, 3 eval run
  dirs) are gitignored, not committed. The auto-published copy under
  `src/slm_training/resources/data/train/lever_exposure12_v1` was removed
  after use, matching the finding doc's precedent.

## Validation

```text
python -m scripts.verify_version_stamps --check
# version-stamps: ok (vs HEAD; 0 changed file(s), 0 component(s) touched)

python -m scripts.repo_policy
# repo-policy: ok (tracked + untracked)

python -m scripts.verify_decode_invariants
# exit 0, agent_surfaces/canonical_defaults/strict_policies/weakening_levers unchanged
```

No source files were touched this session, so no `pytest`/`ruff` targets
apply beyond the eval runs above.

## Next steps

1. Root-cause why `compiler_ms` is still pinned at ~30s post-fix — both fixes
   are in place, yet wall time is unchanged. Compare
   `compiler_prefill_batches` / candidate-set size between this run and the
   finding's pre-fix numbers to isolate whether this checkpoint's
   compiler-tree search genuinely needs >30s per record independent of lexer
   cost, or whether the lexer-cache fix's benefit is concentrated on
   repeated-state cache hits this record's decode path doesn't exercise.
2. Re-run the seeded multi-rep `lever-hard-decode-timeout-wall` protocol on a
   fuller (non-smoke-scale) checkpoint to see whether the honest 0/3
   meaningful rate observed here is specific to this 16-step scratch
   checkpoint or persists at exposure12's originally-reported quality-champion
   training recipe.
3. Investigate the compiler-tree search cost itself (the finding's original
   next-step #2, still not done): whether the ~29-30s sink is Node-bridge
   round-trip latency, candidate-set size, or cache misses — neither fix
   applied so far actually reduced wall-clock spend for this checkpoint.

## Cleanup note

`outputs/data/train/lever_exposure12_v1`, the scratch checkpoint run dir, and
the three `outputs/runs/eval_off{0,1,2}_postfix` eval dirs used for this
session are not committed (`outputs/` is gitignored). The repo's existing
`.venv` and `node_modules` were reused; only `NODE_OPTIONS= npm ci` at repo
root was newly run this session (also gitignored).

Captured: 2026-07-28T12:02:49Z
