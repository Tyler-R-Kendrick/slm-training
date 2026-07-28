# Finding: the deadline-swallow fix reclassifies decode_outcome, but drops the exposure12 hero's one prior "meaningful" fallback

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic re-run
(suite `n=3`, 1 rep each, identical scope to the finding chain this re-runs).
**Not ship. Not a fix — a diagnostic finding**, same status tier as
[`decode-compiler-tree-lexer-fix-eval-rerun-finding.md`](decode-compiler-tree-lexer-fix-eval-rerun-finding.md)
(PR #1182) and
[`decode-compiler-tree-deadline-swallow-fix.md`](decode-compiler-tree-deadline-swallow-fix.md)
(PR #1183). `is_fix: false`.

## Task

[`decode-compiler-tree-deadline-swallow-fix.md`](decode-compiler-tree-deadline-swallow-fix.md)'s
"Scope / next steps" left one item explicitly open: *"re-running the
isolated per-record `exposure12` protocol (PR #1178 / #1182) against this
fix, to see whether `decode_outcome` correctly reclassifies as
`runtime_timeout` or shifts to an on-time completion."* This diagnostic
answers that question directly instead of leaving it open for a further
scheduled iteration.

## Reproduction

Identical recipe and protocol to
[`decode-compiler-tree-lexer-fix-eval-rerun-finding.md`](decode-compiler-tree-lexer-fix-eval-rerun-finding.md)
(PR #1182), re-run against the current checkout with PR #1183's fix
additionally merged (confirmed present: `except TimeoutError: raise` at
`compiler_draft.py`, `check_decode_deadline()` at the top of
`_compiler_ltr_decode_one`'s per-position loop):

```bash
python3.12 -m venv .venv-diag
.venv-diag/bin/pip install --quiet -e .
.venv-diag/bin/pip install --quiet "pytest>=8.0,<9" "pytest-asyncio>=0.23,<2" "ruff>=0.9,<0.16"
.venv-diag/bin/pip install --quiet "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu
env -u NODE_OPTIONS npm ci --silent                                  # repo root -- AgentV SDK publish step
(cd src/apps/openui_bridge && env -u NODE_OPTIONS npm ci --silent)   # Node DSL bridge, subshell so cwd stays at repo root

env -u NODE_OPTIONS .venv-diag/bin/python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train
# 107 records -- matches the original finding exactly

env -u NODE_OPTIONS .venv-diag/bin/python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --context-backend scratch --steps 16 --batch-size 2 \
  --lr 1e-3 --structural-bias 1.5 --seed 47 \
  --run-id exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47 \
  --no-sync-checkpoints --device cpu

for OFFSET in 0 1 2; do
  env -u NODE_OPTIONS .venv-diag/bin/python -m scripts.evaluate_model \
    --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
    --suite smoke \
    --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
    --model twotower --device cpu \
    --checkpoint outputs/runs/exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
    --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
    --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
    --eval-limit 1 --eval-offset "$OFFSET"
done
```

`code_git_sha=90273d6a46f2dd8880a081080c2c5430a66601af` (PR #1183's fix
commit, fast-forward merged onto this session's branch), `code_dirty=false`
at the top-level eval report, identical `checkpoint_sha256`
(`cadeb1d346d863e5a727a64847e59bfcb5c38db65fd6d5e3ef8c996c757539e9` —
byte-identical to every prior finding in this chain) across all three eval
runs.

## Measured: before (PR #1178/#1182, fix unapplied) vs. after (PR #1183's fix applied)

| record | decode_outcome (before→after) | meaningful_program_v1 (before→after) | total_ms (before→after) |
| --- | --- | --- | --- |
| smoke_hero_01 (offset 0) | fallback_output → **runtime_timeout** | 0.0 → 0.0 | 30003.0 → 30000.36 |
| smoke_button_01 (offset 1) | fallback_output → **runtime_timeout** | 0.0 → 0.0 | 30003.8 → 30000.66 |
| smoke_callout_01 (offset 2) | fallback_output → **runtime_timeout** | **1.0 → 0.0** | 30003.7 → 30067.37 |

Every after-run's `decode_outcome_counts` is
`{"runtime_timeout": 1, "fallback_output": 0, ...}` and
`failure_breakdown={"parse_error": 1}` — none of the three records produce
any parseable output at all, versus the empty-completion-forest fake dead
end + fallback-output path all three took pre-fix.

## Interpretation

**The fix's own predicted effect is confirmed exactly as documented in
`decode-compiler-tree-deadline-swallow-fix.md`'s interpretation field**:
`decode_outcome` reclassifies from the misleading `fallback_output` (a
grammar-dead-end lookalike, per PR #1171's original finding) to the honest
`runtime_timeout` on all three records. No record shifts to an on-time
completion — the `exposure12` quality-champion hero genuinely needs more
compiler-tree search than fits inside `decode_timeout_seconds=30` at this
`steps=16` smoke scale, confirming (not contradicting) every prior finding
in this chain (#1171, #1178, #1182) that diagnosed the wall as
compute-bound, not a classification artifact alone.

**But this is not a free honesty win — it has a real, measured cost.**
`smoke_callout_01` (offset 2) had `meaningful_program_v1_rate=1.0` in
*every* prior finding in this chain, before and after the lexer-cache fix
(#1173/#1178/#1182) — its `fallback_output` path happened to still emit a
parseable, meaningful program despite hitting the same dead-end pattern as
the other two records. With PR #1183's fix applied, that record's
`TimeoutError` now propagates immediately instead of falling through to
whatever fallback-output logic previously ran, so this record's
`meaningful_program_v1_rate` drops to 0.0 along with the other two. The fix
trades a **known-wrong classification** (`fallback_output` masking a
timeout) for a **known-correct classification** (`runtime_timeout`) that,
for this one record, also removes the only previously-passing output in
the entire 3-record smoke suite. This is exactly the kind of tradeoff the
iron law exists to surface rather than let a "fix" quietly regress a
metric — recorded here as a finding, not swept into the fix PR's own
"tests pass" framing.

**No ship-gate implication.** All three records already failed every
`smoke:*` ship-gate criterion pre-fix except `smoke:decode_timeout_count`
(which was already failing since `decode_timeout_count=1`, unaffected by
this change's outcome relabeling); post-fix, `smoke:decode_timeout_count`
still fails identically (`actual=1, expected=0` on all three). This finding
does not change any ship/no-ship determination — the suite was already
categorically failing under `honest-ship-eval`'s gates and remains so.

## Why this is a finding, not a fix

No code changed this session — `git status` is clean except for transient
`outputs/` and the reproduction copies of the published train-data
directory and diagnostic venv, neither committed. This diagnostic's only
job was to resolve the explicitly flagged open question from
`decode-compiler-tree-deadline-swallow-fix.md` by direct re-measurement,
and it does: `decode_outcome` reclassifies as `runtime_timeout` on all
three records as hypothesized, with the previously-undocumented side
effect (the offset-2 meaningful-output regression) surfaced directly
rather than left implicit.

## Non-goals

No larger or different held-out corpus, no code fix applied, no
statistical significance test beyond "identical protocol, one rep per
record, matching the finding chain it re-runs." No promotion or ship-gate
claim.

## Scope note

`n=3` records, 1 rep each — identical scope to every prior finding in this
chain, not a multi-rep confirm, and does not characterize records beyond
this fixed 3-record smoke subset. Does not investigate why each
`build_completion_forest` call still costs the wall-clock it does even
post lexer-cache fix (carried forward unchanged from
`decode-compiler-tree-deadline-swallow-fix.md`'s own next_steps item 2) —
that thread remains open for a follow-up scheduled iteration, along with
deciding whether the offset-2 meaningful-output regression warrants its
own follow-up investigation (e.g. whether the pre-fix `fallback_output`
path was itself relying on the swallow bug to produce that output, versus
some other latent difference).
