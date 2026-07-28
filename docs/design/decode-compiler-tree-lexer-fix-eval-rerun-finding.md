# Finding: the lexer-cache fix does not close the exposure12 hero's decode deadline

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic re-run
(suite `n=3`, 1 rep each, identical scope to the finding this re-runs).
**Not ship. Not a fix — a diagnostic finding**, same status tier as
[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173) and
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
(PR #1171).

## Task

`decode-compiler-tree-lexer-cache-fix.md`'s "Next steps" item 1: *"Re-run
the PR #1171 isolated per-record eval protocol ... to see whether
`compiler_ms` drops enough in aggregate for the `exposure12`
quality-champion hero to finish inside `decode_timeout_seconds=30` without
the deadline fix from PR #1171."* This diagnostic answers that question
directly instead of leaving it open.

## Reproduction

Identical recipe and protocol to the original finding, re-run against the
current checkout with PR #1173's fix already merged (confirmed present:
`_load_lexer` at `engine.py:58`, `self._lexer` at `engine.py:97`):

```bash
python3.12 -m venv .venv-diag
.venv-diag/bin/pip install --quiet -e .
.venv-diag/bin/pip install --quiet "pytest>=8.0,<9" "pytest-asyncio>=0.23,<2" "ruff>=0.9,<0.16"
.venv-diag/bin/pip install --quiet "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu
npm ci --silent                                  # repo root -- AgentV SDK publish step
cd src/apps/openui_bridge && npm ci --silent     # Node DSL bridge

python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train
# 107 records -- matches the original finding exactly

python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --context-backend scratch --steps 16 --batch-size 2 \
  --lr 1e-3 --structural-bias 1.5 --seed 47 \
  --run-id exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47 \
  --no-sync-checkpoints --device cpu
```

Confirmed the rebuilt checkpoint matches the original finding's recipe
(`last.meta.json` → `config.compiler_decode_mode == "off"`; `evaluate_model`
still forces the compiler-tree path via `DEFAULT_EVALUATION_POLICY` /
`STRICT_COMPILER_TREE_POLICY` regardless — same setup as the original).

Each of the three smoke records evaluated in isolation, same as the
original finding:

```bash
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

`code_git_sha=7bb77c9191dc1410f6b4af10670f81e27ecc43b8`, `code_dirty=false`,
identical `checkpoint_sha256` (`cadeb1d3...757539e9`) across all three eval
runs (same checkpoint, three isolated records).

## Measured: before (PR #1171) vs. after (PR #1173's fix applied)

| record | meaningful (before→after) | outcome (before→after) | total_ms (before→after) | compiler_ms (before→after) | forwards (before→after) | tokens_emitted (before→after) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| smoke_hero_01 (offset 0) | 0.0 → 0.0 | fallback_output → fallback_output | 30006.5 → 30003.0 | 29209.7 → 29188.7 | 3 → **29** | 7 → 46 |
| smoke_button_01 (offset 1) | 0.0 → 0.0 | fallback_output → fallback_output | 30006.5 → 30003.8 | 29378.1 → 28347.3 | 4 → **102** | 8 → 176 |
| smoke_callout_01 (offset 2) | 1.0 → 1.0 | fallback_output → fallback_output | 30010.2 → 30003.7 | 29186.4 → 28813.5 | 4 → **51** | 8 → 86 |

`dead_ends=1` and `certified_fallbacks=1` on every record, before and
after, unchanged. Every after-run's `constrained_dead_end_traces[0]` still
shows `phase="compiler_tree"`, `reason="empty_completion_forest"` — the
exact pattern PR #1171 diagnosed as the deadline-swallow bug
(`compiler_draft.py:2313`'s bare `except Exception` converting a deadline
`TimeoutError` into a legal-looking grammar dead-end).

## Interpretation

**The lexer-cache fix works exactly as designed and independently
confirmed**: every record now executes far more compiler-tree
forward/prefill steps inside the identical 30s budget — 3→29 (9.7x),
4→102 (25.5x), 4→51 (12.75x) — consistent with the ~2-2.1x per-call
speedup PR #1173 measured directly, compounding across many more steps
before the deadline lands.

**But this answers the open "next steps" question negatively.**
`compiler_ms_sum` does *not* drop enough in aggregate: every record still
consumes 94.5-97.3% of the 30s budget in `compiler_ms`, still lands at
`total_ms≈30003ms`, and still terminates in `decode_outcome=fallback_output`
via the same `empty_completion_forest`/`compiler_tree` dead-end. Every
`meaningful_program_v1_rate` value is byte-identical before and after
(0.0, 0.0, 1.0). The `exposure12` quality-champion hero needs far more
compiler-tree steps to converge than even fit inside 30s at the faster
per-step cost — for `smoke_button_01`, the decoded prefix at the dead-end
position (177 tokens in) shows the model stuck emitting a long repeated
`Separator("vertical")` run, consistent with the original finding's noted
undertrained-checkpoint symptom (`steps=16` is smoke-scale), not a
lexer-cost artifact.

**The conditional in `decode-compiler-tree-lexer-cache-fix.md`'s next
steps item 3 does not trigger.** That item asked whether the
deadline-swallow bug would still be "practically reachable" *if*
`compiler_ms` dropped substantially. It did not drop substantially in
wall-clock terms (both before and after are deadline-bounded at ~30s), so
the deadline-swallow bug remains confirmed reachable on every record,
unchanged from PR #1171's finding — it was not "mostly masking" the
lexer-rebuild cost; the two are separate, both still-open problems.

## Why this is a finding, not a patch

No code changed this session — `git status` is clean except for transient
`outputs/` and a since-removed reproduction copy of the published
train-data directory. This diagnostic's only job was to resolve an
explicitly flagged open question from PR #1173 by direct re-measurement
rather than inference, and it does: of the two candidate follow-ups listed
in that doc's own next steps (fix the deadline-swallow bug vs. keep
chasing lexer/compiler-tree perf), the deadline-swallow bug is the one
still load-bearing for closing this eval gap — a 10-25x increase in
steps-per-budget did not change the outcome.

## Non-goals

No larger or different held-out corpus, no code fix applied, no
statistical significance test beyond "identical protocol, one rep per
record, matching the finding it re-runs." No promotion or ship-gate claim.

## Scope note

`n=3` records, 1 rep each — identical scope to the original finding this
re-runs, not a multi-rep confirm, and does not characterize records beyond
this fixed 3-record smoke subset. Does not apply proposed-fix-sketch item 2
(making `_incremental_sync` genuinely incremental at the lex layer, still
open) or PR #1171's deadline-swallow fix itself — both remain unapplied
after this session.

## Next steps

1. Fix PR #1171's deadline-swallow bug directly: propagate
   `check_decode_deadline()` into the compiler-tree per-position loops
   (`_compiler_ltr_decode_one` / `_compiler_ltr_decode_batch` in
   `twotower.py`) the same way it was added to `_constrained_ltr_repair`
   and the MaskGIT loop in PR #1167, and stop
   `compiler_draft.py:2313`'s bare `except` from reclassifying a deadline
   `TimeoutError` as a legal grammar dead-end.
2. Separately, apply proposed-fix-sketch item 2 from the lexer-rebuild
   finding (make `_incremental_sync` genuinely incremental at the lex
   layer) — still open, and plausibly more relevant now given how many
   more compiler-tree steps these records execute per eval.
3. Investigate the undertrained-checkpoint symptom flagged again here
   (`smoke_button_01`'s repeated `Separator("vertical")` emission at
   position 177) as a separate quality issue, out of scope for this
   decode-latency thread — `steps=16` is smoke-scale, not a real training
   run.

## Cleanup note

`.venv-diag`, `node_modules/` (repo root and `src/apps/openui_bridge/`,
both gitignored), and the auto-published
`src/slm_training/resources/data/train/lever_exposure12_v1` copy
(reproduction scaffolding, not new curated training data) are not
committed — removed after the runs above, matching the original finding's
own cleanup convention.

`python -m scripts.verify_version_stamps --check`: 0 components touched
(no file watched by `versions.json` was edited this session).

Captured: 2026-07-28T05:10:00Z
