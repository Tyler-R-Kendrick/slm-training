# Capstone: does the 6-PR decode-latency/correctness stack move anything at held_out scale? — NOT SHIP

**Honesty:** `fixture_or_scratch`, n=5 (`held_out` suite), 2 seeded reps.
**Not ship.** Closes the `claude/great-dirac-*` decode-latency/correctness
thread (PRs #1189-#1195); every doc in that thread has already said "not
ship" and this one is no exception.

## Task

PR #1190's own "Next steps" item 2 and this iteration's instructions: re-run
the seeded multi-rep
[`lever-hard-decode-timeout-wall`](lever-hard-decode-timeout-wall-measured-results.md)
protocol on a suite bigger than the n=3 `smoke` slice used throughout this
thread's diagnostics, to see whether the thread's combined 6 fixes — correct
deadline classification (#1189), the pre-existing lexer cache (#1173),
witness-search `_tail` cache hoist (#1191), resolve-path cache (#1192), and
`InteractiveParser.accepts()` state-stack memoization (#1195), all riding on
top of the pre-existing hard SIGALRM decode wall (#1167/#1170) — change any
real decode outcome/quality number, or only internal call counts.

## Recipe

Same checkpoint this thread has used throughout (`exposure12` recipe: 107
fixture records, twotower/scratch, steps=16, lr=1e-3, batch=2,
structural_bias=1.5, seed=47, `--no-sync-checkpoints`) — already on disk from
the prior session in this working tree, rebuilt from the finding's exact
recipe. `code_commit=639f22c` (this stack's full HEAD, `model.twotower=v262`).

```bash
python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite held_out \
  --model twotower --device cpu \
  --checkpoint outputs/runs/exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --run-id capstone_heldout_rep{1,2}_exp12_seed47
```

**What was shrunk, and why:** the prior protocol ran 3 reps on `smoke`
(n=3). `held_out` has n=5 — bigger, as the task required — but 5 records x
30s hard-wall-per-record = 150s worst case, close to the repo's
`MAX_RUN_MINUTES=3` (170s harness-interrupt budget) per invocation. Rep count
was shrunk from 3 to 2 to stay bounded; `rico_held` (n=34) was ruled out
entirely — even 1 record's worth of margin would blow the per-invocation cap
if several records hit the timeout, which they did.

## Measured: 0/10 meaningful, 10/10 `runtime_timeout`, stable across reps

| rep | n | parse_rate | meaningful_program_rate | decode_outcome_counts | latency_ms p50 | latency_ms p95 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 5 | 0.0 | 0.0 | `{runtime_timeout: 5, fallback_output: 0, model_valid: 0}` | 30000.79 | 30000.95 |
| 2 | 5 | 0.0 | 0.0 | `{runtime_timeout: 5, fallback_output: 0, model_valid: 0}` | 30000.83 | 30011.86 |

Pooled: 0/10 meaningful, 0/10 parsed, 10/10 `runtime_timeout`,
`failure_breakdown={parse_error: 5}` each rep. Every one of the 5 `held_out`
records (`held_out_form_01`, `held_out_dual_card_01`, `held_out_input_01`,
`held_out_tabs_01`, `held_out_settings_01`) hit the 30s hard wall on both
reps — identical `decode_outcome_counts` and near-identical latency
(`~30000-30012ms`, both pinned at the wall) across reps, so this is a stable,
deterministic result under this recipe, not run-to-run noise.
`evidence_class: fixture_under_minimum_n` on every rate (n=5 < the ship
minimum n=20) — the harness itself already flags this as sub-ship evidence.

## Interpretation: same honest pattern as smoke, now confirmed at a bigger n

[PR #1190's smoke-suite remeasurement](decode-compiler-tree-deadline-swallow-remeasure.md)
found 0/3 meaningful and 3/3 `runtime_timeout` after only the deadline-fix +
lexer-cache. This run adds the three later fixes (#1191/#1192/#1195) and
grows the suite to n=5, and finds the identical pattern: 0/10 meaningful,
10/10 `runtime_timeout`. **The combined stack does not move a single
ship-relevant decode outcome or quality number for this checkpoint at this
scale.** `accepts()`-state memoization's own doc measured a genuine 1.5-1.6x
speedup, but only on isolated `_openui_completion_domain` micro-benchmarks on
two fixed prefixes — that speedup does not surface here because every
`held_out` record's real, full-record compiler-tree decode still needs
substantially more than 30 wall-clock seconds end-to-end (Node-bridge round
trips for `_generated_ast_is_complete` plus the witness search's own
recursive fan-out dominate; PR #1195's own "named next lever" already flagged
these as the new bottlenecks, not `accepts()`).

## Ship-gate check (`honest-ship-eval` default `held_out` bars)

| criterion | bar | actual | pass? |
| --- | ---: | ---: | --- |
| parse | ≥ 0.40 | 0.0 | fail |
| structural_similarity | ≥ 0.30 | 0.0 | fail |
| component_type_recall | ≥ 0.30 | 0.0 | fail |
| placeholder_fidelity | ≥ 0.15 | 0.0 | fail |

Fails every bar, and `evidence_class` self-reports
`fixture_under_minimum_n`. **Verdict: `fixture_or_scratch`, not ship** — this
stack has never claimed readiness and this run does not change that. No gate
was weakened.

## Decision

**Not a green light for anything.** The stack (#1189-#1195) is 6 real,
individually-tested, individually-honest fixes — one correctness fix
(deadline classification no longer masquerades as empty-forest fallback) and
several genuine micro-cache wins (lexer, witness-`_tail`, resolve-path,
`accepts()`-state) — but at held_out scale their combined effect on this
checkpoint's actual decode outcome is unmeasurable: every record was already,
and remains, over the 30s wall. Decode-side speed alone cannot fix a model
that needs longer than the wall to find a legal completion.

## Named next lever (for the next iteration)

**Pivot away from `src/slm_training/dsl/grammar/fastpath/engine.py`.** The
next bounded, concrete lever: **a training-data / SFT lever**, not another
decode microcost session. Specifically:

1. `slm data build-train` (or reuse a larger existing published corpus, e.g.
   the default `e937_role_safe_all_targets_v2` train dir) — bigger than the
   107-record `exposure12` fixture this thread has used throughout.
2. `slm sft train` a short run (steps ≤ 32, matching `MAX_RUN_MINUTES=3`)
   against that data.
3. Re-eval with the **exact same recipe as this doc** (`held_out` suite,
   `--decode-timeout-seconds 30`, `--seed 47`, `fixed_asap`) as a direct A/B
   baseline against this run's `0/5` meaningful, `5/5 runtime_timeout`
   numbers.

This tests the complementary hypothesis to everything this thread has probed:
whether a **better-trained** checkpoint can complete decode inside the
existing wall, since faster decode plumbing provably cannot on its own. See
`references/train-data.md` and `references/sft.md` in
`.agents/skills/autotrain`.

## Validation

```text
python -m scripts.verify_version_stamps --check
# version-stamps: ok (vs HEAD; 0 changed file(s), 0 component(s) touched)

python -m scripts.repo_policy
# repo-policy: ok (tracked + untracked)

python -m scripts.verify_decode_invariants
# exit 0, agent_surfaces/canonical_defaults/strict_policies/weakening_levers unchanged
```

No source files changed this session — pure eval re-run, so 0 version bumps
required and no `pytest`/`ruff` targets apply beyond the eval runs above.

## Scope note

- Diagnostic re-measurement only. No `--ship-gates` scoreboard claim, no
  checkpoint promotion, no MODEL_CARD update — checkpoint is scratch/16-step,
  `--no-sync-checkpoints`, unchanged from the prior session.
- `outputs/runs/capstone_heldout_rep{1,2}_exp12_seed47/` (this session's two
  eval run dirs) are gitignored, not committed.

Captured: 2026-07-28T13:20:00Z
