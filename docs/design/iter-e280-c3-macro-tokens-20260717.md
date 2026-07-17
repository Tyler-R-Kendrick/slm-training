# E280 — C3 corpus-mined macro tokens with deterministic expansion (2026-07-17)

Fixture-grade wiring row for Track C3 (Linear SLM-27). Machine-readable
evidence:
[quality-matrix-results-iter-v13-c3-20260717.json](quality-matrix-results-iter-v13-c3-20260717.json).
Code: [`src/slm_training/data/macro_induction.py`](../../src/slm_training/data/macro_induction.py)
(miner) and [`src/slm_training/models/dsl_tokenizer.py`](../../src/slm_training/models/dsl_tokenizer.py)
(tokenizer v3 `<MACRO_i>` channel).

## What was built

A Stitch/LILO-style (**Adapted** — compression objective only, see
[research-lineage.md](research-lineage.md)) macro-token channel on the
lexer-native tokenizer:

- **Tokenizer v3**: 64 reserved `<MACRO_i>` rows appended after the
  `<STATE_k>` pool (`DSL_TOKENIZER_VERSION = 3`; every pre-existing id
  unchanged). A macro id substitutes for a fixed span of **fixed-vocabulary**
  tokens only (`MACRO_EXPANDABLE_KINDS = {struct, component, builtin, lit,
  byte}` — never `NL`, never `<SYM_i>`/`<BIND_j>`/`<STATE_k>`), so macros are
  alpha-independent by construction and the context-sensitive
  alpha-equivalence hashing pitfall ([arXiv:2401.02948](https://arxiv.org/abs/2401.02948))
  cannot arise.
- **Offline greedy MDL induction** (`induce_macros`): canonicalize sources,
  lex, count n-grams (length 2–8) over expandable kinds, iteratively pick the
  span maximizing `net_gain = freq * (len - 1) - len`, collapse, recount;
  stop below `min_gain_tokens` or at `max_macros`. Deterministic and lossless
  — no learning, no anti-unification.
- **Fail-closed contract**: `set_macro_expansions` rejects spans containing
  dynamic-pool or unknown tokens (and spans shorter than 2); `decode` drops a
  `<MACRO_i>` with no table entry instead of emitting the raw sentinel;
  `induce_macros` refuses a tokenizer that already carries a table. The table
  is persisted inside the tokenizer sidecar, so train and decode can never
  disagree.
- **Plumbing**: `macro_tokens` flag on `TwoTowerConfig` / `ModelBuildConfig` /
  matrix `Experiment` (default `False` — zero behavior change for existing
  rows); `TwoTowerModel.from_records` mines the table from the training
  records when enabled. Grammar-gate surface functions
  (`grammar._token_surface_piece`, `fastpath/compiler_draft._token_piece`)
  treat macro tokens as opaque non-surface ids — pre-empting the E257-class
  bug where an unhandled token kind leaked raw sentinels into the Lark prefix.
- **Diffusion policy**: new `macro_substitution` corruption policy masks every
  macro token (one id = one bound block edit), uniform fallback when a row
  has no macros.

## Induction result on the fixture corpus (108 records, train v1)

16 macros mined (hits the `max_macros=16` cap): corpus 4,964 → 3,261 tokens
including the 80-token table (−34.3%); description length 24,125 → 15,566 bits
(−35.5%). Top spans by net gain: `= Stack ( [` (freq 115, +341), `=
TextContent (` (freq 150, +297), `] , STR:column )` (freq 85, +251), and a
whole byte-literal macro `LIT_STR B:65 B:6d B:61 B:69 B:6c LIT_END`
("email", freq 20, +113) — the miner crosses the literal channel when a
placeholder string recurs verbatim.

The compression is visible in training throughput: at the identical fixture
recipe (80 steps, batch 4, seed 0, lexer tokenizer, same corpus),
`seen_target_tokens` drops 15,417 → 10,118 (−34.4%) with `macro_tokens=true`.

## Fixture result (wiring evidence only)

E280 recipe: `--matrix v13 --only E280 --steps 80 --device cpu
--context-backend scratch --no-design-md-context --rico-limit 3
--scratch-control`; suites smoke 3 / held_out 5 / adversarial 4 / ood 4 /
rico_held 3.

| Suite | syntax | meaningful | struct sim | comp recall |
| --- | --- | --- | --- | --- |
| smoke | 0.0 | 0.0 | 0.11 | 0.0 |
| held_out | 0.0 | 0.0 | 0.17 | 0.0 |
| adversarial | 0.0 | 0.0 | 0.16 | 0.0 |
| ood | 0.0 | 0.0 | 0.13 | 0.0 |
| rico_held | 0.0 | 0.0 | 0.05 | 0.0 |

Train loss 5.61 @80 steps; AgentV 0/5. All honest gates fail, consistent with
every 80-step fixture row in this program — the row exists to prove the
channel is wired end-to-end (mining → substituted training targets →
persisted table → deterministic expansion at decode → evals on expanded
output), which the persisted checkpoint sidecar confirms (16-entry
`macro_expansions` table in `last.tokenizer.json`).

## Verification

- `tests/test_harnesses/model_build/test_dsl_tokenizer.py`:
  `test_macro_induction_round_trip_and_persistence` (determinism, fixed-kind
  restriction, MDL accounting including table cost, per-source shortening +
  `canonical_equal` round-trip on 20 seed programs, sidecar save/load) and
  `test_macro_expansions_fail_closed_on_dynamic_tokens` (rejection of
  `<SYM_0>`-bearing and too-short spans; orphaned macro id decodes to
  nothing).
- `tests/test_data/test_diffusion.py::test_macro_substitution_policy_masks_whole_blocks`
  (all macro positions predicted, exact reconstruction, uniform fallback).
- Full pass over `tests/test_data`, `tests/test_harnesses/model_build`,
  `tests/test_models`, `tests/test_dsl`, `tests/test_evals`; `repo_policy`,
  `ruff`, `git diff --check` clean.

## Honesty and limits

- **Fixture/scratch wiring evidence only** — 108-record corpus, tiny suites,
  no ship claim, no gate weakened, nothing promoted. The E2
  `component_type_recall` floors guard this lever at ship scale and are
  computed on expanded output, so macros cannot game them.
- This v13 run has **no matched no-macro control row**; the
  `seen_target_tokens` comparison above reuses local 80-step artifacts from
  earlier same-recipe runs on the same corpus and is a throughput
  observation, not a quality comparison. A matched-pair (macro on/off) row at
  a real budget is the open E-row.
- Whether shorter sequences actually buy quality (fewer denoiser decisions
  per program vs. rarer, higher-entropy macro ids) is exactly what the
  frontier-scale matched pair must answer; fixture scale cannot.
- The `max_macros=16` cap binds on this corpus; frontier corpora may want a
  gain-threshold stop instead. The byte-literal macro shows the miner will
  memorize recurring placeholder strings — acceptable under deterministic
  expansion, but worth watching for OOD placeholder generalization.
