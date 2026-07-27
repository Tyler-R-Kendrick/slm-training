# VAR3-05 (SLM-433): seed-replication check for VAR3-04's `held_out_improvement`

Date: 2026-07-27
Status: implemented; fixture/small-scale wiring evidence only
Scope: `scripts/run_slm433_05_turn_disposition_seed_sweep.py` (new). No
change to VAR3-04's `run_slm433_04_turn_disposition_real_corpus.py` or any
harness/model/scoring code -- this issue calls VAR3-04's own
`run_training` unmodified, across seeds.
Honesty: **fixture_or_scratch / wiring, not a capability or ship claim.** No
promotion, gate change, or production-readiness claim is made anywhere in
this document.

## Decision

VAR3-04 (`docs/design/var3-04-turn-disposition-real-corpus-20260727.md`)
reported a single real `held_out_improvement` result and named its own
scale limits explicitly: "one seed, one small held-out split, no
statistical test, no cross-seed replication." Its "Non-goals" section
named the next honest step directly -- does this replicate, or was it that
one seed's noise? This issue is that check, and nothing more: it calls
VAR3-04's own `run_training` at the **identical recipe**
(`n_train_roots=8, n_dev_roots=4, dim=8, steps=30, learning_rate=0.05`,
`max_combinations_per_operator=32`) across 5 seeds
(`433_04, 1, 2, 3, 4`) and reports each seed's `disposition` and held-out
`composite_penalized_error_rate` exactly as measured.

Every seed rebuilds the same real corpus from `openui_verified_v1` /
`e763_symbol_only_eval_r2_20260722/suites/held_out` independently (record
selection is deterministic by sorted id, not seed-dependent) -- only the
classifier's initialization and training trajectory vary by seed. This
isolates the question this issue asks (is the *training* effect
seed-robust) from a question it does not ask (would a different corpus
sample show the same effect).

## Result: `replicates_every_seed`

| Seed | `disposition` | held-out `composite_penalized_error_rate` (baseline 0.5625) | `abstention_rate` |
| ---: | --- | ---: | ---: |
| 43304 | `held_out_improvement` | 0.4375 | 0.625 |
| 1 | `held_out_improvement` | 0.5000 | 0.500 |
| 2 | `held_out_improvement` | 0.3750 | 0.750 |
| 3 | `held_out_improvement` | 0.5000 | 0.500 |
| 4 | `held_out_improvement` | 0.4375 | 0.625 |

All 5 seeds tried produced `held_out_improvement` -- `disposition_trained`'s
held-out `composite_penalized_error_rate` was strictly lower than the
always-emit baselines' (0.5625) on every seed, ranging 0.375-0.500. No seed
was excluded, re-run, or cherry-picked; all 5 rows in the committed JSON are
exactly what each independent run produced.

**`claim_class` stays `"wiring"`, not `"capability"`.** This rules out
single-seed noise as the explanation for VAR3-04's result -- it does not
establish that the effect generalizes beyond this one n_dev=4-root, n=8-row
real corpus sample, and it does not survive to a larger held-out set (the
`e763_symbol_only_eval_r2_20260722/suites/held_out` source has only 5
admitted document records total, so a materially larger held-out split
would need a different, larger source -- out of scope for this issue). The
trained arm's held-out predictions still lean on abstaining (`clarify`)
rather than a more accurate `emit` (`abstention_rate` 0.5-0.75 vs. the
baselines' 0.0 on every seed) -- consistent with, but not proof of, the
classifier learning that uncertain rows are better deferred than guessed.

## Non-goals

No larger or different held-out corpus (the current held-out source is
capped at 5 admitted documents), no statistical significance test beyond
"replicated on every seed tried," no promotion or ship-gate claim, no
change to VAR3-04's recipe, corpus, or scoring.

## Tests

`pytest -q tests/test_scripts/test_run_slm433_05_turn_disposition_seed_sweep.py`
(new, 3 tests): stubs `run_training` (VAR3-04's own test file already
covers that the real corpus pipeline itself is correct) and exercises only
this issue's new logic -- the seed-sweep aggregation into
`disposition_counts`/`verdict` -- across all three possible verdicts
(`replicates_every_seed`, `does_not_replicate`, `seed_sensitive`).

## Required artifacts

This JSON/Markdown pair
(`docs/design/var3-05-turn-disposition-seed-sweep-20260727.{json,md}`).

## Acceptance criteria

- [x] VAR3-04's `run_training` is called completely unmodified, at the
  identical recipe, across every seed.
- [x] Every seed's row is reported (seed id, disposition, both composite
  error rates, abstention rate, skipped-root ids) -- no seed silently
  dropped or excluded from the committed JSON.
- [x] The verdict (`replicates_every_seed` / `does_not_replicate` /
  `seed_sensitive`) is derived directly from the per-seed disposition
  counts, never asserted independently of them.
- [x] No promotion or ship claim is made (`claim_class: "wiring"`,
  `honesty: "fixture_or_scratch"`), regardless of how strongly the result
  replicated.

## Falsification / stop rule

This result licenses one narrow conclusion: VAR3-04's `held_out_improvement`
was not an artifact of its one original seed. It does not license scaling
this corpus, training loop, or head to anything resembling a ship claim --
that would require a materially larger and/or differently-sourced held-out
split than this repo's current `e763_symbol_only_eval_r2_20260722` fixture
provides, which is out of scope here. If a future larger-held-out rerun
shows the effect vanish, that is the legitimate outcome to report, not a
reason to discard this seed-sweep's own (narrower) finding.
