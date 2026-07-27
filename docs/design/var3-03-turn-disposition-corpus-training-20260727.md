# VAR3-03 (SLM-433): TurnDispositionHead trained on a real corpus

**Disposition: `trained_head_improves_on_derived_only`** (wins=6, ties=4, losses=0 across 10 seeds)

Fixture-scale (real, freshly-enumerated symbolic corpus, not natural-language data). **Not a ship claim.**

## Why a seed sweep, not a single run

A single (corpus_seed, train_seed) point is not a reproducible claim: CPU BLAS reduction order can differ across environments/thread counts even with a fixed `torch.manual_seed`, and this toy corpus/model is small enough that one flipped held-out prediction flips the whole comparison. A reviewer re-run on a different machine reproduced the corpus/split exactly but got a different `final_train_loss` and the opposite single-seed disposition. This sweep across 10 independent seeds (each redrawing both the corpus and the trained head) is the real evidence surface.

## Preregistered

- Hypothesis: A TurnDispositionHead trained on real, corpus-derived clarify-margin examples reduces composite_penalized_error_rate on a held-out split relative to derived_only (free rules, clarify never fires).
- Aggregate stop rule: Across the preregistered seed sweep (each seed independently redraws both the corpus and the trained head, corpus_seed == train_seed), compare disposition_trained's composite_penalized_error_rate against derived_only's per seed. The hypothesis is supported ('trained_head_improves_on_derived_only') only if wins > losses across the sweep, where a seed counts as a win if trained is strictly lower, a loss if trained is strictly higher, and a tie otherwise (ties count toward neither). wins == losses (including 0-0) is 'no_held_out_improvement' -- a legitimate negative/inconclusive result, not evidence for the hypothesis. No seed is excluded or cherry-picked; every seed in DEFAULT_SWEEP_SEEDS is reported.
- Leakage safety: split is trace-level; all four rows of one trace share one split

## Per-seed results (every seed in the sweep, none excluded)

| seed | trained_composite | derived_composite | outcome | final_train_loss |
| ---: | ---: | ---: | --- | ---: |
| 433 | 0.0375 | 0.0437 | trained_wins | 0.6752 |
| 434 | 0.0469 | 0.0469 | tie | 1.3858 |
| 435 | 0.0469 | 0.0719 | trained_wins | 1.4735 |
| 436 | 0.0500 | 0.0500 | tie | 0.1061 |
| 437 | 0.0312 | 0.0344 | trained_wins | 0.4575 |
| 438 | 0.0438 | 0.0594 | trained_wins | 0.6142 |
| 439 | 0.0344 | 0.0469 | trained_wins | 1.1157 |
| 440 | 0.0500 | 0.0625 | trained_wins | 0.1139 |
| 441 | 0.0437 | 0.0437 | tie | 0.0013 |
| 442 | 0.0594 | 0.0594 | tie | 0.5776 |

## Honesty

fixture_or_scratch, capability class -- not a ship claim. A single-seed run of this experiment is not a reproducible claim (CPU BLAS reduction order can differ across environments/thread counts even with a fixed torch seed, and this toy corpus/model is small enough that one flipped held-out prediction flips the comparison); this sweep across 10 independent seeds is the real evidence surface. No promotion, ship-gate, or production-readiness claim is made regardless of outcome.
