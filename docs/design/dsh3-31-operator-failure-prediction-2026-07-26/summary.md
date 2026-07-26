# DSH3-31 operator-decode failure-prediction feature-arm ablation (SLM-406)

Status: fixture-scale synthetic demonstration; not a ship claim

## Per-arm breakdown (18 synthetic rows, 12 replay-grounded failures, 6 SAFE_DEFER)

| Arm | # features | AUROC | AUPRC | Calibration error | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all_counters` | 134 | 1.000 | 1.000 | 0.157 | 18 |
| `compact_subset` | 5 | 1.000 | 1.000 | 0.174 | 18 |
| `entropy_margin_baseline` | 1 | 0.451 | 0.622 | 0.336 | 18 |
| `context_free_baseline` | 0 | 0.500 | 1.000 | 0.167 | 18 |

Best arm by AUROC: `all_counters`.

## Honesty

This is a hand-built, 18-row synthetic fixture spanning all 7 `ReplayOutcomeLabel`s plus 6 SAFE_DEFER rows, not real `DecodeStats` production/replay traces -- it demonstrates the feature-arm scoring, AUROC/AUPRC, and calibration wiring end to end, and shows a fixture-scale positive signal for the compact counter subset over the entropy/margin-only baseline. It is not a claim about real held-out generalization, real checkpoint cross-generalization, or a production early-abort deployment decision. No time-indexed 'earliest decode fraction reaching target precision' metric is computed here -- that requires real time-indexed decode traces which do not exist yet in this repo; this is explicitly out of scope, not approximated. This module changes no production decode/abort behavior (shadow-analysis only).

Decision: `fixture-scale-positive-signal` -- compact_subset strictly exceeds entropy_margin_baseline's AUROC on this 18-row synthetic fixture -- fixture-scale positive signal for DSH3-31's hypothesis that a compact DecodeStats compiler-lattice/constrained-decode counter subset predicts replay-grounded operator-decode failure better than a selector_regret-only margin baseline. This is not real held-out generalization evidence, not a real checkpoint cross-generalization claim, and not a production early-abort deployment decision -- DSH3-31's own stop rule ('if no feature set beats entropy/margin within confidence bounds, reject telemetry-based early abort') requires real decode-trace-scale evidence this fixture cannot provide. The re-test trigger is once real per-decision DecodeStats rows with replay-grounded outcome labels exist at volume -- only then should this ablation be re-run against real traces and confidence bounds computed.
