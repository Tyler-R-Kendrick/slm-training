# DSH3-30 CONFLICT vs DIFFERENT_RESULT collapse hard-negative ablation (SLM-405)

Status: fixture-scale synthetic demonstration; not a ship claim

## Per-arm breakdown (9 synthetic rows: 4 DIFFERENT_RESULT, 2 CONFLICT, 3 no-negative)

| Arm | Total | Usable | Unusable | Usable rate | Ranking correct | Mean margin loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| `conflict_only` | 2 | 0 | 2 | 0.000 | 0 | 0.000 |
| `different_result_only` | 4 | 4 | 0 | 1.000 | 2 | 1.000 |
| `equal_mix` | 4 | 2 | 2 | 0.500 | 1 | 1.000 |
| `curriculum_mix` | 6 | 4 | 2 | 0.667 | 2 | 1.000 |

Best arm by usable-negative-rate: `different_result_only`.

## Honesty

This is a hand-built, small synthetic fixture (4 DIFFERENT_RESULT rows, 2 CONFLICT rows, 3 no-negative rows), not a real operator_policy_corpus sample -- it demonstrates the arm-selection and usable-negative-rate wiring end to end, not a production readiness claim, and not a real gradient-trained-model quality claim (the scorer here is a fixed, deterministic, label-free proxy over declared cost/effect-signature, never a trained model). Real verified-conversation-trace incidence of CONFLICT versus DIFFERENT_RESULT hard negatives remains unmeasured, per DSH3-24's own documented scope gap.

Decision: `fixture-scale-positive-signal` -- different_result_only strictly exceeds conflict_only's usable-negative-rate on this small synthetic fixture -- fixture-scale positive signal for DSH3-30's hypothesis that DIFFERENT_RESULT hard negatives carry denser, more sample-efficient ordering supervision than CONFLICT ones. This is not a claim about downstream trained-model quality (no real gradient training happened here -- the scorer is a fixed deterministic label-free proxy) and not a real-corpus-scale incidence measurement (real verified-trace CONFLICT/DIFFERENT_RESULT incidence remains unmeasured, per DSH3-24's own documented scope gap). It does not by itself justify wiring hard-negative-aware loss into DSH3-28's typed_operator_policy_loss at production scale -- that needs the real corpus's actual CONFLICT/DIFFERENT_RESULT incidence measured first.
