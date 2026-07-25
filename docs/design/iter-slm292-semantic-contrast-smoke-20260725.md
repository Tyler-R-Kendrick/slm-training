# SLM-292 (AP-010): semantic-contrast objective -- fixture-scale wiring smoke

- claim_class: `fixture_wiring` (NOT a promotion claim)
- Fixture-scale wiring evidence only (2 training records, d_model=32, 15 steps). This is NOT the AP-010 promotion claim: meaning-v2 +0.05 absolute (or binder/reference F1 +0.10) with paired CI excluding zero, replicated across >=3 seeds on the full held-out suites, is unmeasured here. Constrained/raw/repaired decode-outcome comparisons are not run in this smoke at all (a single constrained TwoTowerModel.generate call takes >60s even on this toy architecture and would blow the repo MAX_RUN_MINUTES hard run cap) -- explicit follow-up via scripts.evaluate_model.

## Matched control/treatment arms

| arm | semantic_contrast_loss_weight | seed | steps | d_model | final total loss |
| --- | --- | --- | --- | --- | --- |
| control | 0.0 | 123 | 15 | 32 | 12.0381 |
| treatment | 1.0 | 123 | 15 | 32 | 13.5721 |

## Per-mutation-family effect (treatment arm, aggregated over all logged steps)

| family | n_samples | mean positive distance | mean negative distance | mean margin (neg - pos) |
| --- | --- | --- | --- | --- |
| binding | 31 | 0.0030 | 0.0037 | 0.0008 |
| content | 40 | 0.0038 | 0.0048 | 0.0011 |
| contract | 19 | 0.0029 | 0.0038 | 0.0009 |

## Required logged fields (first vs. last treatment step)

| field | first step | last step |
| --- | --- | --- |
| semantic_contrast_loss_weight | 1.0 | 1.0 |
| semantic_contrast_objective | margin | margin |
| semantic_contrast_margin | 0.2 | 0.2 |
| semantic_contrast_pairs | 6 | 6 |
| semantic_contrast_sampling_seed | 0 | 0 |
| semantic_contrast_family_counts | `{'contract': 2, 'content': 3, 'binding': 1}` | `{'content': 2, 'binding': 1, 'contract': 3}` |
| semantic_contrast_positive_distance_mean | 0.0040 | 0.0030 |
| semantic_contrast_negative_distance_mean | 0.0044 | 0.0046 |

## Follow-up (not run in this session)

- Full >=3-seed promotion campaign on the real training corpus + frozen held-out suites, gated on meaning-v2 +0.05 absolute or binder/reference F1 +0.10 with paired CI excluding zero, and syntax/contract validity regression <=0.01.
- Raw/constrained/repaired decode-outcome comparisons (skipped entirely here -- see disclosure).
