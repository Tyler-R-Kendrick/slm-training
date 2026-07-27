# E850-E853: surgical typed-collection supplement

## Harness and data

E850 proved that composing E826 with the entire fixture registry was not
surgical: it admitted four fixtures and removed one semantic near-duplicate.
That snapshot is rejected. The shared train-data builder now supports
`existing+fixture` plus an explicit, fail-closed fixture-id selection, avoiding
temporary merged files and duplicated corpus definitions.

E851 derived from committed E826 under unchanged strict gates and selected only
`train_form_three_controls_01` and `train_switch_group_three_items_01`. It
admitted 351 rows: exactly those two fixtures were added and
`rico_train_100_aug_dir` was removed as a 0.9502 semantic duplicate of
`rico_train_0_aug_dir`. There were no source, normalization, verification,
quality, decontamination, or sanitizer-fallback failures. The quality report
and synthesis feedback contain no warnings, recommendations, or experiment
candidates. Content fingerprint:
`b5193be89e42fb052a056584f22a9159cf1dccbb1edf2271ec1941cd41e56fb9`.

## Local train and smoke

E852 trained the canonical TwoTower locally on CPU with scratch context, lexer
output, batch size 4, AdamW, and 600 steps. It completed in 70.15 seconds under
the 95-second harness cap; final loss was 4.0737. The explicit no-sync scratch
checkpoint SHA-256 is
`76cd2dc28b921cd9de3efff29e2b07ebf8df453abbca31559f95368525b09819`.

E853 used the unchanged E842 smoke suite and strict compiler-tree policy with
semantic-plan decode weight 4 and margin 2.

| n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | Fallback / timeout | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6000 | 0.6589 | 0.7500 | 0.9490 | 3569.97 / 3862.12 ms | 0 / 0 | 0/1 |

This improves E843 structure from 0.6033 to 0.6589 and component recall from
0.6667 to 0.7500 while preserving its perfect parse, strict meaning, fidelity,
and reward. Retain E851 and E852 as the stronger local diagnostic baseline.
AgentV still fails and only smoke `n=3` ran, so there is no promotion, bucket
sync, deployment, or ship claim. No remote workflow ran.
