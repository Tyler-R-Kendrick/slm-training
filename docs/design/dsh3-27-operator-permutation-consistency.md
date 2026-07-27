# SLM-402 (DSH3-27): local operator-permutation stop-rule probe

**Status:** fixture / wiring only.
**Honest verdict:** `fixture_wiring_stop_rule_negative`.
**Disposition:** `closed_negative_unaugmented_invariant`.

The typed encoder’s evaluator-side scores were unchanged across 7 compatible
reference/candidate permutations on unseen seeds. The stop rule therefore retains the alignment
and cache regression contract and does **not** add augmentation or KL/JS consistency training.

| Metric | Value |
| --- | ---: |
| Top-1 semantic flips | 0 |
| Accepted-set mass drift | 0.000000 |
| Calibration-confidence drift | 0.000000 |
| Score disagreement | 0.000000 |
| Cache hits / misses | 4 / 4 |

- Local CPU; 30 synthetic typed training decisions; 150 steps; K=[1, 2, 4]; suite n=7.
- Fresh legal sets were re-enumerated after `ReferenceTableV1.permuted`; semantic action IDs and descriptor fingerprints were used only by evaluator alignment, never by model input.
- Unequal PARTIAL membership, cross-request/state/branch comparisons, and any KL/JS over those mismatched domains remain rejected by the contract.
- This is not a CAP2, meaningful-parse, ship-gate, or model-promotion result. AgentEvals/AgentV records only the fixture stop-rule assertion.
