# E723 symbol-only slot-owner diagnostic

**Date:** 2026-07-21  
**Decision:** retain the generalized decode lever; reject checkpoint promotion  
**Evidence:** [`iter-e723-slot-owner-symbol-only-20260721.json`](iter-e723-slot-owner-symbol-only-20260721.json)

E723 adds the existing prompt-conditioned slot-to-component owner objective to
E722's plan+edge training stack. It targets placeholder ownership without
adding literals, strings, component-specific rules, or remote dependencies.

The local CPU scratch run used the exact 141-record symbol-only snapshot
(`78096191…b2b45`), grammar-LTR/tree decode, honest slot contract, plan/edge/slot
loss weights 1.0, plan decode 1.0, edge decode 0, slot decode 1.0, and a
160-symbol canvas. It completed 140 steps in 77.16 seconds under
`max_wall_minutes=2`, seeing 72,021 prompt and 14,976 target tokens. Checkpoint
SHA `787d2d21d7c29d56637355fd364f16a0d67b1f452fc0f4ce3a7d486b2bd62795`
is local-only via explicit `--no-sync-checkpoints`.

At step 140, total loss was 9.1764, primary reconstruction loss 5.0753, slot
loss 1.4252, and slot accuracy 0.5 (majority baseline 0.75 on four active
rows). Despite weak aggregate head calibration, the legal-choice bias is causal:
all seven treatment applications change a choice.

| Suite / arm | n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Structure | Recall | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke, slot 1 | 3 | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 | 1108.40 |
| smoke, slot 0 | 3 | 1.0000 | 0.3333 | 0.0000 | 0.3333 | 0.3283 | 0.1667 | 0.3203 | 1845.50 |
| held_out, slot 1 | 4 | 1.0000 | 0.2500 | 0.0000 | 0.2667 | 0.3940 | 0.3208 | 0.7290 | 860.16 |
| held_out, slot 0 | 4 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2988 | 0.0000 | 0.0000 | 1418.77 |

Every arm has zero timeouts and AgentV 0/1. The gain generalizes from smoke to
held-out, so retain the slot-owner decoder as the first causal output-contract-v2
semantic improvement. The checkpoint still fails strict-v2 because requested
components remain missing (2/3 smoke and 4/4 held-out); do not upload, promote,
or claim ship readiness. The next arm should target required-component coverage
on top of this retained recipe.
