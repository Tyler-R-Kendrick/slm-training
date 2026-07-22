# E722 symbol-only component-edge diagnostic

**Date:** 2026-07-21  
**Decision:** reject edge decoder and checkpoint; retain non-semantic Pareto evidence  
**Evidence:** [`iter-e722-component-edge-symbol-only-20260721.json`](iter-e722-component-edge-symbol-only-20260721.json)

E722 targets E721's duplicate/incorrect hierarchy with the existing generalized
resolved-AST parent-child edge objective. It adds no literals, strings, or
component-specific rules.

The local CPU scratch run used the exact 141-record symbol-only snapshot
(`78096191…b2b45`), grammar-LTR/tree decode, honest slot contract, plan and edge
loss/decode weights 1.0, and a 160-symbol canvas. It completed 150 steps in
77.52 seconds under `max_wall_minutes=2`, with 76,707 prompt and 15,924 target
tokens. Checkpoint SHA is `08873bf0940eec19d0e90f50bfbd801f8547b45e450fa6379abe21c90a25597d`;
it is local-only via explicit `--no-sync-checkpoints`.

At step 150, total loss was 8.1400, primary reconstruction loss 4.2572, edge
loss 0.1959, and edge top-k recall 0.3333. Root-plan accuracy was 1.0 and bound
top-k recall 0.4167.

| Edge decode | Parse | Strict-v2 | Fidelity | Structure | Recall | Reward | AST edge F1 | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 1 | 1.0000 | 0.0000 | 0.5278 | 0.2861 | 0.5000 | 0.8313 | 0.1852 | 3890.05 / 3903.84 ms |
| weight 0 | 1.0000 | 0.0000 | 0.5278 | 0.2861 | 0.5000 | 0.8313 | 0.1852 | 3815.32 / 3985.96 ms |

Both arms have zero timeouts and AgentV 0/1. Edge weight 1 records nine
applications but zero choice changes; outputs and aggregate metrics match the
edge-off control. Compared with E721, the changed training trajectory improves
structure, recall, reward, and latency, but all outputs remain strict-v2 failures
because of placeholder spam/semantic-role mismatch and missing requested
components.

Reject the causal edge-decoder hypothesis and checkpoint. This is a useful
non-semantic Pareto diagnostic, not promotable or ship evidence. Do not upload.
