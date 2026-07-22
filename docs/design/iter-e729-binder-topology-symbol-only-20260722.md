# E729 symbol-only binder-reference topology

**Date:** 2026-07-22  
**Decision:** reject the checkpoint and keep topology decode off  
**Evidence:** [`iter-e729-binder-topology-symbol-only-20260722.json`](iter-e729-binder-topology-symbol-only-20260722.json)

## Question

Can the lexer-native parent-to-child binder topology head improve reference
identity when added to E723's effective slot-owner recipe? Unlike E726's
choice-only root heads, this lever is executable on lexer compiler paths and
passed the gradient/path-score preflight before training.

## Recipe

- Local CPU scratch TwoTower; output contract `symbol_only/v2`, lexer output,
  grammar-LTR/tree decode, honest slot contract, and no free-form target channel.
- Exact strict 141-record snapshot, manifest
  `78096191b5272658a1f6c48b6cdceaba5492f363630c0bf7d1b19d2f465b2b45`.
- 140 steps, batch 4, plan loss/decode 1, edge loss 1/decode 0, slot-owner
  loss/decode 1, and binder-topology loss/decode 1.
- Completed in 77.50 seconds under cumulative `max_wall_minutes=2`; 72,021
  prompt and 14,976 target tokens seen.
- Local-only checkpoint SHA
  `c5bafb8d88a0897e3c9c2d4727b04134042ae2944cdeabcb3c65fb7a9d18c43d`.
  `--no-sync-checkpoints` was explicit.

The topology head learned: final loss 0.6602, accuracy 0.7143 across seven
active rows, and 3.71 legal candidates per row. The shared primary
reconstruction loss (5.0753) and slot-owner diagnostics reproduce E723/E727.

## Matched smoke result

All accepted arms use the same checkpoint and three records, compiler-tree
decode, constrained slot contracts, one attempt, an eight-second per-record
timeout, a 160-symbol canvas, no unconstrained fallback, and AgentV.

| Topology weight | Parse | Meaning-v1 | Strict-v2 | Fidelity | Structure | Recall | Reward | Applications / changes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 | 0 / 0 |
| 0.25 | 1.0000 | 0.3333 | 0.0000 | 0.5278 | 0.4642 | 0.2500 | 0.7953 | 3 / 3 |
| 1 | 1.0000 | 0.3333 | 0.0000 | 0.5278 | 0.4642 | 0.2500 | 0.7953 | 3 / 3 |

Weights 0.25 and 1 produce identical prediction hashes. The control exactly
reproduces E723 smoke quality. Therefore the learned reference-ranking signal
causally changes all three applicable choices in the wrong direction. AgentV
is 0/1 for each arm with no execution errors; no arm times out.

The initial invocation stopped before model execution because the fresh
worktree lacked the default eval-data directory. A second completed scoreboard
omitted slot-contract constrained decode and is retained only as a weaker-policy
failure, not included in the matched comparison.

## Disposition

Reject the checkpoint, do not upload or promote it, and keep binder-topology
decode disabled for this recipe. Smoke already regresses materially, so a
held-out run would spend cycles without a promotion path. E723 remains the
minimal effective checkpoint recipe; the next experiment should improve the
slot-owner head or root-list closure without overriding correct reference
choices.
