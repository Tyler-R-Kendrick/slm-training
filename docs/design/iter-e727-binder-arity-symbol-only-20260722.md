# E727 symbol-only binder-reference arity

**Date:** 2026-07-22  
**Decision:** reject the checkpoint and close binder-arity weights 0/1/2  
**Evidence:** [`iter-e727-binder-arity-symbol-only-20260722.json`](iter-e727-binder-arity-symbol-only-20260722.json)

## Question

E726 proved that choice-only root-reference arity cannot run on the required
lexer output path. The existing binder-reference arity objective is lexer
native: it predicts declaration reference counts and scores legal compiler
continue/stop paths. E727 combines that count signal with E723's slot-owner
identity signal.

## Recipe

- Local CPU scratch TwoTower; output contract `symbol_only/v2`, lexer output,
  grammar-LTR primary, compiler-tree evaluation, and honest slot contract.
- Exact strict 141-record snapshot, manifest
  `78096191b5272658a1f6c48b6cdceaba5492f363630c0bf7d1b19d2f465b2b45`.
- 140 steps, batch 4, plan loss/decode 1, edge loss 1/decode 0, slot-owner
  loss/decode 1, and binder-arity loss/decode 1.
- Completed in 77.46 seconds under cumulative `max_wall_minutes=2`; 72,021
  prompt and 14,976 target tokens seen.
- Local-only checkpoint SHA
  `c211d2eae1028334a33d16adc2c29b26a908ade3f90e8c86c2d3da914136a857`.
  `--no-sync-checkpoints` was explicit.

At step 140, total loss was 9.9616 and primary reconstruction loss was 5.0753.
The arity head was active on 10 declaration rows with loss 0.7852 and accuracy
0.80. Slot-owner loss remained 1.4252 with accuracy 0.50.

## Matched evaluation

Accepted arms use compiler-tree decode, the same checkpoint and suite records,
one attempt, an eight-second per-record timeout, a 160-symbol canvas, no
unconstrained fallback, and AgentV.

| Suite / arity weight | n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Structure | Recall | Reward | Applications / changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke / 0 | 3 | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 | 0 / 0 |
| smoke / 1 | 3 | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 | 3 / 0 |
| smoke / 2 | 3 | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 | 3 / 0 |
| held_out / 0 | 4 | 1.0000 | 0.2500 | 0.0000 | 0.2667 | 0.3940 | 0.3208 | 0.7290 | 0 / 0 |
| held_out / 1 | 4 | 1.0000 | 0.2500 | 0.0000 | 0.2667 | 0.3940 | 0.3208 | 0.7290 | 4 / 0 |

Every treatment/control prediction hash is identical within its suite. AgentV
passes 0/1 in each arm with no execution errors. The accepted treatment has no
timeouts. Sequential latency differences are not treated as performance
evidence.

An earlier smoke invocation (`e727-binder-arity1-smoke-r1`) left
`compiler_decode_mode=off`; it is a valid scoreboard for that configuration but
cannot test binder-arity path scoring and is excluded from the causal result.

## Disposition

Reject the checkpoint and close binder-arity scalar tuning for this recipe.
The generalized lexer-native head learns and applies, but its score changes no
decision at weights 1 or 2. E723 remains the better minimal checkpoint recipe;
the next structural experiment should target reference identity or root-list
closure directly on the lexer path. Nothing was uploaded or promoted.
