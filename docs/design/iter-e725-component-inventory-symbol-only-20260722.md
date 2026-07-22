# E725 cumulative symbol-only component inventory

**Date:** 2026-07-22  
**Decision:** reject the checkpoint and inventory decode lever  
**Evidence:** [`iter-e725-component-inventory-symbol-only-20260722.json`](iter-e725-component-inventory-symbol-only-20260722.json)

## Question

E723's slot-owner objective causally improved semantic overlap, but its outputs
still omitted prompt-requested components. E724 showed that post-hoc coverage
closure cannot help when no compatible continuation reaches the ranking stage.
E725 therefore adds the existing learned prompt component-inventory objective
to E723's plan, edge, and slot-owner training recipe. This changes no grammar,
metric, gate, or free-form output contract.

## Recipe

- Local CPU scratch TwoTower; output contract `symbol_only/v2`, lexer output,
  grammar-LTR primary, honest slot contract, and constrained decode.
- Exact strict 141-record snapshot, manifest
  `78096191b5272658a1f6c48b6cdceaba5492f363630c0bf7d1b19d2f465b2b45`.
- 130 steps, batch 4, plan loss/decode 1, edge loss 1/decode 0, slot-owner
  loss/decode 1, and component-inventory loss/decode 1.
- Completed in 73.22 seconds under cumulative `max_wall_minutes=2`; 66,368
  prompt and 13,898 target tokens seen.
- Local-only checkpoint SHA
  `897208bf4bf0ce12b137145a3a6c88f2140faa6579080b0fe54c6794fde8ba1e`.
  `--no-sync-checkpoints` was explicit because this is a scratch diagnostic.

At step 130, total loss was 12.5112 and primary reconstruction loss was
5.5197. The inventory head produced 0.9026 loss, 0.6833 top-k recall, a 2.0878
positive score margin, and 3.75 mean positive components.

Two setup invocations failed before loading any records because their requested
ignored eval directories did not contain `smoke`; they emitted no metrics and
are not evidence. The accepted runs below use the committed frozen remediated
suite explicitly.

## Bounded smoke result

Both accepted arms use the same three frozen smoke prompts, checkpoint, plan
and slot-owner weights, one attempt, eight-second per-record timeout, 160-symbol
canvas, no unconstrained fallback, and AgentV.

| Decode arm | Parse | Meaning-v1 | Strict-v2 | Fidelity | Structure | Component recall | Reward | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inventory 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3094 | 0.0000 | 0.0000 | 0/3 |
| Inventory 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3094 | 0.0000 | 0.0000 | 0/3 |

All three prediction hashes are identical across arms. Inventory decode records
zero applications and zero choice changes. AgentV is 0/1 in both arms with no
execution errors. The checkpoint also underperforms E723's accepted smoke
scoreboard, although the different step count means that comparison is not a
causal estimate of inventory-loss harm.

## Disposition

Reject the checkpoint and inventory decode lever. The head learns its training
target, but the compiler-legal component decision path never exposes a choice
where its bias can act. Do not spend another cycle on inventory scalar tuning;
the next experiment must target the structural decision path that makes
requested components reachable. Nothing was uploaded or promoted.
