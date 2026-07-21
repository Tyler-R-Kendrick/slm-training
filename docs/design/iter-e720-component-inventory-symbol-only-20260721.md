# E720 symbol-only component-inventory diagnostic

**Date:** 2026-07-21  
**Decision:** reject the checkpoint and decode bias; retain no new default  
**Evidence:** [`iter-e720-component-inventory-symbol-only-20260721.json`](iter-e720-component-inventory-symbol-only-20260721.json)

## Question

E714 repeatedly emits frequent training components such as `TextContent` while
missing prompt-requested `Button`, `Callout`, and `Card` families. This is not a
simple corpus-coverage gap: the strict 141-record symbol-only snapshot contains
111 `TextContent`, 82 `Stack`, 31 `Card`, 28 `Button`, 3 `Callout`, and 3
`CardHeader` targets, including 16 button, 12 callout, and 9 hero prompts. E720
therefore tests the existing generalized component-inventory objective instead
of adding examples or special-casing component names.

## Recipe

- Local CPU scratch TwoTower, output contract v2, lexer tokenizer, grammar-LTR
  primary, honest slot contract, tree compiler.
- Exact E714 strict snapshot: 141 records, manifest
  `78096191b5272658a1f6c48b6cdceaba5492f363630c0bf7d1b19d2f465b2b45`.
- 600 steps, batch 4, inventory loss weight 1.0, configured inventory decode
  weight 4.0, 160-symbol canvas.
- Completed in 72.38 seconds under cumulative `max_wall_minutes=2`; 307,003
  prompt and 63,841 target tokens seen.
- Local-only checkpoint SHA
  `842a1a21fb9897fe5ee594d9c9d2835315d63d4a12905e3c3640eec348f91a11`.
  `--no-sync-checkpoints` was explicit because this is a scratch diagnostic.

At step 600, total loss was 3.8415 and primary reconstruction loss was 2.5074.
The auxiliary head did learn its training target: inventory loss 0.5906,
top-k recall 0.6875, score margin 2.5212, and mean positive count 1.75.

## Bounded smoke result

Both evaluations use the same three frozen smoke prompts, one attempt, an
eight-second per-example decode timeout, no unconstrained fallback, and AgentV.

| Decode arm | Parse | Strict-v2 meaning | Fidelity | Structure | Component recall | Reward | Timeouts | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inventory bias 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3/3 | 8000.83 / 8000.98 ms |
| Inventory bias 0 | 0.0000 | 0.0000 | 0.8056 | 0.2903 | 0.3333 | 0.0000 | 0/3 | 3465.93 / 6258.84 ms |

AgentV failed 0/1 in both arms with zero execution errors. Bias 4 forces all
three examples into timeout/empty-output failure. Bias 0 recovers non-empty
outputs and useful diagnostic overlap, but every prediction remains invalid
(including required-schema failures), so parse, strict meaning, and reward are
all zero.

## Disposition

The learned inventory signal does not transfer into valid programs, and the
configured decode bias is actively harmful. E720 is rejected, not promoted,
not uploaded, and not a ship checkpoint. This closes stronger inventory-bias
tuning on this checkpoint: the next experiment must target generalized
grammar/schema decision completion rather than another scalar inventory or
canvas sweep.
