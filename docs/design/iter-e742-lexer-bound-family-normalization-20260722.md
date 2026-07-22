# E742 — lexer bound-family normalization

**Date:** 2026-07-22  
**Decision:** retain reachability fix; reject weights and checkpoint promotion  
**Evidence:** [`iter-e742-lexer-bound-family-normalization-20260722.json`](iter-e742-lexer-bound-family-normalization-20260722.json)

E742 fixes a remaining codec-kind mismatch in the existing semantic-plan
family scorer. The scorer intentionally owns bound families rather than root
selection, but its compiler adapter recognized only the synthetic
`component_bound` kind. Real lexer paths use `component`, leaving enabled
semantic-plan weights inert. The adapter now normalizes both representations to
the same bound-family role. No new lever, template channel, or free-form output
path was added.

The matched arms reuse the unchanged local E735 checkpoint and three frozen
smoke records. They ran locally on CPU with `strict_compiler_tree`, honest slot
contracts, visible semantic roles, schema-role and semantic-role weights 2,
coverage-close weight 2, component-plan and root-arity weights 1, a 160-symbol
canvas, an eight-second per-record guard, and the two-minute command cap. Their
complete 235-field effective configurations are byte-identical after removing
only semantic-plan family weight and margin (0/0 versus 4/2).

| Arm | Plan apps / changes | Close apps / changes | Tokens | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v208 plan 0/0 | 0 / 0 | 3 / 3 | 61 | 1.0000 | 0.6667 | 0.0000 | 1.0000 | 1.0000 | 0.5464 | 0.4167 | 0.9370 | 1733 / 2536 ms | 0/1 |
| v208 plan 4/2 | 3 / 2 | 3 / 3 | 61 | 1.0000 | 1.0000 | 0.0000 | 0.8889 | 0.9333 | 0.6628 | 0.5000 | 0.9037 | 1050 / 3090 ms | 0/1 |

The treatment restores the prompt-required `Card` and `Callout` families and
keeps coverage-close active. Meaning-v1 rises by 0.3333, structural similarity
by 0.1164, component recall by 0.0833, and median latency falls by 39.4%.
However, `Card.children` still receives one role-incompatible symbol and the
three-argument `Callout` consumes only title/body, so fidelity falls 0.1111,
validity falls 0.0667, reward falls 0.0333, strict-v2 stays zero, and AgentV
stays 0/1.

Retain model v208 because an enabled canonical lever must not silently no-op on
one tokenizer representation. Keep the behavior default-off and reject weights
4/2, checkpoint promotion, and ship claims on this recipe. No checkpoint was
created, synced, or promoted. The next intervention should use the existing
schema/semantic-role contract to place visible template symbols in legal typed
properties inside the restored planned family.
