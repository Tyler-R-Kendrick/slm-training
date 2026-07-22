# E738 lexer planned-subtree nonempty margin

**Date:** 2026-07-22  
**Decision:** retain the generalized structural fix; reject checkpoint promotion  
**Evidence:** [`iter-e738-lexer-planned-subtree-nonempty-20260722.json`](iter-e738-lexer-planned-subtree-nonempty-20260722.json)

E736 proved that prompt-semantic family scoring reached the lexer compiler, but
its selected hero `Card` immediately closed as `Card([])`. The existing
`semantic_plan_typed_array_nonempty_margin_decode_weight` already prevented this
failure for the choice codec. E738 routes that same shared bias through lexer
restricted/tree selection and uses the enabled semantic-plan margin as its
structural margin, so no second overlapping knob is required.

Both accepted arms reuse the unchanged local E735 checkpoint and the same three
frozen smoke records. They ran locally on CPU under `strict_compiler_tree`, an
honest visible slot contract, a 160-symbol canvas, an eight-second per-record
decode timeout, and the two-minute command cap. No remote compute, training,
checkpoint creation, sync, promotion, or serving change occurred.

| Arm | Plan apps / changes | Parse | Meaning-v1 | Strict-v2 | Fidelity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v203 control | 0 / 0 | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 | 1222 / 1238 ms | 0/1 |
| v203 plan 4 / margin 2 | 3 / 2 | 1.0000 | 1.0000 | 0.0000 | 0.6111 | 0.6503 | 0.5000 | 0.8363 | 2011 / 3029 ms | 0/1 |

The hero treatment now emits a nonempty grammar/AST subtree:
`Card([Stack([b2, Button(":smoke.hero.subtitle")]),
TextContent(":smoke.hero.body")])`. The button record remains correct, and the
callout family remains recovered. All tracked quality metrics improve and no
record times out, so retain `config.levers` v10 and `model.twotower` v203.

This is still a bounded scratch diagnostic, not a ship evaluation. Binding-aware
strict meaningfulness remains zero; its only covered record reports a
`placeholder_semantic_role_mismatch`, while two records still lack authoritative
inventory coverage. The next experiment should repair typed visible-slot role
assignment inside the retained subtree without adding any free-form target.

The preliminary r1 pair used a 256-symbol canvas and no explicit per-record
timeout. It reproduced the same aggregate quality deltas but is excluded from
the final comparison because its recipe did not match E736's bounds.
