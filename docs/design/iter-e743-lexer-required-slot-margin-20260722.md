# E743 — lexer required-slot margin reachability

**Date:** 2026-07-22  
**Decision:** retain reachability fix; reject weight and checkpoint promotion  
**Evidence:** [`iter-e743-lexer-required-slot-margin-20260722.json`](iter-e743-lexer-required-slot-margin-20260722.json)

E743 fixes the harness rather than masking an invalid treatment. Preflight
correctly rejected the first lexer treatment before writing an artifact because
`required_slot_margin_decode_weight` was declared choice-only. The canonical
lever registry now declares the existing scorer dual-path, and one shared
adapter applies it to both restricted and tree lexer compiler selection. Its
schema-position gate resolves either choice-decoder frames or the lexer
compiler's active call, while continuing to exclude enum and opaque string
properties. No parallel lever, template channel, free-form completion target,
or checkpoint was added.

The accepted matched arms reuse the unchanged local E735 checkpoint and three
frozen smoke records. They ran locally on CPU with `strict_compiler_tree`,
honest slot contracts, semantic-plan family weights 4/2, coverage-close weight
2, schema-role and semantic-role weights 2, component-plan and root-arity
weights 1, a 160-symbol canvas, an eight-second per-record guard, and the
two-minute command cap. Their complete 235-field effective configurations are
byte-identical after removing only required-slot margin weight (0 versus 2).

| Arm | Margin apps / changes | Plan apps / changes | Close apps / changes | Tokens | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v209 margin 0 | 0 / 0 | 3 / 2 | 3 / 3 | 61 | 1.0000 | 1.0000 | 0.0000 | 0.8889 | 0.9333 | 0.6628 | 0.5000 | 0.9037 | 1164 / 3361 ms | 0/1 |
| v209 margin 2 | 6 / 0 | 3 / 2 | 3 / 3 | 61 | 1.0000 | 1.0000 | 0.0000 | 0.8889 | 0.9333 | 0.6628 | 0.5000 | 0.9037 | 1221 / 3252 ms | 0/1 |

The treatment is no longer inert: the shared scorer applies six times. It
changes zero choices, however, so all predictions and quality metrics are
identical. The remaining hero failure is typed ownership, where the subtitle
symbol is placed in `Button.label` inside `Card.children`; the callout still
uses only its title and description symbols. Strict-v2 remains zero and AgentV
remains 0/1. The small latency movement is not promoted from this three-record
diagnostic.

Retain model v209 and lever registry v14 because enabled canonical levers must
execute on every declared decode path. Keep the behavior default-off and reject
required-slot margin 2, checkpoint promotion, and ship claims. The next
intervention should constrain planned-family role ownership at the active
schema property rather than increasing a global missing-symbol floor.
