# E745 — lexer bound-role continuation

**Date:** 2026-07-22  
**Decision:** retain model v210; no checkpoint promotion  
**Evidence:** [`iter-e745-lexer-bound-role-continuation-20260722.json`](iter-e745-lexer-bound-role-continuation-20260722.json)

E745 repairs a representation gap inside the existing
`schema_role_slot_decode_weight` owner. Choice decoding can score a missing
bound symbol directly at an optional property. Lexer compiler decoding first
chooses between call-close and comma, so the same scorer never reached that
property and instead emitted another component later. The shared scorer now
continues a lexer call only when its next official schema property owns a
still-missing bound semantic role. Focused tests cover restricted and tree
selection. No new lever, free-form completion channel, or fixture-specific
component rule was added.

The v209 control is E744's accepted treatment; the v210 replay uses the same
unchanged E735 checkpoint, frozen smoke records, and byte-identical 235-field
effective configuration. Both are local CPU `strict_compiler_tree` diagnostics
with schema-wide declared role candidates, honest slot contracts,
semantic-plan family weights 4/2, coverage-close weight 2, schema-role and
semantic-role weights 2, component-plan and root-arity weights 1, a 160-symbol
canvas, an eight-second per-record guard, and the two-minute command cap.

| Arm | Plan apps / changes | Close apps / changes | Tokens | Parse | Meaning-v1 | Strict-v2 | Coverage | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v209 control | 5 / 3 | 4 / 3 | 77 | 1.0000 | 1.0000 | 0.0000 | 0.3333 | 1.0000 | 1.0000 | 0.7535 | 0.7500 | 0.9370 | 1714 / 4074 ms | 0/1 |
| v210 treatment | 5 / 3 | 4 / 4 | 64 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.8308 | 0.7500 | 0.9370 | 1707 / 2551 ms | 0/1 |

The hero changes from two one-argument `CardHeader` nodes plus three repeated
body leaves to one schema-correct `CardHeader(title, subtitle)`, one kicker,
and one body. Duplicate-subtree spam, placeholder spam, and semantic-role
mismatch all disappear. Structure improves 0.0773, emitted tokens fall 16.9%,
and p95 falls 37.4%; the three-record timing is diagnostic, not a performance
claim. Every remaining strict-v2 reason is `required_inventory_unknown`, which
also explains coverage moving from one known failing row to zero known rows.

Retain v210 because the same declared lever must reach the same schema property
across tokenizer representations. Do not promote a checkpoint or ship claim:
this is smoke `n=3`, strict-v2 is not defined on any covered row, and AgentV is
0/1. The next iteration should make declared runtime symbols authoritative
required-inventory evidence for the evaluator without inspecting marker text.
