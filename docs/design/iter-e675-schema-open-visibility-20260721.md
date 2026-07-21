# E675 — prompt-authored open-property visibility

Date: 2026-07-21
Status: completed positive semantic scratch; retained; not ship

One capped CPU OOD `n=4` evaluation-only scratch run reused E620's local
checkpoint under the exact E674 policy, adding only a default-off
`schema_open_decode_weight=2`. It emitted AgentEvals and AgentV with no timeout
or fallback after 153 compiler, factory, and eval tests passed.

The lever reads the public positional schema and applies only when the active
property is boolean `open` and the enclosing component family is present in the
authored prompt plan. It abstains for unplanned components and every other
boolean property. This is a schema-derived visibility rule, not a Modal name or
fixture-string special case.

Exactly one token changes: the requested Modal becomes
`Modal(title, true, children)` instead of `Modal(title, false, children)`.
Dashboard, Gallery, and Auth are byte-identical. The public Modal schema says
`open=true` displays the dialog, and the committed gold uses `true`; therefore
the change repairs render semantics. Existing aggregate metrics do not score
this literal, so meaningful v1, strict v2, fidelity, validity, structure,
recall, reward, and AST F1 are all unchanged. No performance claim is made from
the small latency difference.

Retain v132 and the default-off lever as a narrow semantic correction, not a
quantitative or ship improvement. The suite is diagnostic `n=4`, AgentV remains
0/1 because full ship minimums are unmet, and no checkpoint was created, synced,
or promoted.

Evidence: [JSON](iter-e675-schema-open-visibility-20260721.json).
