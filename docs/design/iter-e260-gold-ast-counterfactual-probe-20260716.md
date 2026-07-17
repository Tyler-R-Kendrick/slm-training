# E260 — grammar/AST-aligned counterfactual probe

Date: 2026-07-16
Status: **completed; bounded hypothesis confirmed; full-corpus run required**

E260 replaces model-rollout state mining with exact branch states derived from
`gold_compiler_decisions()`. Gold remains outside model context: it supplies the
parsed offline state and selected completion, while competing verifier-legal
branches are completed by the unchanged E228 policy. Every candidate is then
graded by the independent judge plus meaningful-program/Pareto verifier before
an event can qualify.

The bounded probe used the first 10 document records from E230, four stratified
gold-AST states per record, up to four candidates per state, CPU, and seed 260.
It started from merged commit
`1b4cb50c2b10f66ebc93ceea0d9b8175926a390f` after a fresh fetch/rebase and a
clean `0 behind / 0 ahead` proof. Trace ID:
`1211a687b0b3464bf67409ee31cdb806`.

## Measured result

| Measure | Result |
| --- | ---: |
| Accepted document traces | 10 / 10 |
| Exact states replayed | 40 |
| Grammar-legal candidates | 116 |
| Independent-judge pass | 114 / 116 |
| Fully verified candidates | 48 / 116 |
| Qualified events | 30 |
| Qualified decision kinds | 11 |
| Qualified prompt groups | 9 |
| Train / held-out events | 24 / 6 |
| Train / held-out groups | 7 / 2 |
| Set-valued events | 17 |

All 30 retained probes declare `state_source=gold_ast` and contain both a
gold-AST selected completion and at least one policy-completed legal
alternative. The stable held-out groups are `program_2b63e36ffdc1226c` and
`program_36eb21e823ac68c9`; no split was manually reassigned.

## Decision

The bounded hypothesis is confirmed. E260 exceeds the group support of the
65-record E258/E259 rollout probes while using only 10 records, and it clears
the minimum two-held-out-group prerequisite. Do not train from this bounded
export. Run the identical recipe over all 65 document records, persist one
immutable source-controlled corpus with full judge evidence, and inspect its
group/role distribution before unblocking a new set-FTPO experiment.

No checkpoint was written. Machine-readable evidence:
[`quality-matrix-v10-e260-gold-ast-probe-results.json`](quality-matrix-v10-e260-gold-ast-probe-results.json).
