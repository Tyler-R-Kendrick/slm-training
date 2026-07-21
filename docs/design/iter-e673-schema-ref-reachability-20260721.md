# E673 — public schema-reference reachability

Date: 2026-07-21
Status: completed positive scratch; retained research baseline; not ship

One capped CPU OOD `n=4` evaluation-only scratch run reused E620's rejected
local checkpoint under the exact E653 policy. It emitted AgentEvals and AgentV
with no timeout or fallback after all 119 compiler tests passed.

E673 resolves public `$ref` nodes with cycle protection when deciding whether a
typed-array item can reach a missing visible slot. Dashboard replaces the wrong
nested `Carousel([[Form(...)]] )` route with direct
`Card([TextContent(":ood.dash.m1.value")])`. Strict v2 stays 0.7500, while
structure rises 0.7230→0.7931, node/edge F1 rise 0.7987/0.6845→0.8556/0.7486,
and p95 falls 9332.10→6621.10 ms. Meaningful v1, fidelity, validity, recall,
and reward hold.

Retain v130 as the next research baseline, not a ship result. The metric slot
still fails semantic-role meaning, and AgentV remains 0/1. No checkpoint was
created, synced, or promoted.

Evidence: [JSON](iter-e673-schema-ref-reachability-20260721.json).
