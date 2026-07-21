# E687 — decoder trace record identity

Date: 2026-07-21
Status: completed positive observability; retained; not ship

E687 reverts E686's ineffective model-local trace budget and attaches stable
eval record IDs to each per-call `DecodeStats` object before aggregation. The
independently capped full Held-out replay completed with exit 0, no timeout or
fallback, and emitted AgentEvals JSONL plus an AgentV SDK bundle.

Quality is intentionally prediction- and metric-identical to E685: strict
v2.6.0 remains 2/5, structure 0.5108, component recall 0.5733, reward 0.8602,
and AgentV 0/1. Observability succeeds: all 104 selection traces carry one of
the five stable record IDs, including 30 traces for `held_out_tabs_01`.

The tabs failure is now concrete. Decode enters a `Tabs` owner and repeatedly
prefers `TabItem`. It consumes heading, overview, tab1, and tab2, but cannot
route `details.title` and `details.body` through a TabItem content array. That
missing pair is unchanged from token positions 18 through 148 while decode
keeps adding sibling TabItems until the 160-token cap. The selected prediction
therefore remains the short `TextContent(tab2)` candidate.

Retain eval harness v35 and the v141 restoration as positive observability
work, not quality or ship evidence. The next lever must keep one TabItem's
content open for a schema-compatible carrier of the remaining details roles,
not add more sibling TabItems. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e687-trace-record-identity-20260721.json).
