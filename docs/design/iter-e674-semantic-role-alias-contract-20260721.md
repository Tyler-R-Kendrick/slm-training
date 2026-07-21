# E674 — canonical semantic-role alias contract

Date: 2026-07-21
Status: completed positive scratch; retained research baseline; not ship

One capped CPU OOD `n=4` evaluation-only scratch run reused E620's local
checkpoint under the exact E653 decode policy. It emitted AgentEvals and an
AgentV bundle with no timeout or fallback after 45 evaluator/canary tests and
124 compiler/quality tests passed.

The diagnosis first replayed Dashboard's committed gold through strict v2. The
gold failed its own role check because `status.body` occupies
`Callout.description` and metric `*.value` slots occupy `TextContent.text`, but
those schema-valid aliases were absent. The metric-gaming canaries also exposed
an older omission for `action` in `Button.label`. E674 adds only those three
public-schema aliases and advances binding-aware meaningful v2 from 2.2.1 to
2.3.0; ship thresholds remain disabled and unchanged.

Dashboard now emits the gold-aligned assignments:

```text
Button(refresh)
Callout(info, status.title, status.body)
Card(TextContent(m1.value))
Card(TextContent(m2.value))
```

Gallery, Modal, and Auth are byte-identical to E673. Strict v2 rises
0.7500→1.0000, while meaningful v1, fidelity, validity, structure (0.7931),
recall (0.8750), reward (0.9730), and node/edge F1
(0.8556/0.7486) are unchanged. The strict-rate delta is version-bound because
the metric contract changed in this arm; the independently useful evidence is
the gold-aligned Dashboard prediction and the unchanged control records.

Retain v131 and metric v2.3.0 as the next research baseline, not a ship result.
The suite is diagnostic `n=4`, AgentV remains 0/1 because ship evidence requires
the full suite minimum, and no checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e674-semantic-role-alias-contract-20260721.json).
