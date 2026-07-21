# E676 — joint semantic-role carrier

Date: 2026-07-21
Status: completed positive scratch; retained research baseline; not ship

One capped CPU OOD `n=4` evaluation-only scratch run reused E620's local
checkpoint under the exact E675 policy. It emitted AgentEvals and AgentV with
no timeout or fallback after 127 compiler and quality tests passed.

E676 groups visible roles by namespace, then uses one component only when a
bipartite schema match proves that every role has a distinct direct string
property. The visible candidate restrictions remain authoritative. For
Gallery, `Callout.title` and `Callout.description` jointly cover
`hint.title/body`; `TextContent.text` cannot cover both roles in one instance.

Only Gallery changes:

```text
Callout("info", ":ood.gallery.hint.title", ":ood.gallery.hint.body")
```

replaces two loose `TextContent` leaves. Dashboard, Modal, and Auth are
byte-identical. Structure improves 0.7931→0.8181 and node/edge F1 improve
0.8556/0.7486→0.8722/0.7665. Meaningful v1, strict v2, fidelity, validity, and
recall hold. Reward slips 0.973→0.970; that tradeoff is retained explicitly and
requires broader evaluation before promotion. No latency claim is made.

Retain v133 as the next research baseline, not a ship result. The suite is
diagnostic `n=4`, AgentV remains 0/1 because full minimums are unmet, and no
checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e676-joint-role-carrier-20260721.json).
