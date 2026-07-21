# E706 — bounded missing-slot carrier

Date: 2026-07-21  
Status: completed five-suite negative; rejected; not ship

E706 combines E704's schema-value weight 5 with a decoder change that, at a
verified top-level root boundary, binds at most one still-missing visible slot
to the simplest legal direct carrier before normal root closure. The complete
five-suite `n=19` scratch replay finished within the three-minute cap with no
decode timeout or unconstrained fallback and emitted AgentEvals plus a five-case
AgentV SDK bundle.

The treatment is exactly neutral versus E704 weight 5 on every tracked quality
metric. Rico remains at strict 0.0, fidelity 0.8750, validity 0.9250, structure
0.7611, reward 0.9625, node F1 0.7942, and edge F1 0.7757. The trace shows the
missing-slot carrier does not activate at the relevant boundary: the bad enum
assignment has already consumed the visible symbol, so the failure is semantic
role correctness rather than an absent symbol at root closure.

Reject the decoder change and restore v175 behavior as v177. Schema-value weight
4 and semantic-plan root margin 2 remain the retained recipe. AgentV is 0/5 and
the Rico length budget still fails. This is scratch-matrix evidence, not a ship
evaluation; no checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e706-bounded-slot-carrier-20260721.json).
