# E602 — first-component plan score trace

Date: 2026-07-20
Status: diagnostic success; not promotable or ship

E602 adds bounded score-decomposition evidence for the E601 first-component
semantic-plan seed. Each trace records the legal candidate chosen before the
plan, the candidate chosen after it, and the top eight candidates' pre-plan
score, plan bias, and post-plan score. It changes no candidate legality or
ranking policy.

One capped CPU OOD `n=4` diagnostic at seed weight 32 completed within 170
seconds. In deterministic suite order, the seed changes every first choice:

- dashboard: `TextArea` → `Button`; planned `Button`, `Card`, and `Callout`
  each receive +36, with their pre-plan scores deciding the winner;
- gallery: `Image` → `ImageGallery`;
- modal: `TextContent` → `Button`, ahead of the also-planned `Modal`;
- auth: `TextArea` → `Input`, narrowly ahead of the also-planned `Button`.

Despite those four interventions, the final serialized predictions and every
quality aggregate remain identical to E601/E600: syntax 1.0, meaningful-v1
0.5, strict meaning-v2 0, fidelity 0.5917, validity 0.7550, structure 0.5169,
component recall 0.6250, reward 0.8115, AST-node F1 0.5754, and AST-edge F1
0.4143. AgentV remains 0/1.

The failure is therefore downstream of first-family ranking: a later
attempt/final-candidate path discards the changed first choice. E603 should
trace candidate-attempt outcomes and final selection before changing scores
again. No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e602-plan-seed-score-trace-20260720.json).
