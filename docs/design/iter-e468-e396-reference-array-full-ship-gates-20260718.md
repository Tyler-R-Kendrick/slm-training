# E468 E396 reference-array full ship gates — 2026-07-18

E468 merges E467's complete four-suite bounded result with E460's exact
1,500-row RICO result after a prediction-level impact audit. The repaired
branch can fire only when the first emitted component opens a reference-only
child array before any component reference exists. None of E460's 1,500
stored predictions begins with a component array, so every RICO prediction is
unaffected and the exact E460 artifact is reusable.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, current prompt-role constrained decode,
reference-array fail-closed semantics, honest constrained slot contracts,
eight generation steps, three attempts, and no fallback. The merge completed
normally in about two seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6279 | 0.8750 | 0.9865 |
| rico_held | 1500 | 1.0 | 1.0 | 1.0 | 0.8740 | 1.0 | 0.9939 |

All five authoritative local ship gates pass with zero failures, fallback,
and decode timeouts. AgentV passes 5/5 with zero execution errors. Relative
to E465, OOD structure/recall/reward improve
0.5835/0.8125/0.9835→0.6279/0.8750/0.9865; held structure falls
0.8023→0.7838 while recall and reward hold or improve. Smoke, adversarial,
and full RICO are unchanged.

**Verdict:** E396 remains the current-policy local ship-gate champion. This is
authoritative local five-suite evidence, not a production HF ship: the
unchanged checkpoint still lacks a durable bucket sync.
