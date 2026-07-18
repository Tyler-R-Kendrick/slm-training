# E458 E396 semantic slot arguments — 2026-07-18

E458 prevents visible content slots from being consumed by internal string
arguments on `SwitchItem`, `Slider`, and `TabItem`. It builds on E457's
conservative explicit prompt-component contracts and leaves all other
components on the prior behavior.

The unchanged E396 checkpoint uses CPU, local HF context, 320-token grammar
LTR, automatic content floor, component-plan weight 2, slot-component weight
8, prompt-component constrained decode, honest constrained slot contracts,
eight generation steps, three attempts, and no unconstrained fallback. The
process completed normally in 27 seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7553 | 0.8048 | 0.9838 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |

Fallback and decode timeout counts are zero. AgentV passes 4/4 with zero
execution errors. Relative to E457, held-out structure improves
0.7527→0.7553, recall 0.7381→0.8048, and reward 0.9790→0.9838; the other
three suites are unchanged. The tab-panel row becomes type-complete. The
settings row still misses Slider because an earlier density-feasibility layer
counts its internal strings as content slots.

**Verdict:** keep the semantic slot positions, but align the density layer
before refreshing full-suite evidence. This remains bounded diagnostic
evidence, not a promotion or production HF claim.
