# E338 bounded HF-context plan-off ablation — 2026-07-17

E338 evaluates the frozen E337 checkpoint with component-plan decode weight
overridden from 1 to 0. E337 telemetry showed the plan head changing choices
while the slot head never applied, making this the smallest causal decode
ablation. The four-suite run completed in 25.1s under the hard 300-second cap.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.0 | 0.2067 | 0.0 | 0.0 | 0.0 |
| held_out | 5 | 1.0 | 0.0 | 0.2196 | 0.0 | 0.0 | 0.0 |
| adversarial | 4 | 1.0 | 0.0 | 0.2311 | 0.0 | 0.0 | 0.0 |
| ood | 4 | 0.75 | 0.0 | 0.1447 | 0.0 | 0.0 | 0.0 |

AgentV passes 0/4 with no execution errors. Disabling the plan restores parse
on smoke, held-out, and adversarial, but every completed suite still has zero
fidelity, meaningful-program rate, component recall, and reward. RICO was
intentionally omitted; this is not a ship evaluation.

**Verdict:** reject E338. Component-plan bias contributes to parse instability
but is not the semantic-collapse cause. No checkpoint was written or promoted.

