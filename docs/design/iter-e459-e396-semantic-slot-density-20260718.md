# E459 E396 semantic slot density — 2026-07-18

E459 aligns density feasibility with E458's semantic slot-argument map.
Components with internal required strings are now judged by their visible
content-slot demand rather than every schema string. This keeps Slider legal
after SwitchItem consumes its two visible slots.

The unchanged E396 checkpoint uses CPU, local HF context, 320-token grammar
LTR, automatic content floor, component-plan weight 2, slot-component weight
8, prompt-component constrained decode, honest constrained slot contracts,
eight generation steps, three attempts, and no unconstrained fallback. The
process completed normally in 28 seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.8023 | 0.9048 | 0.9862 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |

Fallback and decode timeout counts are zero. AgentV passes 4/4 with zero
execution errors. Relative to E458, held-out structure improves
0.7553→0.8023, recall 0.8048→0.9048, and reward 0.9838→0.9862; all other
suites are unchanged. Both the tab-panel and settings rows are type-complete.

**Verdict:** keep the semantic density alignment. Bounded evidence improves
without a measured regression. RICO rows explicitly requesting Switch or
Slider must be refreshed before a new five-suite or promotion claim.
