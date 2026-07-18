# E457 E396 explicit prompt-component contracts — 2026-07-18

E457 generalizes E454's visible prompt-role parser to conservative natural
language contracts when no `(roles: ...)` tag exists. It recognizes only
explicit counted component nouns: cards, buttons, switches, sliders, and
`N-tab panel`. Role-tagged RICO prompts retain precedence. An audit of all 16
bounded prompts produces contracts for only five directly stated cases.

The unchanged E396 checkpoint uses CPU, local HF context, 320-token grammar
LTR, automatic content floor, component-plan weight 2, slot-component weight
8, prompt-component constrained decode, honest constrained slot contracts,
eight generation steps, three attempts, and no unconstrained fallback. The
process completed normally in 27 seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7527 | 0.7381 | 0.9790 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |

Fallback and decode timeout counts are zero. AgentV passes 4/4 with zero
execution errors. Relative to E455, held-out meaningful rate improves
0.6→1.0, structure 0.6400→0.7527, recall 0.5048→0.7381, and reward
0.5922→0.9790. Adversarial structure improves 0.7661→0.8061; smoke and OOD
are unchanged.

**Verdict:** keep the generalized visible-contract parser. It materially
improves bounded quality without a measured regression. Required TabItem and
Slider counts are still only partially realized, so this is not yet a
five-suite, promotion, or production HF claim.
