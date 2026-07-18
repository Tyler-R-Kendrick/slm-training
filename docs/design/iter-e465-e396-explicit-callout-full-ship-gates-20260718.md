# E465 E396 explicit-callout full ship gates — 2026-07-18

E465 merges E464's complete four-suite bounded result with E460's exact
1,500-row RICO result after a fail-closed impact audit. The new explicit
Callout prompt pattern matches zero E451 RICO prompts, so every RICO
prediction is unaffected and the exact E460 artifact is reusable.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, explicit Callout plus prior prompt-role constrained
decode, honest constrained slot contracts, eight generation steps, three
attempts, and no fallback. The merge completed normally in about two seconds
under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.8023 | 0.9048 | 0.9862 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |
| rico_held | 1500 | 1.0 | 1.0 | 1.0 | 0.8740 | 1.0 | 0.9939 |

All five authoritative local ship gates pass with zero failures, fallback,
and decode timeouts. AgentV passes 5/5 with zero execution errors. Relative
to E462, smoke structure/recall improve 0.5600/0.5000→0.6822/0.6667 while
reward changes 0.9770→0.9730; the other four suites are unchanged.

**Verdict:** E396 remains the current-policy local ship-gate champion. This is
authoritative local five-suite evidence, not a production HF ship: the
unchanged checkpoint still lacks a durable bucket sync.
