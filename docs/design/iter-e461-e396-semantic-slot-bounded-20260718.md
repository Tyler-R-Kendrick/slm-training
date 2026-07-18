# E461 E396 semantic-slot bounded refresh — 2026-07-18

E461 refreshes all four complete bounded suites after the final E459 decoder
and position-aware reward scrubber changes. It uses the unchanged E396
checkpoint and E451 repaired corpus.

Recipe: CPU, local HF context, 320-token grammar LTR, automatic content floor,
component-plan weight 2, slot-component weight 8, visible prompt-component
constrained decode, semantic slot arguments/density, honest constrained slot
contracts, eight generation steps, three attempts, and no fallback. The
process completed normally in 30 seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.8023 | 0.9048 | 0.9862 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |

All suites have zero failures, fallback, and decode timeouts. AgentV passes
4/4 with zero execution errors. The result exactly reproduces E459 under the
corrected evaluator.

**Verdict:** complete current-policy bounded evidence. All bounded gates pass,
but this is not yet the authoritative five-suite or production HF result.
