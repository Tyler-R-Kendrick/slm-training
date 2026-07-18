# E462 E396 semantic-slot full ship gates — 2026-07-18

E462 merges the fresh E461 bounded suites with E460's exact full-RICO
refresh. The checkpoint is unchanged from E396; the evidence uses the E451
repaired corpus and E459's final decoder plus the corrected position-aware
reward evaluator.

Recipe: CPU, local HF context, 320-token grammar LTR, automatic content floor,
component-plan weight 2, slot-component weight 8, visible prompt-component
constrained decode, semantic slot arguments/density, honest constrained slot
contracts, eight generation steps, three attempts, and no fallback. The merge
completed normally in 1.8 seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.8023 | 0.9048 | 0.9862 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |
| rico_held | 1500 | 1.0 | 1.0 | 1.0 | 0.8740 | 1.0 | 0.9939 |

All five authoritative ship gates pass with zero failures, fallback, and
decode timeouts. AgentV passes 5/5 with zero execution errors. Relative to the
previous E456 five-suite evidence, the decoder/evaluator repairs raise full
RICO structure from 0.8683 to 0.8740 and type recall from 0.9960 to 1.0.

**Verdict:** E396 remains the local ship-gate champion under the refreshed
current policy. This is authoritative local five-suite evidence, not a
production HF ship: the unchanged checkpoint still lacks a durable bucket
sync.
