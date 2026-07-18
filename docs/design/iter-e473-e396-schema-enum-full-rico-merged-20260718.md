# E473 E396 schema-enum full RICO merge — 2026-07-18

E473 canonically merges E472's 16 normally completed RICO shards. The merger
verifies one checkpoint and evaluation policy, unique records, and exact
contiguous `[0,1500)` coverage before producing full-suite metrics.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, E470 schema-enum constraints, prompt-role constrained
decode, honest constrained slot contracts, eight generation steps, three
attempts, and no fallback. Every generating process was externally capped at
290 seconds; the merge completed normally in 1.8 seconds under the same cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RICO held | 1500 | 1.0 | 1.0 | 1.0 | 0.8736 | 1.0 | 0.9939 |

The merged suite has zero failures, fallback, or decode timeouts and AgentV
passes 1/1. Relative to E460, structure changes 0.8740→0.8736 while reward is
effectively unchanged at 0.9939; parse, meaningful output, fidelity, and
component recall remain perfect.

**Verdict:** accept the fresh full-RICO result. Five-suite ship-gate assembly
with E470–E471 remains required before replacing E468 as authoritative.
