# E478 E396 schema-array item full RICO merge — 2026-07-18

E478 canonically merges E477's 21 normally completed RICO shards. The merger
verifies one checkpoint and evaluation policy, unique records, and exact
contiguous `[0,1500)` coverage before producing full-suite metrics. Two
earlier 96-row attempts that reached the external timeout emitted no
evaluation JSON and are not merge inputs.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, schema-enum and array-item constrained decode,
prompt-role constrained decode, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback. Every process was
externally capped at 290 seconds; the merge completed normally in about one
second under the same cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RICO held | 1500 | 1.0 | 1.0 | 1.0 | 0.8736 | 1.0 | 0.9939 |

The merged suite has zero failures, fallback, or decode timeouts and AgentV
passes 1/1. Aggregate metrics are identical to E474, while the audited invalid
`AreaChart(["…"], [Button])` output is now schema-valid `AreaChart([], [])`.

**Verdict:** accept the fresh full-RICO result. Five-suite ship-gate assembly
with E476 remains required before replacing E474 as authoritative.
