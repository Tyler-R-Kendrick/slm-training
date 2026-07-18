# E488 E396 JSON-number and typed-any full RICO merge — 2026-07-18

E488 canonically merges E487's 77 normally completed RICO shards. The merger
verified one checkpoint and evaluation policy, unique records, and exact
contiguous `[0,1500)` coverage before producing full-suite metrics. One earlier
48-row attempt exceeded the then-active external cap and is not a merge input.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, component-plan weight 2, slot-component weight 8,
schema-enum, array-item, JSON-number-frame, and typed-any constrained decode,
prompt-role constrained decode, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback. The active policy capped
each process at 170 seconds plus a 10-second kill grace, for a hard maximum of
three minutes. The merge completed normally in about two seconds.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RICO held | 1500 | 1.0 | 1.0 | 1.0 | 0.8737 | 1.0 | 0.9939 |

The merged suite has zero failures, fallback, or decode timeouts, and AgentV
passes 1/1. Compared with E478, three predictions change. JSON-number
constraints replace invalid or untyped Slider bounds on
`rico_hf_test_1810`, `rico_hf_test_2249`, and `rico_hf_test_2644`. The first
improves structure from 0.5103 to 0.5750; the other two are metric-neutral.
Full-suite structure rises by 0.0000431 and reward is unchanged within floating
point precision.

**Verdict:** accept the fresh full-RICO diagnostic result. Five-suite
ship-gate assembly remains required before any promotion or ship claim.
