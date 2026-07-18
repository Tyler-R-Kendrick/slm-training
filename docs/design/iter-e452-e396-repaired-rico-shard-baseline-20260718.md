# E452 E396 repaired-RICO shard baseline — 2026-07-18

E452 establishes the no-prompt-role control on E451's repaired RICO rows
1344–1439. It uses E396 unchanged on CPU with local HF context, 320-token
grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, honest constrained slot contracts, eight generation
steps, three attempts, and no unconstrained fallback.

| n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96/1500 | 1.0 | 0.9792 | 1.0000 | 0.6421 | 0.8681 | 0.9778 |

The run completes normally in about 149 seconds under the external 290-second
cap, with no fallback or decode timeout. Its failure breakdown is one
`trivial_layout` and one `low_component_recall`. AgentEvals JSONL and an
AgentV bundle are present; AgentV is 0/5 with zero execution errors because
this is one partial suite.

**Verdict:** E452 is a diagnostic control, not a ship result. Run the identical
repaired shard with only prompt-role constrained decode enabled for the
authoritative matched comparison.
