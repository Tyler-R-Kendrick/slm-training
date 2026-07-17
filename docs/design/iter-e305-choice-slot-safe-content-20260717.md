# E305 slot-safe connected content (2026-07-17)

E305 repairs the two E304 parse failures without changing its checkpoint.
When the opt-in minimum-content policy is constructing a bound content
component, each required string-bearing argument must consume a visible request
slot when legal, and the component closes after its required arguments. This
prevents enum/direction literals such as `"row"` from occupying TextContent's
required content position.

The authoritative run uses E304 SHA
`2081378f2a3f11530a2193e79a0b98d4f487c2631c3f814018117bbd2677d420`,
plan decode weight 1, concise connected topology, prompt-only inputs, and no
unconstrained fallback.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Recall | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.5278 | 0.4642 | 0.1667 | 0.2497 | 162.49 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2800 | 0.3369 | 0.0000 | 0.0000 | 144.63 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.5417 | 0.4744 | 0.3750 | 0.4245 | 144.56 |
| ood | 4 | 1.0000 | 0.0000 | 0.2583 | 0.3750 | 0.0000 | 0.0000 | 148.73 |
| rico_held | 3 | 1.0000 | **1.0000** | 0.5417 | 0.3397 | 0.5556 | 0.8515 | 329.14 |

E305 restores parse 1.0, reduces E304's failed thresholds 10→7, and recovers
AgentV 1/5→2/5. It matches E301's global failure count while retaining E304's
RICO gain. Held-out and OOD remain meaningful/component-recall 0.0, so this is
still not a ship or promotion result.

**Verdict:** keep the generalized slot-safe completion repair as part of the
opt-in choice minimum-content policy. Stop decoder forcing here; the remaining
bottleneck is component-type supervision/data coverage.

Artifacts:

- `outputs/runs/e305-choice-slot-safe-connected-honest-r1/`
- [machine-readable result](choice-slot-safe-results-iter-e305-20260717.json)
