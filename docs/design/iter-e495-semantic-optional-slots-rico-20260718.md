# E495 semantic optional-slot RICO slice — 2026-07-18

E495 evaluates the generalized E491–E494 decoder fix on `rico_held` rows
`[144,168)`. The unchanged E396 checkpoint runs on CPU with local HF context
and completes normally in about 124 seconds under the hard three-minute cap.

| Metric | E487 matched slice | E495 | Δ |
| --- | ---: | ---: | ---: |
| Structural similarity | 0.84037 | 0.89108 | +0.05072 |
| Parse rate | 1.0 | 1.0 | 0.0 |
| Meaningful rate | 1.0 | 1.0 | 0.0 |
| Placeholder fidelity | 1.0 | 1.0 | 0.0 |
| Component type recall | 1.0 | 1.0 | 0.0 |

Three predictions improve and none regress. `rico_hf_test_293` rises from 0.35
to 1.0 and now emits exactly four two-slot `ImageBlock` components.
Reliability remains clean: zero failures, fallbacks, timeouts, or AgentV
execution errors. The 0/5 AgentV envelope is expected for a single-suite
diagnostic because four policy suites are intentionally absent.

**Verdict:** accept the diagnostic slice. Full RICO evidence remains required
before changing the five-suite ship claim.
