# E495 semantic optional-slot RICO slice — 2026-07-18

E495 evaluates the E491–E494 generalized prompt-role/root-stop and
schema-optional semantic-slot fix on `rico_held` rows `[144,168)`. It uses the
unchanged E396 checkpoint and E451 corpus on CPU with local HF context,
320-token grammar LTR, component-plan weight 2, slot-component weight 8,
honest prompt-role/slot constraints, eight generation steps, three attempts,
and no unconstrained fallback.

The 24-row process completed normally in about 124 seconds under the hard
three-minute limit.

| Metric | E487 matched slice | E495 | Δ |
| --- | ---: | ---: | ---: |
| Structural similarity | 0.84037 | 0.89108 | +0.05072 |
| Parse rate | 1.0 | 1.0 | 0.0 |
| Meaningful rate | 1.0 | 1.0 | 0.0 |
| Placeholder fidelity | 1.0 | 1.0 | 0.0 |
| Component type recall | 1.0 | 1.0 | 0.0 |

Three predictions change and all improve; none regress. The largest changes
are `rico_hf_test_293` (0.35 → 1.0) and `rico_hf_test_319`
(0.495 → 1.0). The former now emits exactly four two-slot `ImageBlock`
components from the visible “4 images” contract.

Reliability remains clean: zero failures, fallbacks, or decode timeouts.
AgentV artifacts were published with zero execution errors. Its 0/5 summary is
expected for a diagnostic single-suite run because the other four policy
suites are intentionally absent.

**Verdict:** accept this diagnostic slice. It is not a full-RICO or production
ship claim; a sharded full evaluation remains required.
