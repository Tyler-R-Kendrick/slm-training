# Continuous autotrain: 2026-08-03 (session 60a9ge) cycle 3 — component-plan regresses, non-positive

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `7f856181`

**Verdict:** the `component-plan` knob regresses vs its size-matched control
on every quality axis. Non-positive; ruled out as a candidate. Fixture
screening only — not a ship or promotion claim.

| Arm | Params | meaningful_program_rate | structural_similarity | component_type_recall | binder_reference_f1 | placeholder_fidelity | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,760 | .33333 | .23083 | .16667 | .73333 | .63889 | 11044.17 |
| component-plan | 1,755,760 | 0 | .1725 | 0 | .63333 | .52778 | 10381.34 |

`component-plan` is size-matched to `control` (1,755,760 trainable params,
both parse all 3 smoke docs) but regresses on every quality metric:
`structural_similarity` drops from `.23083` to `.1725` (Δ `-.0583`),
`meaningful_program_rate` and `component_type_recall` both drop to `0`, and
`binder_reference_f1` drops from `.73333` to `.63333` — a
`non_regression_fail`. `component-plan` decodes marginally faster
(10381.34ms vs 11044.17ms p50), but that is not treated as an offsetting win
since quality regressed across the board (the SDLC quality-aware tradeoff
rule requires a latency win to hold quality/mpr, not trade it away).

Ship gates fail as expected on fixture scale (`insufficient_n`, n=3 need 20,
plus every quality threshold); `held_out`/`adversarial`/`ood`/`rico_held`
were not run.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n` + negative primary-metric delta +
`binder_reference_f1` non-regression-gate fail). Per `sdlc`
autotrain-iteration-delivery, no stacked PR layer is opened for this cycle —
local commit and docs only. This rules out `component-plan` as a
quality-improving knob under this recipe.

## Next priorities (ranked by the driver)

1. Test the distinct size-matched `component-edge` quality hypothesis next
   (confidence 0.90) — `component-plan` is now exhausted and regressive.
2. Keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).
3. Rotate thrash recommendation across the lever bank (confidence 0.65).
4. Soft ship-gate fails on fixture `n` never stop the continuous loop
   (confidence 0.80).

Machine evidence:
[`continuous-openui-local-60a9ge-c3-results.json`](continuous-openui-local-60a9ge-c3-results.json).
