# Continuous autotrain: 2026-08-03 (session 60a9ge) cycle 2 — clean null, non-positive

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `95bbfecf` (frozen replay of cycle 1's `-bounds` vs
`-control` arm, now executed after the venv bootstrap gap documented in
[`continuous-openui-local-60a9ge-c1-results.md`](continuous-openui-local-60a9ge-c1-results.md)
was closed)

**Verdict:** `bounds` ties `control` exactly on every quality metric and is
*slower* on decode latency. Clean null result. Fixture screening only — not a
ship or promotion claim.

| Arm | Params | Seed | last_loss | structural_similarity | binder_reference_f1 | placeholder_fidelity | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,960 | 100001 | 22.6219 | .0575 | .63333 | .52778 | 4046.13 |
| bounds | 1,608,960 | 100001 | 22.6219 | .0575 | .63333 | .52778 | 4280.03 |

Both arms parse all 3 smoke documents with identical training loss, identical
trainable parameters, and identical quality metrics across the board
(`meaningful_program_rate`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, and `reward_score` are all **0 on both arms**). `bounds`
is 5.8% *slower* than `control` here (4280.03ms vs 4046.13ms p50), so unlike
some other same-day sessions there is not even a latency offset to weigh —
this cycle is a clean null on every axis.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), every quality
threshold (`meaningful_program_rate`, `structural_similarity`,
`component_type_recall`, `ast_beq_rate`, `canonical_beq_rate`,
`reward_score`), and `held_out`/`adversarial`/`ood`/`rico_held` were not run.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n` + null primary-metric delta,
`smoke.structural_similarity` control=candidate=`.0575`, improvement=`0`, no
latency offset either). Per `sdlc` autotrain-iteration-delivery, no stacked
PR layer is opened for this cycle — local commit and docs only.

## Next priorities (ranked by the driver)

1. Test the distinct size-matched `component-plan` quality hypothesis next
   instead of re-running the now twice-exhausted `bounds` arm (confidence
   0.90).
2. Keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).
3. Rotate thrash recommendation across the lever bank rather than
   bounds-only (confidence 0.65).
4. Soft ship-gate fails on fixture `n` never stop the continuous loop
   (confidence 0.80).

Machine evidence:
[`continuous-openui-local-60a9ge-c2-results.json`](continuous-openui-local-60a9ge-c2-results.json).
