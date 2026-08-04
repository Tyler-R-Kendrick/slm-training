# Continuous autotrain: 2026-08-04 (scheduled loop `peuum8`) cycle 1 — non-positive fixture screen

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `98d1fb2c` (`origin/main` tip at cycle start)

**Verdict:** non-positive, fixture-honesty screening cycle. Not a ship claim.

## Recipe

- Device: CPU, `train_version=wf_smoke_v2`, `eval_version=e938_role_safe_all_targets_v2`
- Suite: `smoke` (`n=3`, well under the `n>=20` ship-gate evidence-volume floor —
  expected on a 20-step fixture screen, not a lever regression)
- Arms: matched `control` (grammar levers off) vs size-matched `bounds`
  (grammar-completion-bounds candidate), both `1,608,962` trainable params

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.0 | 0.0575 | 0.6333 | 4391.05 | fail (gate reject) |
| bounds | 1.0 | 0.0 | 0.0575 | 0.6333 | 3536.21 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie between control and the `bounds` candidate. `meaningful_program_rate`,
`component_type_recall`, `ast_beq_rate`, `canonical_beq_rate`, and
`reward_score` were all `0.0` on both arms; `held_out` / `adversarial` / `ood`
/ `rico_held` suites are not published for this fixture recipe
(`missing_suite`). None of this indicates a lever regression — the fixture
`n=3` smoke suite cannot clear the `n>=20` evidence-volume gate by design at
20 steps.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`, `primary_metric_null_or_worse`
with `improvement=0.0`). Per `sdlc` autotrain-iteration-delivery, no stacked
PR layer opens for this cycle — docs + local commit only.

## Next priorities

1. (rank 1, confidence 0.90) The completed `bounds` arm is now exhausted;
   test the distinct size-matched `component-plan` quality hypothesis next
   rather than re-running `bounds` (`c20260804-continuous-openui-local-8c0b60dd-c1-component-plan`).
2. (rank 2, confidence 0.70) Keep the matched `control` as the size-matched
   baseline every cycle.
