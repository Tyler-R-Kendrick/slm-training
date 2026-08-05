# Continuous autotrain: 2026-08-05 (scheduled loop `0805a`) cycle 1 — screening, non-positive

**Loop:** `continuous-openui-scheduled-0805a`
**Campaign:** `continuous-loop-20260805-continuous-openui-schedu-e7f55102-c1`
**Integration commit:** `bdf143cd` (`origin/main` tip at cycle start; branch was already current)

**Recipe:** CPU, `train_version=wf_smoke_v2`, `eval_version=e938_role_safe_all_targets_v2`,
`suite=smoke`, `steps=20`, `--ship-gates` on (honest, not weakened).

**Verdict:** measurement complete, honest ship-gate reject on fixture scale —
not a model win, not a harness fault. Both the matched `control` and
size-matched `bounds` arm (`1,608,962` trainable params each) trained and
evaluated cleanly (exit 0).

## Results

| Arm | latency p50 (ms) | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 |
| --- | --- | --- | --- | --- | --- |
| control | 3312.91 | 1.0 | 0.0 | 0.0575 | 0.6333 |
| bounds  | 3095.98 | 1.0 | 0.0 | 0.0575 | 0.6333 |

Primary metric `smoke.structural_similarity`: control = bounds = 0.0575 →
**delta = 0.0** (null).

Ship gates fail on evidence-volume (`smoke:insufficient_n actual=3 need>=20`,
plus missing `held_out`/`adversarial`/`ood`/`rico_held` suites — expected at
fixture `n=3` smoke scale) and on quality thresholds
(`meaningful_program_rate`, `structural_similarity`, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score` all below gate). This is
the expected shape of a fixture-scale smoke run and is not evidence of a
regression.

## SDLC Phase A

**Non-positive**: primary metric delta is null (`0.0`) and both arms hit
`fixture_insufficient_n`. Per `sdlc` autotrain-iteration-delivery, no new
stack layer opens for this cycle — local commit + docs only.

## Checkpoints

Two fixture-scale checkpoints were written (`control`, `bounds`, both
`1,608,962` trainable params) — see `docs/MODEL_CARD.md` update in the same
commit for roster/eval-table entries. Fixture scale only; not a promotion
candidate (fails ship gates, `n=3`).

## Next priorities

1. **Rank 1 (confidence 0.90):** the completed non-positive `bounds` arm is
   exhausted at this size; test the distinct size-matched `component-plan`
   quality hypothesis next
   (`c20260805-continuous-openui-schedu-e7f55102-c1-component-plan`).
2. **Rank 2 (confidence 0.70):** keep the matched `control` as the
   size-matched baseline every cycle.
3. **Rank 3–5 (monitor):** rotate thrash recommendation across the lever
   bank; soft ship-gate fails on fixture `n` never stop the continuous loop;
   confirmed champions promote under cadence, thrash only screens.
