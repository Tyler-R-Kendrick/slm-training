# Autotrain c4 (continuous-openui-local): component-plan quality-null — not ship

**Verdict:** quality-null fixture screen, not positive for SDLC Phase A, not
ship. Per the driver's ranked successor priority from c3, this cycle tests
the size-matched `component-plan` head (`c...-c4-component-plan`, 1,755,764
params, prebuilt component-plan head) against a fresh matched control at the
same size.

Both arms tie exactly on every smoke metric: `structural_similarity=0.4167`,
`meaningful_program_rate=0.3333`, `component_type_recall=0.25`,
`ast_beq_rate=0`, `canonical_beq_rate=0`. `--ship-gates` rejects for the
expected reasons at this scale (`insufficient_n actual=3 need>=20`, plus the
quality thresholds below bar).

SDLC Phase A: **not positive** — `primary_metric_null_or_worse` (control ==
candidate, `improvement=0.0`) and `fixture_insufficient_n_alone`. No stack
layer opened; local commit + docs only.

Checkpoints (`0f1cb67a...4dc9` control / `4c5f57ea...940a` component-plan)
stay local, explicit no-sync, not reusable/promotable/ship evidence.

Next (per the driver's ranked successor priorities): the size-matched
`component-edge` quality hypothesis
(`c20260803-continuous-openui-local-8c0b60dd-c4-component-edge`).

Machine evidence:
[`autotrain-cycle-c4-component-plan-quality-null.json`](autotrain-cycle-c4-component-plan-quality-null.json).
