# Autotrain c5 (continuous-openui-local): component-edge quality-null — not ship

**Verdict:** quality-null fixture screen, not positive, not ship. Per the
driver's ranked successor priority from c4, this cycle tests the
size-matched `component-edge` head (`c...-c5-component-edge`, 1,766,987
params) against a fresh matched control at the same size.

Both arms tie exactly on every smoke metric: `structural_similarity=0.0431`,
`meaningful_program_rate=0`, `component_type_recall=0`,
`binder_reference_f1=0.8222`. `--ship-gates` rejects for the expected
reasons at this scale (`insufficient_n actual=3 need>=20`, plus quality
thresholds below bar).

SDLC Phase A: **not positive** — `primary_metric_null_or_worse` (control ==
candidate, `improvement=0.0`) and `fixture_insufficient_n_alone`. No stack
layer opened; local commit + docs only.

Checkpoints (`29be9c13...640d2` control / `23d794ca...b13890` component-edge)
stay local, explicit no-sync, not reusable/promotable/ship evidence.

Next (per the driver's ranked successor priorities): the size-matched
`component-inventory` quality hypothesis
(`c20260803-continuous-openui-local-8c0b60dd-c5-component-inventory`).

Machine evidence:
[`autotrain-cycle-c5-component-edge-quality-null.json`](autotrain-cycle-c5-component-edge-quality-null.json).
