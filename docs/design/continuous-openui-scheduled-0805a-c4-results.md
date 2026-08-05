# Continuous autotrain: 2026-08-05 (scheduled loop `0805a`) cycle 4 — batch-size runtime diagnostic, non-positive

**Loop:** `continuous-openui-scheduled-0805a`
**Campaign:** `continuous-loop-20260805-continuous-openui-schedu-e7f55102-c4`
**Integration commit:** `e3079bd4` (previous cycle's docs commit, merged clean onto `origin/main` tip `bdf143cd`)

**Recipe:** CPU, `train_version=wf_smoke_v2`, `eval_version=e938_role_safe_all_targets_v2`,
`suite=smoke`, `steps=20`, `--ship-gates` on. Rank-3 priority from cycle 3:
`batch1` runtime diagnostic (deliberately not a quality hypothesis) while a
new quality-targeted objective is preregistered.

**Verdict:** non-positive, as expected for a runtime-only diagnostic. Both
arms (`1,608,962` matched params) train/evaluate cleanly.

## Results

| Arm | latency p50 (ms) | structural_similarity | binder_reference_f1 |
| --- | --- | --- | --- |
| control | 22109.8 | 0.4167 | 0.9524 |
| batch1  | 22688.8 | 0.4167 | 0.9524 |

Primary metric tied exactly (`0.4167` both) — expected, since `batch1` only
varies a throughput knob, not a quality lever. Ship gates still honestly
reject on fixture evidence volume and `meaningful_program_rate`/
`component_type_recall`/`ast_beq_rate`/`canonical_beq_rate` thresholds.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `fixture_insufficient_n`).
No stack layer.

## Note on the re-proposed successor

The driver's rank-1 successor for cycle 5 is again the `component-plan`
family. Cycles 2–3 of this loop already showed that lever is seed-sensitive
and does not survive fresh-seed confirmation (agrees with session `peuum8`,
commit `6d97009`). Re-running it without a new hypothesis variant would just
repeat a closed approach — deferred to a future cycle that preregisters a
distinct quality-targeted objective instead of recycling the exhausted
fingerprint.
