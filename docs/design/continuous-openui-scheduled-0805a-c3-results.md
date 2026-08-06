# Continuous autotrain: 2026-08-05 (scheduled loop `0805a`) cycle 3 — component-plan champion rejected on fresh-seed confirmation

**Loop:** `continuous-openui-scheduled-0805a`
**Campaign:** `continuous-loop-20260805-continuous-openui-schedu-e7f55102-c3`
**Integration commit:** `0bb9ed32` (previous cycle's docs commit, merged clean onto `origin/main` tip `bdf143cd`)

**Recipe:** CPU, `train_version=wf_smoke_v2`, `eval_version=e938_role_safe_all_targets_v2`,
`suite=smoke`, `steps=20`, `--ship-gates` on (honest, not weakened). Rank-1
priority from cycle 2: fresh-seed confirmation of the queued `component-plan`
champion (`champ-continuous-openui-scheduled-0805a-2-6cfba5d6fd08579f`).

**Verdict:** **confirmation rejected.** On a fresh seed, the primary metric
delta collapses to null and a real quality regression appears that the
screening cycle didn't show.

## Results

| Arm | latency p50 (ms) | meaningful_program_rate | structural_similarity | binder_reference_f1 |
| --- | --- | --- | --- | --- |
| control | 7512.17 | 0.3333 | 0.2308 | 0.4889 |
| confirm (component-plan) | 6778.62 | 0.3333 | 0.2308 | **0.2667** |

Primary metric `smoke.structural_similarity`: control = confirm = `0.2308` →
**delta = 0.0** (the +0.0561 win from cycle 2 did not reproduce on a fresh
seed). `meaningful_program_rate` also ties. `binder_reference_f1` **regresses**
`0.4889 → 0.2667` — a real quality loss the screening cycle's metric slice
didn't surface. The only lever that held was efficiency
(`mpr_per_ms` +10.8%, above the 5% minimum) and training loss ("quality_held"
on parse/MPR), which is exactly the failure mode this repo's diagnostics warn
about: token loss diverging from certified program quality.

Ship gates still honestly reject on fixture evidence volume (`n=3`) and
remaining quality thresholds.

## SDLC Phase A

**Non-positive**: `confirmation_rejected:primary_quality_not_reheld` — the
champion fingerprint is disposed `rejected`, not `climb_accepted`. Per `sdlc`
autotrain-iteration-delivery, no stack layer opens; docs + champion-status
record only.

## Interpretation

This is the same qualitative finding as the prior `continuous-openui-local`
session (`peuum8`, commit `6d97009`): a `component-plan`-family fixture win
that improves structural_similarity while trading away `binder_reference_f1`
does not survive fresh-seed confirmation. Two independent sessions now agree
the `component-plan` lever, as currently implemented, is **seed-sensitive and
not a reproducible quality gain** — it should be treated as exhausted for
this size class rather than re-queued without a new hypothesis.

## Next priorities

1. **Rank 1 (confidence 0.95, monitor):** exhaust the `component-plan`
   champion fingerprint; test a distinct size-matched quality-targeted
   objective instead of spending more scalar steps on it.
2. **Rank 2 (confidence 0.90, monitor):** keep training loss as a diagnostic
   only, never a promotion proxy — this cycle is direct evidence of why.
3. **Rank 3 (confidence 0.65, experiment_next):** `c20260805-continuous-openui-schedu-e7f55102-c3-batch1`
   as a runtime diagnostic only (not a quality hypothesis) while a new
   quality-targeted objective is preregistered.
