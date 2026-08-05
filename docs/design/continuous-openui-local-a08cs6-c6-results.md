# Continuous autotrain: 2026-08-05 (scheduled loop `a08cs6`) cycle 6 — component-plan exact tie (new seed)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c6`
**Integration commit:** `bec42b3c`
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c5`

**Verdict:** non-positive. Ran the `component-plan` arm flagged as priority
rank 1 from [cycle 5](continuous-openui-local-a08cs6-c5-results.md), on a new
seed (100006).

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.0 | 0.0964 | 0.82222 | 3597.92 | fail (gate reject) |
| component-plan | 1.0 | 0.0 | 0.0964 | 0.82222 | 3385.20 | fail (gate reject) |

Primary metric improvement: **0.0** — exact tie, confirming no lever effect
for `component-plan` at this fixture scale across two distinct seeds (cycle
4's 100004 and this cycle's 100006).

## SDLC Phase A

**Non-positive.** No stacked PR layer; local commit only.

## Session summary (cycles 1–6, loop `continuous-openui-local`, session `a08cs6`)

| Cycle | Outcome |
| --- | --- |
| 1–2 | hard-blocked — `generate_batch_size` schema rejection |
| 3 | harness fix applied + replayed (positive, **merged as PR #1444**); model delta null |
| 4 | component-plan vs control — exact tie |
| 5 | component-edge vs control — exact tie (weak-seed baseline) |
| 6 | component-plan vs control (new seed) — exact tie |

This session's stack-worthy result is the `generate_batch_size` harness fix
(PR #1444, merged). All model-level screening arms this session (component-plan
×2 seeds, component-edge) tie their matched control exactly at fixture scale —
no lever demonstrated a quality effect yet. Loop continues; next priority is
`component-edge` on a fresh seed.
