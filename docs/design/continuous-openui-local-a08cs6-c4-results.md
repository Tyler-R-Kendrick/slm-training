# Continuous autotrain: 2026-08-05 (scheduled loop `a08cs6`) cycle 4 — component-plan exact tie

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `038934ea` (`origin/main` tip `34111e6e` + the
`generate_batch_size` schema fix and its measured-results docs)
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`

**Verdict:** non-positive. Ran the `component-plan` arm flagged as priority
rank 1 from [cycle 3](continuous-openui-local-a08cs6-c3-results.md).

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.3333 | 0.41667 | 0.95238 | 22274.34 | fail (gate reject) |
| component-plan | 1.0 | 0.3333 | 0.41667 | 0.95238 | 22792.94 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie. This does **not** reproduce the historical `peuum8`-session component-plan
win (which was itself rejected on fresh-seed confirmation in that session's
cycle 3); the champion fingerprint remains rejected/exhausted.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`). No stacked PR layer;
docs + local commit only.

## Next priorities

1. (rank 1, confidence 0.90) The component-plan arm is exhausted; test the
   distinct size-matched `component-edge` quality hypothesis next
   (`c20260805-continuous-openui-local-8c0b60dd-c4-component-edge`).
2. (rank 2, confidence 0.70) Keep the matched control as the size-matched
   baseline every cycle.

## Session summary (cycles 1–4, loop `continuous-openui-local`, session `a08cs6`)

| Cycle | Focus | Primary metric delta | Outcome |
| --- | --- | --- | --- |
| 1 | (harness-blocked; no experiment ran) | — | hard-blocked, `generate_batch_size` schema rejection |
| 2 | (harness-blocked; no experiment ran) | — | hard-blocked, same failure (reproducibility confirmed) |
| 3 | control vs both (post-fix replay) | 0.23083 → 0.23083 (Δ0.0) | non-positive, exact tie — but harness unblock itself is positive (stacked, PR #1444) |
| 4 | component-plan vs control | 0.41667 → 0.41667 (Δ0.0) | non-positive, exact tie |

No model-quality win this session; the harness fix is the session's stacked
result. All fixture (`n=3`) screening evidence stays below honest ship gates.
