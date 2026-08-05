# Continuous autotrain: 2026-08-05 (scheduled loop `a08cs6`) cycle 3 — harness unblocked, model delta null

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `421603fc` (`origin/main` tip `34111e6e` + the
`generate_batch_size` schema fix)
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`
(hard-blocked, see
[unblock record](continuous-openui-local-a08cs6-generate-batch-size-unblock.md))

**Verdict:** non-positive on its own model-level terms (exact primary-metric
tie). This cycle exists only because the harness fix in the same commit
unblocked it — see the harness-unblock doc for the positive/stack-worthy
result.

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.3333 | 0.23083 | 0.73333 | 10847.95 | fail (gate reject) |
| both | 1.0 | 0.3333 | 0.23083 | 0.73333 | 9116.46 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie. The `both` candidate arm was faster (9116.46ms vs 10847.95ms p50,
`efficiency_win` reason logged) but with no held quality delta, so the
quality-aware tradeoff classifier keeps this **non-positive**
(`mpr_per_ms` gain_fraction=0.18993 alone doesn't clear the latency-win bar).

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`, `quality_held` only).
No stacked PR layer for this cycle's model result specifically — but the
**harness fix that made this cycle runnable at all** is stacked (see
[`continuous-openui-local-a08cs6-generate-batch-size-unblock.md`](continuous-openui-local-a08cs6-generate-batch-size-unblock.md)).

## Next priorities

1. (rank 1) The completed non-positive arm is exhausted; test the distinct
   size-matched `component-plan` quality hypothesis next
   (`c20260805-continuous-openui-local-8c0b60dd-c3-component-plan`).
2. (rank 2) Keep the matched control as the size-matched baseline every
   cycle.

## Session summary (cycles 1–3, loop `continuous-openui-local`, session `a08cs6`)

| Cycle | Outcome |
| --- | --- |
| 1 | hard-blocked — `generate_batch_size` schema rejection, 0 valid hypotheses, exit 2 |
| 2 | hard-blocked — identical failure, confirming reproducibility |
| 3 | harness fix applied + identical arm replayed successfully (exit 0); model-level result is a null primary-metric tie |

No cycle in this session produced a model-quality win; all fixture (`n=3`)
screening evidence stays below honest ship gates. The stack-worthy result
this session is the harness fix that unblocked the loop, documented
separately.
