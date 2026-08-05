# Continuous autotrain: 2026-08-05 (scheduled loop `z0fvm2`) cycle 1 — bounds arm exact tie

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `bdf143cd` (`origin/main` tip, includes the
`generate_batch_size` schema fix from #1444)
**Predecessor:** none — fresh campaign lineage in a new container; continues
the shared `continuous-openui-local` loop already advanced by other sessions
on `origin/main`.

**Verdict:** non-positive. Exact primary-metric tie between `control` and the
`bounds` candidate arm.

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.0 | 0.0575 | 0.6333 | 4108.71 | fail (gate reject) |
| bounds | 1.0 | 0.0 | 0.0575 | 0.6333 | 3108.98 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie. The `bounds` candidate was faster (3108.98ms vs 4108.71ms p50) but with
no held quality delta (`meaningful_program_rate=0.0` on both arms), so the
quality-aware tradeoff classifier keeps this **non-positive**.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`). No stacked PR layer
for this cycle — local commit + docs only, per `sdlc` autotrain-iteration-delivery.

## Next priorities

1. (rank 1) The completed non-positive arm is exhausted; test the distinct
   size-matched `component-plan` quality hypothesis next
   (`c20260805-continuous-openui-local-8c0b60dd-c1-component-plan`).
2. (rank 2) Keep the matched control as the size-matched baseline every cycle.

## Honesty

Fixture (`n=3`) screening evidence only — below honest ship gates
(`smoke:insufficient_n need>=20`; `held_out`/`adversarial`/`ood`/`rico_held`
all `missing_suite`). Not a ship claim.
