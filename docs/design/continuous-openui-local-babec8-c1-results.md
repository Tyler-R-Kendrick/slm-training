# Continuous autotrain: 2026-08-04 (session babec8) cycle 1 — exact tie, duplicate of PR #1401

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `eba6db30` (`main` tip at cycle start; branch
`claude/great-dirac-babec8`, scheduled `autotrain` loop task)

**Verdict:** `bounds` reproduces its size-matched `control` byte-for-byte on
training and ties exactly on every smoke quality metric. This is a
**byte-identical duplicate** of session `krjpdg`'s same-day cycle 1, already
committed on open PR #1401
([`continuous-openui-local-krjpdg-c1-results.md`](continuous-openui-local-krjpdg-c1-results.md)) —
same checkpoint SHAs, same params, same seed. Fixture screening only — not a
ship or promotion claim.

> **Why a duplicate exists:** `outputs/autoresearch/` is gitignored and local
> to each ephemeral session container, so cycle numbering restarts at 1 every
> session. The driver's campaign id is `sha256(loop_id)[:8]` plus the UTC
> date, which is **session-independent** — every session that runs cycle 1 of
> `continuous-openui-local` on 2026-08-04 gets the identical nominal campaign
> id `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`, and with
> a fixed fixture corpus/recipe/seed the measurement is deterministic, so
> independent sessions produce byte-identical evidence. This doc exists only
> to satisfy the driver's per-cycle `document` action receipt (required
> before cycle 2 can start) — it adds no new evidence. See "Next priorities"
> below for a proposed harness fix.

| Arm | Params | Seed | last_loss | structural_similarity | binder_reference_f1 | placeholder_fidelity | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 100001 | 22.6219 | .0575 | .63333 | .52778 | 3461.75 |
| bounds | 1,608,962 | 100001 | 22.6219 | .0575 | .63333 | .52778 | 3123.35 |

Both arms parse all 3 smoke documents, use the same 1,608,962 trainable
parameters, and produce identical training loss and identical quality
metrics across the board: `meaningful_program_rate`, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, and `reward_score` are all **0 on both
arms**. `bounds` decodes 9.8% faster this run (3123.35ms vs 3461.75ms p50),
but as a pure latency delta with no accompanying quality/held/mpr signal, per
the SDLC quality-aware tradeoff rule (`_classify_metric_tradeoff`) it is
**not** a positive result.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), every quality
threshold (`meaningful_program_rate`, `structural_similarity`,
`component_type_recall`, `ast_beq_rate`, `canonical_beq_rate`,
`reward_score`), and `held_out`/`adversarial`/`ood`/`rico_held` were not run.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n` + null primary-metric delta).
Per `sdlc` autotrain-iteration-delivery, no stacked PR layer is opened for
this cycle — local commit and docs only.

## Next priorities (ranked by the driver, plus one harness observation)

1. Test the distinct size-matched `component-plan` quality hypothesis next
   instead of re-running the now-exhausted `bounds` arm (confidence 0.90).
2. Keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).
3. Rotate thrash recommendation across the lever bank rather than
   bounds-only (confidence 0.65).
4. Soft ship-gate fails on fixture `n` never stop the continuous loop
   (confidence 0.80).
5. **New, this session:** scope `continuous-loop` campaign ids by
   session/branch in addition to `loop_id` + date, so that independent
   same-day sessions stop re-measuring and re-documenting byte-identical
   cycle-1 (and, per PR #1401's cycle 2, potentially cycle-2) evidence. At
   least 3 independent sessions today alone (`krjpdg`, this session
   `babec8`) have now produced an identical `8c0b60dd-c1` doc. Flagged as an
   observation, not a repair attempted in this cycle (no reproduced
   `HarnessSignalV1` beyond duplicate documentation overhead — not a
   correctness defect).

Machine evidence:
[`continuous-openui-local-babec8-c1-results.json`](continuous-openui-local-babec8-c1-results.json).
