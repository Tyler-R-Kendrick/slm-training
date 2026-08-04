# Continuous autotrain: 2026-08-04 (scheduled loop `peuum8`) cycle 4 — batch-size diagnostic, exact tie

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `1dc21c5d` (`origin/main` tip `0c2c6bc7` + merge)

**Verdict:** non-positive. Ran the batch-size arm flagged as a runtime-only
diagnostic in [cycle 3](continuous-openui-local-peuum8-c3-results.md)
priority rank 3, while a new quality-targeted hypothesis is preregistered.

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- |
| control | 0.3333 | 0.41667 | 0.95238 | 24553.6 | fail (gate reject) |
| batch1 | 0.3333 | 0.41667 | 0.95238 | 25612.8 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie, as expected for a batch-size-only diagnostic arm. `batch1` was
marginally slower (+1059ms p50) with no quality delta — no lever effect at
this fixture scale.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`). No stacked PR layer;
docs + local commit only.

## Next priorities

1. (rank 1, confidence 0.90) The batch1 diagnostic arm is exhausted; test a
   distinct size-matched quality hypothesis next
   (`c20260804-continuous-openui-local-8c0b60dd-c4-component-plan`).
2. (rank 2, confidence 0.70) Keep the matched control as the size-matched
   baseline every cycle.

## Session summary (cycles 1–4, loop `continuous-openui-local`, session `peuum8`)

| Cycle | Experiment focus | Primary metric delta | Outcome |
| --- | --- | --- | --- |
| 1 | grammar-completion-bounds vs control | 0.0575 → 0.0575 (Δ0.0) | non-positive, exact tie |
| 2 | component-plan vs control | 0.32667 → 0.38280 (Δ+0.05613) | **positive**, champion queued (`candidate_queued`) |
| 3 | component-plan fresh-seed confirmation | 0.23083 → 0.23083 (Δ0.0) | confirmation rejected — win did not reproduce; `binder_reference_f1` regressed 0.48889→0.26667 |
| 4 | batch-size diagnostic vs control | 0.41667 → 0.41667 (Δ0.0) | non-positive, exact tie (expected) |

No cycle in this session produced a stacked PR: cycle 2's win did not
survive fresh-seed confirmation (cycle 3), and the champion fingerprint
`6cfba5d6fd08579f` is now rejected/exhausted. All four cycles are fixture
(`n=3`) screening evidence only — none clears the honest ship gates (which
require `n>=20` plus quality thresholds on `meaningful_program_rate`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score`). Per SDLC Phase A, this
session's work stays as local commits with docs, pushed to the working
branch as a documentation-only PR at session end (no code/harness change
this session; nothing to stack).
