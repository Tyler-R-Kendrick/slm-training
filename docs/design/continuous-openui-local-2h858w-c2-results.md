# Continuous autotrain: 2026-08-04 (session 2h858w) cycle 2 — decode timeout, likely environmental (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `d38b45a9` (this session's cycle-1 docs commit, on top
of `main` tip `5ba8e430`)

**Verdict:** Both arms of the size-matched `component-plan` hypothesis
(1,755,764 params, seed 100002 — the same recipe/seed that produced a
reproduced `+0.05613` `structural_similarity` win in three independent prior
sessions, most recently [PR #1376](https://github.com/Tyler-R-Kendrick/slm-training/pull/1376))
hit `decode_timeout_count=3/3` on both arms and produced no scoreboard.
Measurement incomplete, not a model result.

## Diagnosis: environmental slowness, not a code defect

`thrash_timing.json` records `arm_wall_seconds=70.0`,
`fitted_decode_timeout_seconds=8.0` (unclamped — the fitting formula did not
further reduce the policy-configured 8s/record budget), `min_train_floor_seconds=20.0`,
`eval_overhead_seconds=8.0`.

| Signal | This cycle (container) | Prior successful session, same recipe/seed |
| --- | --- | --- |
| `compiler_ms_mean` | ~23,150–23,249 ms | — |
| end-to-end `latency_ms_p50` | timed out (no scoreboard) | control 14,554.69 ms, component-plan 12,549.58 ms |
| decode timeout budget | 8,000 ms/record | 8,000 ms/record |

The measured `compiler_ms_mean` alone (~23.1–23.2s) is 6–7x the per-record
decode budget and 1.6–1.9x the **entire** prior session's end-to-end p50
latency for the identical recipe
([`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)).
Since the identical frozen recipe already completed within budget on other
same-loop sessions, this reads as environmental slowness specific to this
container (likely residual CPU contention right after this session's
`.venv`/torch/npm bootstrap) rather than a code-level harness defect.

`scripts/run_autotrain_continuous.py`'s own comment on
`_fit_screening_decode_timeout_seconds` explicitly warns: "if this clamp
always binds, either the arm share model or the thrash recipe (steps/n)
needs recalibration — not silent wall++." Given the identical recipe has a
demonstrated in-budget history, bumping the timeout here would be exactly
that discouraged speculative fix, so this session does not touch the
fitting formula or the policy-configured decode timeout.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure`,
`primary_metric_unavailable`). Per `sdlc` autotrain-iteration-delivery, no
stacked PR layer is opened for this cycle — local commit and docs only.

## Handoff action disposition

- `repair_harness` (owner `improve-openui-harnesses`, family `model_build`):
  acknowledged **blocked** with this diagnostic evidence, not claimed fixed —
  no canonical-owner code change made this cycle; root cause reads as
  container-local resource contention, not a reproducible harness defect.
- `document`: this doc + JSON.

## Next priorities

1. Frozen-replay the identical `c2-component-plan` / `c2-control` arms
   (`retry_measurement`) once container load has settled, before trying a
   new hypothesis.
2. If decode timeouts persist across ≥3 consecutive frozen replays with no
   new information, escalate to a dedicated `improve-openui-harnesses`
   session to recalibrate the screening arm-wall/decode-timeout share model
   for size-matched arms above ~1.75M params (the fitting formula assumes a
   flat per-role wall regardless of arm size).
3. Do not raise `decode_timeout_seconds` or the arm wall speculatively
   without that dedicated investigation.

Machine evidence:
[`continuous-openui-local-2h858w-c2-results.json`](continuous-openui-local-2h858w-c2-results.json).
