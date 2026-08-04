# Continuous autotrain: 2026-08-04 (session krjpdg) cycle 2 — decode timeout, measurement incomplete

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `cc9671d7` (branch `claude/great-dirac-krjpdg`)

**Verdict:** Both the size-matched `control` and the `component-plan`
quality-hypothesis candidate (the rank-1 priority carried over from cycle 1)
hit an internal AgentV decode timeout on all 3 smoke records
(`compiler_ms_mean` ~23.5-23.9s/record against the wall-adaptive per-record
budget). SLM-303 decode-outcome classification correctly reports these as
incomplete documents rather than false parse/quality-0s, so **no quality
metric is available for either arm** — the comparison is inconclusive, not a
model result. Infrastructure/measurement signal, not a model regression.

| Arm | Params | Seed | Exit | decode_timeout_count | incomplete_document_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,764 | 100002 | 2 | 3 | 3 |
| component-plan | 1,755,764 | 100002 | 2 | 3 | 3 |

## Diagnosis

The driver's typed handoff routes this to a `repair_harness` action against
the `model_build` harness family (frozen manifest
`d5cd43c70fc4faa379ec8b07d65daca66e4a5340e058b38bcccf2a990f3003e3`), with a
paired `retry_measurement` action to replay the identical frozen
`control`/`component-plan` arms once repaired. Given the wall-adaptive
per-record decode-timeout budget (`_effective_record_decode_timeout` in
`src/slm_training/harnesses/model_build/eval_runner.py`) and a ~24s/record
compile cost on this container, this reads as container-speed/eval-budget
contention on a single occurrence rather than a confirmed harness code
defect — the same fixture, `component-plan` knob, and seed have produced
completed measurements (including positive structural-similarity wins) in
multiple prior independent sessions
([`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md),
PR #1387, PR #1369). Per the continuous-loop self-heal law this is the
**first** occurrence of this exact blocker this session — not the
3-consecutive-failure hard-block threshold — so the loop continues by
re-invoking the driver, which consumes the queued `retry_measurement` action
before considering any new model hypothesis.

## SDLC Phase A

**Non-positive** (`measurement_incomplete` + `primary_metric_unavailable`).
Per `sdlc` autotrain-iteration-delivery, no stacked PR layer is opened for
this cycle — local commit and docs only.

## Next priorities (ranked by the driver)

1. Replay the exact frozen `control` and `component-plan` candidate (seed
   100002, 1,755,764 params) before testing a new hypothesis
   (confidence 0.95).

Machine evidence:
[`continuous-openui-local-krjpdg-c2-results.json`](continuous-openui-local-krjpdg-c2-results.json).
