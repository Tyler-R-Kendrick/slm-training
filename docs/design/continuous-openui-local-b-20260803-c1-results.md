# Autotrain continuous-openui-local-b c1: bounds screen, null delta

**Verdict:** non-positive. Independent fresh loop lineage started this session
(`continuous-openui-local-b`) after `continuous-openui-local`'s c2 hit a
symmetric decode-timeout blocker requiring a dedicated governed-threshold
follow-up (see
[`continuous-openui-local-20260803-c2-decode-timeout-budget-diagnosed.md`](continuous-openui-local-20260803-c2-decode-timeout-budget-diagnosed.md)).
This lineage screens the same `grammar_completion_bounds` hypothesis at the
same recipe (`wf_smoke_v2`, 20 steps, smoke `n=3`, seed `100001`) to keep
making training progress while that blocker's meta-campaign is pending.

| Arm | Params match | Smoke complete | Structural sim | MPR | Binder F1 | Parse rate | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | matched | 3/3 | 0.0575 | 0.0 | 0.6333 | 1.0 | 1200.72 |
| bounds | matched | 3/3 | 0.0575 | 0.0 | 0.6333 | 1.0 | 1271.92 |

Same result shape as `continuous-openui-local` c1: primary metric identical
between arms (improvement = 0.0) — a null delta, not a latency win. Both arms
fail `insufficient_n` (fixture `n=3`) and the honest ship gates, as expected
at fixture scale. No checkpoint promoted/synced; no `MODEL_CARD.md`/README
change. No canonical harness file governed by
`src/slm_training/resources/versions.json` changed this cycle
(`code_dirty=false`, `code_commit=905cf794`).

Per SDLC Phase A, non-positive keeps no new stack layer — local docs commit
only.

**Next priority (driver-ranked, rank 1):** run the distinct size-matched
`component-plan` quality hypothesis next
(`c20260803-continuous-openui-local--a8257b0d-c1-component-plan`).

Machine evidence:
[`continuous-openui-local-b-20260803-c1-results.json`](continuous-openui-local-b-20260803-c1-results.json).
