# Autotrain continuous-openui-local c1: bounds screen, null delta

**Verdict:** non-positive. Fresh local continuous-loop worktree (torch/AgentV
bootstrapped this session; no prior `outputs/` state carried over). Screened
`grammar_completion_bounds` against the matched fixture control at `wf_smoke_v2`
/ 20 steps / smoke `n=3` only.

| Arm | Params match | Smoke complete | Structural sim | MPR | Binder F1 | Parse rate | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | matched | 3/3 | 0.0575 | 0.0 | 0.6333 | 1.0 | 1406.14 |
| bounds | matched | 3/3 | 0.0575 | 0.0 | 0.6333 | 1.0 | 1218.84 |

Primary metric `smoke.structural_similarity` is identical between arms
(improvement = 0.0), so this is a null delta, not a latency win — no held/mpr
tradeoff to spend under `_classify_metric_tradeoff`. Both arms also fail
`insufficient_n` (fixture `n=3` vs. gate floor `20`) and the honest ship gates
(`meaningful_program_rate`, `structural_similarity`, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score` all below threshold).
Expected at fixture scale; not evidence of a broken harness.

No checkpoint was promoted or synced (both checkpoints are local scratch run
artifacts only), so `docs/MODEL_CARD.md` / README are unchanged this cycle.
No canonical harness file governed by `src/slm_training/resources/versions.json`
changed in this cycle, so no version-stamp component bump applies
(`code_dirty=false`, `code_commit=62540f38`).

Per SDLC Phase A (`autotrain-iteration-delivery`), a non-positive cycle keeps
no new stack layer — local docs commit only.

**Next priority (driver-ranked, rank 1):** the `bounds` screen is exhausted;
run the distinct size-matched `component-plan` quality hypothesis next
(`c20260803-continuous-openui-local-8c0b60dd-c1-component-plan`), keeping the
same matched control every cycle (rank 2).

Machine evidence:
[`continuous-openui-local-20260803-c1-results.json`](continuous-openui-local-20260803-c1-results.json).
