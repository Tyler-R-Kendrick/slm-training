# Continuous autotrain: 2026-08-04 (scheduled loop `peuum8`) cycle 2 — component-plan structural-similarity win, awaiting confirmation

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `92ceb3ea` (`origin/main` tip `b04d6636` + merge)

**Verdict:** positive fixture-honesty screening result. **Not a ship claim,
not a promotion.** SDLC Phase A: `positive=true`,
`stack_action=positive_no_tracked_delta_skip_stack` — the driver's own
classification is "metric win recorded; no code/docs delta — skip stack PR;
continue loop." Nothing here changed tracked harness/model code, so the win
is queued as a champion candidate pending fresh-seed confirmation rather than
stacked.

## Recipe

- Device: CPU, `train_version=wf_smoke_v2`, `eval_version=e938_role_safe_all_targets_v2`
- Suite: `smoke` (`n=3`, seed `100002`)
- Arms: matched `control` vs size-matched `component-plan` candidate
  (component-plan loss/decode weights on), both `1,755,764` trainable params

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.0 | 0.32667 | 0.0 | 34102.31 | fail (gate reject) |
| component-plan | 1.0 | 0.0 | 0.38280 | 0.0 | 29684.18 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **+0.05613**
(0.32667 → 0.38280), a quality win, not a bare latency blip — latency also
improved (34.1s → 29.7s p50) but that is a secondary observation, not the
qualifying metric. Both arms still fail the honest ship gates (fixture
`n=3` misses the `n>=20` evidence-volume floor by design; `meaningful_program_rate`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score` all `0.0`).

**Caveat:** `binder_reference_f1` is `0.0` on both arms this cycle vs `0.6333`
in cycle 1 — a knob/recipe difference between cycles 1 and 2, not something
attributable to the component-plan treatment in isolation (the comparison is
control-vs-candidate within cycle 2, which is valid; cross-cycle absolute
levels are not).

## Champion queue

`CHAMPION_ENQUEUE entry_id=champ-continuous-openui-local-2-6cfba5d6fd08579f
fingerprint=6cfba5d6fd08579f candidate=c20260804-continuous-openui-local-8c0b60dd-c2-component-plan`
— `climb_state=candidate_queued`, `ship_state=blocked`. Formal/Lean promotion
preflight stays locked until fresh-seed confirmation (priority rank 2,
confidence 1.0).

## SDLC Phase A

**Positive**, but `stack_layer=false` / `has_tracked_delta=false` — no PR
opened this cycle. Docs + local commit only, per the driver's explicit
`agent_required` guidance.

## Next priorities

1. (rank 1, confidence 0.95) Confirm the `component-plan` candidate on a
   fresh seed with the exact size-matched treatment/control recipes before
   any promotion (`c20260804-continuous-openui-local-8c0b60dd-c2-component-plan-fresh-confirmation`).
2. (rank 2, confidence 1.00, `lean_assumption`) Keep the formal promotion
   preflight locked until fresh confirmation establishes a champion.
