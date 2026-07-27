# AP-027 (SLM-322): discrete-plan Pareto screening

`claim_class: wiring`. **Not a ship claim.** Screening-scale only (1 seed,
1 prompts, scratch CPU checkpoint `outputs/runs/slm322_ap027_scratch_v1/checkpoints/last.pt`) -- far
below the issue's own >=3/5-seed acceptance bar; that full run is out of
scope for this PR (see "What this PR does not claim" below). Refinement
rounds measured this invocation: [1]; rounds
[2, 4, 8] remain pending under the repo's
155s harness wall-clock cap (AP-022's own
partial-shard precedent: a bounded run reports exactly what it measured).

## AP-022 gating disposition (I14)

AP-022/SLM-313 already produced a complete locked-matrix verdict for
`learned_abstract_plan`: **ignored_or_collapsed**
(docs/design/abstract-plan-functional-evidence.md). Per decode-invariant I14, that
verdict is never re-litigated here. `learned_plan_binder_precision` --
named in this issue's own arm list, and never run by AP-022 -- is the
successor approach and the only arm this harness's `build_campaign()` will
ever mark `role="candidate"`.

## What ran

The committed `playground_demo` checkpoint carries no trained AP-024
`AbstractPlanConnector`, so `no_plan` and `current_auxiliary_heads` decode
through an identical configuration on this checkpoint today -- both labels
below report the same measured series rather than a fabricated split.
Measured across refinement rounds [1, 2, 4, 8]:

| refinement_round | total_latency_ms | meaning_v2 (parse_rate proxy) | non-dominated |
| --- | --- | --- | --- |
| 1 | 118829.03 | 0.0000 | yes |

Non-dominated (arm@round) points on (lower latency, higher meaning_v2):
no_plan@1, current_auxiliary_heads@1.

## What this PR does not claim

- No `>=3`/`5`-seed run (this is 1 seed, screening scale).
- No measurement of `oracle_plan`, `learned_plan_binder_precision`, or
  `retrieval_baseline` -- each needs an artifact (trained connector,
  binder-precision channel, or retrieval index) this PR does not build.
  Recorded as `arms_pending`: empty, gold_semantic_plan, learned_plan_binder_precision, length_matched_verbal, prefix_truncation, random_norm_matched, retrieval_baseline, shuffle_between_examples, within_plan_permutation.
- No Pareto-frontier promotion. `ship_eligible: false`.

## Evidence

- `docs/design/iter-slm322-discrete-plan-pareto-20260726.json`
- Harness: `src/slm_training/harnesses/experiments/discrete_plan_pareto.py`
- Runner: `scripts/run_discrete_plan_pareto.py`
