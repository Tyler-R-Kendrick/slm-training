# Continuous autotrain: 2026-08-05 (scheduled loop `0805a`) cycle 2 — component-plan fixture win, candidate queued

**Loop:** `continuous-openui-scheduled-0805a`
**Campaign:** `continuous-loop-20260805-continuous-openui-schedu-e7f55102-c2`
**Integration commit:** `96fa4dd7` (previous cycle's docs commit, merged clean onto `origin/main` tip `bdf143cd`)

**Recipe:** CPU, `train_version=wf_smoke_v2`, `eval_version=e938_role_safe_all_targets_v2`,
`suite=smoke`, `steps=20`, `--ship-gates` on (honest, not weakened). Hypothesis:
rank-1 priority from cycle 1 — the size-matched `component-plan` quality lever.

**Verdict:** fixture-scale **primary-metric win**, both arms measurement-complete
(exit 0). Candidate and control are size-matched (`1,755,764` trainable params
each).

## Results

| Arm | latency p50 (ms) | parse_rate | structural_similarity | component_type_recall |
| --- | --- | --- | --- | --- |
| control | 30920.89 | 1.0 | 0.3267 | 0.0 |
| component-plan | 24683.42 | 1.0 | 0.3828 | 0.1667 |

Primary metric `smoke.structural_similarity`: control `0.3267` → candidate
`0.3828`, **improvement = +0.0561** (+17.2% relative), with candidate latency
*lower* than control (no latency tradeoff spent). `component_type_recall`
also moves `0.0 → 0.1667`.

Ship gates still honestly reject on fixture evidence volume
(`smoke:insufficient_n actual=3 need>=20`, missing `held_out`/`adversarial`/
`ood`/`rico_held` suites) and on remaining quality thresholds
(`structural_similarity` still below the `0.35` gate on control and only
narrowly above on the candidate at `n=3`; `meaningful_program_rate`,
`ast_beq_rate`, `canonical_beq_rate`, `placeholder_fidelity`, `reward_score`
all `0`). This is expected at smoke scale and is not a ship claim.

## SDLC Phase A

**Positive** (`primary_metric_win`, `smoke.structural_similarity` improves in
the declared beneficial direction under a size-matched control). The driver
classifies `has_tracked_delta=false` / `stack_action=positive_no_tracked_delta_skip_stack`
— this cycle changed no harness/knob code, only ran an existing lever
combination, so per `sdlc` autotrain-iteration-delivery there is no
independent code delta to stack; the win is recorded as documentation only.
The candidate is enqueued (`champ-continuous-openui-scheduled-0805a-2-6cfba5d6fd08579f`)
for fresh-seed confirmation before any promotion/climb claim.

## Checkpoints

Two fixture-scale checkpoints (`control`, `component-plan`, both
`1,755,764` trainable params) — see `docs/MODEL_CARD.md` update in the same
commit. Fixture scale only; **not** a promotion candidate yet (fails ship
gates, `n=3`, still awaiting fresh-seed confirmation).

## Next priorities

1. **Rank 1 (confidence 0.95):** confirm the fixture candidate on a fresh
   seed with the exact size-matched treatment/control recipes before any
   promotion claim
   (`c20260805-continuous-openui-schedu-e7f55102-c2-component-plan-fresh-confirmation`).
2. **Rank 2 (confidence 1.0, monitor):** keep promotion formal preflight
   locked until fresh confirmation establishes a champion.
