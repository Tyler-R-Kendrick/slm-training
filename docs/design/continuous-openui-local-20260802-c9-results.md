# Continuous autotrain cycle 9 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c9` |
| Cycle intent | `screening` (role) — driver's hypothesizer selected `binder-topology`, its own rank-1 recommendation carried over from cycle 8 |
| Upstream / integration | `b8188a49` / `81275113` |
| Device | CPU |
| Steps | **40** (escalated from the prior 5 cycles' 20-22 — see scale investigation below) / seed 100009 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` only (screening role; `held_out`/`adversarial`/`ood`/`rico_held` are `missing_suite` at this role), `--ship-gates` |
| Wall cap | 3 minutes |
| Hypothesis | Low-weight `binder_topology_loss_weight=0.25` (`structural_aux_head_profile=binder-topology`) improves `smoke.structural_similarity` without lowering `parse_rate` or `binder_reference_f1` |
| Primary metric | `smoke.structural_similarity` (direction: increase) |

## Diagnostic scale-investigation step (this cycle's mandate)

Before running the screen, this cycle spent bounded effort checking whether a
larger, already-published fixture/eval combination was available:

1. **Eval suite size.** Every published eval snapshot under
   `src/slm_training/resources/data/eval/` was checked
   (`e763_symbol_only_eval_r2_20260722`, `e827_target_slots_only_v4`,
   `e842_harness_owned_slots_v1`, `e938_role_safe_all_targets_v2`). All four
   share identical `suite_counts` for `smoke` (3), `held_out` (5),
   `adversarial` (4), and `ood` (4); only `rico_held` differs (7-34), and
   `e938` already has the largest `rico_held` (34) and is the snapshot in use.
   `openui_hard_valid_v1` is a different eval *kind* (`semantic_contrast`
   mutation-pair scoreboard, not a smoke/held_out AgentV suite) and is not a
   drop-in substitute without harness work — out of scope for a bounded
   diagnostic step and not attempted. **Conclusion: no larger published
   smoke/held_out snapshot exists; `run_autotrain_continuous` exposes no flag
   to grow suite `n` (only `--train-version`/`--steps`/`--objective`/
   `--primary-metric`).**
2. **Training steps.** `run_autotrain_continuous.py` enforces a *symmetric
   per-arm* wall budget inside the 3-minute campaign cap
   (`_arm_wall_minutes`: `(MAX_HARNESS_WALL_SECONDS - 15) / 2` ≈ 70s per arm).
   Cycle 8's from-scratch control/candidate pair (`steps=22`) used only ~19s
   of that ~70s per-arm budget each (event timestamps: control
   10:15:53.215→10:16:12.042, candidate 10:16:14.298→10:16:33.046 — 27%
   utilization), so `--steps 40` (2x) was judged to have comfortable headroom
   without a separate timing dry-run. **This cycle's real run confirmed it:**
   whole campaign 10:23:58.458Z→10:24:29.188Z (~31s wall), both arms
   `stopped_on: "steps"` (not a timeout), comfortably inside both the 180s
   `MAX_RUN_MINUTES` cap and the ~70s-per-arm harness budget.

**Knobs used this cycle:** `--train-version wf_smoke_v2 --steps 40` (vs. the
prior 5 cycles' `--steps 20`, recorded as 20-22 after the driver's
`steps+(cycle%3)` jitter). Eval snapshot unchanged (`e938_role_safe_all_targets_v2`
— no larger one exists).

## Run matrix

| Arm | Levers | Params | steps | smoke n | smoke structural_similarity | latency_ms_p50 | last_loss | Gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c9-control | binder-topology off | 2,137,346 | 40 | 3 | 0.51 | 5614.17 | 6.802424 | **fail** (insufficient_n, quality thresholds) |
| c9-binder-topology | binder-topology **on** (weight 0.25) | 2,137,346 | 40 | 3 | 0.51 | 5999.52 | 6.833416 | **fail** (same) |

`measurement_complete: true` — both arms have a full AgentV ship-gate
scoreboard (`gates.pass=false`; `held_out`/`adversarial`/`ood`/`rico_held`
suites `missing_suite` at this cycle's screening role, as expected).

Both arms trained cleanly from scratch at seed 100009 with distinct recipe
values (confirmed via `train_summary.json`/manifest diff): the candidate
applies `binder_topology_loss_weight=0.25`, the control keeps it at `0.0`.
**Training loss measurably diverges** between arms at this 2x step count
(`last_loss` 6.802424 control vs 6.833416 candidate) — proof the escalated
run was not a no-op and the candidate's extra loss term is actually
optimizing something different from the control.

**Despite that measurable training divergence, every decode-facing quality
metric is still bit-identical between the two arms on the smoke suite:**
`parse_rate`, `meaningful_program_rate`, `structural_similarity`,
`binder_reference_f1`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, `placeholder_fidelity`, and `reward_score` all match
exactly. Only `latency_ms_p50` differs (+385.35 ms, candidate slower —
consistent with fixture-scale timing noise, not a real speed effect).
Primary metric (`smoke.structural_similarity`, candidate − control):
**0.0** exactly.

## SDLC Phase A classification

`positive: false`, `stack_layer: false`, `action: no_stack_layer_non_positive`.

Reasons (from `sdlc_delivery.json`):

1. `fixture_insufficient_n:c9-control` (smoke n=3 < 20)
2. `fixture_insufficient_n:c9-binder-topology` (same)
3. `primary_metric_null_or_worse:smoke.structural_similarity:control=0.51 candidate=0.51 improvement=0.0`
4. `fixture_insufficient_n_alone`

## The key question this cycle answers: does escalated scale reveal a signal?

**No — the escalation revealed no new signal.** This is the fourth
consecutive bit-identical null this session (after cycle 6 `component-edge`,
cycle 7 `component-plan` re-screen, cycle 8 `component-inventory`), and the
**first at 2x training steps**. Training loss diverged measurably between
arms (proving the escalated run genuinely trained differently), yet every
decode-facing output metric on the fixed n=3 smoke suite still matched
exactly between control and candidate. This is meaningful evidence that:

- The bottleneck is **eval suite size (`n`)**, not training duration/steps.
  Doubling `--steps` (within the wall cap) does not, by itself, make small
  aux-loss effects detectable on a 3-example decode suite — the outputs
  apparently saturate to the same greedy/constrained-decode result regardless
  of the small optimizer-level loss differences at this model/data scale.
- No larger published eval snapshot exists to test the alternate hypothesis
  (bigger `n` might reveal the signal instead); growing `n` would require a
  new `slm data build-test` run, which is out of scope for a bounded
  diagnostic step and was correctly not attempted (would count as inventing
  a new fixture).

## Next-run priorities

1. **model:** the completed non-positive binder-topology arm is exhausted;
   test the distinct size-matched `component-structure` quality hypothesis
   next (rank 1, confidence 0.90).
2. **evaluation:** keep the matched control as the size-matched baseline
   every cycle.
3. **model:** rotate thrash recommendation across the lever bank (not
   bounds-only) — the completed candidate is exhausted and cannot be
   selected again without a new preregistered hypothesis.
4. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop.
5. **model_build:** confirmed champions promote under cadence; thrash only
   screens.

## Screening-bank assessment

Cycles 3/5 (`component-plan`, rejected — genuine regression), 6
(`component-edge`, bit-identical null), 7 (`component-plan` re-screen,
bit-identical null), 8 (`component-inventory`, bit-identical null), and now 9
(`binder-topology`, bit-identical null **even at 2x steps**) — five
consecutive non-positive from-scratch cycles this session, four of them
bit-identical on every decode-facing metric. Escalating `--steps` did not
change the outcome pattern; it only added evidence that the null is a
detection-power problem tied to `n`, not a training-duration problem. The
orchestrating session's open question from cycle 8 ("would more steps reveal
a real effect?") is now answered: **no, not within the wall-cap-safe range
tested (2x steps).** A genuine harness-improvement gap exists if more power
is wanted: growing published smoke/held_out `n` beyond 3/5 (a
`slm data build-test` fixture-scale change, out of scope for this cycle) is
the only remaining lever this session's evidence supports.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c9/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c9-{control,binder-topology}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c9-results.json`
- Predecessor: [cycle 8 results](continuous-openui-local-20260802-c8-results.md) (`component-inventory`, bit-identical null, steps=20-22)
