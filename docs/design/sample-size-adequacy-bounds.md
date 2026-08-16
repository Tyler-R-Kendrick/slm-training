# Sample-size adequacy (certified coverage + measured marginal utility)

**Status:** infrastructure + live continuous-loop wiring; signal-only
(recommends, never gates). **Claim class:** fixture/wiring — no train, eval,
or ship claim is made in this change.

## Motivation

The climb loop never had a principled answer to "how much training data is
enough — and when is more a waste?". The record shows both failure
directions: corpora in the low hundreds pinned by policy (`wf_smoke_v2` =
101 records, Gold tier = 20) with the SLM-267 scale-up blocked at the
generator's measured 1,781-root grid, and a (thin, n=3) measured REJECT for
volume alone (350/524 rows vs 103,
[lever-train-corpus-e1291-e937-vs-smoke-measured-results.md](lever-train-corpus-e1291-e937-vs-smoke-measured-results.md)).
No prior sample-size number was derived or validated; no Lean quantity ever
certified data volume.

An earlier draft of this design bounded volume with a global coverage floor
and a capacity-prior ceiling. Critique (recorded here so the reasoning
survives): a global floor driven by the scarcest component compiles into
uniform volume scaling — the most expensive way to buy tail coverage — and
a memorization-capacity prior is wrong-signed as a stop signal, because
data beyond memorization capacity is precisely what forces the compression
regime. v2 keeps the certified arithmetic but re-points both actions.

## The v2 design

### 1. Coverage drives generation — per component, targeted, fail-closed

`autoresearch.sample_adequacy.compute_sample_adequacy` observes the build's
`component_histogram` (every build already emits it in `stats.json`) and
reports **per-component witness deficits** against the witness target
(default 4). Any deficit — including zero-witness components — compiles via
`climb_policy.sample_adequacy_intervention` onto the existing
`rebuild_data` action as a **targeted** rebuild:

- `generation_mode: until_coverage`
- `component_coverage_minimum: k` (new overlay on the plan's existing
  component-axis coverage minimum; `--component-coverage-minimum` /
  `DataGenerationKnobs.component_coverage_minimum`)
- `unique_root_target` raised toward the projected floor, capped by the
  knob schema ceiling

The build is fail-closed (`exhaustion: fail` + unmet minima), so **the
build, not a projection, decides whether coverage is reachable** — a failed
targeted build is the evidence that closes the approach (successor:
generator widening, see
[compiler-inverted-program-data.md](compiler-inverted-program-data.md)).

The projected record floor `ceil(k·n/w_min)` is still computed and
reported for planning through the EVID-04 registry
(`bound.sample_size.coverage_lower.v1`, proved:
`LeverProofLean.SampleAdequacy.coverageLowerBound_covers`), with the
rate-persistence assumption declared on every report.

### 2. Saturation requires measurement — the data-adequacy ladder

"Stop generating / change trajectory" may only come from the **measured
marginal utility of data**:
`harnesses.experiments.data_adequacy_ladder` + `scripts/run_scaling_ladder.py
--family data-adequacy` train the same recipe on nested prefix subsets of
one immutable corpus (echoing the generator's `scaling_ladder` semantics),
measure held-out weighted NLL per rung, and classify the last gain:

- `rising` — more data still pays; keep the data axis open.
- `flat` — measured saturation; `sample_adequacy` may now emit
  `saturated_change_trajectory`, which compiles to a `change_trajectory`
  action (quality/coverage levers, or **charged** capacity growth via
  `EG_params` — never silent scaling).
- `undecidable` — underpowered or variance-free measurement. **A flat claim
  is refused** unless the eval suite meets
  `power.required_n_for_effect(mde, sd)`; this is the same eval-decidability
  arithmetic whose absence stalled hill climbing at n=3 smoke suites
  ([harness-evolution-architecture-review-20260809.md](harness-evolution-architecture-review-20260809.md)).
  The ladder converts that lesson into a hard rule: an undecidable eval can
  never become a stop signal.

Ladder subsets and artifacts are `claim_class: fixture`,
`promotion_authorized: false`.

### 3. Capacity prior is a diagnostic, never a verdict

The Collins et al. 3–6 task-bits/parameter interval
(`bound.sample_size.capacity_upper.v1`, assumption-backed; arithmetic
proved: `capacityUpperBound_within_budget`) is computed only when the
caller deliberately supplies `trainable_params` + description bits — the
live loop does not. Exceeding the generous endpoint sets
`memorization_regime_expected: true` (compression regime ahead) and changes
nothing else. `web/routes.py`'s refusal to derive whole-model sizing from
this prior still stands.

### 4. Live caller (no dormant vocabulary)

`scripts/run_autotrain_continuous.py`'s `SELF_HEAL_REBUILD_DATA` path:

1. `_sample_adequacy_report(cwd)` — observes the latest local heal build's
   `stats.json` (fallback: the policy corpus fixture stats) plus the latest
   `data_adequacy_ladder.json` classification.
2. Measured-flat verdict → the heal **refuses to rebuild**
   (`SAMPLE_ADEQUACY_SATURATED`), leaving the action pending for a
   trajectory decision instead of faking progress.
3. `generate_more` → the rebuild argv is shaped by
   `sample_adequacy_intervention` (`until_coverage` targeting). The
   wall-capped local CPU heal keeps the plan's own coverage minima so the
   heal stays reliable; the raised component minimum is a promotion-scale
   instruction carried in the action payload.
4. The report is written to the campaign directory as
   `sample_adequacy.json` and bound into the action receipt's evidence.

### Verdict table

| verdict | condition | compiled action |
|---|---|---|
| `insufficient_evidence` | no records / no histogram | none |
| `generate_more` | any component below the witness target | targeted `until_coverage` rebuild, fail-closed |
| `sufficient` | coverage met, no measured-flat evidence | none |
| `saturated_change_trajectory` | coverage met **and** ladder measured `flat` (powered) | `change_trajectory` |

Every input is re-observed each cycle, so the signal recalibrates as the
model, generator, and synthesis distribution improve — and the ladder
re-measures marginal utility whenever it is re-run, which is the genuinely
recursive part.

## Honesty constraints

- **Never a gate.** Reports carry `promotion_authority: false`; findings
  carry `authority: climb_signal_not_gate`
  (`sample_size_below_coverage_floor`, `sample_size_above_capacity_ceiling`
  in `harnesses.train_data.feedback.FINDING_CODES`, mapped to targeted
  knobs). Admission, ship, and promotion gates are untouched.
- **Capacity is charged.** Trajectory changes route through quality /
  coverage levers or `EG_params`-charged growth (decode-invariants VI).
- **No undecidable stop signals.** `flat` requires the powered suite floor;
  `undecidable` feeds nothing.
- **Volume is necessary, not sufficient.** `sufficient` means "not provably
  under-covered and not measured-saturated"; quality/shortcut gates still
  own admission (`capability-driven-data-synthesis.md` still holds: row
  count is not the scale variable).

## Successors

- Run the data-adequacy ladder at promotion scale (powered suites) and file
  the first measured marginal-gain record via
  `documenting-experiment-results`.
- Tighten `bits_per_example` (compressed / entropy description length) if
  the capacity diagnostic is ever promoted beyond a diagnostic.
- Component-axis coverage floors for prompt-surface and reference-topology
  axes (the overlay currently targets the component axis).

Eval-side sample floors remain owned by `autoresearch/power.py`,
`default_min_suite_n`, and the `rico_held` floor; the ladder consumes that
machinery rather than duplicating it.
