# Sample-size adequacy bounds (certified data-volume climb signal)

**Status:** infrastructure landed; wiring is signal-only (recommends, never
gates). **Claim class:** fixture/wiring — no train, eval, or ship claim is
made in this change.

## Motivation

The climb loop has never had a principled answer to "how much training data
is enough?". The historical record shows both failure directions:

- Corpora in the low hundreds (`wf_smoke_v2` = 101 records; `remediated_unique`
  = 198; Gold tier = 20) with the hill-climb policy pinned to them, and the
  scale-up attempt (SLM-267) blocked by the generator's measured 1,781-root
  reachable grid ([compiler-inverted-program-data.md](compiler-inverted-program-data.md)).
- A measured REJECT for volume alone: 350/524-row corpora did not beat the
  103-row baseline at matched recipe
  ([lever-train-corpus-e1291-e937-vs-smoke-measured-results.md](lever-train-corpus-e1291-e937-vs-smoke-measured-results.md)),
  on thin (n=3 smoke) evidence.

No prior sample-size number was ever derived or formally validated; no Lean
quantity certified data volume (the RESEARCH-19 Rademacher filing is an
explicit Python-only fixture). This change gives the loop a *derived,
self-recalibrating* band instead of a guessed constant.

## The band

For each cycle, from the current build's evidence and the current
architecture:

**Coverage floor (lower bound)** — records required for every tracked
component to accumulate `k` witnesses, projecting the scarcest component's
observed witness rate:

```
N_lower = ceil(k * n_observed / w_min)
```

- Registry: `bound.sample_size.coverage_lower.v1`; Lean:
  `LeverProofLean.SampleAdequacy.coverageLowerBound` with theorem
  `coverageLowerBound_covers` (`k*n ≤ wMin * bound`).
- Authority: `theorem_backed_projection` — the arithmetic is proved; the
  rate-persistence projection ("synthesis keeps witnessing the scarcest
  component at its observed rate") is a declared assumption restated on
  every report.

**Capacity ceiling (upper bound)** — records whose total description length
fits the parameter budget under the Collins et al. 3–6 task-bits/parameter
prior (the same prior `web/routes.py` applies to grammar-memory sizing):

```
N_upper[prior] = floor(params * bits_per_param / bits_per_example),  prior ∈ {3, 6}
```

- Registry: `bound.sample_size.capacity_upper.v1`; Lean:
  `LeverProofLean.SampleAdequacy.capacityUpperBound` with theorem
  `capacityUpperBound_within_budget` (`bound * bits_per_example ≤ params *
  bits_per_param`).
- Authority: `assumption_backed` — the prior interval is an assumption box;
  only the budget arithmetic is proved. This is **not** a generalization or
  quality claim ("optimal whole-model parameter size" remains explicitly
  non-identifiable, see `web/routes.py` `unavailable_targets`).

Both bounds evaluate through the EVID-04 `bound_ast/v1` registry: exact
`Fraction` arithmetic, digest-pinned documents, Lean `#guard` parity anchors
in `bound_ast_parity_fixtures.v1.json`.

## Verdicts (the recursive-improvement signal)

`slm_training.autoresearch.sample_adequacy.compute_sample_adequacy` emits a
`sample_adequacy/v1` report whose verdict drives the loop:

| verdict | condition | climb action |
|---|---|---|
| `insufficient_evidence` | no records / no witness counts | none |
| `coverage_gap_blocked` | a tracked component has zero witnesses | `change_trajectory` → generator widening (volume cannot fix a zero rate) |
| `generate_more` | `n < N_lower` | `rebuild_data` with `unique_root_target` raised toward `N_lower` (capped by the generator's reachable grid and the knob schema ceiling 32768) |
| `sufficient` | `N_lower ≤ n ≤ N_upper[6]` | none |
| `saturated_change_trajectory` | `n > N_upper[6]` | `change_trajectory` — more volume is waste at this parameter budget |
| `conflict_change_trajectory` | `N_lower > N_upper[6]` | `change_trajectory` — coverage demand cannot fit the budget; capacity growth must be charged (`EG_params`) |

Because every input (witness counts, record count, mean description bits,
trainable params) is re-observed each cycle, the band tightens or moves as
the model, generator, and parameters change — the bounds *are* the feedback
loop, not a one-time constant.

Wiring:

- `climb_policy.sample_adequacy_intervention(policy, report)` compiles the
  verdict onto the existing typed `autotrain_data_intervention/v1` actions
  (`rebuild_data` reuses `data_intervention_action`; no shadow path).
- `climb_policy.data_intervention_indicated(..., sample_adequacy=...)`
  accepts the report as an additional indicator.
- Feedback vocabulary: `sample_size_below_coverage_floor` and
  `sample_size_above_capacity_ceiling` join
  `harnesses.train_data.feedback.FINDING_CODES` with executable knobs
  (`data_generation.unique_root_target` / `max_records_per_parent`). They
  are *not* blocking codes: the signal recommends, gates are unchanged.

## Worked example (current smoke fixture, illustrative only)

`wf_smoke_v2` (n=101), witness target k=4, scarcest tracked component
witnessed twice → `N_lower = ceil(4·101/2) = 202` → `generate_more`, and
with the generator's measured 1,781-root grid as `reachable_unique_roots`
the recommendation stays reachable. A 1000-parameter toy at 60
bits/example has `N_upper = [50, 100]` → 101 records would already be
`saturated_change_trajectory`. These are fixture numbers for wiring tests,
not measurements of any production corpus.

## Honesty constraints

- **Never a gate.** Reports carry `promotion_authority: false` and findings
  carry `authority: climb_signal_not_gate`. Admission gates, ship gates,
  and promotion criteria are untouched.
- **Capacity is charged.** A `conflict`/`saturated` verdict routes to
  trajectory change; growing `trainable_params` to raise the ceiling
  remains subject to decode-invariants VI (`EG_params` ≥ 1, size-matched
  arms).
- **The projection is falsifiable.** If a rebuild at the recommended target
  fails to reach the witness target, the rate-persistence assumption is
  refuted for that generator — that closes the *projection*, and the
  successor is generator widening, not a weakened target.
- **Volume is necessary, not sufficient.** `sufficient` means "not
  provably starved and not provably saturated"; quality/shortcut gates
  still own admission (`capability-driven-data-synthesis.md` still holds:
  row count is not the scale variable — this band bounds it, it does not
  chase it).

## Successors

- Feed real per-component witness counts from the coverage manifest /
  quality report into the continuous loop caller (today the caller supplies
  them; the schema is ready).
- Learning-curve saturation (marginal metric gain per added record) as a
  second, evidence-side ceiling estimator alongside the capacity prior.
- Tighten `bits_per_example` from raw serialized bytes to compressed /
  entropy-based description length.

Eval-side sample floors are already owned by `autoresearch/power.py`
(`required_n_for_effect`), `default_min_suite_n`, and the `rico_held`
floor; this change deliberately does not duplicate them.
