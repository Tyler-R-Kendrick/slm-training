# Screening sample-size range (certified n for the climb loop)

**Status:** infrastructure + live continuous-loop wiring; signal-only
(recommends, never gates). **Claim class:** fixture/wiring — no train, eval,
or ship claim is made in this change.

## Motivation

The continuous climb loop screened at a hard-coded `screening_smoke_n: 3`.
Cycle history (c430–c473) shows the cost: n=3 is below the paired sign-test
decidability floor (minimum two-sided p = 0.25 > alpha = 1/20 — the exact
arithmetic in `autoresearch.power`), so nearly every cycle classified
`fixture_insufficient_n` → `positive=false`, screens oscillated ~0.06↔0.36
with no causal signal, and confirm runs never reheld screening spikes
(`primary_quality_not_reheld`). Nothing in the system *calculated* what n
should be; the number was a constant, not a derived quantity.

This change makes screening n a **computed range** — the same Lean-proved
bound pattern as [sample-size-adequacy-bounds.md](sample-size-adequacy-bounds.md),
pointed at the eval side instead of the data side.

## The design

### 1. Floor: exact sign-test attainability (theorem-backed)

`LeverProofLean.ScreeningSampleSize.signTestDecidabilityFloor` searches upward
from n=1 for the least per-arm n whose minimum two-sided sign-test p
(`2/2^n`) reaches the policy alpha (`power_gate.alpha`, default 1/20).
Soundness and minimality of the search are proved
(`signTestFloorFrom_sound`, `signTestFloorFrom_minimal`); at alpha = 1/20 the
floor is **6** (n=5 floors at p = 0.0625 > 0.05). The loop consumes it through
the EVID-04 registry predicate `bound.screening_n.decidability_lower.v1`.

An optional **power floor** (`power.required_n_for_effect`, normal
approximation over paired differences) composes when a caller supplies both
`minimum_effect` and an observed SD. It is labeled `assumption_backed` —
budgeting arithmetic, never theorem-labeled — per the
improve-lean-optimums discipline (Lean proves arithmetic, not calibration).

### 2. Ceilings: arm-wall budget (theorem-backed) and suite volume

`LeverProofLean.ScreeningSampleSize.screeningBudgetUpperBound` is the most
records whose decode fits the arm wall beside the train floor and suite
overhead (`floor((wall − train − overhead) / minDecode)`; fitness of every n
under the ceiling proved by `screeningBudgetUpperBound_fits`). Consumed as
`bound.screening_n.budget_upper.v1`.

The **suite-volume ceiling** is the resolved smoke suite's record count — all
committed smoke suites currently carry exactly 3 records
(`resources/data/eval/*/suites/smoke/records.jsonl`).

### 3. Composition: `screening_sample_size/v1`

`autoresearch.screening_sample_size.compute_screening_sample_size` evaluates,
per cycle:

| field | source | authority |
|---|---|---|
| `decidability_floor_n` | registry predicate search | theorem-backed exact |
| `power_floor_n` | `power.required_n_for_effect` | assumption-backed |
| `n_min` | max of the floors | — |
| `budget_ceiling_n` | `bound.screening_n.budget_upper.v1` | theorem-backed exact |
| `suite_ceiling_n` | resolved smoke suite records | measured |
| `n_max` | min of the ceilings | — |

| verdict | condition | meaning |
|---|---|---|
| `feasible` | `n_min ≤ n_max` | climb at `chosen_n = n_min` (smallest sufficient) |
| `infeasible_range_empty` | `n_min > n_max` | range empty; `binding_constraints` names `wall_budget` / `suite_volume` |
| `insufficient_evidence` | no decode/wall/suite observation | exact floor still reported; caller keeps fallback n |

`screeningRangeFeasible_fits_budget` proves the smallest sufficient choice of
a feasible range fits the arm wall.

### 4. Live wiring (fail closed)

Climb policy v10/v12: `measurement.screening_smoke_n_mode: "auto"` with
`screening_smoke_n: 3` kept as the documented **fallback only**. The resolver
`climb_policy.screening_smoke_n_for_policy` is consumed by
`run_autotrain_continuous.py` at the decode-fit path (which now embeds the
full report in the `thrash_timing.json` `decode_fit` meta) and at the
power-gate arm-closure calculation:

- `feasible` → screen at `chosen_n`.
- `infeasible_range_empty` + `suite_volume` → **do not screen**. Set
  `must_generate`, append smoke fixtures, `build_test_data` + `DataStore.publish`
  a new eval snapshot under `resources/data/eval/`, commit it, then resume at
  `n_min`. Fallback n=3 is not a runnable screen size.
- `infeasible_range_empty` + `wall_budget` only → park; recalibrate decode/steps
  (never silent wall++).
- `insufficient_evidence` → advisory fallback n until wall/suite/decode inputs
  exist; not promotion authority.
- Resolution failure returns the configured fallback with no report.

### 5. Latency pre-check probe (`latency_preflight/v1`)

Screening arms carry a latency **pre-check**: `compile_commands` clones the
full eval command at `--eval-limit <probe_records>` (policy
`measurement.latency_probe.probe_records`, default 1) under a `-latprobe` run
id, inserted immediately before the full eval. `execute_commands` then
applies the calculated verdict before spending the full eval:

- **Probe timeout** — one record hitting the stage wall stops the arm there
  instead of burning `planned_n` × the fitted per-record decode timeout.
- **Over-budget projection** — `probe_wall_seconds × planned_n` (planned n is
  the resolved screening n from §4) past the wall the full eval would receive
  skips the full eval. The skip is recorded as typed stage telemetry
  (`latency_preflight_infeasible`) and the delivery reports the arm as
  `measurement_incomplete:<run>:latency_preflight_infeasible` — harness
  incomplete, **never a model verdict**.
- **Feasible / failed probe** — the full eval runs unchanged (probe failure
  fails open). Probe metrics never merge into arm metrics (separate run id;
  the executor excludes probe stages from the metrics fold).

The projection threshold is the same arithmetic as the certified budget
ceiling (per-record cost × n against the wall); the one-record projection
itself is a declared assumption (`authority: climb_signal_not_gate`), not
proved arithmetic. At the current 3-record suites the probe caps latency
discovery at one record instead of three; the saving grows linearly as the
screening suite grows past the §2 floor.

## What the certified range says today

At the declared defaults (alpha = 1/20, ~70s arm share under the 3-minute run
cap, 20s train floor, 8s overhead, 2s decode floor):

- floor = **6**, budget ceiling = **21**, suite ceiling = **3** →
  **`infeasible_range_empty`, binding: `suite_volume`**.

The wall budget is *not* the binding constraint at the declared decode floor —
the 3-record smoke suite is. The actionable successor is therefore a data
action (grow the screening suite past 5 records), not a wall or config bump.
This replaces the bare recurring `fixture_insufficient_n` with a computed,
auditable cause.

## Honesty constraints

- **Never a gate.** Reports carry `promotion_authority: false`; findings carry
  `authority: climb_signal_not_gate`. Ship, admission, promotion, and
  positive-classification semantics are untouched.
- **Fallback is explicit.** The configured `screening_smoke_n` remains the
  fail-closed default; an infeasible range never silently raises or lowers n.
- **Exact vs approximate is labeled.** Only the sign-test floor and budget
  ceiling are theorem-backed; the power floor is a declared approximation.
- **Timeouts remain optima.** If the decode-fit clamp always binds, the
  recipe or arm-share model is recalibrated — never a silent wall++ (the
  existing `_fit_screening_decode_timeout_seconds` rule stands).
- **Probe is a pre-check, not a verdict.** The latency probe may only skip
  measurement spend (typed incomplete); it never classifies an arm, and its
  failure fails open to the full eval.

## Successors

- **Grow the smoke screening suite past the floor** (≥6 records, target ≥20
  for decidable gates) via the frontier/synthesis pipeline — the measured
  binding constraint as of policy v10/v12.
- Wire an observed per-record decode p95 from thrash-timing rows into the
  budget ceiling (currently the declared `default_decode_floor_seconds`).
- Re-derive `promotion_suite_n` (currently 6 = the alpha floor) from the same
  report at promotion scale.
- Revisit the declared `minimum_effect: 0.01`, which is ~33x below measurement
  granularity at n=3 (see `autoresearch.power` docstring): at observed fixture
  noise the assumption-backed power floor (~785) dwarfs the attainability
  floor, so the declared effect size — not just n — needs recalibration before
  screening can be both decidable and sensitive.
