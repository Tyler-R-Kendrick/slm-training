# Power/decidability preflight (WP-2)

Status: implemented 2026-08-09.
Code: `src/slm_training/autoresearch/power.py` (pure arithmetic),
`src/slm_training/autoresearch/preflight/power_check.py` (loop plugin).
Tests: `tests/test_autoresearch/test_power_preflight_math.py`,
`tests/test_autoresearch/test_power_preflight_check.py`.

## Why (RC1 of the 2026-08-09 architecture review)

[`harness-evolution-architecture-review-20260809.md`](harness-evolution-architecture-review-20260809.md)
§2 RC1: screening measurements are below the decidability floor, and every
consequence is derivable *before* any run executes:

- Screening runs `screening_smoke_n: 3` documents at 20–22 train steps
  (`policy.v1.json`, `MAX_RUN_MINUTES = 3`). The screening metrics move in
  quanta of ~1/3 per flipped document, so the policy's
  `minimum_effect: 0.01` is **~33x below measurement granularity**.
- The loop's screening decision statistic — the paired sign test in
  `credit_engine._two_sided_p_from_signed_effects` — has a minimum
  attainable two-sided p of **0.25 at n=3**. No screening result at n=3 can
  ever reject at alpha=0.05. Arithmetic, not an empirical finding.
- `fixture_insufficient_n` is the single most frequent blocker in the
  durable record (**449 hits**), yet the climb loop was instructed to ignore
  exactly that check where it binds.

Review §4-R1 successor approach (binding): an experiment is inadmissible
unless its minimum detectable effect at the declared budget is <= its
preregistered plausible effect. Computable pre-run; fails closed.

## What was reused from darkfactory phase 1 vs added

Darkfactory phase 1 (HEAD `26c38ce`) already shipped the exact power gate in
`src/slm_training/autoresearch/evidence_ledger.py` ("Power feasibility
(exact arithmetic)"): `sign_test_min_two_sided_p`, `sign_test_feasible`,
`required_sign_test_n`, `parse_alpha`, `power_scaled_null_seeds`,
`power_feasibility_report` — all for the **paired sign test** with exact
`fractions.Fraction` arithmetic.

`power.py` **reuses** (imports, does not re-implement):

- `sign_test_min_two_sided_p` — the paired branch of `min_attainable_p`.
- `required_sign_test_n` + `parse_alpha` — the paired two-sided branch of
  `min_attainable_n`.

`power.py` **adds** what phase 1 did not have:

- the **unpaired two-sample permutation** p-value floor
  (`C(n_a + n_b, n_a)` labelings),
- the normal-approximation **minimum detectable effect** and its inverse
  `required_n_for_effect` (phase 1 answered only "can this n reject at
  alpha at all", never "how big an effect could it see"),
- the combined `is_decidable(...) -> DecidabilityReport` verdict, and
- the loop-facing preflight plugin with observed-SD lookup.

## Paired vs unpaired reconciliation

Two exact tests with different p-value floors appear in the repo/review;
both are exposed behind `paired`:

| variant | arrangement space | min one-sided p | min two-sided p | at n=3 (two-sided) |
|---|---|---|---|---|
| unpaired two-sample permutation, n vs n (`paired=False`, default) | `C(2n, n)` labelings | `1/C` | `2/C` | `2/20 = 0.1` |
| paired sign test / sign-flip permutation (`paired=True`) | `2^n` sign patterns | `(1/2)^n` | `2*(1/2)^n` | `2/8 = 0.25` |

At n=3 vs n=3 the unpaired floors are one-sided `1/20 = 0.05` and two-sided
`2/20 = 0.1`; the review doc's **0.25 at n=3** is the paired variant. Both
exceed alpha=0.05 two-sided, so the n=3 screening design is undecidable
under either reading. **The loop uses the paired variant**: the credit
engine decides screening arms with a paired sign test over signed
per-document effects, so `power_check.py` evaluates `paired=True`. All
defaults in this module are two-sided, matching that sign test; `sided=1`
is available on `min_attainable_p` / `min_attainable_n`.

Smallest decidable n at alpha=0.05 (two-sided): paired **n = 6**
(`2/2^6 = 0.03125`; identical to phase 1's `required_sign_test_n(1/20)`),
unpaired **n = 4 vs 4** (`2/70 ~= 0.0286`).

## Normal-approximation assumptions

`min_detectable_effect(n, sd, alpha=0.05, power=0.8)` is the balanced
two-sample z-power form with known common `sd`:
`MDE = (z_{1-alpha/2} + z_{power}) * sd * sqrt(2/n)`; with `paired=True` it
is the one-sample form on differences, `MDE = z_total * sd / sqrt(n)`,
where `sd` is the SD of the paired differences. `required_n_for_effect` is
its ceil-inverse. These are budgeting approximations; the attainability
floor above is exact. `is_decidable` requires **both** axes: p-floor <=
alpha AND MDE <= `minimum_effect`, and its `required_n_for_effect` already
folds in the attainability floor so re-planning to that n is decidable on
both axes.

## Observed SD source

The plugin pools the committed phase 1 evidence ledger
(`src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json`),
whose per-arm Welford `m2_delta` / `n_delta` accumulate candidate-minus-
control deltas of the loop primary metric — exactly the paired-difference
SD the paired MDE needs: `sd = sqrt(sum m2 / sum (n-1))` =
**0.031158** over **157** pooled degrees of freedom (2026-08-09 artifact).
The ledger does not record a metric name (it is the screening primary,
`smoke.structural_similarity` per `policy.v1.json`), so the pooled SD is
used only for `structural_similarity` endpoints with >= 20 pooled dof;
anything else uses the conservative default `sqrt(0.25/3) ~= 0.2887` — the
worst-case SD of a mean over 3 binary-quantized documents (RC1's ~1/3
quantum). Conservative means *large*: it inflates the MDE so
unknown-variance designs block rather than slip through.

## Worked examples (values produced by the shipped code)

**RC1 screening design — blocked.** `n_seeds=3`, `minimum_effect=0.01`,
`endpoint_metric=smoke.structural_similarity`, ledger sd 0.031158,
paired, alpha=0.05:

- `min_attainable_p = 0.25 > 0.05` — no result at n=3 can ever reject;
- `min_detectable_effect = 0.0504 >> 0.01` (~5x the preregistered effect);
- `required_n_for_effect = 77` paired seeds to power `minimum_effect=0.01`
  at this sd — the verdict carries both numbers so the loop can re-plan
  (raise n, or raise `minimum_effect` to something measurable).

Verdict: `block`, reasons include `min_attainable_p=0.25 > alpha=0.05 at
n=3 (paired sign test)` and `required_n_for_effect=77`.

**n=8 — decidable for effects >= 0.031.** Same sd, `n_seeds=8`:

- `min_attainable_p = 2/2^8 = 0.0078 <= 0.05`;
- `min_detectable_effect = 0.0309` — decidable for any preregistered
  `minimum_effect >= 0.031` (e.g. `minimum_effect=0.05` passes).

With the conservative default sd (0.2887) instead, n=8 gives MDE 0.286:
foreign-metric or missing-ledger candidates need either a large
preregistered effect or a much larger n — by design.

## Plugin behavior

`preflight/power_check.py` exposes the module-level `CHECK`
(`check_id = "power_decidability"`) discovered by the preflight package
seam (`preflight/__init__.py`, RC3 work package). `run(candidate)` reads
`n_seeds`, `minimum_effect` (falling back to the policy screening default
0.01 with an explanatory reason), and `endpoint_metric`; verdicts:

- `block` when candidate data is unusable (e.g. no usable `n_seeds`), **or**
  the design is decidable at neither the current cumulative `n_seeds` nor
  any realistic amount of further accumulation — `required_n_for_effect`
  exceeds `MAX_REASONABLE_N` (64). The §4-R1 gate fails closed on evidence
  questions; reasons always carry `min_attainable_p` and
  `required_n_for_effect` for re-planning, `data` carries the full report;
- `warn` when the design is not yet decidable at the current cumulative
  `n_seeds` but `required_n_for_effect <= MAX_REASONABLE_N` — still
  accumulating, not yet decided (see "Seeds-policy reconciliation" below);
- `pass` when decidable on both axes at the current cumulative `n_seeds`;
- `warn` also when the check itself errors (the preflight package's
  fail-soft law: a check bug must never take down the loop nor silently
  block a candidate — a bug is not evidence).

It never raises.

## Seeds-policy reconciliation (post-launch fix, 2026-08-10)

The continuous loop spends **one seed per screening cycle** and
accumulates evidence for an arm across many cycles — the committed
`evidence_ledger.v1.json`'s per-arm `n_complete` is that cumulative count.
The first cut of this preflight passed a literal `n_seeds=1` (the marginal
contribution of a single cycle) to `power_decidability`. Since
`min_attainable_p` at n=1 (paired sign test) is `2 * (1/2)^1 = 1.0`, that
design is undecidable *by construction* — every screening cycle for every
arm blocked, and the driver's fail-soft override
(`_preflight_screening_slug`, "all_open_arms_blocked_ran_original_pick")
fired every time, making the gate a no-op that only ever logged verdicts.
Worse, blocking a low-n arm's *only* path to accumulating more evidence is
a catch-22: the arm could never cross into decidability if every cycle
that would grow its `n_complete` was preflighted away first.

The fix has two parts:

1. `scripts/run_autotrain_continuous.py::_preflight_screening_slug` now
   reads the arm's cumulative `n_complete` from
   `evidence_ledger.load_ledger()` (best-effort — any load failure
   degrades to the old `n_seeds=1` behavior rather than raising) and
   passes `n_seeds = n_complete + 1` — the projected total after this
   cycle, not the cycle's marginal contribution.
2. `power_check.py`'s verdict is now three-way: an arm below
   `required_n_for_effect` but under the `MAX_REASONABLE_N` ceiling
   **warns** (still accumulating — the cycle proceeds, `n_complete` grows,
   and the next cycle's check moves closer to decidable) rather than
   **blocks**. Only designs whose `required_n_for_effect` exceeds the
   ceiling — i.e. RC1's own case, `minimum_effect=0.01` requiring 77 paired
   seeds — remain a hard block, since no realistic amount of further
   accumulation would help.

`MAX_REASONABLE_N = 64` is generous relative to the conclusion policy's
`adequate_power_requires.min_seeds = 8`
(`docs/design/hypothesis-family-conclusions.md`), so ordinary accumulation
toward an 8-to-64-seed decision is never blocked — only designs that
could never reach a decision at any realistic budget are.

## Process / heal arms (not confirmatory)

A local `rebuild_data` successor (`heal_resume`, or any candidate with
`process_arm=true` / `claim_class` in `{process, heal, wiring}`) is a first
execution on a new snapshot, not a confirmatory screen of a 0.01 effect.
At n=1 the paired floor is 1.0 and `required_n_for_effect` exceeds
`MAX_REASONABLE_N` — the same arithmetic that correctly blocks RC1
screening must **not** skip the heal and rematch a leftover fixture slug.

`power_check` therefore **warns** (never blocks) on process candidates.
`_preflight_screening_slug` also refuses to rotate a process arm away if
another plugin still returns `block`. Observed failure: cycle 198 selected
`simplified-nl-i10-heal` then `PREFLIGHT_BLOCK` rotated to `c78`.
