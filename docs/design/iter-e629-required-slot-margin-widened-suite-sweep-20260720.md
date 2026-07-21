# E629 — a widened-suite margin sweep (1/2/3/4), and a new failure mode it surfaces

Date: 2026-07-20
Status: exploratory, single-checkpoint/single-seed, n=19; not a ship claim; not
confirmatory

## Which deferral this picks

E626/E627/E628 each deferred the same next step: "a powered multi-seed replay
(H19's protocol) sweeping `required_slot_margin_decode_weight` across a small
grid (e.g. 1, 2, 3, 4) on a full held-out/`rico_held` suite, not just `n=4`
OOD". This iteration runs that sweep honestly at the scale actually achievable
in one session, and reports what it actually shows rather than forcing a
confidence claim it can't support.

## Suite: what "full held-out/rico_held suite" means here

`src/slm_training/resources/data/eval/remediated/` already ships 5 immutable,
leakage-checked suites (`stats.json`: 37 candidates rejected against the train
manifest): `held_out` (5), `rico_held` (3), `adversarial` (4), `ood` (4),
`smoke` (3) — 19 records total. Building a materially larger `rico_held` via
`slm data build-test --rico-hf-split test --rico-limit 2600` would need live
HF network access and a fresh leakage-checked build; not attempted, since it
could not be completed and verified within this session alongside the sweep
itself. Retraining multiple seeds was also not attempted — the task
explicitly asked for E626's exact checkpoint to be reused verbatim for an
apples-to-apples comparison, so this stays single-checkpoint/single-seed
(seed 0).

**Suite used: the union of all 5 committed suites, n=19** — roughly 5x
E626-E628's n=4 ood-only replay, and it includes `rico_held`, the suite named
explicitly in the deferred next-step language. It is still far below
`DEFAULT_MIN_SUITE_N=20` **per suite** (`ship_gates.py`), so this remains
exploratory, not a ship-gate scoreboard.

## Checkpoint and recipe (reused, not retrained)

E626's own scratch checkpoint (`outputs/runs/e626-required-slot-margin-scratch800-20260720/checkpoints/last.pt`)
was reused verbatim; sha256 `c5b7c807…dd561221` verified to match before
running. The full E626/E627/E628 matched eval recipe was replayed unchanged
(`honest_slot_contract`, `slot_contract_constrained_decode`,
`slot_contract_in_context`, `semantic_role_contract_in_context`,
`semantic_role_decode_weight=8`, `slot_coverage_close_decode_weight=2`,
`schema_value_decode_weight=4`, `schema_opaque_close_decode_weight=4`,
`schema_role_slot_decode_weight=8`, the full `semantic_plan_*` family at
E617's recorded weights, `grammar_ltr_max_tokens=160`), against
`--suites held_out,rico_held,adversarial,ood,smoke`, varying only
`required_slot_margin_decode_weight` over `{0, 1, 2, 3, 4}`. No code was
changed this session (working tree clean at `2caba977`) — no version bump
required. Five `evaluate_model` runs, 45-57s wall each, well under the
3-minute cap.

The `margin=0` control reproduces E626/E628's own `ood`-subset control exactly
(all 8 headline metrics to 4+ decimal places, matching `checkpoint_sha256`),
confirming a faithful replay before widening to the other 4 suites.

## Headline result: pooled across all 19 records

| Metric | margin=0 (control) | margin∈{1,2} | margin∈{3,4}\* |
| --- | ---: | ---: | ---: |
| meaningful_program_v1 rate | 0.6842 (13/19) | 0.6842 (13/19) | 0.6842 (13/19) |
| reward_score (mean) | 0.7889 | 0.9260 | 0.9260 |
| placeholder_fidelity (mean) | 0.5781 | 0.8991 | 0.8991 |
| placeholder_validity (mean) | 0.7258 | 0.9395 | 0.9395 |
| structural_similarity (mean) | 0.4494 | 0.5295 | 0.5295 |
| component_type_recall (mean) | 0.5965 | 0.6667 | 0.6667 |

\* margins 3 and 4 differ from 1/2 in exactly 1 of 19 predicted programs
(`rico_held:rico_eval_test_25`) — a cosmetic slot-position swap among two
structurally symmetric `Card` children with identical `reward_score` (0.997)
and identical `meaningful_program_v1` (`False`) in both variants. No aggregate
metric moves. Margins 1 and 2 are **byte-identical** (0/19 differing
predictions).

**The continuous metrics move, and move honestly (with real uncertainty):**
using `slm_training.evals.power_protocol`'s `bootstrap_paired_ci` directly on
the matched 19-record pairs (`margin≥1` vs `margin=0`), then
`benjamini_hochberg` across the 4 continuous metrics tested (α=0.05):

| Metric | Δ (margin1 − margin0) | 95% bootstrap CI | BH-significant? |
| --- | ---: | --- | :---: |
| reward_score | +0.137 | [0.058, 0.253] | yes |
| placeholder_fidelity | +0.321 | [0.188, 0.461] | yes |
| structural_similarity | +0.080 | [-0.063, 0.220] | no |
| component_type_recall | +0.070 | [-0.035, 0.184] | no |

`reward_score` and `placeholder_fidelity` survive Benjamini-Hochberg
correction; `structural_similarity` and `component_type_recall` are
directionally positive but their CIs cross zero — not distinguishable from
noise at n=19.

**But the binary `meaningful_program_v1` gate is exactly flat — 13/19 in
every one of the 5 arms.** Inspecting per-record: at `margin≥1`,
`ood_dashboard_01` flips `False → True` (matching E626/E628's own reported
Dashboard fix exactly), and `rico_eval_test_25` flips `True → False` — a real
gain and a real loss cancel out to zero net movement on the pass/fail gate,
purely because the suite happened to widen to include the record that loses.
The paired bootstrap on the binary outcome makes this explicit:
`bootstrap_paired_ci` estimate = `0.0`, 95% CI `[-0.158, +0.158]` — the true
difference could plausibly be anywhere in that range at this n; this is not
"no effect", it is "not enough data to know".

## A new failure mode this widened suite surfaces

`rico_eval_test_25` at `margin=0` correctly instantiates 5 different
components for 5 distinct role-appropriate placeholders. At `margin≥1`, one
`Button` absorbs 5 placeholders that belong to 5 different slots
(`:sliding_tabs.label`, `:cardview.title`, `:toolbar.text`, `:cardview.body`,
`:cardview_1.title`), triggering `schema_value_role_mismatch` on
`Button.size`/`type`/`variant` and `placeholder_semantic_role_mismatch` on 6
placeholders. Since E628 already excludes `frame_depth == 0` from this bias,
this is happening at `frame_depth >= 1` — an **argument-position**
over-stuffing failure, not the root-hijack E627/E628 already fixed.

This is exactly the risk E628's own "Next step" #2 flagged as untested:
*"a different checkpoint/seed could still surface a different failure mode
once the floor is large enough to dominate a legitimate `frame_depth >= 1`
argument-position competition."* It surfaced here on the **same**
checkpoint/seed, at the **smallest** margin tested (1, not "substantially
larger" as E628 speculated) — just on a suite record E626-E628 never
evaluated. It was invisible in three prior sessions purely because
`rico_held` was outside their `n=4` ood-only replay.

## H19's protocol: was it actually applicable here?

Ran `python -m scripts.run_flow_power_protocol --mode analyze-existing
--iter-json outputs/runs/e629-outcomes/e629-margin<X>-outcomes.json` for each
arm. Its Wilson/exact-binomial single-arm interval machinery is real and
applicable: pooled `meaningful_program_v1` (n=19, 13 successes) gives Wilson
`[0.460, 0.846]`, exact binomial `[0.434, 0.874]` — identical in every arm,
since the success count never changes. **Its seed-variance/MDE-simulation
machinery is not meaningfully exercised here** — `analyze-existing` correctly
reports `seed_variance = 0.0` because there is exactly one seed (seed 0) in
every arm, per this session's explicit instruction to reuse E626's checkpoint
rather than retrain. `analyze-existing` also only analyzes one arm's JSON at a
time; it has no built-in two-arm delta. For the actual control-vs-treatment
comparison, `slm_training.evals.power_protocol`'s own `bootstrap_paired_ci`,
`cluster_bootstrap_ci`, and `benjamini_hochberg` functions were called
directly (same library, not a shadow reimplementation) —
`outputs/runs/e629-outcomes/paired_bootstrap_analysis.py`. A cluster-bootstrap
(suite as cluster) was also run as a sanity check but is noted as coarse:
only 5 clusters.

**Honest verdict: this is not a powered confirmatory result.** n=19 with one
seed is enough for real (if wide) Wilson/bootstrap intervals on the pooled
binary gate and enough to detect a large, consistent continuous-metric shift
(reward/fidelity) after multiplicity correction, but nowhere near enough to
resolve whether the flat binary gate reflects a true null or is just two
opposite effects of similar size that happened to net to zero at this n.

## Decision

**Do not change the shipped default** (`required_slot_margin_decode_weight`
stays `0.0`). Margins 1 and 2 remain the best-supported range if this lever
is ever adopted — margins 3/4 show no benefit over 1/2 (byte-identical bar one
cosmetic swap), so there is no reason to prefer a higher margin within this
grid. But this session found a genuine new failure mode (`rico_eval_test_25`,
`frame_depth >= 1` argument over-stuffing) that E626-E628 never saw, and that
finding — not a clean "margin=2 is safe" story — is the honest headline.
**Recommend margin≈2 as a candidate default for a future iteration only if
and after the rico_held over-stuffing mode is root-caused and fixed** the way
E627/E628 root-caused and fixed the `frame_depth == 0` hijack. Adopting it
before that would trade one already-fixed failure mode for a newly-discovered
one on different data.

## Next step (deferred)

1. Root-cause the `rico_eval_test_25` argument-position over-stuffing failure
   the same way E627/E628 root-caused the root hijack — almost certainly
   another per-position bias-stack magnitude interaction, this time between
   `required_slot_margin_bias` and whatever bias is meant to stop one
   component from absorbing multiple unrelated slots at `frame_depth >= 1`.
2. A genuinely powered confirmatory pass still needs either a freshly built,
   materially larger `rico_held` (live HF fetch + leakage check, not attempted
   here) or multiple retrained seeds (intentionally not done this session) —
   this n=19/single-seed result is a real step up from E626-E628's n=4 but is
   not that.
3. `binding_aware_meaningful_v2_rate_strict` is 0.0 across `held_out`/
   `rico_held`/`adversarial`/`ood` at every margin (only `smoke` shows 0.333,
   unchanged by margin) — E620's coverage-aware component/property closure
   work remains the next lever after (1) and (2) above.

Raw evidence:
[JSON](iter-e629-required-slot-margin-widened-suite-sweep-20260720.json).
Supplementary analysis (untracked, `outputs/` is gitignored):
`outputs/runs/e629-outcomes/paired_bootstrap_analysis.py` and its output
`outputs/runs/e629-outcomes/paired_bootstrap_analysis.json`; per-arm
Wilson/exact-binomial reports under
`outputs/runs/e629-power-protocol-margin{0,1,2,3,4}/`; per-suite scoreboards
under `outputs/runs/e629-margin{0,1,2,3,4}/scoreboard.json`.
