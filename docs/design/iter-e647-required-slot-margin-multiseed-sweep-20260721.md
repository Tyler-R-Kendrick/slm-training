# E647 — a genuinely powered multi-seed replay of `required_slot_margin_decode_weight`

Date: 2026-07-21
Status: completed negative/mixed; no checkpoint promoted; no code changed

## Which lever this picks up, and why not E620's literal recommendation

Every iteration from E639 through E646 closed with boilerplate next-step text
claiming "E620's coverage-aware component/property closure recommendation
remains the next untried lever." Before touching code, this session read
E620's own entry (`docs/design/iter-e620-required-slot-coverage-scratch800-
20260720.md`) plus every intervening matrix entry E621–E646 in full. That
reading falsifies the boilerplate: E620's recommendation ("the next useful
lever is coverage-aware closure across component and property boundaries,
with role-correctness preserved") is exactly the mechanism E621–E638 built,
generalized, and root-caused over eighteen iterations —
`slot_coverage_close_decode_weight` (`TwoTowerModel._slot_coverage_close_bias`,
`src/slm_training/models/twotower.py`), with role-compatible owner selection
(E621), an intervention trace (E622), a rejected prompt-owned filter (E630), a
frame-aware incompatible-owner escape (E631, the strongest single win in that
arc), a rejected broad string-slot penalty (E632), active Input role routing
(E633/E634), property-compatible slot coverage (E635, first strict-v2 record
fix), Modal schema reachability (E636), and nested family accounting (E637)
before E638's root-slot-coverage gate was rejected as a dead end. That is not
"never tried" — it is a fully explored, partially retained (v73–v75
default-off), ultimately capacity-limited arc that the lineage itself then
deliberately pivoted away from. PR #625/E639 explicitly frames its own
`required_slot_margin_decode_weight` lever as "a more concrete framing" of
*the same* E620 recommendation, not a different one. The phrase kept
propagating into E640–E646's `next_step` fields as unexamined copy-paste,
inherited from E639's own text without anyone re-checking it against E621–
E638's history in between. Filing another closure-mechanism experiment under
this banner would duplicate eighteen already-committed iterations.

Per this session's own instructions, the honest move is to say so in writing
(this entry) rather than manufacture a duplicate "closure" experiment — and,
since time allowed, to pick up the other item repeatedly and explicitly
flagged as open across the *same* six iterations: **"a genuinely powered
multi-seed/retrained comparison of `required_slot_margin_decode_weight`
remains open (E626/E639's own deferral, still not picked up by E627–E646)."**
That is what this session ran.

## Recipe

E639's exact scratch recipe (E620's own recipe): `e530_visible_semantic_
roles_r2_20260719` corpus (244 records), TwoTower, scratch context, choice
output tokenizer, batch size 1, 800 steps, `--no-sync-checkpoints`, seed
varied. E645's own checkpoint (seed 0, already present on disk this session,
sha256 `a4c24987…2041b7e7`, `last_loss=4.062225`) was reused verbatim as one
arm. Two fresh same-recipe checkpoints were trained for seeds 1 and 2:

| Seed | Run ID | Wall time | `last_loss` | Checkpoint sha256 |
| --- | --- | ---: | ---: | --- |
| 0 (E645, reused) | `e645-smoke-hero-trace-scratch800-20260721` | 55.8s (E645) | 4.062225 | `a4c24987…2041b7e7` |
| 1 | `e647-multiseed-s1-scratch800-20260721` | 54.85s | 4.603209 | `6fe9c20b…4b25afea` |
| 2 | `e647-multiseed-s2-scratch800-20260721` | 57.28s | 3.395596 | `1162ae51…4abedb3d` |

All three finished well inside the three-minute hard run cap. The three seeds
produce materially different losses (4.60/4.06/3.40), confirming these are
genuinely different training trajectories, not near-duplicates.

## Eval sweep

Each checkpoint was replayed at `required_slot_margin_decode_weight` in
`{0, 2}` (E642's best-supported margin) against the exact same union-of-5
suite E639–E646 used (`held_out`=5, `rico_held`=3, `adversarial`=4, `ood`=4,
`smoke`=3, n=19; all other flags identical to E645/E646's command template:
`honest_slot_contract`, `slot_contract_constrained_decode`,
`slot_contract_in_context`, `semantic_role_contract_in_context`, the full
`schema_*`/`semantic_plan_*` family, `slot_coverage_close_decode_weight=2`,
`grammar_ltr_max_tokens=160`). Six eval runs total (seed 0's margin-0/margin-2
runs reused verbatim from E645, `e645-fix-suite19-margin{0,2}-20260721`; four
new runs for seeds 1/2). Every eval command completed inside the three-minute
cap; two initial attempts (seed 1 margin 2, seed 2 margin 0) were interrupted
by an unrelated tool-timeout artifact of this session's own shell and were
discarded and re-run to completion rather than used as evidence.

## Pooled per-seed result (n=19 each)

| Metric (pooled, n=19) | seed0 margin0 | seed0 margin2 | seed1 margin0 | seed1 margin2 | seed2 margin0 | seed2 margin2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| meaningful_program_v1 | 0.7895 | 0.8421 | 0.7895 | 0.7368 | 0.6842 | 0.6316 |
| reward_score | 0.7990 | 0.8107 | 0.8617 | 0.8739 | 0.7700 | 0.7002 |
| placeholder_fidelity | 0.6232 | 0.6645 | 0.6996 | 0.7276 | 0.6794 | 0.5987 |
| structural_similarity | 0.4521 | 0.5001 | 0.5273 | 0.6193 | 0.4791 | 0.5531 |
| component_type_recall | 0.6842 | 0.6579 | 0.7061 | 0.7167 | 0.6904 | 0.6140 |

Net binary delta (margin2 − margin0) out of 19 records, per seed: **seed0
+1, seed1 −1, seed2 −1.** Seed 0 (the checkpoint every prior E639–E646 report
traced) reproduces E645's own previously-reported gain. Seeds 1 and 2 —
independently trained, same recipe — both show the *opposite* sign on the
binary gate, and seed 2 regresses on four of five headline continuous metrics
as well.

## Pooled 3-seed statistics (n=57 paired records)

Using `slm_training.evals.power_protocol.bootstrap_paired_ci` (57 records
treated as i.i.d., replicating E642's own method) and
`cluster_bootstrap_ci` (57 records resampled by seed-cluster, respecting the
fact each seed contributes 19 correlated records — 3 clusters, so this is
still thin, but honestly bounds the seed-level uncertainty rather than
hiding it inside a falsely-large n):

| Metric | control mean | treatment mean | delta | naive 95% CI (n=57 i.i.d.) | cluster 95% CI (by seed, k=3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| meaningful_program_v1 | 0.7544 | 0.7368 | -0.0175 | [-0.0702, 0.0351] ns | [-0.0526, 0.0526] ns |
| reward_score | 0.8102 | 0.7949 | -0.0153 | [-0.0635, 0.0191] ns | [-0.0698, 0.0122] ns |
| placeholder_fidelity | 0.6674 | 0.6636 | -0.0038 | [-0.0623, 0.0471] ns | [-0.0807, 0.0412] ns |
| structural_similarity | 0.4862 | 0.5575 | +0.0713 | [+0.0184, +0.1264] **SIG** | [+0.0480, +0.0920] **SIG** |
| component_type_recall | 0.6936 | 0.6629 | -0.0307 | [-0.0728, 0.0000] ns | [-0.0763, +0.0105] ns |

## Honest verdict

**Not a confirmatory result; not a ship claim; no checkpoint trained,
promoted, or synced; no code changed this session (clean tree throughout, no
version bump required).** This is the first time this lineage has replayed
`required_slot_margin_decode_weight` on more than one training seed. The
result is a genuine negative for the "moderate margin is a real, generalizing
win" reading that E639/E641/E645 each built up one checkpoint at a time:
across 3 independently-trained same-recipe checkpoints, the binary
`meaningful_program_v1` gate and three of four continuous headline metrics
(`reward_score`, `placeholder_fidelity`, `component_type_recall`) show no
statistically distinguishable effect once seed variance is honestly
represented (CIs cross zero both ways), and the per-seed binary sign flips
between seeds (+1/-1/-1). Only `structural_similarity` shows a
seed-cluster-robust positive delta (CI excludes 0 in both the naive and
seed-clustered bootstrap) — a real, reproducible-across-seeds effect on that
one metric, but not evidence that the lever raises meaning-bearing program
correctness. This matches this lineage's own repeated caution (E639, E642,
E645) that single-800-step-checkpoint results on this recipe are dominated by
which-bias-fires-first path sensitivity rather than a stable underlying
effect; three seeds is still not `DEFAULT_MIN_SUITE_N`-powered, but it is
enough to overturn the single-checkpoint "consistent gain" narrative that had
accumulated across E639/E641/E645/E646's shared seed-0 checkpoint. Default
`required_slot_margin_decode_weight` stays `0.0` (already default-off; this
session recommends *against* treating margin≈2 as a promising default based
on the prior single-seed evidence alone).

## Decision

Do not promote or default-enable `required_slot_margin_decode_weight` on the
strength of prior single-checkpoint results. Retain it default-off exactly as
before (unchanged). Correct the lineage's own `next_step` boilerplate: E620's
coverage-aware component/property closure recommendation is not an untried
lever — it is the E621–E638 arc, already explored to a capacity limit and
consciously superseded by the `required_slot_margin_decode_weight` framing
this session just multi-seed-tested. If this lever is pursued further, do it
with more seeds (5+) and/or a genuinely powered per-suite `n>=20` sample
before drawing a promotion conclusion from it; `structural_similarity`'s
seed-robust gain is the one thread worth following up in isolation.

Raw evidence: [JSON](iter-e647-required-slot-margin-multiseed-sweep-20260721.json).
