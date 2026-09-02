# iter-s5 — decode-only authority ladder (N4)

**Date** 2026-09-02 · **Card** S5 · **Hypothesis** N4 · **Verdict: N4 holds
(falsifier did not fire)** · **Honesty: fixture-demo scratch-model diagnostic,
not ship.** JSON mirror:
[`iter-s5-decode-authority-ladder-20260902.json`](iter-s5-decode-authority-ladder-20260902.json).
Preregistered campaign:
[`campaign.v1.json`](../../src/slm_training/resources/experiments/decode_only_authority_ladder/campaign.v1.json)
(`ExperimentCampaignV1`, `manifest_sha256`
`8338a5b283f367a9eafbe2de8e25ba2ed378891cf7f7a42dbc5f6addef385425`).

No checkpoint was created, synced, bootstrapped or promoted; no gate threshold,
default, or production config changed; the only repository changes are this
doc, its JSON mirror, the campaign manifest, matrix rows, the regenerated
evidence ledger and a version-stamp history note. **No production code was
touched** — every arm is a decode-time config setting that already exists.

## Hypothesis and falsifier

**N4** — under `block_diffusion_decode=True` the left-prefix `admit_fill`
false-admit rate is non-zero, i.e. `block_joint_rejections > 0` on the fixture,
unlike the 0-rejection rate S14 measured on sequential positionwise commits
(0 of 2,132 real probes,
[`iter-s14-exhaustion-rate-20260902.md`](iter-s14-exhaustion-rate-20260902.md)).
**Falsifier:** zero joint rejections across block sizes {2, 4, 8}.

**The falsifier did not fire. N4 holds.**

| | probe canvases | admit-probe rejections | `block_joint_rejections` |
| --- | ---: | ---: | ---: |
| `m1_left_prefix_admit` (sequential) | 2,554 | **0** | **0** |
| `m2_block_diffusion_b2` | 5,334 | 186 | **146** |
| `m2_block_diffusion_b4` | 8,362 | 184 | **80** |
| `m2_block_diffusion_b8` | 7,846 | 240 | **162** |

Totals over 3 suites × 2 seeds. The exact `multi_region_support` joint check
proved a parallel-commit canvas impossible and reverted the step 388 times
across the three block sizes, on a configuration where the same admit machinery
rejected nothing at all when commits were sequential.

The mechanism is visible in the probe shape. S14's explanation for the 0/2,132
sequential rejection rate was that **every** probe canvas carried a committed
token after the first hole — the suffix-blind span `admit_fill` structurally
cannot validate — so the probe was a pure over-approximation. That share drops
under block diffusion:

| arm | probes with committed suffix / probes (smoke, seed 0) |
| --- | --- |
| `m1_left_prefix_admit` | 514 / 544 = **94.5 %** |
| `m2_block_diffusion_b2` | 1034 / 1228 = 84.2 % |
| `m2_block_diffusion_b4` | 1280 / 1665 = 76.9 % |
| `m2_block_diffusion_b8` | 1098 / 1664 = **66.0 %** |

Block scheduling reveals contiguous spans, which produces probe canvases whose
left prefix is clean — and those are exactly the canvases that can reject.

## Preregistration

The campaign manifest was written and its `CampaignLockV1` digest recorded
**before any endpoint was read**, at `source_commit`
`d64d65001daf169f93fef13848e46f92fc980126`, `source_dirty=false`. Two path-
scoped ladders, nine arms, six candidates, `selection_rule =
best_by_primary_then_smallest` (the only value the contract allows, and the one
several sibling manifests were red for omitting).

**Disclosure — wiring probes preceded the lock.** Five `n=2` smoke decodes
(one per lever family) were run before the manifest was written, purely to
confirm each config routes to the intended lane. They produced no campaign
endpoint number and none of the numbers below descend from them; they are
recorded here rather than omitted. Everything in the measured-results tables
comes from the 54 post-lock cells.

**Contract deviation, stated.** The card specifies
`claim_class="fixture_or_scratch"`. `ExperimentCampaignV1.ClaimClass` admits
only `wiring | fixture | diagnostic | screening | promotion_candidate |
ship_gate`, so the manifest carries `claim_class="fixture"` — the nearest legal
literal — and the scratch-model caveat is carried in the doc and the results
JSON instead. **Manifest location:** the card names
`src/slm_training/resources/campaigns/…`; that directory does not exist. The
canonical home for an `ExperimentCampaignV1` manifest in this repository is
`src/slm_training/resources/experiments/<experiment_id>/campaign.v1.json`
(precedent: `recursive_adaptive_depth_h7`), and the manifest is there.

### Ladders

The two decode paths are disjoint, so every candidate is scored against **its
own** ladder's control and the two ladders are never pooled
(`kill:cross_ladder_comparison`).

**Ladder M — positionwise MaskGIT** (`grammar_ltr_primary=False`,
`compiler_decode_mode="off"`, `unmask_mode="positions"`, `gen_steps=8`):

| arm | role | lever |
| --- | --- | --- |
| `m0_admit_probes_off` | **diagnostic negative control** (`mechanism_off_arm_ids`, never a candidate — it is a legality-weakening configuration) | `grammar_fastpath_mode="force"` |
| `m1_left_prefix_admit` | ladder-M control | `grammar_fastpath_mode="mask"` |
| `m2_block_diffusion_b{2,4,8}` | candidates | `+ block_diffusion_decode=True`, `block_diffusion_block_size ∈ {2,4,8}`, exact `multi_region_support` joint check |

**Ladder L — compiler LTR** (`grammar_ltr_primary=True`,
`compiler_decode_mode="tree"`):

| arm | role | lever |
| --- | --- | --- |
| `l1_singleton_only` | ladder-L control | `speculative_rank="off"` |
| `l2_ngram_margin_{0p5,1p0,2p0}` | candidates | `speculative_rank="ngram"` over the committed table `resources/decode/speculative_ngram_v1.json`, `speculative_rank_margin ∈ {0.5, 1.0, 2.0}` |

### Endpoints (locked)

* **Primary — `legal_choice_flip_rate`.** Index-aligned per-position
  disagreement between an arm's emitted canvas and its **own ladder control's**,
  restricted to **legal-choice** positions: positions `t` of the control's
  emitted token sequence where `force_emit_token_id(tokenizer, control_ids[:t])`
  returns `None` — i.e. positions the grammar does not already prove, where a
  choice was actually available. The predicate is the same one `_propose` uses
  in the decoder; it is evaluated **outside** the model from the emitted ids, so
  no decode path is instrumented and no production code changed. Positions the
  DFA proves are excluded by construction and cannot flip.
* **Secondary — `forwards_saved_at_matched_output`**: control `forwards_count`
  minus arm `forwards_count`, defined **only** when every record's emitted
  program is byte-identical to the control's. Otherwise reported as
  `null` with the raw `forwards_delta_all_records` and the matched-record count,
  because a forward delta at different output is not a saving.
* **Secondary gates (non-regression):** `meaningful_program_rate`,
  `structural_similarity`, `placeholder_fidelity` — each as a delta against the
  arm's own ladder control.
* **Secondary telemetry:** `forwards_count`, `forced_tokens`,
  `speculative_rank_*`, `block_joint_rejections`, `admit_probe_*`.

## Recipe

* **Suites** — `e938_role_safe_all_targets_smoke96_v2` smoke (first 8 of 96)
  and held_out (first 8 of 24); `e938_role_safe_all_targets_v2` adversarial
  (all 4). Seeds 0 and 1.
* **Runs** — 9 arms × 3 suites × 2 seeds = **54 bounded subprocesses**, each
  `timeout 170`, CPU, `torch.set_num_threads(2)`. **0 timeouts**; slowest run
  **75.98 s**, well inside `MAX_RUN_MINUTES = 3`. There are no timeout rows.
* **Seed invariance** — seeds 0 and 1 produced **identical** numbers in every
  one of the 27 arm × suite pairs. This constrained decode is greedy and
  seed-invariant, exactly as S6 found; one row per arm × suite is shown below
  and covers both seeds.

### Measurement checkpoint — named, and honest about it

The card names
`src/slm_training/resources/checkpoints/playground_demo/last.pt`. **It cannot
be loaded.** `TwoTowerModel.from_checkpoint` raises `OutputContractError:
checkpoint output contract v0 is incompatible with required symbol_only/v2`;
`scripts/bootstrap_playground.py` cannot regenerate it either. Both breakages
predate this card and are already documented by S6 and S14.

Following that precedent, every number here comes from an **uncommitted
session-scratch twin**: `from_records` over the 524 records of
`e937_role_safe_all_targets_v2` with `TwoTowerConfig(d_model=96, n_heads=4,
context_layers=2, denoiser_layers=3, gen_steps=8, context_backend="scratch",
design_md_in_context=False, seed=0)`, AdamW `lr=3e-3`, batch 8, 900 steps, no
scheduler, no clipping — **818,210 parameters, final loss 4.8031, 47.6 s to
build.** Recipe-reproducible, not bit-reproducible (CPU reduction order moves
the final loss between builds). It is undertrained, and it is not the shipped
model; every claim below is about this twin.

## Measured results — ladder M (positionwise MaskGIT)

Deltas are against `m1_left_prefix_admit` on the same suite. `flips` /
`legal-choice positions` is the primary endpoint. `fwd Δ` is
`forwards_count(control) − forwards_count(arm)` over **all** records; it is not
a matched-output saving except on the `m0` rows, where every record matched.

| arm | suite | n | flips / legal-choice pos | flip rate | matched outputs | fwd Δ (all rec.) | fwd saved @ matched | Δ meaningful | Δ SS | Δ placeholder | `block_joint_rejections` | `admit_probe_rejections` | s/run |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `m0_admit_probes_off` | smoke | 8 | 0 / 79 | **0.0000** | 8/8 | 0 | **0** | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 32.8 / 32.4 |
| `m0_admit_probes_off` | held_out | 8 | 0 / 81 | **0.0000** | 8/8 | 0 | **0** | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 32.3 / 31.6 |
| `m0_admit_probes_off` | adversarial | 4 | 0 / 36 | **0.0000** | 4/4 | 0 | **0** | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 13.3 / 13.2 |
| `m2_block_diffusion_b2` | smoke | 8 | 31 / 79 | 0.3924 | 5/8 | 8 | n/a | 0.0000 | **−0.0865** | 0.0000 | **23** | 39 | 32.7 / 32.5 |
| `m2_block_diffusion_b2` | held_out | 8 | 20 / 81 | 0.2469 | 6/8 | 0 | n/a | 0.0000 | +0.0427 | 0.0000 | **26** | 24 | 28.2 / 28.4 |
| `m2_block_diffusion_b2` | adversarial | 4 | 0 / 36 | 0.0000 | 3/4 | 8 | n/a | 0.0000 | **−0.1083** | 0.0000 | **24** | 30 | 17.5 / 16.4 |
| `m2_block_diffusion_b4` | smoke | 8 | 23 / 79 | 0.2911 | 5/8 | 6 | n/a | 0.0000 | +0.0106 | 0.0000 | **21** | 35 | 39.2 / 40.2 |
| `m2_block_diffusion_b4` | held_out | 8 | 24 / 81 | 0.2963 | 5/8 | 8 | n/a | 0.0000 | +0.0713 | 0.0000 | **11** | 41 | 70.4 / 65.1 |
| `m2_block_diffusion_b4` | adversarial | 4 | 4 / 36 | 0.1111 | 2/4 | 16 | n/a | 0.0000 | **−0.1140** | 0.0000 | **8** | 16 | 25.7 / 25.3 |
| `m2_block_diffusion_b8` | smoke | 8 | 23 / 79 | 0.2911 | 5/8 | 3 | n/a | 0.0000 | +0.0271 | 0.0000 | **46** | 54 | 26.2 / 25.5 |
| `m2_block_diffusion_b8` | held_out | 8 | 20 / 81 | 0.2469 | 6/8 | 0 | n/a | 0.0000 | +0.0437 | 0.0000 | **35** | 58 | 25.2 / 24.5 |
| `m2_block_diffusion_b8` | adversarial | 4 | 6 / 36 | 0.1667 | 1/4 | 24 | n/a | 0.0000 | **−0.1412** | 0.0000 | **0** | 8 | 68.0 / 65.0 |

Control absolutes (`m1_left_prefix_admit`, per seed): `structural_similarity`
0.4106 / 0.3413 / 0.3481 on smoke / held_out / adversarial;
`meaningful_program_rate` 0.00 and `placeholder_fidelity` 0.00 on all three;
`forwards_count` 64 / 64 / 36; `admit_probe_canvases` 544 / 503 / 230;
`admit_probe_rejections` **0** everywhere; `block_joint_rejections` **0**
everywhere.

`block_joint_unknowns` (budget-limited verdicts, fail-open by design and counted
so the open share stays visible): b2 7 / 0 / 7, b4 2 / 8 / 2, b8 0 / 5 / 14 on
smoke / held_out / adversarial. `admit_probe_reject_run_max` was **1** on every
M2 cell — rejections never ran consecutively, so S14's `N12-tau` restart trigger
still has no evidence even here.

## Measured results — ladder L (compiler LTR)

Deltas are against `l1_singleton_only` on the same suite.

| arm | suite | n | flips / legal-choice pos | flip rate | matched outputs | fwd Δ (all rec.) | fwd saved @ matched | Δ meaningful | Δ SS | Δ placeholder | `spec_rank` eval / commit / declined | s/run |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `l2_ngram_margin_0p5` | smoke | 8 | 16 / 72 | 0.2222 | 0/8 | +32 | n/a | **+0.500** | **−0.0118** | 0.0000 | 168 / 136 / 32 | 28.3 / 27.0 |
| `l2_ngram_margin_0p5` | held_out | 8 | 16 / 72 | 0.2222 | 0/8 | +32 | n/a | **+0.375** | +0.0848 | 0.0000 | 168 / 136 / 32 | 28.6 / 28.2 |
| `l2_ngram_margin_0p5` | adversarial | 4 | 8 / 36 | 0.2222 | 0/4 | +16 | n/a | **+0.500** | +0.0566 | 0.0000 | 84 / 68 / 16 | 15.2 / 14.0 |
| `l2_ngram_margin_1p0` | smoke | 8 | 18 / 72 | 0.2500 | 0/8 | −28 | n/a | 0.0000 | **−0.1233** | 0.0000 | 242 / 150 / 92 | 46.8 / 49.3 |
| `l2_ngram_margin_1p0` | held_out | 8 | 18 / 72 | 0.2500 | 0/8 | −35 | n/a | 0.0000 | **−0.0539** | 0.0000 | 258 / 159 / 99 | 47.9 / 48.6 |
| `l2_ngram_margin_1p0` | adversarial | 4 | 10 / 36 | 0.2778 | 0/4 | −5 | n/a | **+0.250** | +0.0631 | 0.0000 | 85 / 58 / 27 | 17.4 / 17.4 |
| `l2_ngram_margin_2p0` | smoke | 8 | 16 / 72 | 0.2222 | 0/8 | −22 | n/a | 0.0000 | **−0.0053** | 0.0000 | 154 / 68 / 86 | 35.3 / 34.4 |
| `l2_ngram_margin_2p0` | held_out | 8 | 16 / 72 | 0.2222 | 0/8 | −44 | n/a | 0.0000 | +0.0796 | 0.0000 | 188 / 80 / 108 | 42.8 / 44.4 |
| `l2_ngram_margin_2p0` | adversarial | 4 | 8 / 36 | 0.2222 | 0/4 | −6 | n/a | 0.0000 | +0.0390 | 0.0000 | 67 / 29 / 37 | 15.2 / 15.6 |

Control absolutes (`l1_singleton_only`, per seed): `structural_similarity`
0.3979 / 0.3605 / 0.3481; `meaningful_program_rate` 0.00 and
`placeholder_fidelity` 0.00 on all three; `forwards_count` 64 / 64 / 32;
`forced_tokens` 56 / 56 / 28 of 96 / 96 / 48 committed tokens;
`speculative_rank_evaluations` **0** everywhere (the ranker is genuinely off).

**No `forwards_saved_at_matched_output` number exists on ladder L.** Every L2
arm changed the emitted program on every record (0/8, 0/8, 0/4 matched), so the
forward deltas above are not savings — they are the cost of decoding a different
program. `l2_ngram_margin_0p5` halves `forwards_count` (32 vs 64 on smoke)
while committing **280** tokens against the control's 96, so per committed token
it is roughly 6× cheaper in forwards; but the output is a different program, so
that is a throughput observation, not a like-for-like saving.

## What the numbers say

1. **N4 holds; the falsifier did not fire.** `block_joint_rejections` is
   non-zero on 8 of the 9 `m2` arm × suite cells and on every block size
   ({2,4,8} → 146 / 80 / 162 over both seeds), against **0** on every
   `m1_left_prefix_admit` cell over 2,554 probes. The `kill:m1_joint_rejection`
   criterion — which would have voided the campaign had the joint check fired on
   sequential commits — never tripped.
2. **The single zero is a small-n cell, not a counter-example.**
   `m2_block_diffusion_b8` on adversarial (n=4) recorded 0 joint rejections but
   **14 `block_joint_unknowns`** — the joint check ran out of node budget and
   failed open rather than proving anything. That is a budget outcome, not a
   clean canvas, and it is the only cell where block-8 committed as few as 12
   tokens. It does not falsify anything; it says the exact check is not free.
3. **The admit probe is causally inert on the sequential path.**
   `m0_admit_probes_off` — probes off entirely — flipped **0 of 196
   legal-choice positions**, produced byte-identical output on **20 of 20**
   records, and cost exactly the same `forwards_count` as `m1`. This is the
   clean matched-output measurement the campaign asked for:
   `forwards_saved_at_matched_output = 0`. Combined with S14's 0/2,132
   rejections, the left-prefix probe on this configuration changes nothing and
   saves nothing when commits are sequential — its value only appears once
   parallel commits give it a clean prefix to reason about.
4. **Every candidate is decision-bearing, and most trip a gate somewhere.**
   Flip rates are 0.11–0.39 (ladder M, excluding the b2/adversarial 0.0000 cell)
   and 0.22–0.28 (ladder L) — far above the locked 0.01 threshold, so no arm is
   inert. But `structural_similarity` regressed against the arm's own control on
   9 of the 18 candidate cells, tripping `kill:gate_regression` for
   `m2_b2` (smoke, adversarial), `m2_b4` (adversarial), `m2_b8` (adversarial),
   `l2_0p5` (smoke), `l2_1p0` (smoke, held_out) and `l2_2p0` (smoke).
   **No arm is promotable from this campaign, and none was ever eligible to be:
   `claim_class` is fixture.**
5. **Ladder L's margin is non-monotonic.** `speculative_rank_commits /
   evaluations` falls cleanly with the margin (0.81 → 0.62 → 0.44 on smoke), as
   the lever intends, but quality does not follow: margin 0.5 is the only arm
   that moves `meaningful_program_rate` off the floor on all three suites, while
   margin 1.0 is the worst `structural_similarity` arm in the whole campaign
   (−0.1233 on smoke) and costs 44 % **more** forwards than the control. A
   tighter margin is not monotonically safer here.

## Caveats — what these numbers do not say

1. **Fixture-demo, never ship.** An 818 k-parameter, 900-step CPU twin on
   8 / 8 / 4 records, two seeds. No ship gate is evaluated, no checkpoint is
   written, no promotion is implied. The named measurement checkpoint is
   unloadable (see Recipe); every claim is about the substitute.
2. **`placeholder_fidelity` is 0.00 on every arm of both ladders, including
   both controls.** Every Δ is exactly 0.0000. That metric has no headroom here,
   so **its zero delta is uninformative — it is not a pass.**
3. **`meaningful_program_rate` is 0.00 on every ladder-M arm including the
   control**, so the ladder-M zero deltas are likewise uninformative and are not
   passes. On ladder L the picture is different from what S6/S14 measured on the
   MaskGIT lane: the control is still 0.00, but `l2_ngram_margin_0p5` reaches
   0.50 / 0.375 / 0.50 and `l2_ngram_margin_1p0` reaches 0.25 on adversarial.
   The metric **does** have upward headroom on the compiler-LTR lane, so those
   positive deltas are real signal — but a *zero* delta on ladder L is still
   uninformative, because the control sits on the floor and nothing can go
   below it.
4. **Flips are decisions, not outcomes.** A flipped position can still be
   remasked, reverted by the joint check, or overwritten in LTR repair. The
   `m2_b2`/adversarial cell shows the converse: 0.0000 flips at legal-choice
   positions yet `ΔSS = −0.1083`, because the arm changed the emitted program's
   *length* on 1 of 4 records and the primary endpoint only compares
   index-aligned positions. Flip rate neither predicts nor bounds the quality
   effect when lengths diverge; the matched-output and length-mismatch columns
   are reported for exactly that reason.
5. **The primary endpoint is measured on re-encoded output, not on live commit
   decisions.** Flips are computed by re-encoding each arm's emitted program
   with the model's own tokenizer and comparing index-aligned positions. This
   deliberately avoids instrumenting the decoder (unlike S6's `ctx_shadow`
   probe), at the cost of being blind to a decision that was made and then
   overwritten before emission. It under-counts, never over-counts, real
   decision changes.
6. **Two lanes, never pooled.** Ladder M numbers and ladder L numbers are not
   comparable and were never combined; `l1_singleton_only` and
   `m1_left_prefix_admit` happen to share `structural_similarity` 0.3481 on
   adversarial by coincidence, not by construction.

## Invariants

I1–I6 are untouched. No production code changed, so nothing could be weakened.
Specifically: every commit on every arm still goes through the constrained pick
(I6); the deterministic/singleton bypass is what `l1_singleton_only` *is*, and
no arm reorders it below a learned score (I2); speculation on the `l2` arms is
the existing forest-verified path that re-derives the domain and verifies before
commit (I3); no arm adds a parameter, so `EG_params` is not in play and the
"capability is never bought with parameters" law is not engaged. The
`m0_admit_probes_off` arm is a legality-*weakening* configuration and is
therefore declared in `mechanism_off_arm_ids` and permanently barred from
candidacy.

## Interface notes

* **`forced_tokens` is 0 on the entire MaskGIT ladder** (all 30 `m0`/`m1`/`m2`
  cells) while it is 28–245 on the compiler-LTR ladder. S6 flagged the same
  asymmetry: `_record_exact_bypass` on the constrained-LTR repair lane
  increments `forced_row_tokens_without_forward`, not `forced_tokens`, and only
  the MaskGIT one-hole `exact_commit` branch increments `forced_tokens` — which
  this configuration never reaches. Any consumer reading
  `forced_token_fraction` as "share of positions the grammar proved" gets 0.0
  from a MaskGIT decode that did prove positions. Not changed here (S4 owns
  those properties); re-flagged with a second, independent set of runs.
* **`docs/design/quality-experiment-matrix.md` contained committed merge
  conflict markers** (`<<<<<<< HEAD` at line 6541, `=======` at 6575,
  `>>>>>>> worktree-agent-a75a3c186b0c92e54` at 6616), introduced by merge
  commit `7bdc348` on the integration branch, not by this card. Both sides were
  additive append-only sections from different cards (S14, and S6/S12), so the
  three marker lines were removed and **both** sections kept in full. No content
  was dropped.
* **`src/slm_training/resources/campaigns/` does not exist.** If that path is
  intended to become canonical for `ExperimentCampaignV1` manifests, nothing
  currently reads it and `versions.json` does not watch it; the
  `resources/experiments/<experiment_id>/` convention is what the contract's
  existing manifests and the version registry use.
* **`admit_probe_reject_run_max` never exceeded 1** on any arm of this campaign,
  including all nine block-diffusion cells. S14's preregistered `N12-tau` gate
  (a corpus with `max_consecutive_rejection_run >= 2` on ≥ 10 % of documents)
  is still unmet, even now that real admit rejections exist.
