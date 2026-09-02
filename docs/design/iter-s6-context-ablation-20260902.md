# iter-s6 — context-tower causal ablation on the real decode path (N2)

**Date** 2026-09-02 · **Card** S6 · **Hypothesis** N2 · **Verdict: falsified**
· **Honesty: fixture-demo, not ship.** JSON mirror:
[`iter-s6-context-ablation-20260902.json`](iter-s6-context-ablation-20260902.json).

## Hypothesis and falsifier

**N2** — the frozen context tower contributes approximately nothing to
legal-set decisions at fixture scale: **flips < 5% and no quality delta**.
Falsifier as preregistered by the card: **≥ 5% flips or a measurable quality
delta**.

**The falsifier fires.** Blanking the projected context (`zero`) flips
**16.25% / 16.67% / 11.07%** of non-singleton legal-domain decisions on
smoke / held_out / adversarial, and moves `structural_similarity` by
**−0.037 / −0.046 / 0.000**. N2 is rejected: the context tower is *not*
causally inert on this decode path.

## Instrument

New default-off lever `TwoTowerConfig.context_ablation ∈ {off, zero,
shuffle_batch, shuffle_positions}` applied at the **one** decode-time seam
where the projected context tower output enters the denoiser
(`_generate_batch_once`, right after `_encode_context`). `off` returns the
caller's tensor **object** — not a copy — so the production decode is
byte-identical; `tests/test_models/test_context_ablation.py::
test_off_is_byte_identical_to_the_unablated_decode` proves it by deleting the
field from the dataclass (reproducing a pre-S6 config) and comparing outputs,
`forwards_count` and `tokens_emitted`.

Training never sees it: `training_loss` / `forward` go through
`_encode_context` directly and the ablation is applied only at the generate
call site.

**Decision-level probe.** On an ablated arm the decode is driven by the
ablated tensor and the intact tensor is carried as `ctx_shadow` into
`_generate_maskgit_one`. At every **non-singleton** legal-domain position
(`force_emit_token_id` returned `None` — where the grammar already proves the
token, context cannot matter by construction) the same
`pick_constrained_token` call is repeated against the shadow logits, over the
identical legal candidate set, and disagreements are counted:
`DecodeStats.context_ablation_applications` / `_choice_changes`. The shadow
logits are never assigned to `logits`, so the probe cannot steer a commit.

**Fail-closed.** `context_ablation` is registered in
`levers.DIAGNOSTIC_ONLY_LEVERS` with safe value `"off"` and is refused by
`require_constrained_production_config`, which guards the playground serving
config (`web/service.py`) and every evaluation policy. Registering it also
exposed a real hole: the guard compared levers with `bool(value) ==
bool(safe_value)`, and `bool("zero") == bool("off") == True`, so **any**
enum-valued lever would have passed. `_lever_value_is_safe` now compares
non-bool levers by value; boolean levers keep truthiness semantics. Strictly
fail-closed — no lever weakened, no default changed.

I6 is not weakened by the arm itself: every commit still goes through the
constrained pick, and `parse_rate == 1.00` on every arm of every suite.

## Recipe

* **Decode path** — positionwise MaskGIT `_generate_maskgit_one`, reached with
  `grammar_ltr_primary=False`. This is a diagnostic routing choice: the
  checkpoint's declared policy and `MANDATORY_GENERATION_POLICY` are
  LTR-primary, so MaskGIT is the fallback lane in a default eval. Stated
  because it bounds what these numbers cover.
* **Suites** — smoke (first 8 of 96), held_out (first 8 of 24), adversarial
  (all 4). Seeds 0 and 1. CPU, torch threads = 2. 22 runs, **0 timeouts**,
  every run inside `MAX_RUN_MINUTES = 3` (slowest 141.0 s).
* **Measurement checkpoint** — the card names
  `src/slm_training/resources/checkpoints/playground_demo/last.pt`. **It could
  not be loaded.** `TwoTowerModel.from_checkpoint` raises
  `OutputContractError: checkpoint output contract v0 is incompatible with
  required symbol_only/v2`; the payload has no `output_contract_version` key at
  all. `scripts/bootstrap_playground.py` cannot regenerate it either — its
  `DEMO_RECORDS` use non-opaque template markers and fail
  `assert_canonical_template_markers`. Both are pre-existing breakages, not
  caused by this card, and they mean **every `run_perf_matrix` /
  `run_discrete_plan_pareto` path that defaults to `PLAYGROUND_DEMO_CHECKPOINT`
  is currently unrunnable.**
* **Substitute** — a playground-scale model rebuilt under the current
  contract, scratch only and never committed: `from_records` over the 524
  records of `e937_role_safe_all_targets_v2` with
  `TwoTowerConfig(d_model=96, n_heads=4, context_layers=2, denoiser_layers=3,
  gen_steps=8, context_backend="scratch", design_md_in_context=False, seed=0)`,
  AdamW lr 3e-3, batch 8, 900 steps, no scheduler, no clipping; final loss
  4.4353. Recipe-reproducible, not bit-reproducible — CPU thread reduction
  order moves the final loss by ~0.4 between builds.

## Measured results

Flip rate = `context_ablation_choice_changes / context_ablation_applications`.
`ΔSS` is `structural_similarity` minus the same suite's `off` arm. Seeds 0 and
1 produced **identical** numbers in every arm, so one row is shown per arm and
the seed column records which seeds were run; this constrained decode is greedy
and seed-invariant.

| Suite | n | Arm | applications | flips | flip rate | ΔSS | Δmeaningful | outputs = `off` | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 8 | `off` (seeds 0,1) | 0 | 0 | — | 0.0000 | 0.0000 | yes | 41.9 / 41.3 |
| smoke | 8 | `zero` (seeds 0,1) | 1680 | 273 | **0.1625** | **−0.0373** | 0.0000 | no | 141.0 / 137.3 |
| smoke | 8 | `shuffle_batch` (0,1) | 1456 | 31 | 0.0213 | **+0.0223** | 0.0000 | no | 53.1 / 54.2 |
| smoke | 8 | `shuffle_positions` (0,1) | 1480 | 0 | **0.0000** | 0.0000 | 0.0000 | yes | 54.6 / 54.1 |
| held_out | 8 | `off` (seeds 0,1) | 0 | 0 | — | 0.0000 | 0.0000 | yes | 35.9 / 41.5 |
| held_out | 8 | `zero` (seeds 0,1) | 1680 | 280 | **0.1667** | **−0.0458** | 0.0000 | no | 140.8 / 137.4 |
| held_out | 8 | `shuffle_batch` (seed 0) | 1444 | 39 | 0.0270 | **+0.0187** | 0.0000 | no | 43.5 |
| held_out | 8 | `shuffle_positions` (seed 0) | 1441 | 0 | **0.0000** | 0.0000 | 0.0000 | yes | 51.2 |
| adversarial | 4 | `off` (seeds 0,1) | 0 | 0 | — | 0.0000 | 0.0000 | yes | 40.5 / 39.8 |
| adversarial | 4 | `zero` (seeds 0,1) | 840 | 93 | **0.1107** | 0.0000 | 0.0000 | no | 72.1 / 71.7 |
| adversarial | 4 | `shuffle_batch` (0,1) | 743 | 33 | 0.0444 | 0.0000 | 0.0000 | no | 45.7 / 46.4 |
| adversarial | 4 | `shuffle_positions` (0,1) | 717 | 0 | **0.0000** | 0.0000 | 0.0000 | yes | 44.5 / 45.1 |

`parse_rate = 1.00` in all 22 runs (I6 holds on every diagnostic arm).
`meaningful_program_rate = 0.00` in all 22 runs — see the caveat below.
`context_ablation_degenerate_rows = 0` everywhere: every batch had ≥ 2 rows and
every prompt ≥ 2 context tokens, so no arm silently failed to ablate.

Forward cost of the arms (`forwards_count`, smoke): `off` 238, `zero` 336,
`shuffle_batch` 242, `shuffle_positions` 238. The `zero` arm needs ~40% more
forwards because a blank context produces more grammar-rejected proposals that
have to be re-proposed. The probe's own shadow forwards are deliberately *not*
counted in `forwards_count` — they are not production work — and cost roughly
+25 % wall on `shuffle_*` and +140 % on `zero` (which pays for the extra real
forwards too).

## Verdict

**N2 is falsified.** Two of the three ablations move real decisions:

* `zero` (no context at all) — **11.1–16.7 % flips**, well over the 5 %
  falsifier, and a measurable `structural_similarity` delta on both non-trivial
  suites (−0.037 smoke, −0.046 held_out). Context is load-bearing.
* `shuffle_batch` (each row reads another prompt's context) — 2.1–4.4 % flips,
  under the flip threshold, **but** a measurable quality delta on smoke
  (+0.0223) and held_out (+0.0187). The falsifier is disjunctive, so this arm
  falsifies N2 on the quality limb alone. The delta being *positive* is not a
  win: at this fixture scale `structural_similarity ≈ 0.10–0.16` is near the
  floor and a mismatched prompt happens to nudge the model toward shapes that
  score marginally better against the gold. It is evidence that the tower's
  signal is present and poorly aimed, not that scrambling helps.
* `shuffle_positions` — **exactly 0 flips in all six runs, 4 138 probed
  decisions**, and byte-identical output. This is the expected result, not a
  finding: cross-attention over context keys is permutation-invariant, so
  reversing the context sequence *cannot* change any decision. It is the
  instrument's null control, and it passing means the 273/280/93 flips above
  are not probe artefacts.

## Caveats — what these numbers do not say

1. **Fixture-demo, never ship.** A 785 k-parameter, 900-step CPU model on 8–8–4
   records. No ship gate is touched, no promotion is implied.
2. **`meaningful_program_rate` is 0.00 on every arm including `off`.** That
   metric has no headroom here, so its zero delta is uninformative — the
   ablation could not have moved it in either direction. Only
   `structural_similarity` and the flip probe carry signal.
3. **MaskGIT-only.** The LTR-primary / compiler-tree lane that a default
   evaluation actually takes was not probed; `ctx_shadow` is wired into
   `_generate_maskgit_one` alone.
4. **Flips are decisions, not outcomes.** A flipped position can still be
   remasked by the stream checker or overwritten in constrained LTR repair, and
   the adversarial suite shows exactly that: 11.07 % flips with a 0.0000 quality
   delta and identical output. Flip rate upper-bounds, never predicts, the
   quality effect.
5. **The named measurement checkpoint is unloadable** (see Recipe). Every claim
   here is about the substitute model.

## Interface notes

* `forced_tokens` stayed **0** in all 22 runs while
  `semantic_singleton_bypasses` was 14–32. `_record_exact_bypass` fires on the
  constrained-LTR-repair lane, which increments
  `forced_row_tokens_without_forward` but never `forced_tokens`; only the
  MaskGIT one-hole `exact_commit` branch increments `forced_tokens`. S4's
  `forced_token_fraction` therefore reads 0.0 on a decode that did bypass 32
  positions. Not changed here (S4 owns those properties) — flagged.
* The enum-lever truthiness hole in `constraint_weakening_violations` was live
  for every non-bool lever, not just this one.
