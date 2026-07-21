# RSC-A05: matched recursive control arms (SLM-241)

Run id: `iter_rsc_a05_matched_recursive_control_arms`
Status: **partial_implementation** (wiring/resource-accounting only; no quality claim)
Date: 2026-07-21 (arm F added in a same-day follow-up iteration; arm E added
in a second same-day follow-up -- only arm H remains deferred)

## What this is

Linear SLM-241 (RSC-A05) asks for eight architecture control arms (A-H) plus a
fair-initialization contract and full resource accounting, so a future matched
recursive-depth quality campaign (SLM-233, **out of scope here**) can attribute
a result to one resource dimension at a time. The historical SLM-138 fixture
compares a 64,994-parameter stacked model with a 74,242-parameter recursive
model and cannot attribute any result to recurrence, because the recursive arm
also adds `z_latent`/`ctx_proj` (SLM-240/RSC-A04 already made this delta
explicit; it is unchanged here).

This is a genuinely large issue. Per its own prioritization ("implement A, B,
G first ... then C and D ... treating E/F/H as stretch goals"), an earlier
same-day iteration built **A, B, C, D, G** fully (each constructs through the
canonical factory, trains one step, and round-trips a checkpoint) and
explicitly deferred **E, F, H** — never fabricated, never half-implemented.
A first follow-up iteration built **F** (unshared depth-matched tower), per
that iteration's own recommendation (cheapest of the three deferred arms,
reuses `DenoiserTower`'s existing structure). This second follow-up iteration
builds **E** (stacked + matched state capacity) — the mirror image of D: a
new `StackedMatchedStateDenoiserTower` class (unshared, non-recursive blocks,
same block-evaluation count as A) with a learned `state`/`state_ctx_proj`
pair shape-matched to B's `z_latent`/`ctx_proj`, injected once before the
transition blocks run. Only **H** (stop-gradient recurrence) remains
deferred.

## Built arms

| Arm | `denoiser_arch` | `z_state_mode` | What it isolates |
| --- | --- | --- | --- |
| **A** | `stacked` | n/a | Existing baseline: no z state, unshared blocks. Already existed; now a named, reportable arm. |
| **B** | `shared_recursive` | `full` | Existing V1: shared transition blocks + explicit learned z state. Already existed; now a named, reportable arm. |
| **C** | `shared_recursive_y_only` | `y_only` | Shared repeated depth **without** the y/z split — both the F- and G-update layers run on `y` alone each recursion step; no `z_latent`/`ctx_proj` tensor exists at all. |
| **D** | `shared_recursive_no_extra_capacity` | `parameter_free` | Keeps the y/z split structurally, but `z`'s initial value is a deterministic pooled-context broadcast (no learned `max_len` bank, no learned projection) — removes exactly the two parameter tensors `recursive_zstate_parameter_delta` accounts for. |
| **E** | `stacked_matched_state` | n/a (no z state, same as A) | Stacked + matched state capacity: a new `StackedMatchedStateDenoiserTower` class — the same unshared, non-recursive block structure as A (`n_layers` blocks, each called once) — plus a learned `state`/`state_ctx_proj` pair shape-matched to B's `z_latent`/`ctx_proj`, injected into the initial hidden state exactly once (never recurrently re-applied). Mirror image of D: D removes the z-state parameters from a recursive tower; E adds them (once) to a non-recursive one. Its parameter delta over a same-`n_layers` arm A equals `recursive_zstate_parameter_delta(d_model, max_len)` **exactly** — see "Arm E: matched-state formula + gradient-consumption test" below. |
| **F** | `stacked_depth_matched` | n/a (no z state, same as A) | Unshared depth-matched tower: the exact same `DenoiserTower` class as A, no new tower code, built with `recursive_steps * recursive_transition_layers` independent transition blocks instead of `n_layers`. Isolates weight sharing from block-evaluation count — same total block evaluations as B, no weight sharing at all (vs B's fully shared transition). **Necessarily has MORE parameters than B** (nothing is shared); see "Arm F: two honest matching views" below. |
| **G** | `shared_recursive` | `full` | Same constructor as B with `recursive_steps` forced to 1 — an architecture-change control. Interface-compatible with A, **not** behaviorally equivalent (SLM-240's framing, reused, not re-derived). |

## Deferred arms (explicit, not fabricated)

| Arm | Why deferred |
| --- | --- |
| **H** — stop-gradient recurrence | Needs `y`/`z` detached between recursive steps inside `recursive_outputs`, plus a paired test proving identical forward values but divergent cross-step gradient paths vs the non-detached B. Not built yet. |

`slm_training.models.recursive_control_arms.construct_arm_tower` raises
`NotImplementedError` (never a silent no-op or a fabricated construction) for
both — see `tests/test_models/test_recursive_denoiser.py::test_deferred_arms_fail_closed_not_silently_built`.

## Resource accounting: real, measured `RecursiveControlArmReportV1` table

Built with `vocab_size=23, d_model=32, n_layers=recursive_transition_layers=2,
n_heads=2, max_len=256, recursive_steps=2` (same denoiser-only config as the
SLM-138 fixture) — tower-level, not whole-`TwoTowerModel` (the context tower is
identical across arms and cancels out of any delta, same convention as
RSC-A04). Full instance in the sibling JSON's `control_arm_table` key.

| Arm | Total params | Δ vs A | Block evals | Est. FLOPs | Matched? |
| --- | --- | --- | --- | --- | --- |
| A | 42,752 | +0 | 2 | 304,128 | true (self) |
| B | 52,000 | +9,248 (+21.63%) | 4 | 608,256 | false (no target declared) |
| C | 42,752 | +0 | 4 | 608,256 | **true** |
| D | 42,752 | +0 | 4 | 608,256 | **true** |
| E | 52,000 | +9,248 (+21.63%) | 2 | 304,128 | delta-vs-B-formula **true**; delta-vs-A false (no target declared) |
| F | 76,544 | +33,792 (+79.04%) | 4 | 608,256 | block-evals **true** (vs B); params false |
| G | 52,000 | +9,248 (+21.63%) | 2 | 304,128 | false (no target declared) |

Notes:

- **C and D exactly match A's parameter count** (residual = 0) at this
  `recursive_transition_layers == n_layers` configuration — this is the
  concrete, measured result of removing `z_latent`/`ctx_proj` (C) or making
  the z-state parameter-free (D). Neither carries any undeclared z-state
  parameter (`undeclared_zstate_parameter_names == []` for both, enforced by
  `RecursiveControlArmReportV1.__post_init__` raising if it is ever non-empty
  for arm C/D).
- **B and G's +9,248-parameter delta is exactly
  `recursive_zstate_parameter_delta(d_model=32, max_len=256)`** — the same
  formula RSC-A04 introduced, cross-checked against the real measured delta
  in `parameter_delta_formula_check` (sibling JSON). No matching target is
  declared for B/G: they are not intended to parameter-match A.
- **E's +9,248-parameter delta over A is also exactly
  `recursive_zstate_parameter_delta(d_model=32, max_len=256)`** — the same
  formula, the same numeric value as B/G's delta, because E's `state`/
  `state_ctx_proj` tensors are shape-matched to B's `z_latent`/`ctx_proj`
  by construction (not by coincidence). Unlike B/G, E's declared matching
  target is real: "match arm B's parameter delta over A", not "match A's raw
  total" (that is C's/D's kind of target) — see "Arm E: matched-state
  formula + gradient-consumption test" below.
- **F's row above is the block-evaluation-matched construction only**
  (`n_layers = recursive_steps * recursive_transition_layers = 4`, exactly
  B's block-evaluation count) — its +33,792-parameter delta over A (+24,544
  over B) is real, measured, and reported plainly, never hidden. See "Arm F:
  two honest matching views" below for the paired parameter-nearest
  construction and its own (nonzero) block-evaluation residual.
- **Block evaluations**: A performs `n_layers=2`; B performs
  `recursive_steps * recursive_transition_layers = 4`; G forces
  `recursive_steps=1`, so it performs `2` — the *same* block-evaluation count
  as A despite being architecturally distinct (shared blocks + z state vs
  unshared blocks, no z state). E performs `n_layers=2` (same dial as A,
  never `recursive_steps * recursive_transition_layers` — that is F's dial),
  so it matches A's block-evaluation count *and* has an added, real,
  measured parameter cost — the two dimensions E is built to keep
  independent. F performs `4`, matching B, by construction.
  Self-attention/cross-attention/MLP call counts are identical to the
  block-evaluation count in this codebase (every `TransformerBlock` call
  always performs exactly one of each).
- **Estimated FLOPs** are the same analytic per-block estimator RSC-A04
  introduced (`estimate_transformer_block_flops`, now public) — an explicit
  relative-cost proxy, never a profiler measurement or latency claim.
  Profiler-measured FLOPs / peak activation memory / wall time remain a
  stretch goal, not attempted this session.
- **No field anywhere is named `parity` or `winner`** —
  `RecursiveControlArmReportV1.__post_init__` raises if either is present.

## Arm F: two honest matching views

`DenoiserTower` (arm F's tower) has exactly one free dial, `n_layers`, so it
cannot simultaneously match arm B's block-evaluation count *and* its
parameter count at this fixture scale. `build_arm_f_dual_view` builds and
measures **both** real constructions instead of asserting one is "matched" —
see the sibling JSON's `arm_f_dual_view` key for the full instance.

Per-layer parameter cost is derived from two real constructed 1-layer/2-layer
`DenoiserTower` instances (never a hard-coded constant): `16,896` parameters
per independent transition block, `8,960` common (non-block) parameters —
same numbers `recursive_zstate_parameter_delta`'s sibling accounting already
implies (`42,752 = 8,960 + 2 * 16,896`).

| View | `n_layers` | Block evals | Δ block evals vs B | Total params | Δ params vs B |
| --- | --- | --- | --- | --- | --- |
| `block_evaluation_matched` (the `control_arm_table` "F" row above) | 4 | 4 | 0 | 76,544 | +24,544 (+47.20%) |
| `parameter_nearest` (a separate construction, reported only here) | 3 | 3 | **-1** (25% fewer block evaluations than B) | 59,648 | **+7,648** (+14.71%) |

Neither row is a "matched" claim on both dimensions at once:

- `block_evaluation_matched` is exact on block evaluations (target: B's `4`);
  its parameter residual (+24,544 vs B) is real, measured, and reported
  plainly.
- `parameter_nearest` picks the integer `n_layers` (found from the measured
  per-layer/common-parameter formula above, `round((52,000 - 8,960) /
  16,896) = 3`) whose real total parameter count is closest to B's 52,000 —
  closer than the block-evaluation-matched view's residual (+7,648 vs
  +24,544) — but its block-evaluation count (3) is honestly reported as `-1`
  relative to B's `4`, never hidden or rounded away.

`construct_arm_tower("F", ...)` and the `control_arm_table` row always return
the `block_evaluation_matched` construction (the primary, block-evaluation-
matched view); `parameter_nearest` is a real, measured, but *separate*
construction available only through `build_arm_f_dual_view`.

## Arm E: matched-state formula + gradient-consumption test

Arm E (`StackedMatchedStateDenoiserTower`, `denoiser_arch=
"stacked_matched_state"`) is the mirror image of D: D keeps the recursive
y/z split but strips the z-state parameters; E keeps the stacked baseline's
unshared, non-recursive block structure but adds z-state-shaped parameters
back, consumed **once** rather than every recursion step.

**Construction.** `n_layers` independent `TransformerBlock` instances (same
`DenoiserTower` block loop, same `n_layers` dial arm A uses — never
`recursive_steps * recursive_transition_layers`, that is arm F's dial), plus:

```text
y_0 = token + position + kind (+ symbol features)
y_0 = y_0 + state[position] + state_ctx_proj(mean_pool(context))
for block in layers:                # n_layers blocks, each called exactly once
    y = block(y, context)            # no recurrence, no re-injection
```

`state` is `[max_len, d_model]` (indexed by target position, same shape as
B's `z_latent`); `state_ctx_proj` is `Linear(d_model, d_model)` (same shape
as B's `ctx_proj`). Both are injected exactly once, before the first
transition block runs — never inside the block loop, never re-applied per
layer or per recursion step (E has no recursion at all).

**Parameter-match verification (real, measured, this session).** At the
fixture config (`vocab_size=23, d_model=32, n_layers=2, max_len=256`):

| Quantity | Value |
| --- | --- |
| Arm A total parameters | 42,752 |
| Arm E total parameters | 52,000 |
| Measured Δ (E − A) | 9,248 |
| `recursive_zstate_parameter_delta(d_model=32, max_len=256)` | 9,248 |
| Δ matches formula? | **true** (exact, not approximate) |
| Arm B total parameters (same config) | 52,000 |
| E total == B total? | **true** — both add the identical 9,248-parameter delta over the same stacked baseline, even though E has zero recurrence and B's transition blocks are `recursive_steps`-shared rather than independent |
| Block evaluations (E) | 2 (== arm A's `n_layers`, verified by a real `register_forward_hook` call-counter, not just `len(layers)`) |

`test_arm_e_parameter_count_matches_zstate_delta_formula_exactly`
(`tests/test_models/test_recursive_denoiser.py`) checks this against real
constructed A/B/E towers at a second, distinct `(d_model=32, n_layers=3,
max_len=256)` configuration too — the match is a formula property, not a
one-off coincidence of the fixture's numbers.

**Gradient-consumption test (the issue's required "E consumes its matched
capacity and receives gradients" test).**
`test_arm_e_consumes_matched_state_and_receives_gradients` does both halves
required:

1. **Ablation**: construct two arm-E towers with identical weights, zero one
   copy's `state`/`state_ctx_proj` parameters, run both through the same
   forward pass, and assert the outputs differ
   (`not torch.allclose(out_full, out_zeroed)`) — measured `True`, so `state`/
   `state_ctx_proj` are not dead padding that never affects the forward pass.
2. **Gradient flow**: run one real forward + `.backward()` pass and assert
   `tower.state.grad is not None` and `torch.any(tower.state.grad != 0)`
   (likewise for `state_ctx_proj.weight.grad`) — both measured `True` (real
   run: `state.grad` absolute sum ≈ 1,112.6, `state_ctx_proj.weight.grad`
   absolute sum ≈ 10,907.1 at a small `d_model=16` smoke config — nonzero,
   not merely "not None").

Neither half is asserted from configuration alone — both are measured from a
real constructed tower and a real forward/backward pass.

## Fair initialization: `RecursiveControlInitializationV1`

Built by reseeding the global RNG to
`derive_seed(base_seed, "model_initialization")` (offset 0 — i.e. `base_seed`
itself) immediately before constructing **each** arm's tower, exactly the
discipline `TwoTowerModel.__init__` already applies per model instance. This
is *sufficient* for common-tensor identity because every arm's constructor
registers `tok`, `pos`, `kind` (if enabled), `layers.*`, `norm`, `lm_head` (tied
to `tok.weight`, itself consuming — then discarding — one RNG draw identically
across every arm) **before** any architecture-specific tensor
(`z_latent`/`ctx_proj` for B/G only), so the arch-specific draws that come
after never perturb the already-drawn common tensors.

Measured for A/B/C/D/E/F at the same fixture-scale config (G is excluded from
this specific report instance because it shares B's `denoiser_arch`
(`"shared_recursive"`) and therefore B's `arch_specific:shared_recursive`
seed — `RecursiveControlInitializationV1.__post_init__` requires pairwise
*disjoint* architecture-specific seeds, so a report spanning both B and G
together is a distinct, separately-testable case, exercised by
`test_arm_g_is_r1_shared_recursive_and_not_behaviorally_equivalent` instead):

- **49 common tensor names** — every `tok.weight`/`pos.weight`/`layers.*`/
  `norm.weight` — with `common_tensor_hashes_match_across_arms: true`
  (measured, not assumed; `__post_init__` raises otherwise, exercised by
  `test_recursive_control_initialization_rejects_mismatched_common_tensors`).
  F's 2 shared-prefix transition blocks (`layers.0`/`layers.1`, at this
  `recursive_transition_layers == n_layers == 2` configuration) hash-match
  every other arm's, since F's constructor draws them from the same
  `model_initialization`-seeded RNG stream before any of its extra blocks.
  E's `layers.0`/`layers.1` hash-match too, for the identical reason — its
  `state`/`state_ctx_proj` tensors are registered strictly after them.
- **Architecture-specific tensors**: `{}` for A/C/D, `{"z_latent": [256, 32],
  "ctx_proj.weight": [32, 32], "ctx_proj.bias": [32]}` for B — exactly the
  RSC-A04 delta tensors, never present for C/D. For F: its two *extra*
  unshared transition blocks beyond the shared prefix (`layers.2.*`,
  `layers.3.*` at `recursive_steps=2, recursive_transition_layers=2`) —
  real parameters absent from every other arm, not a z-state tensor. For E:
  `{"state": [256, 32], "state_ctx_proj.weight": [32, 32],
  "state_ctx_proj.bias": [32]}` — the exact same *shapes* as B's z-state
  tensors under distinct names, real parameters absent from every other arm.
- **Architecture-specific seeds** (declared, pairwise-disjoint):
  `{"A": 60000, "B": 70000, "C": 80000, "D": 90000, "E": 110000, "F":
  100000}` (measured, this session) from the `arch_specific:<denoiser_arch>`
  namespaces in `rng_contract.NAMESPACE_OFFSETS` (additive — E's
  `arch_specific:stacked_matched_state` namespace is a new entry this
  iteration; every earlier namespace, including F's from the prior
  iteration, is unchanged). **Honesty note**: these seeds are *reserved
  contract surface*, not literally consumed mid-construction by the current
  single-pass tower constructors — the class docstring says so explicitly,
  including for E's new namespace. They exist so a future two-phase
  constructor (or standalone telemetry) can draw architecture-specific
  tensors independently of the common-tensor draw count; today's
  constructors already get correct disjointness for free (arch-specific
  tensor names simply do not exist in the common set).
- **Optimizer group membership**: one `"base"` group per arm (these towers
  have no auxiliary heads) — B's group additionally contains `z_latent`/
  `ctx_proj.*`; F's group additionally contains its two extra unshared
  blocks' parameters; E's group additionally contains `state`/
  `state_ctx_proj.*`.
- **Optimizer initial-state hash**: the hash of an empty dict — `AdamW`
  lazily allocates `exp_avg`/`exp_avg_sq` on the first `.step()` call, so no
  nonempty initial state exists to hash; this is stated honestly rather than
  fabricating a "zero" tensor that was never actually materialized.

## Tests

`tests/test_models/test_recursive_denoiser.py` grew from 70 to 89 tests in the
earlier A/B/C/D/G iteration, to 93 in the F follow-up (net +5: 6 new arm-F
tests, minus `test_deferred_arms_fail_closed_not_silently_built[F]` correctly
removed since `F` left `DEFERRED_ARM_IDS`; 93, not 94, is the actual measured
collection count — reproduced by running `pytest --collect-only` against the
committed pre-E revision), and now to **100** in this E follow-up: **+7** new
tests, none removed (E was already `DEFERRED_ARM_IDS`-parametrized, so no
`[E]` case existed to delete the way `[F]` did last iteration), all 100
passing. One existing test
(`test_control_arm_table_reports_every_built_arm_no_parity_or_winner`) was
amended in place to add E assertions rather than duplicated, same convention
as the F iteration. Mapping onto the issue's 11 required tests (only those
applicable to what has been built so far):

1. **Construct + round-trip** — `test_arm_c_d_train_one_step_and_roundtrip_checkpoint`
   (C/D); `test_arm_e_denoiser_arch_wired_through_twotower_config_and_roundtrips`
   (E, through `TwoTowerConfig.denoiser_arch="stacked_matched_state"`);
   `test_arm_f_denoiser_arch_wired_through_twotower_config_and_roundtrips`
   (F, through `TwoTowerConfig.denoiser_arch="stacked_depth_matched"`); B
   already covered by the existing `test_twotower_shared_recursive_trains_and_roundtrips`; A by the existing suite generally.
2. **Parameter counts match formula** — `test_arm_c_d_parameter_count_matches_stacked_baseline`;
   `test_arm_e_parameter_count_matches_zstate_delta_formula_exactly` (E's delta
   over A equals `recursive_zstate_parameter_delta` exactly, and equals B's
   own delta over the same baseline);
   `test_control_arm_table_reports_every_built_arm_no_parity_or_winner` cross-checks B/G against `recursive_zstate_parameter_delta`, E against the same formula, and F against B (amended to add E/F assertions);
   `test_arm_f_parameter_count_exceeds_arm_b_real_measured`;
   `test_build_arm_f_dual_view_reports_honest_residuals`.
3. **B independent of R** — reused/confirmed via the existing (SLM-237/238)
   `test_recursive_parameter_count_independent_of_recursive_steps`; not re-implemented.
4. **C/D no undeclared z-state params** — `test_arm_c_d_have_no_zstate_parameters`;
   F's absence of z-state parameters is asserted in
   `test_arm_f_is_unshared_depth_matched_tower_with_no_zstate`; E's is
   asserted in `test_arm_e_is_unshared_non_recursive_tower_with_matched_state`
   (no `z_latent`/`ctx_proj` attribute or parameter name -- `state`/
   `state_ctx_proj` are distinct tensors, never collide with B's names).
5. **F: unshared blocks, correct block-eval count** — `test_arm_f_is_unshared_depth_matched_tower_with_no_zstate`
   (distinct block objects, no weight sharing);
   `test_arm_f_block_evaluations_match_arm_b_verified_by_hook_count` (requirement
   #6's "concretely instrument/count actual forward calls" — a real
   `register_forward_hook` call-counter on every transition block during one
   forward pass, not just a structural `len(layers)` claim).
6. **E: matched state capacity, correct block-eval count (== A)** —
   `test_arm_e_is_unshared_non_recursive_tower_with_matched_state` (distinct
   block objects, no weight sharing, `state`/`state_ctx_proj` present with
   the exact expected shapes); `test_arm_e_block_evaluations_match_arm_a_verified_by_hook_count`
   (same requirement-#6 real-hook-counter discipline as F's analogous test,
   applied to arm A instead of arm B since E's target block-eval count is
   A's, not B's); **`test_arm_e_consumes_matched_state_and_receives_gradients`
   — the issue's specifically required "E consumes its matched capacity and
   receives gradients" test**: zeroing `state`/`state_ctx_proj` changes the
   forward output (ablation), and both receive real nonzero `.grad` after a
   backward pass (gradient flow) — see "Arm E: matched-state formula +
   gradient-consumption test" above for the measured numbers.
7. **H tests** — not applicable; H remains the only deferred arm.
8. **Common init hashes + disjoint seeds** — `test_recursive_control_initialization_common_tensors_match_and_seeds_disjoint`;
   fail-closed case: `test_recursive_control_initialization_rejects_mismatched_common_tensors`;
   F extension: `test_recursive_control_initialization_includes_arm_f_with_disjoint_seed`;
   E extension: `test_recursive_control_initialization_includes_arm_e_with_disjoint_seed`.
9. **Runtime compatibility** — `test_deep_supervision_works_for_arm_c_and_d`
   (recursive_outputs duck-typing in `TwoTowerModel.training_loss` — no
   architecture-specific plumbing needed, C/D reuse `SharedRecursiveDenoiserTower`
   unmodified); the existing runtime-symbol-feature/attention-return/checkpoint
   tests already cover `SharedRecursiveDenoiserTower` generically. F reuses
   `DenoiserTower` unmodified, and E subclasses it (overriding only `encode`),
   so both are already covered the same way (no `recursive_outputs`
   attribute -- deep supervision correctly stays unavailable for E/F, same as
   A).
10. **Invalid combinations fail closed** — `test_zstate_mode_rejects_unknown_value`;
    `test_denoiser_arch_rejects_unknown_value` (TwoTowerModel construction no
    longer silently falls back to `stacked` on a typo); `test_deferred_arms_fail_closed_not_silently_built`
    (now parametrized over `DEFERRED_ARM_IDS = ("H",)` -- `test_deferred_arm_ids_now_only_contains_h`
    checks this set explicitly);
    `test_construct_arm_tower_rejects_unknown_arm_id`; the mismatched-init
    rejection above.
11. **Fixture emits a complete comparison table** — `test_control_arm_table_reports_every_built_arm_no_parity_or_winner`
    (now over `BUILT_ARM_IDS = ("A","B","C","D","E","F","G")`);
    `scripts/run_slm138_recursive_denoiser_fixture.py` now embeds a real
    `control_arm_table` field (E's and F's rows included, both driven purely
    by the imported `BUILT_ARM_IDS` constant -- no fixture-script code change
    needed to add E's row) plus an `arm_f_dual_view` field/markdown section
    for F's block-evaluation-matched vs parameter-nearest views (regenerated
    clean-tree artifact: `docs/design/iter-slm138-recursive-denoiser-20260721.{json,md}`).

## Configuration contract coverage

- **Config serialization / compatibility fingerprint**: C/D/E/F are ordinary
  `denoiser_arch` string values on the existing `TwoTowerConfig`/
  `ModelBuildConfig` dataclasses — no new field, so existing serialization/
  fingerprint code paths apply unchanged. F reuses `recursive_steps`/
  `recursive_transition_layers` (already-existing fields) to derive its layer
  count instead of `denoiser_layers` directly — no new field either. E reuses
  the plain `denoiser_layers` field (arm A's own dial), no new field either.
- **Checkpoint manifest / migration**: C/D/F never declare `z_latent`/
  `ctx_proj`, so the existing `_missing_ok` allowance (only active for
  `denoiser_arch == "shared_recursive"`) needs no change for them; B/G are
  unchanged from RSC-A04. E declares its own `state`/`state_ctx_proj`
  parameters (distinct names, no collision with B's `z_latent`/`ctx_proj`),
  so `twotower.py`'s checkpoint loader gains a matching, symmetric warm-start
  allowance gated on `denoiser_arch == "stacked_matched_state"` — an older
  plain-`"stacked"` checkpoint can warm-start into arm E the same way an
  older stacked checkpoint already warm-starts into arm B.
- **Active optimizer parameter groups**: `TwoTowerModel.optimizer_parameter_groups`
  needs no change — C/D/F simply have no extra parameters to group differently
  (F's extra unshared blocks are ordinary `nn.Module` parameters, same
  grouping path as A's blocks); E's `state`/`state_ctx_proj` are likewise
  ordinary parameters in the same `"base"` group, no special-casing needed.
- **Architecture comparison report**: this document's `control_arm_table`
  (`RecursiveControlArmReportV1`) extends SLM-240's
  `ArchitectureComparisonReportV1` idea to an N-arm table; F additionally gets
  `build_arm_f_dual_view`, a paired-view report specific to its two-matching-
  dimension tradeoff. E needs no analogous dual-view helper — unlike F, its
  parameter delta over A matches its declared target (B's z-state delta
  formula) exactly, with no residual on either matching dimension to report.
- **CLI/factory selection**: `scripts/train_model.py --denoiser-arch` gains
  `shared_recursive_y_only`/`shared_recursive_no_extra_capacity` (earlier
  iteration), `stacked_depth_matched` (F follow-up), and now
  `stacked_matched_state` (this iteration, arm E) as choices.
- **Model card / run report**: no checkpoint was created or promoted this
  session (wiring/fixture-scale only) — no `docs/MODEL_CARD.md` change is
  triggered; this note is the required "mention, not a new checkpoint row".

## Acceptance criteria check (for the built subset)

- State capacity (D vs B, and now E vs A), recurrence (C vs A), and weight
  sharing (F vs B) are now all independently attributable for the arms
  built: C isolates shared repeated depth from the y/z split; D isolates
  state capacity removal from recurrence; G isolates the R=1 architecture
  change from stacked; **F isolates weight sharing from block-evaluation
  count** — same total block evaluations as B, no weight sharing at all,
  real measured (necessarily larger) parameter cost reported plainly; **E
  isolates matched state capacity from recurrence** — same block-evaluation
  count as A (no recurrence added), exact real-measured parameter delta over
  A matching `recursive_zstate_parameter_delta` (the same formula B/G's
  delta over A matches), with the added state/state_ctx_proj verified (not
  merely asserted) to actually be consumed by the forward pass and receive
  real gradients. Only **H** (stop-gradient recurrence) remains deferred, so
  that one dimension is not yet isolated — noted, not hidden.
- Every remaining mismatch is explicit and machine-readable: B/G's
  `parameter_count_delta_vs_baseline`/`within_matching_tolerance: false` is
  reported plainly, never hidden behind a "matched" label; F's row is
  likewise `within_matching_tolerance: false` (parameter dimension), with the
  full block-evaluation-matched-vs-parameter-nearest tradeoff in
  `arm_f_dual_view`. E's row is *also* `within_matching_tolerance: false`
  against the generic "delta vs A" check (E was never intended to match A's
  raw total, only B's delta over A) — its real match (against
  `recursive_zstate_parameter_delta`) is reported in this document's "Arm E:
  matched-state formula + gradient-consumption test" section instead of
  overloading the generic per-A tolerance field with a target it was not
  designed to check.
- No quality or efficiency conclusion is drawn anywhere in this document or
  the underlying code (`claim_class` is always `"wiring"`, enforced in
  `RecursiveControlArmReportV1.__post_init__`).

## Recommended next iteration (a paragraph, not implementation)

Only **H** (stop-gradient recurrence) remains. It needs `y`/`z` detached
between recursive steps inside `SharedRecursiveDenoiserTower.recursive_outputs`
(likely a `detach_between_steps: bool = False` constructor flag, following
this module's existing `z_state_mode`-style single-class-plus-flag
convention rather than a new class, since H's forward math is otherwise
identical to B's), plus a careful paired test proving **identical forward
values** (detaching `.detach()` never changes the numeric value passed
forward, only what receives gradient) but **divergent cross-step gradient
paths** vs the non-detached B — concretely, that gradients w.r.t. an early
step's parameters differ (in the detached case, are structurally excluded
from the backward graph for later-step-only paths) between H and B for an
otherwise-identical forward pass. This is a genuine test-design problem, not
just a constructor change: having now built E, F, C, D, and G, the
constructor-and-wiring half of this issue (config/factory/checkpoint/CLI
surface, RNG-namespace reservation, control-arm-table/init-report inclusion)
is a well-worn, low-risk path — every one of those arms followed it without
surprises. H's risk is concentrated entirely in the gradient-path test
itself: proving a *negative* (a specific gradient path does NOT exist/is
zero) is easy to get subtly wrong (e.g. a residual connection or tied weight
elsewhere in the tower could still let gradient flow through an "detached"
path indirectly), so the next iteration should budget real effort for
constructing a minimal, sufficiently deep (`recursive_steps >= 2`) synthetic
case where the detached-vs-non-detached gradient divergence is unambiguous
and directly attributable to the `y`/`z` detach point, not to some other
confound. SLM-233 itself remains untouched and out of scope.

## Non-goals (explicitly out of scope, unchanged)

No multi-seed training campaign; no semantic-latent slot implementation; no
architecture adoption or production default change; no stochastic recursive
width; SLM-233's own campaign is out of scope. H is deferred, not partially
implemented.
