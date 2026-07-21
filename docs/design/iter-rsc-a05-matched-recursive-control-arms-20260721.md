# RSC-A05: matched recursive control arms (SLM-241)

Run id: `iter_rsc_a05_matched_recursive_control_arms`
Status: **partial_implementation** (wiring/resource-accounting only; no quality claim)
Date: 2026-07-21 (arm F added in a same-day follow-up iteration)

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
This follow-up iteration builds **F** (unshared depth-matched tower), per that
iteration's own recommendation (cheapest of the three deferred arms, reuses
`DenoiserTower`'s existing structure). **E** and **H** remain deferred.

## Built arms

| Arm | `denoiser_arch` | `z_state_mode` | What it isolates |
| --- | --- | --- | --- |
| **A** | `stacked` | n/a | Existing baseline: no z state, unshared blocks. Already existed; now a named, reportable arm. |
| **B** | `shared_recursive` | `full` | Existing V1: shared transition blocks + explicit learned z state. Already existed; now a named, reportable arm. |
| **C** | `shared_recursive_y_only` | `y_only` | Shared repeated depth **without** the y/z split — both the F- and G-update layers run on `y` alone each recursion step; no `z_latent`/`ctx_proj` tensor exists at all. |
| **D** | `shared_recursive_no_extra_capacity` | `parameter_free` | Keeps the y/z split structurally, but `z`'s initial value is a deterministic pooled-context broadcast (no learned `max_len` bank, no learned projection) — removes exactly the two parameter tensors `recursive_zstate_parameter_delta` accounts for. |
| **F** | `stacked_depth_matched` | n/a (no z state, same as A) | Unshared depth-matched tower: the exact same `DenoiserTower` class as A, no new tower code, built with `recursive_steps * recursive_transition_layers` independent transition blocks instead of `n_layers`. Isolates weight sharing from block-evaluation count — same total block evaluations as B, no weight sharing at all (vs B's fully shared transition). **Necessarily has MORE parameters than B** (nothing is shared); see "Arm F: two honest matching views" below. |
| **G** | `shared_recursive` | `full` | Same constructor as B with `recursive_steps` forced to 1 — an architecture-change control. Interface-compatible with A, **not** behaviorally equivalent (SLM-240's framing, reused, not re-derived). |

## Deferred arms (explicit, not fabricated)

| Arm | Why deferred |
| --- | --- |
| **E** — stacked + matched state capacity | Needs a *new* constructor: a learned target-position state + context projection injected **once** into an otherwise-unshared stacked tower (the mirror image of D — capacity without recurrence, instead of recurrence without capacity). Not built yet. |
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
  unshared blocks, no z state). F performs `4`, matching B, by construction.
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

Measured for A/B/C/D/F at the same fixture-scale config:

- **49 common tensor names** — every `tok.weight`/`pos.weight`/`layers.*`/
  `norm.weight` — with `common_tensor_hashes_match_across_arms: true`
  (measured, not assumed; `__post_init__` raises otherwise, exercised by
  `test_recursive_control_initialization_rejects_mismatched_common_tensors`).
  F's 2 shared-prefix transition blocks (`layers.0`/`layers.1`, at this
  `recursive_transition_layers == n_layers == 2` configuration) hash-match
  every other arm's, since F's constructor draws them from the same
  `model_initialization`-seeded RNG stream before any of its extra blocks.
- **Architecture-specific tensors**: `{}` for A/C/D, `{"z_latent": [256, 32],
  "ctx_proj.weight": [32, 32], "ctx_proj.bias": [32]}` for B — exactly the
  RSC-A04 delta tensors, never present for C/D. For F: its two *extra*
  unshared transition blocks beyond the shared prefix (`layers.2.*`,
  `layers.3.*` at `recursive_steps=2, recursive_transition_layers=2`) —
  real parameters absent from every other arm, not a z-state tensor.
- **Architecture-specific seeds** (declared, pairwise-disjoint):
  `{"A": 60000, "B": 70000, "C": 80000, "D": 90000, "F": 100000}` from the
  `arch_specific:<denoiser_arch>` namespaces in
  `rng_contract.NAMESPACE_OFFSETS` (additive — F's
  `arch_specific:stacked_depth_matched` namespace is a new entry this
  iteration; every earlier namespace is unchanged). **Honesty note**: these
  seeds are *reserved contract surface*, not literally consumed
  mid-construction by the current single-pass tower constructors — the class
  docstring says so explicitly, including for F's new namespace. They exist
  so a future two-phase constructor (or standalone telemetry) can draw
  architecture-specific tensors independently of the common-tensor draw
  count; today's constructors already get correct disjointness for free
  (arch-specific tensor names simply do not exist in the common set).
- **Optimizer group membership**: one `"base"` group per arm (these towers
  have no auxiliary heads) — B's group additionally contains `z_latent`/
  `ctx_proj.*`; F's group additionally contains its two extra unshared
  blocks' parameters.
- **Optimizer initial-state hash**: the hash of an empty dict — `AdamW`
  lazily allocates `exp_avg`/`exp_avg_sq` on the first `.step()` call, so no
  nonempty initial state exists to hash; this is stated honestly rather than
  fabricating a "zero" tensor that was never actually materialized.

## Tests

`tests/test_models/test_recursive_denoiser.py` grew from 70 to 89 tests in the
earlier A/B/C/D/G iteration, then to **94** in this F follow-up: 6 new arm-F
tests, minus 1 (`test_deferred_arms_fail_closed_not_silently_built[F]`,
correctly removed since `F` is no longer in `DEFERRED_ARM_IDS` — it is
parametrized over that tuple), net **+5**, all passing. One existing test
(`test_control_arm_table_reports_every_built_arm_no_parity_or_winner`) was
amended in place to add F assertions rather than duplicated. Mapping onto the
issue's 11 required tests (only those applicable to what has been built so
far):

1. **Construct + round-trip** — `test_arm_c_d_train_one_step_and_roundtrip_checkpoint`
   (C/D); `test_arm_f_denoiser_arch_wired_through_twotower_config_and_roundtrips`
   (F, through `TwoTowerConfig.denoiser_arch="stacked_depth_matched"`); B
   already covered by the existing `test_twotower_shared_recursive_trains_and_roundtrips`; A by the existing suite generally.
2. **Parameter counts match formula** — `test_arm_c_d_parameter_count_matches_stacked_baseline`;
   `test_control_arm_table_reports_every_built_arm_no_parity_or_winner` cross-checks B/G against `recursive_zstate_parameter_delta` and F against B (amended to add F assertions);
   `test_arm_f_parameter_count_exceeds_arm_b_real_measured`;
   `test_build_arm_f_dual_view_reports_honest_residuals`.
3. **B independent of R** — reused/confirmed via the existing (SLM-237/238)
   `test_recursive_parameter_count_independent_of_recursive_steps`; not re-implemented.
4. **C/D no undeclared z-state params** — `test_arm_c_d_have_no_zstate_parameters`;
   F's absence of z-state parameters is asserted in
   `test_arm_f_is_unshared_depth_matched_tower_with_no_zstate`.
5. **F: unshared blocks, correct block-eval count** — `test_arm_f_is_unshared_depth_matched_tower_with_no_zstate`
   (distinct block objects, no weight sharing);
   `test_arm_f_block_evaluations_match_arm_b_verified_by_hook_count` (requirement
   #6's "concretely instrument/count actual forward calls" — a real
   `register_forward_hook` call-counter on every transition block during one
   forward pass, not just a structural `len(layers)` claim).
6-7. **E/H tests** — not applicable; those arms remain deferred.
8. **Common init hashes + disjoint seeds** — `test_recursive_control_initialization_common_tensors_match_and_seeds_disjoint`;
   fail-closed case: `test_recursive_control_initialization_rejects_mismatched_common_tensors`;
   F extension: `test_recursive_control_initialization_includes_arm_f_with_disjoint_seed`.
9. **Runtime compatibility** — `test_deep_supervision_works_for_arm_c_and_d`
   (recursive_outputs duck-typing in `TwoTowerModel.training_loss` — no
   architecture-specific plumbing needed, C/D reuse `SharedRecursiveDenoiserTower`
   unmodified); the existing runtime-symbol-feature/attention-return/checkpoint
   tests already cover `SharedRecursiveDenoiserTower` generically. F reuses
   `DenoiserTower` unmodified, so it is already covered the same way (no
   `recursive_outputs` attribute -- deep supervision correctly stays
   unavailable for F, same as A).
10. **Invalid combinations fail closed** — `test_zstate_mode_rejects_unknown_value`;
    `test_denoiser_arch_rejects_unknown_value` (TwoTowerModel construction no
    longer silently falls back to `stacked` on a typo); `test_deferred_arms_fail_closed_not_silently_built`
    (now parametrized over `DEFERRED_ARM_IDS = ("E", "H")`);
    `test_construct_arm_tower_rejects_unknown_arm_id`; the mismatched-init
    rejection above.
11. **Fixture emits a complete comparison table** — `test_control_arm_table_reports_every_built_arm_no_parity_or_winner`
    (now over `BUILT_ARM_IDS = ("A","B","C","D","F","G")`);
    `scripts/run_slm138_recursive_denoiser_fixture.py` now embeds a real
    `control_arm_table` field (F's row included) plus an `arm_f_dual_view`
    field/markdown section for F's block-evaluation-matched vs
    parameter-nearest views (regenerated clean-tree artifact:
    `docs/design/iter-slm138-recursive-denoiser-20260721.{json,md}`).

## Configuration contract coverage

- **Config serialization / compatibility fingerprint**: C/D/F are ordinary
  `denoiser_arch` string values on the existing `TwoTowerConfig`/
  `ModelBuildConfig` dataclasses — no new field, so existing serialization/
  fingerprint code paths apply unchanged. F reuses `recursive_steps`/
  `recursive_transition_layers` (already-existing fields) to derive its layer
  count instead of `denoiser_layers` directly — no new field either.
- **Checkpoint manifest / migration**: C/D/F never declare `z_latent`/
  `ctx_proj`, so the existing `_missing_ok` allowance (only active for
  `denoiser_arch == "shared_recursive"`) needs no change for them; B/G are
  unchanged from RSC-A04.
- **Active optimizer parameter groups**: `TwoTowerModel.optimizer_parameter_groups`
  needs no change — C/D/F simply have no extra parameters to group differently
  (F's extra unshared blocks are ordinary `nn.Module` parameters, same
  grouping path as A's blocks).
- **Architecture comparison report**: this document's `control_arm_table`
  (`RecursiveControlArmReportV1`) extends SLM-240's
  `ArchitectureComparisonReportV1` idea to an N-arm table; F additionally gets
  `build_arm_f_dual_view`, a paired-view report specific to its two-matching-
  dimension tradeoff.
- **CLI/factory selection**: `scripts/train_model.py --denoiser-arch` gains
  `shared_recursive_y_only`/`shared_recursive_no_extra_capacity` (earlier
  iteration) and `stacked_depth_matched` (this iteration, arm F) as choices.
- **Model card / run report**: no checkpoint was created or promoted this
  session (wiring/fixture-scale only) — no `docs/MODEL_CARD.md` change is
  triggered; this note is the required "mention, not a new checkpoint row".

## Acceptance criteria check (for the built subset)

- State capacity (D vs B), recurrence (C vs A), and weight sharing (F vs B)
  are now all independently attributable for the arms built: C isolates
  shared repeated depth from the y/z split; D isolates state capacity
  removal from recurrence; G isolates the R=1 architecture change from
  stacked; **F isolates weight sharing from block-evaluation count** — same
  total block evaluations as B, no weight sharing at all, real measured
  (necessarily larger) parameter cost reported plainly. E (matched state
  capacity without recurrence) and H (stop-gradient recurrence) remain
  deferred, so those two dimensions are not yet isolated — noted, not hidden.
- Every remaining mismatch is explicit and machine-readable: B/G's
  `parameter_count_delta_vs_baseline`/`within_matching_tolerance: false` is
  reported plainly, never hidden behind a "matched" label; F's row is
  likewise `within_matching_tolerance: false` (parameter dimension), with the
  full block-evaluation-matched-vs-parameter-nearest tradeoff in
  `arm_f_dual_view`.
- No quality or efficiency conclusion is drawn anywhere in this document or
  the underlying code (`claim_class` is always `"wiring"`, enforced in
  `RecursiveControlArmReportV1.__post_init__`).

## Recommended next iteration (a paragraph, not implementation)

Build **E** (stacked + matched state capacity) next: it needs a genuinely new
constructor (a learned target-position state + context projection injected
once into an otherwise-unshared stacked tower — the mirror image of D), but
is a pure construction/wiring task, same shape as C/D/F. **H** (stop-gradient
recurrence) should come after E: it needs the same y/z-detach-between-steps
change inside `recursive_outputs` plus a careful paired test proving identical
forward values but divergent cross-step gradient paths vs the non-detached
B — a test-design problem, not just a constructor change, so it carries more
risk of a subtly wrong implementation than E's straightforward wiring.
SLM-233 itself remains untouched and out of scope.

## Non-goals (explicitly out of scope, unchanged)

No multi-seed training campaign; no semantic-latent slot implementation; no
architecture adoption or production default change; no stochastic recursive
width; SLM-233's own campaign is out of scope. E/H are deferred, not
partially implemented.
