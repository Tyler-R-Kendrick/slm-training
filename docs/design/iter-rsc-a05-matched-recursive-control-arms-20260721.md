# RSC-A05: matched recursive control arms (SLM-241)

Run id: `iter_rsc_a05_matched_recursive_control_arms`
Status: **partial_implementation** (wiring/resource-accounting only; no quality claim)
Date: 2026-07-21

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
G first ... then C and D ... treating E/F/H as stretch goals"), this session
builds **A, B, C, D, G** fully (each constructs through the canonical factory,
trains one step, and round-trips a checkpoint) and explicitly defers **E, F,
H** — never fabricated, never half-implemented.

## Built arms

| Arm | `denoiser_arch` | `z_state_mode` | What it isolates |
| --- | --- | --- | --- |
| **A** | `stacked` | n/a | Existing baseline: no z state, unshared blocks. Already existed; now a named, reportable arm. |
| **B** | `shared_recursive` | `full` | Existing V1: shared transition blocks + explicit learned z state. Already existed; now a named, reportable arm. |
| **C** | `shared_recursive_y_only` | `y_only` | Shared repeated depth **without** the y/z split — both the F- and G-update layers run on `y` alone each recursion step; no `z_latent`/`ctx_proj` tensor exists at all. |
| **D** | `shared_recursive_no_extra_capacity` | `parameter_free` | Keeps the y/z split structurally, but `z`'s initial value is a deterministic pooled-context broadcast (no learned `max_len` bank, no learned projection) — removes exactly the two parameter tensors `recursive_zstate_parameter_delta` accounts for. |
| **G** | `shared_recursive` | `full` | Same constructor as B with `recursive_steps` forced to 1 — an architecture-change control. Interface-compatible with A, **not** behaviorally equivalent (SLM-240's framing, reused, not re-derived). |

## Deferred arms (explicit, not fabricated)

| Arm | Why deferred |
| --- | --- |
| **E** — stacked + matched state capacity | Needs a *new* constructor: a learned target-position state + context projection injected **once** into an otherwise-unshared stacked tower (the mirror image of D — capacity without recurrence, instead of recurrence without capacity). Not built this session. |
| **F** — unshared depth-matched tower | Needs a *new* tower with `recursive_steps * recursive_transition_layers` independent (non-shared-object) `TransformerBlock`s, plus both an equal-block-evaluation/FLOP view and a nearest-parameter/checkpoint-byte view. Not built this session. |
| **H** — stop-gradient recurrence | Needs `y`/`z` detached between recursive steps inside `recursive_outputs`, plus a paired test proving identical forward values but divergent cross-step gradient paths vs the non-detached B. Not built this session. |

`slm_training.models.recursive_control_arms.construct_arm_tower` raises
`NotImplementedError` (never a silent no-op or a fabricated construction) for
all three — see `tests/test_models/test_recursive_denoiser.py::test_deferred_arms_fail_closed_not_silently_built`.

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
- **Block evaluations**: A performs `n_layers=2`; B performs
  `recursive_steps * recursive_transition_layers = 4`; G forces
  `recursive_steps=1`, so it performs `2` — the *same* block-evaluation count
  as A despite being architecturally distinct (shared blocks + z state vs
  unshared blocks, no z state). Self-attention/cross-attention/MLP call
  counts are identical to the block-evaluation count in this codebase (every
  `TransformerBlock` call always performs exactly one of each).
- **Estimated FLOPs** are the same analytic per-block estimator RSC-A04
  introduced (`estimate_transformer_block_flops`, now public) — an explicit
  relative-cost proxy, never a profiler measurement or latency claim.
  Profiler-measured FLOPs / peak activation memory / wall time remain a
  stretch goal, not attempted this session.
- **No field anywhere is named `parity` or `winner`** —
  `RecursiveControlArmReportV1.__post_init__` raises if either is present.

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

Measured for A/B/C/D at the same fixture-scale config:

- **49 common tensor names** — every `tok.weight`/`pos.weight`/`layers.*`/
  `norm.weight` — with `common_tensor_hashes_match_across_arms: true`
  (measured, not assumed; `__post_init__` raises otherwise, exercised by
  `test_recursive_control_initialization_rejects_mismatched_common_tensors`).
- **Architecture-specific tensors**: `{}` for A/C/D, `{"z_latent": [256, 32],
  "ctx_proj.weight": [32, 32], "ctx_proj.bias": [32]}` for B — exactly the
  RSC-A04 delta tensors, never present for C/D.
- **Architecture-specific seeds** (declared, pairwise-disjoint):
  `{"A": 60000, "B": 70000, "C": 80000, "D": 90000}` from the new
  `arch_specific:<denoiser_arch>` namespaces added to
  `rng_contract.NAMESPACE_OFFSETS` (additive — the existing five namespaces
  are unchanged). **Honesty note**: these seeds are *reserved contract
  surface*, not literally consumed mid-construction by the current
  single-pass tower constructors — the class docstring says so explicitly.
  They exist so a future two-phase constructor (or standalone telemetry) can
  draw architecture-specific tensors independently of the common-tensor draw
  count; today's constructors already get correct disjointness for free
  (arch-specific tensor names simply do not exist in the common set).
- **Optimizer group membership**: one `"base"` group per arm (these towers
  have no auxiliary heads) — B's group additionally contains `z_latent`/
  `ctx_proj.*`.
- **Optimizer initial-state hash**: the hash of an empty dict — `AdamW`
  lazily allocates `exp_avg`/`exp_avg_sq` on the first `.step()` call, so no
  nonempty initial state exists to hash; this is stated honestly rather than
  fabricating a "zero" tensor that was never actually materialized.

## Tests

`tests/test_models/test_recursive_denoiser.py` grew from 70 to **89** tests
(19 new, all passing). Mapping onto the issue's 11 required tests (only
those applicable to what was actually built):

1. **Construct + round-trip** — `test_arm_c_d_train_one_step_and_roundtrip_checkpoint`
   (C/D); B already covered by the existing `test_twotower_shared_recursive_trains_and_roundtrips`; A by the existing suite generally.
2. **Parameter counts match formula** — `test_arm_c_d_parameter_count_matches_stacked_baseline`;
   `test_control_arm_table_reports_every_built_arm_no_parity_or_winner` cross-checks B/G against `recursive_zstate_parameter_delta`.
3. **B independent of R** — reused/confirmed via the existing (SLM-237/238)
   `test_recursive_parameter_count_independent_of_recursive_steps`; not re-implemented.
4. **C/D no undeclared z-state params** — `test_arm_c_d_have_no_zstate_parameters`.
5-7. **E/F/H tests** — not applicable; those arms are deferred.
8. **Common init hashes + disjoint seeds** — `test_recursive_control_initialization_common_tensors_match_and_seeds_disjoint`;
   fail-closed case: `test_recursive_control_initialization_rejects_mismatched_common_tensors`.
9. **Runtime compatibility** — `test_deep_supervision_works_for_arm_c_and_d`
   (recursive_outputs duck-typing in `TwoTowerModel.training_loss` — no
   architecture-specific plumbing needed, C/D reuse `SharedRecursiveDenoiserTower`
   unmodified); the existing runtime-symbol-feature/attention-return/checkpoint
   tests already cover `SharedRecursiveDenoiserTower` generically.
10. **Invalid combinations fail closed** — `test_zstate_mode_rejects_unknown_value`;
    `test_denoiser_arch_rejects_unknown_value` (TwoTowerModel construction no
    longer silently falls back to `stacked` on a typo); `test_deferred_arms_fail_closed_not_silently_built`;
    `test_construct_arm_tower_rejects_unknown_arm_id`; the mismatched-init
    rejection above.
11. **Fixture emits a complete comparison table** — `test_control_arm_table_reports_every_built_arm_no_parity_or_winner`;
    `scripts/run_slm138_recursive_denoiser_fixture.py` now embeds a real
    `control_arm_table` field/markdown section (regenerated clean-tree
    artifact: `docs/design/iter-slm138-recursive-denoiser-20260721.{json,md}`).

## Configuration contract coverage

- **Config serialization / compatibility fingerprint**: C/D are ordinary
  `denoiser_arch` string values on the existing `TwoTowerConfig`/
  `ModelBuildConfig` dataclasses — no new field, so existing serialization/
  fingerprint code paths apply unchanged.
- **Checkpoint manifest / migration**: C/D never declare `z_latent`/
  `ctx_proj`, so the existing `_missing_ok` allowance (only active for
  `denoiser_arch == "shared_recursive"`) needs no change for them; B/G are
  unchanged from RSC-A04.
- **Active optimizer parameter groups**: `TwoTowerModel.optimizer_parameter_groups`
  needs no change — C/D simply have no extra parameters to group differently.
- **Architecture comparison report**: this document's `control_arm_table`
  (`RecursiveControlArmReportV1`) extends SLM-240's
  `ArchitectureComparisonReportV1` idea to an N-arm table.
- **CLI/factory selection**: `scripts/train_model.py --denoiser-arch` gains
  `shared_recursive_y_only`/`shared_recursive_no_extra_capacity` as choices.
- **Model card / run report**: no checkpoint was created or promoted this
  session (wiring/fixture-scale only) — no `docs/MODEL_CARD.md` change is
  triggered; this note is the required "mention, not a new checkpoint row".

## Acceptance criteria check (for the built subset)

- State capacity (D vs B), recurrence (C vs A), and weight sharing (B/C/D vs
  F, deferred) are independently attributable for the arms built: C isolates
  shared repeated depth from the y/z split; D isolates state capacity
  removal from recurrence; G isolates the R=1 architecture change from
  stacked. F (the weight-sharing-vs-block-count control) is deferred, so the
  weight-sharing dimension is not yet *fully* isolated — noted, not hidden.
- Every remaining mismatch is explicit and machine-readable: B/G's
  `parameter_count_delta_vs_baseline`/`within_matching_tolerance: false` is
  reported plainly, never hidden behind a "matched" label.
- No quality or efficiency conclusion is drawn anywhere in this document or
  the underlying code (`claim_class` is always `"wiring"`, enforced in
  `RecursiveControlArmReportV1.__post_init__`).

## Recommended next iteration (a paragraph, not implementation)

Build **F** (unshared depth-matched tower) next: it needs no new z-state
machinery — it reuses `DenoiserTower`'s existing per-layer-independent
structure, just with `recursive_steps * recursive_transition_layers`
distinct (non-shared) `TransformerBlock` instances instead of `n_layers` — so
it is the cheapest of the three deferred arms to build and gives SLM-233 a
real "same total compute, no weight sharing" control before tackling **E** (a
new learned-state-without-recurrence constructor, the mirror image of D) or
**H** (stop-gradient recurrence, which needs a careful forward-equivalence/
gradient-divergence test pair rather than a simple construction change).
SLM-233 itself remains untouched and out of scope.

## Non-goals (explicitly out of scope, unchanged)

No multi-seed training campaign; no semantic-latent slot implementation; no
architecture adoption or production default change; no stochastic recursive
width; SLM-233's own campaign is out of scope. E/F/H are deferred, not
partially implemented.
