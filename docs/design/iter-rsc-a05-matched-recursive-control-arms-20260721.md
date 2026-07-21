# RSC-A05: matched recursive control arms (SLM-241)

Run id: `iter_rsc_a05_matched_recursive_control_arms`
Status: **complete** (wiring/resource-accounting only; no quality claim -- all
eight control arms A-H are now built, none deferred)
Date: 2026-07-21 (arm F added in a same-day follow-up iteration; arm E added
in a second same-day follow-up; arm H -- the last remaining arm -- added in a
third same-day follow-up, closing SLM-241)

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
reuses `DenoiserTower`'s existing structure). A second follow-up iteration
built **E** (stacked + matched state capacity) — the mirror image of D: a
new `StackedMatchedStateDenoiserTower` class (unshared, non-recursive blocks,
same block-evaluation count as A) with a learned `state`/`state_ctx_proj`
pair shape-matched to B's `z_latent`/`ctx_proj`, injected once before the
transition blocks run. This third follow-up iteration builds **H**
(stop-gradient recurrence) — the last remaining arm: arm B's exact
construction (`denoiser_arch="shared_recursive"`, `z_state_mode="full"`,
reused, not a new arch string, same convention arm G already established)
plus a new `detach_between_steps: bool = False` flag on
`SharedRecursiveDenoiserTower` that, when `True`, detaches the carried-forward
`y`/`z` between recursive steps. **All eight control arms (A-H) are now
built; SLM-241 (RSC-A05) is complete.**

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
| **H** | `shared_recursive` | `full` | Arm B's exact construction plus `detach_between_steps=True` — identical forward recurrence (same shared blocks, same `y`/`z` update equations, byte-identical forward values), only the backward graph differs: `y`/`z` are detached between recursive steps so a later step's loss cannot backpropagate through the recurrent state into an earlier step's block application. Isolates whether any gain from recurrence needs genuine backprop-through-recurrence, or merely re-applying the same shared weights repeatedly. |

## Deferred arms

None. All eight control arms (A-H) are built as of this iteration.
`slm_training.models.recursive_control_arms.construct_arm_tower` still raises
`NotImplementedError` (never a silent no-op or a fabricated construction) for
any arm id a future issue might add to `DEFERRED_ARM_IDS` (currently empty) —
see `tests/test_models/test_recursive_denoiser.py::test_deferred_arms_fail_closed_not_silently_built`
(now parametrized over the empty set, correctly collecting/skipping zero
cases) and `test_all_eight_control_arms_now_built`.

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
| H | 52,000 | +9,248 (+21.63%) | 4 | 608,256 | false vs A (no target declared, same as B/G); **true** vs B (exact, see below) |

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
- **H's row is exactly B's row** — same total parameters (52,000), same
  block evaluations (4), same estimated FLOPs — because H's construction is
  literally B's construction (`denoiser_arch="shared_recursive"`,
  `z_state_mode="full"`) plus `detach_between_steps=True`, which changes no
  parameter and no block evaluation, only the backward graph. The generic
  per-A `within_matching_tolerance` field is `false` for H, same as B/G (H
  was never intended to match A's zero-delta baseline, only B's exact
  count) — reported plainly, never hidden behind a "matched" label. H's
  real, declared matching target — total parameters and block-evaluations
  equal to B's *exactly* — is checked directly against real constructed B/H
  towers instead (`test_arm_h_parameter_count_and_block_evaluations_match_arm_b_exactly`),
  see "Arm H: forward-identity + gradient-divergence evidence" below.
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

## Arm H: forward-identity + gradient-divergence evidence

Arm H (`denoiser_arch="shared_recursive"` + `detach_between_steps=True`,
i.e. `SharedRecursiveDenoiserTower(z_state_mode="full",
detach_between_steps=True)`) is arm B's exact construction with one
orthogonal flag added. `recursive_outputs` calls `.detach()` on the
carried-forward `y` (and `z`, since H uses the `"full"` z-state path) at the
end of every recursion step except the last, before the *next* step reads
them:

```text
for r in 1..R:
    z_r = z_{r-1} + F_theta(norm(z_{r-1} + y_{r-1}), context)
    y_r = y_{r-1} + G_theta(norm(y_{r-1} + z_r),     context)
    h_r = norm(y_r); logits_r = lm_head(h_r)          # unchanged from B
    if detach_between_steps and r < R:
        y_r, z_r = y_r.detach(), z_r.detach()          # backward graph only
```

`recursive_outputs` also gained an additive, opt-in
`return_step_boundaries: bool = False` parameter (default off, changes no
existing behavior) that exposes the *real* `y`/`z` tensor exactly as computed
at the end of each step, captured **before** any detach is applied — so a
test can register an autograd hook on the identical tensor object regardless
of `detach_between_steps`, and observe whether gradient from a later step's
loss actually reaches it.

**Forward-identity (required property #1: `.detach()` only affects the
backward graph, never the forward numeric value).**
`test_arm_h_forward_values_identical_to_arm_b_before_backward` constructs B
and H with identical weights (same seed immediately before each
construction) and identical inputs, `recursive_steps=3`, and asserts
`torch.equal` (bit-identical, not merely `allclose`) on `logits`, every
`depth_logits[r]`, every `depth_hiddens[r]`, and `hidden` — measured max
absolute difference across all of them: **0.0**. `torch.equal` passing
(rather than only `allclose`) confirms detaching genuinely changes zero
forward values, at every depth, not just the final output.

**Gradient-divergence (required property #2, made mechanism-precise, not a
hand-waved "gradients differ").**
`test_arm_h_blocks_cross_step_gradient_flow_that_arm_b_has` is the concrete
mechanism check the prior session's tractability assessment asked for. With
`recursive_steps=2`:

1. Run `recursive_outputs(..., return_step_boundaries=True)` on both towers;
   capture the real step-1 `y` tensor (`step_boundaries[0]["y"]`) — the exact
   same kind of object for B and H, captured before any detach.
2. Register a backward hook on that tensor: `hook_grads = []; y1.register_hook(lambda g: hook_grads.append(g.clone()))`.
3. Compute the loss as **`depth_logits[-1].sum()` only** — the *last*
   recursion step's logits, deliberately excluding depth 1's own
   contribution — and call `.backward()`.

Measured result:

| Tower | Hook invocation count | Hook gradient (when invoked) |
| --- | --- | --- |
| B (no detach) | **1** | nonzero (abs sum ≈ 244.9 at a `d_model=16` smoke config) |
| H (`detach_between_steps=True`) | **0** | never invoked |

This is exact, not approximate, and directly attributable to the detach
point: for B, step 2's `z`/`y` update reads the step-1 boundary tensor
directly, so it is a genuine ancestor of the step-2-only loss and the hook
fires. For H, step 2 instead reads a **detached copy** of that tensor —
`torch.Tensor.detach()` returns a new tensor with no `grad_fn`, so there is
no autograd edge from the detached copy back to its source. The original
step-1 boundary tensor is therefore not an ancestor of the loss's backward
graph at all (not "receives zero gradient" — genuinely never visited), so
the hook is never invoked. Depth 1's own logits (excluded from this loss)
are irrelevant to this check since autograd's `.backward()` traversal only
follows tensors that are actual ancestors of the loss.

**Same-step gradient is not broken (rules out an over-aggressive detach).**
`test_arm_h_shared_weights_still_receive_same_step_gradient` runs a normal
forward + backward on H alone (loss = full output sum, both steps
contribute) and asserts every shared transition-block parameter has
`grad is not None` and at least one has nonzero grad — confirming
`detach_between_steps` only removes the *cross-step* backward path, never a
step's own contribution to the shared weights' gradient.

**Same params/block-evaluations as B (required, not merely a side effect).**
`test_arm_h_parameter_count_and_block_evaluations_match_arm_b_exactly`
constructs B and H with identical arguments (`recursive_steps=3,
recursive_transition_layers=2`) and asserts `sum(p.numel() for p in
h_tower.parameters()) == sum(p.numel() for p in b_tower.parameters())`
exactly, plus a real `register_forward_hook` call-counter on every
transition block (same discipline as arms E/F) showing both towers evaluate
each block exactly `recursive_steps * recursive_transition_layers = 6`
times.

Neither half of the required test is a hand-waved "gradients differ"
assertion — each names the exact tensor, the exact loss construction, and
the exact autograd mechanism (`detach()` severing a graph edge) responsible.

## Fair initialization: `RecursiveControlInitializationV1`

Built by reseeding the global RNG to
`derive_seed(base_seed, "model_initialization")` (offset 0 — i.e. `base_seed`
itself) immediately before constructing **each** arm's tower, exactly the
discipline `TwoTowerModel.__init__` already applies per model instance. This
is *sufficient* for common-tensor identity because every arm's constructor
registers `tok`, `pos`, `kind` (if enabled), `layers.*`, `norm`, `lm_head` (tied
to `tok.weight`, itself consuming — then discarding — one RNG draw identically
across every arm) **before** any architecture-specific tensor
(`z_latent`/`ctx_proj` for B/G/H only), so the arch-specific draws that come
after never perturb the already-drawn common tensors.

Measured for A/B/C/D/E/F at the same fixture-scale config (G **and now H**
are excluded from this specific report instance because both share B's
`denoiser_arch` (`"shared_recursive"`) and therefore B's
`arch_specific:shared_recursive` seed — `RecursiveControlInitializationV1.__post_init__`
requires pairwise *disjoint* architecture-specific seeds, so a report
spanning B with G, or B with H, together is a distinct, separately-testable
case: `test_arm_g_is_r1_shared_recursive_and_not_behaviorally_equivalent` for
G, and for H, `test_recursive_control_initialization_excludes_arm_h_when_arm_b_present`
(asserts the pairwise-disjoint `ValueError` is actually raised when B and H
are both included) plus
`test_recursive_control_initialization_includes_arm_h_excluding_arm_b`
(builds a real report over A/C/D/H — excluding B — and confirms it succeeds,
with H's `z_latent`/`ctx_proj` reported as its own architecture-specific
tensors, same shapes as B's, absent from A/C/D). H requires **no new
`arch_specific:*` namespace** in `rng_contract.NAMESPACE_OFFSETS` — confirmed
by reading `build_recursive_control_initialization` before assuming it: the
architecture-specific-tensor set for any arm in a report is derived purely
from *which named parameters are absent from that report's common-tensor
intersection*, and H declares literally the same parameter names/shapes as
B (no new tensor at all, only a different gradient-flow rule), so reusing
B's exact `arch_specific:shared_recursive` seed value is correct, not an
oversight — the only cost is the same B+H-together exclusion G already
established, which is now itself directly tested rather than merely
documented):

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
  For H (in a report that excludes B, e.g. A/C/D/H): `{"z_latent": [256, 32],
  "ctx_proj.weight": [32, 32], "ctx_proj.bias": [32]}` — the identical
  *names*, not merely the identical shapes, since H is literally B's
  construction; this is the one arm whose architecture-specific tensor set
  would collide by name (not just by shape) with another arm's (B's) if both
  were ever included in the same report, which is exactly why that
  combination is excluded rather than attempted.
- **Architecture-specific seeds** (declared, pairwise-disjoint):
  `{"A": 60000, "B": 70000, "C": 80000, "D": 90000, "E": 110000, "F":
  100000}` (measured, this session) from the `arch_specific:<denoiser_arch>`
  namespaces in `rng_contract.NAMESPACE_OFFSETS` (additive — E's
  `arch_specific:stacked_matched_state` namespace is a new entry this
  iteration; every earlier namespace, including F's from the prior
  iteration, is unchanged). **H deliberately adds no new namespace entry** —
  a report over A/C/D/H measures `{"A": 60000, "C": 80000, "D": 90000, "H":
  70000}`, i.e. H reuses B's exact `arch_specific:shared_recursive` value
  (70000); this is by design, not an omission, since H has no
  architecture-specific *parameter* to reserve a distinct namespace for
  (only a different gradient-flow rule) — see the "Fair initialization"
  section's opening paragraph above and `rng_contract.py`'s
  `NAMESPACE_OFFSETS` docstring for the full reasoning, now also directly
  tested (not merely documented) by
  `test_recursive_control_initialization_excludes_arm_h_when_arm_b_present`.
  **Honesty note**: these seeds are *reserved contract surface*, not
  literally consumed mid-construction by the current single-pass tower
  constructors — the class docstring says so explicitly, including for E's
  new namespace and H's deliberate non-namespace. They exist so a future
  two-phase constructor (or standalone telemetry) can draw
  architecture-specific tensors independently of the common-tensor draw
  count; today's constructors already get correct disjointness for free
  (arch-specific tensor names simply do not exist in the common set).
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
earlier A/B/C/D/G iteration, to 93 in the F follow-up, to 100 in the E
follow-up, and now to **108** in this H follow-up: **+8** net (9 new arm-H
tests, minus the stale `test_deferred_arm_ids_now_only_contains_h` removed and
replaced by `test_all_eight_control_arms_now_built` since H is no longer
deferred), all 108 passing. Two existing tests were amended in place to add H
assertions (`test_control_arm_table_reports_every_built_arm_no_parity_or_winner`)
rather than duplicated, same convention as the E/F iterations. Mapping onto
the issue's 11 required tests:

1. **Construct + round-trip** — `test_arm_c_d_train_one_step_and_roundtrip_checkpoint`
   (C/D); `test_arm_e_denoiser_arch_wired_through_twotower_config_and_roundtrips`
   (E); `test_arm_f_denoiser_arch_wired_through_twotower_config_and_roundtrips`
   (F); `test_arm_h_denoiser_arch_wired_through_twotower_config_and_roundtrips`
   (H, through `TwoTowerConfig.denoiser_arch="shared_recursive"` +
   `recursive_detach_between_steps=True`); B already covered by the existing
   `test_twotower_shared_recursive_trains_and_roundtrips`; A by the existing
   suite generally.
2. **Parameter counts match formula** — `test_arm_c_d_parameter_count_matches_stacked_baseline`;
   `test_arm_e_parameter_count_matches_zstate_delta_formula_exactly`;
   `test_arm_h_parameter_count_and_block_evaluations_match_arm_b_exactly`
   (H's total parameters and block-evaluations equal B's exactly, at a
   distinct config from the control-arm-table's own);
   `test_control_arm_table_reports_every_built_arm_no_parity_or_winner`
   cross-checks B/G against `recursive_zstate_parameter_delta`, E against the
   same formula, F against B, and now H against B directly (parameters,
   z_state_mode, denoiser_arch, and block-evaluations all asserted equal);
   `test_arm_f_parameter_count_exceeds_arm_b_real_measured`;
   `test_build_arm_f_dual_view_reports_honest_residuals`.
3. **B independent of R** — reused/confirmed via the existing (SLM-237/238)
   `test_recursive_parameter_count_independent_of_recursive_steps`; not re-implemented.
4. **C/D no undeclared z-state params** — `test_arm_c_d_have_no_zstate_parameters`;
   F's absence of z-state parameters is asserted in
   `test_arm_f_is_unshared_depth_matched_tower_with_no_zstate`; E's is
   asserted in `test_arm_e_is_unshared_non_recursive_tower_with_matched_state`.
   H is the mirror case: it *does* declare `z_latent`/`ctx_proj` (deliberately,
   since it is B's exact construction), asserted present in
   `test_arm_h_is_gradient_flow_only_variant_of_arm_b`.
5. **F: unshared blocks, correct block-eval count** — `test_arm_f_is_unshared_depth_matched_tower_with_no_zstate`;
   `test_arm_f_block_evaluations_match_arm_b_verified_by_hook_count`.
6. **E: matched state capacity, correct block-eval count (== A)** —
   `test_arm_e_is_unshared_non_recursive_tower_with_matched_state`;
   `test_arm_e_block_evaluations_match_arm_a_verified_by_hook_count`;
   `test_arm_e_consumes_matched_state_and_receives_gradients`.
7. **H: identical forward values, divergent cross-step gradient paths** —
   the issue's specifically required test, mapped onto three tests, each
   proving a distinct, named half:
   `test_arm_h_forward_values_identical_to_arm_b_before_backward` (forward
   identity, `torch.equal`, max abs diff measured 0.0);
   `test_arm_h_blocks_cross_step_gradient_flow_that_arm_b_has` (the
   mechanism-precise gradient-divergence hook test — see "Arm H:
   forward-identity + gradient-divergence evidence" above for the exact
   mechanism and measured hook-invocation counts);
   `test_arm_h_shared_weights_still_receive_same_step_gradient` (rules out an
   over-aggressive detach that would also kill same-step gradient).
8. **Common init hashes + disjoint seeds** — `test_recursive_control_initialization_common_tensors_match_and_seeds_disjoint`;
   fail-closed case: `test_recursive_control_initialization_rejects_mismatched_common_tensors`;
   F extension: `test_recursive_control_initialization_includes_arm_f_with_disjoint_seed`;
   E extension: `test_recursive_control_initialization_includes_arm_e_with_disjoint_seed`;
   H extensions: `test_recursive_control_initialization_includes_arm_h_excluding_arm_b`
   (the positive case, A/C/D/H) and
   `test_recursive_control_initialization_excludes_arm_h_when_arm_b_present`
   (the required negative case — B+H together must raise, proving the
   documented reasoning is actually correct rather than merely asserted).
9. **Runtime compatibility** — `test_deep_supervision_works_for_arm_c_and_d`;
   the existing runtime-symbol-feature/attention-return/checkpoint tests
   already cover `SharedRecursiveDenoiserTower` generically, and since H is
   an *instance* of `SharedRecursiveDenoiserTower` (not a new subclass), it
   is covered by every one of those generic tests automatically, with no new
   architecture-specific plumbing anywhere in `TwoTowerModel`.
10. **Invalid combinations fail closed** — `test_zstate_mode_rejects_unknown_value`;
    `test_denoiser_arch_rejects_unknown_value`;
    `test_deferred_arms_fail_closed_not_silently_built` (now parametrized over
    the empty `DEFERRED_ARM_IDS = ()` -- correctly collects/skips zero cases);
    `test_construct_arm_tower_rejects_unknown_arm_id`; the mismatched-init
    rejection above; `test_all_eight_control_arms_now_built` (the honest
    "nothing left to defer" assertion, replacing the stale H-specific one).
11. **Fixture emits a complete comparison table** — `test_control_arm_table_reports_every_built_arm_no_parity_or_winner`
    (now over `BUILT_ARM_IDS = ("A","B","C","D","E","F","G","H")`);
    `scripts/run_slm138_recursive_denoiser_fixture.py`'s `control_arm_table`
    field now embeds H's row too, driven purely by the imported
    `BUILT_ARM_IDS` constant -- no fixture-script code change needed
    (regenerated clean-tree artifact:
    `docs/design/iter-slm138-recursive-denoiser-20260721.{json,md}`).

## Configuration contract coverage

- **Config serialization / compatibility fingerprint**: C/D/E/F are ordinary
  `denoiser_arch` string values on the existing `TwoTowerConfig`/
  `ModelBuildConfig` dataclasses — no new field, so existing serialization/
  fingerprint code paths apply unchanged. H is different: it reuses B's
  exact `denoiser_arch` string (`"shared_recursive"`, same reuse convention
  as arm G's `recursive_steps=1`) and adds one new orthogonal dataclass
  field, `recursive_detach_between_steps: bool = False`, on both
  `TwoTowerConfig` and `ModelBuildConfig` (threaded through
  `_twotower_config_from_build` and `scripts/train_model.py
  --recursive-detach-between-steps`) — a real new field, but since both
  configs are ordinary dataclasses serialized via `asdict(self.config)`
  (`compatibility_fingerprint`/checkpoint `config` payload), no manual
  field-list update was needed anywhere the field's value is *read*; only
  the three call sites that *construct* a config from another config's
  fields (`_twotower_config_from_build`, the CLI, `checkpoint_migrate.py`'s
  `valid = {f.name for f in TwoTowerConfig.__dataclass_fields__.values()}`
  filter) needed touching, and the last of those needed none (it derives its
  allowlist from the dataclass fields directly).
- **Checkpoint manifest / migration**: C/D/F never declare `z_latent`/
  `ctx_proj`, so the existing `_missing_ok` allowance (only active for
  `denoiser_arch == "shared_recursive"`) needs no change for them; B/G are
  unchanged from RSC-A04. E declares its own `state`/`state_ctx_proj`
  parameters (distinct names), so `twotower.py`'s checkpoint loader gained a
  matching, symmetric warm-start allowance gated on
  `denoiser_arch == "stacked_matched_state"`. H needs **no new allowance at
  all** — it reuses `denoiser_arch == "shared_recursive"` and B's exact
  `z_latent`/`ctx_proj` parameter names, so the existing B allowance already
  covers it; a plain `"stacked"` checkpoint warm-starts into arm H exactly
  the way it already warm-starts into arm B.
- **Active optimizer parameter groups**: `TwoTowerModel.optimizer_parameter_groups`
  needs no change — H's parameters are literally B's (`z_latent`/
  `ctx_proj.*` plus the shared transition blocks), same `"base"` group, no
  special-casing needed.
- **Architecture comparison report**: this document's `control_arm_table`
  (`RecursiveControlArmReportV1`) extends SLM-240's
  `ArchitectureComparisonReportV1` idea to an N-arm table; F additionally gets
  `build_arm_f_dual_view`. H needs no analogous dual-view helper — its
  parameter/block-evaluation delta over A is *identical* to B's own (not
  merely close), so there is no residual to report beyond what B's row
  already reports; H's own distinguishing evidence (forward-identity,
  gradient-divergence) lives in the "Arm H" section above instead, since
  none of it is a resource-accounting dimension `RecursiveControlArmReportV1`
  was designed to hold.
- **CLI/factory selection**: `scripts/train_model.py --denoiser-arch` is
  unchanged for H (it reuses the existing `"shared_recursive"` choice); a new
  `--recursive-detach-between-steps` boolean flag (`action="store_true"`) was
  added alongside the existing `--recursive-steps`/`--recursive-transition-layers`
  flags.
- **Model card / run report**: no checkpoint was created or promoted this
  session (wiring/fixture-scale only) — no `docs/MODEL_CARD.md` change is
  triggered; this note is the required "mention, not a new checkpoint row".

## Acceptance criteria check (all eight arms)

- Every one of the eight independent resource/behavior dimensions SLM-241
  asks to isolate is now built: C isolates shared repeated depth from the
  y/z split; D isolates state capacity removal from recurrence; G isolates
  the R=1 architecture change from stacked; F isolates weight sharing from
  block-evaluation count; E isolates matched state capacity from recurrence;
  **H isolates recurrent credit assignment (backprop-through-recurrence)
  from merely re-applying shared weights repeatedly** — identical
  parameters, identical block-evaluations, identical forward values to B
  (measured, `torch.equal`, not merely `allclose`), with the cross-step
  backward path verified (not merely asserted) to be structurally absent via
  a concrete, mechanism-precise autograd-hook test, while same-step gradient
  to the shared weights is verified to remain intact. **No arm remains
  deferred.**
- Every mismatch that exists is explicit and machine-readable: B/G's
  `parameter_count_delta_vs_baseline`/`within_matching_tolerance: false` is
  reported plainly; F's row is likewise `within_matching_tolerance: false`
  (parameter dimension), with the full block-evaluation-matched-vs-parameter-
  nearest tradeoff in `arm_f_dual_view`; E's row is `within_matching_tolerance:
  false` against the generic "delta vs A" check (by design — its real match
  is against `recursive_zstate_parameter_delta`, reported in the "Arm E"
  section instead); H's row is *also* `within_matching_tolerance: false`
  against that same generic per-A check (same reason as B/G — H was never
  intended to match A's zero-delta baseline), while its real, meaningful
  match — total parameters and block-evaluations equal to B's *exactly* — is
  reported in the "Arm H" section and verified directly against real
  constructed towers.
- No quality or efficiency conclusion is drawn anywhere in this document or
  the underlying code (`claim_class` is always `"wiring"`, enforced in
  `RecursiveControlArmReportV1.__post_init__`).

## Recommended next iteration

SLM-241 (RSC-A05) is **complete** — all eight control arms (A-H) are built,
resource-accounted, and fairness-init-covered. There is no remaining scope
within this issue. The next step belongs to the parent campaign, **SLM-233**
(matched recursive-depth quality campaign, out of scope here): actually train
and compare these eight arms on real data under matched compute/parameter
budgets to answer the quality/efficiency questions this issue only built the
constructors and honest resource accounting for. That campaign should be able
to reuse every arm's `construct_arm_tower`/`ARM_DENOISER_ARCH`/
`RecursiveControlArmReportV1`/`RecursiveControlInitializationV1` surface
unmodified.

## Non-goals (explicitly out of scope, unchanged)

No multi-seed training campaign; no semantic-latent slot implementation; no
architecture adoption or production default change; no stochastic recursive
width; SLM-233's own campaign is out of scope. Nothing in this document draws
a quality or efficiency conclusion from any number it reports.
