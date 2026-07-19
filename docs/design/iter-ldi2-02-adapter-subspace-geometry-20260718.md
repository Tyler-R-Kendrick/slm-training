# LDI2-02 (SLM-125) — exact objective geometry in the TwoTower adapter subspace

Date: 2026-07-18
Status: **Diagnostic harness landed with tests and a deterministic fixture artifact.
No E283 canonical run, no checkpoint, no quality claim.** This iteration answers the
question "is the E284 no-safe-direction result intrinsic to the evidence/objectives or a
consequence of full-parameter cost and geometry?" only to the extent a *fixture* can: it
builds and verifies the measurement instrument and shows it already surfaces objective
conflict. The scientific answer requires running the instrument on the admitted E283
corpus + the frozen E228 checkpoint (deferred — that corpus/checkpoint is not in-repo).

## Why this exists

E285 (an exact-signature full-parameter gradient profile) and E286 (its batched-VJP
acceleration) both **failed to complete inside a real wall gate and produced no valid
evidence** — E285 ran unbounded past 25 min and was operator-stopped; E286 was killed at
283.6 s and its batched implementation was removed
([iter-e285](iter-e285-exact-signature-profile-aborted-20260717.md),
[iter-e286](iter-e286-batched-signature-profile-rejected-20260717.md)). LDI2-02 replaces
the full-parameter target with a small, explicit **low-rank adapter subspace** on a frozen
parent, so the same protected-objective geometry can be profiled cheaply and honestly
under a real cumulative wall.

## What landed

`src/slm_training/harnesses/preference/adapter_subspace_geometry.py` — a read-only
profiler that, for each `(rank × target-module)` cell:

1. **reports exact objective-signature support before any gradient is computed**
   (`objective_view_support`, train vs held-out coverage);
2. attaches a fresh **zero-init** adapter to a frozen parent (`lora_B = 0`, so the profile
   is taken at the exact parent function point) and takes the differentiable subset as
   `model.adapter_parameters()` — the parent stays frozen (`parent.grad is None`,
   asserted);
3. differentiates the four protected legal-token-space quantities — local objective
   `loss`, `good_probability_mass` (sign-adjusted for maximization), `bad_probability_mass`
   (minimization), and `mean_margin` (maximization) — w.r.t. the adapter tensors with a
   plain reverse pass over one shared forward graph (**no batched VJP**, per E286);
4. reports **raw and unit-normalized** gradient variants;
5. runs the declared solvers/transforms without assuming success — uniform weighted-mean
   control, PCGrad, MGDA/min-norm (with the **common-descent certificate** and mixing
   weights), and the SGD/AdamW first-step transforms aligned against the min-norm
   direction;
6. repeats the profile across strata — `decision_kind`, `abstract_state_role`, and the
   **exact objective-view signature**;
7. runs every cell as one bounded stage under a single cumulative
   [`DiagnosticBudget`](../../src/slm_training/harnesses/preference/decision_diagnostics.py)
   (hard 5-minute cap); an over-budget run emits a stopped record with `result: null`
   and **never a partial artifact**, and emits per-stage wall time, forward/backward
   counts, adapter parameter dimensions, and peak memory.

The whole objective/logits computation reuses the tested legal-token math in
`local_train` (`_guard_objective_tensors(probability_space="legal_tokens")`,
`_project_conflicting_gradients`, `_minimum_norm_gradient`, `_scale_gradient`,
`_fresh_adamw_direction`, `_gradient_alignment`) via a thin, never-persisted
`DecisionStateV2 → DecisionEventV1` shim; only the orchestration is new. Full-parameter
profiling stays refused upstream by `tier2_subspace_gradients`.

## Fixture demonstration (wiring evidence only)

[`iter-ldi2-02-adapter-subspace-geometry-fixture-20260718.json`](iter-ldi2-02-adapter-subspace-geometry-fixture-20260718.json)
is the deterministic output of the profiler on the committed 3-state test corpus across a
`rank ∈ {2, 4, 8}` × `{attn_q+attn_v, attn_q+attn_k+attn_v+attn_out}` matrix (seed 0;
wall-time/peak-memory fields nulled for reproducibility). It is **fixture/scratch evidence
only** — a random-init tiny model over a synthetic corpus, not the E283 evidence.

| Cell | Adapter dims | Support (held-out) | MGDA common descent | Any solver descends all four |
|---|---|---|---|---|
| rank2 · attn_q+attn_v | 128 | passed (1/1 signature) | **false** | **no** |
| rank4 · attn_q+attn_v | 256 | passed | **false** | **no** |
| rank8 · attn_q+attn_k+attn_v+attn_out | 1024 | passed | **false** | **no** |

Pooled unit-normalized objective-pair geometry (rank4) is internally consistent:
`good_probability_mass ⟂ bad_probability_mass` at cosine **−1.0** (the two sides of the
legal-probability simplex), `loss` aligned with raising good mass (cosine +0.48) and
lowering bad mass. The exact-signature stratum splits the two swapped-partition train
states into two distinct objective signatures, as designed.

**Observation, not a claim:** even in the smallest adapter subspace, with unit-normalized
gradients, neither MGDA nor PCGrad nor the weighted-mean control finds a direction that
descends all four protected objectives on this fixture — i.e. the instrument already
reproduces the *shape* of the E284 "no safe direction" result at adapter scale. This is a
property of the synthetic fixture's objective conflict; it is **not** evidence about the
real E283 objective and authorizes nothing.

## Operational cost vs the invalid E285/E286

Unlike E285 (unbounded) and E286 (killed at 283.6 s with the batched impl removed), this
harness runs to completion under the real cumulative 5-minute `DiagnosticBudget` on the
fixture, computes gradients strictly over the adapter subspace (not full parameters),
uses plain reverse passes (no resurrected batched VJP), and — by construction — yields a
stopped `result: null` record rather than a partial artifact if a real corpus ever
exceeds the wall. Tests cover the adapter-only gradient invariant, legal-space
finite-difference agreement, sign conventions, zero-gradient exclusion, unit
normalization, MGDA/PCGrad determinism, the first-step transforms, exact-signature
strata, deadline expiry → stopped record, no-partial-result, telemetry, and the new
fail-closed authorization decision
(`tests/test_models/test_adapter_subspace_geometry.py`, 16 tests;
`tests/test_scripts/test_run_twotower_adapter_subspace_geometry.py`).

## CLI entrypoint

`scripts/run_twotower_adapter_subspace_geometry.py` runs the profiler in two modes:

* `--fixture` — the committed 3-state synthetic corpus and tiny model (wiring evidence only);
* `--checkpoint PATH --events PATH` — a canonical run against a frozen parent checkpoint
  and a V2 decision-events JSONL corpus, materialized with `--materializer`
  (`pareto`, `set_valued`, `single_best_worst`, or `thresholded`).

Both modes accept `--ranks`, `--target-modules`, `--module-restricted`, `--objective`,
`--budget`, `--admit`, and `--out`. The emitted JSON now carries an explicit
`authorization` envelope (`authorized` | `repair_evidence` | `no_safe_direction` |
`expired`) consumed by the downstream LDI2-03 campaign matrix
(`scripts/run_twotower_adapter_matrix.py`).

## Decision

**Insufficient evidence to authorize an adapter training arm; do not repair or stop the
path yet.** The instrument is built and verified but has only been run on a fixture. The
honest next step (LDI2-03 / SLM-126 territory) is to run this canonical diagnostic on the
**admitted E283 objective-view corpus against the frozen E228 checkpoint** under the same
declared 5-minute budget, and read the common-descent certificate and held-out alignment
there before any bounded adapter training arm is authorized. That run needs the E283
corpus and the E228 checkpoint, which are not in this repository — it is deferred, not
claimed. No checkpoint, model-card update, or promotion is produced here.

## Honest remaining scope

- Canonical run on the real admitted corpus + frozen checkpoint (the actual
  intrinsic-vs-cost adjudication).
- Optional reference/locality objective and grouped action-role stratum (the profiler
  leaves clean extension points; only the four core protected quantities are wired).
