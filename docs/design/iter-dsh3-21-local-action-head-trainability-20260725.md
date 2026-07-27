# DSH3-21: make LocalFlatHead optimizer-visible and checkpoint-stable

**Linear issue:** SLM-396 (DSH3-21)
**Date:** 2026-07-25
**Evidence:** [iter-dsh3-21-local-action-head-trainability-20260725.json](iter-dsh3-21-local-action-head-trainability-20260725.json)
**Related:** [iter-cap2-03-state-local-action-heads-20260718.md](iter-cap2-03-state-local-action-heads-20260718.md)
(original CAP2-03 fixture harness; this issue fixes one bug it inherited)

## Decision

Yes, with a documented capacity limit. `LocalFlatHead` is now genuinely
trainable, serializable, and reproducible under a dynamic action vocabulary
via an explicit `materialize()` / checkpoint contract. It remains an
`O(distinct semantic actions)` table with no unseen-action generalization --
that limit is inherent to the head family, not a bug, and is documented rather
than hidden (see "Stop rule" below).

## Why (the bug)

`LocalFlatHead.action_embeddings` was a plain Python `dict[str, nn.Parameter]`
populated lazily inside `score()`. Plain dict values assigned as instance
attributes are **invisible** to `nn.Module.parameters()` /
`named_parameters()`: an optimizer built over `model.parameters()` before any
`score()` call never saw these tensors, and even after `score()` populated the
dict, the optimizer's already-fixed parameter groups still didn't include them.
The per-action embeddings could never receive a gradient update from a normal
training loop.

## What changed

`src/slm_training/models/local_action_head.py`:

- **Registered table.** `action_embeddings` is now an `nn.ParameterDict`
  (a real submodule), so every allocated embedding appears in
  `named_parameters()` and is visible to any optimizer over
  `self.parameters()`.
- **Canonical semantic action key** (`canonical_action_key`, module-level).
  Action identities following the reserved operator serialization
  (`dsl.operators.legal_set`'s
  `"OPERATOR <op_id> <slot>=<kind>:<request_id>:<opaque_id> ..."`) canonicalize
  to `"OPERATOR <op_id> <slot>=<kind> ..."` -- the opaque per-request
  `request_id`/`opaque_id` pair is dropped, so two actions differing *only* in
  that opaque identity collapse onto the same parameter. Every other
  (already-semantic, non-operator) action identity string -- what every
  existing fixture harness in this repo uses, e.g.
  `"component:root:none:card"` -- passes through unchanged. The function has
  no dependency on `dsl.operators`; only the serialization *format* is shared,
  by design, to keep this head decoupled from the full operator/reference
  stack.
- **`materialize(action_views)`.** Pre-registers one parameter per canonical
  key (deduplicating repeats and opaque-id-only variants), and must run
  *before* optimizer construction. After `materialize()`, the vocabulary is
  frozen: an unseen canonical key either raises
  (`unseen_action_policy="error"`, the default -- fail closed) or routes to a
  dedicated `"__unk__"` parameter (`unseen_action_policy="unk"`, an explicit
  composition/UNK path). It never silently lazily allocates a parameter an
  already-constructed optimizer cannot see.
- **Legacy lazy mode preserved.** Before `materialize()` is ever called
  (fixture/wiring mode -- what `cap2_state_local_action.py`'s harness uses
  today), an unseen action still lazily allocates on first sight, exactly as
  before this change, so fixture-mode numeric behavior is unchanged
  (untouched harness file, zero-step parity verified below).
- **Duplicate rejection.** `score()` raises `ValueError` if two entries in one
  `legal_actions` call canonicalize to the same key -- a legality-building bug
  is surfaced, not silently resolved by picking one.
- **Versioned, fail-closed checkpoints.** `checkpoint_state()` /
  `load_checkpoint_state()` add an explicit `CHECKPOINT_FORMAT_VERSION` and
  raise on format-version mismatch, action-registry mismatch (the live head's
  materialized vocabulary must match the checkpoint's exactly), or tensor
  shape mismatch (`load_state_dict(..., strict=True)`).

No other head family (`GlobalMaskedHead`, `TernaryDigitHead`,
`TernaryECOCHead`, `GrammarFactorizedHead`) was touched -- this issue is
scoped to `LocalFlatHead` only.

## Verification matrix

All measured values are in the evidence JSON's `run_evidence` block (real
script output, not hand-typed).

| Check | Result |
| --- | --- |
| Zero-step fixture parity (`local_flat` in the CAP2-03 harness) | oracle_accuracy=1.0, random_init_accuracy=1.0, forced=1, abstain=0, detected_error=0 -- matches the 20260718 baseline table |
| Two-action overfit (Adam, 200 steps) | loss 1.359 -> ~0.0 by step 49; final decisions correct for both actions |
| Save/load exactness | `torch.equal` on both raw parameters and re-scored logits after a round trip through `checkpoint_state()`/`torch.save`/`torch.load`/`load_checkpoint_state()` |
| Candidate/reference permutation | scores keyed by action identity, not list position (order-permutation test); opaque-id-only variants of an operator action collapse to one parameter |
| Unseen action after optimizer construction | `unseen_action_policy="error"` (default) raises `ValueError`; `unseen_action_policy="unk"` routes to `"__unk__"` and records `unk_routed_actions` in output metadata |

## Adversarial controls

| Control | Result |
| --- | --- |
| Opaque-ID-only changes cannot allocate distinct semantic parameters | `canonical_action_key` collapses three opaque-id variants of the same operator/slot/kind to one key; `materialize()` allocates exactly one parameter for them |
| Duplicate semantic actions reject or deduplicate | `materialize()` deduplicates repeated raw identities; `score()` raises `ValueError` if two *distinct* candidate strings in one call canonicalize to the same key |
| Late materialization cannot silently remain unoptimized | after `materialize()` + optimizer construction, an unseen action raises by default -- it cannot lazily allocate an optimizer-invisible parameter |
| Mismatched registry loads fail | `load_checkpoint_state` raises on `format_version` mismatch, action-registry mismatch, and (via `strict=True` state_dict load) shape mismatch |

## Acceptance criteria

- [x] All intended tensors appear in `named_parameters()` before optimizer
      construction (post-`materialize()`).
- [x] Finite, nonzero gradients and a correct one-step score change.
- [x] Exact checkpoint round trip.
- [x] Fixture outputs unchanged before training (harness file untouched;
      zero-step accuracy matches the CAP2-03 baseline).

## Stop rule

`LocalFlatHead` inherently requires an unbounded per-action parameter table --
there is no compositional structure in this head that could generalize to an
action never seen at `materialize()` time. Per the issue's stop rule, this is
kept and documented as a **non-generalizing diagnostic/baseline head**
(`O(distinct semantic actions)` capacity), not forced into a fake
generalization claim. Heads with bounded capacity over an unbounded vocabulary
already exist in this same module for that purpose: `TernaryDigitHead`,
`TernaryECOCHead`, `GrammarFactorizedHead` (see
[iter-cap2-03-state-local-action-heads-20260718.md](iter-cap2-03-state-local-action-heads-20260718.md)).

## Non-goals (unchanged)

- No unseen-action generalization claim for `LocalFlatHead`.
- Opaque reference ids are never used as vocabulary/scoring authority --
  `canonical_action_key` strips them by construction (consistent with
  AGENTS.md's engineering norm that external/opaque identity must not become
  scoring or legal authority).
- No legal-membership changes: `decode()` still only ever returns an
  `action_identity` present in the supplied `legal_actions` (or `None`).

## Honest caveats

- This is a CPU fixture/unit-test verification, not a production
  `TwoTowerModel` training integration or `--ship-gates` run; no checkpoint was
  created or promoted, so no checkpoint-bucket sync or `MODEL_CARD.md` update
  applies.
- `canonical_action_key` only strips the opaque `request_id`/`opaque_id` pair
  from the reserved operator serialization format; it does not resolve a
  reference's `semantic_fingerprint` from a live `ReferenceTableV1` (this head
  has no access to one -- that dereferencing lives in
  `dsl.operators.references`/`legal_set`). Two structurally different
  references of the same kind in the same slot still collapse onto one
  parameter. That is a documented precision limit, not a claim of full
  compositional action identity.

## Tests

```bash
pytest -q tests/test_models/test_local_action_head.py \
  tests/test_harnesses/experiments/test_cap2_state_local_action.py
```

Result: **43 passed**.

Also re-run (all downstream `LocalFlatHead` consumers), **47 passed**:

```bash
pytest -q tests/test_harnesses/quantization/test_sensitivity.py \
  tests/test_harnesses/quantization/test_calibration.py \
  tests/test_harnesses/experiments/test_cap3_03_ternary_falsification.py \
  tests/test_harnesses/experiments/test_cap2_04_state_ablation.py
```

One existing test needed a fix, not a weakening:
`test_sensitivity.py::test_baseline_parameters_restored_after_profiling`
snapshotted "baseline parameters" *before* warming the head's action
embeddings. Under the old plain-dict implementation those embeddings were
invisible to `named_parameters()` either way, so the snapshot/compare loop
silently never covered them. Now that they are real parameters, the lazily
warmed keys legitimately appear post-profiling but weren't in the pre-call
snapshot, raising `KeyError`. The fix warms the head with every corpus action
*before* taking the snapshot (matching the pattern already used by the two
tests above it in the same file), so the restore assertion now actually covers
every parameter profiling touches -- exactly the coverage gap this issue set
out to close.

## Version stamp

`model.quantization` bumped **v5 -> v6** in
`src/slm_training/resources/versions.json` (watches
`src/slm_training/models/local_action_head.py`). This is a real bump, not a
`no-bump:` note: `named_parameters()` membership changes and checkpoint
loading now fails closed on mismatches it previously ignored.
`python -m scripts.verify_version_stamps --check` passes.
