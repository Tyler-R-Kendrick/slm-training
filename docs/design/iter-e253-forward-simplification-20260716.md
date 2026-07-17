# E253 — D1 simplification-consistent forward corruption (2026-07-16)

Data-transform wiring, not a train/ship run. Code:
[`data/diffusion/simplify.py`](../../src/slm_training/data/diffusion/simplify.py).
Builds on D2 ([`dsl/canonicalize.py`](../../src/slm_training/dsl/canonicalize.py)).
Linear SLM-29.

## What and why

Tree simplification belongs in the **forward** diffusion process, not the reverse
loop. If training targets are pre-canonicalized (D2), every noised intermediate
the denoiser sees is a noised version of the *canonical* tree, so the reverse
model learns to reconstruct one canonical form per layout equivalence class.
Simplifying in the reverse loop instead pushes mid-trajectory states
off-distribution — the model's predictions were trained for the unsimplified
tree.

## Mechanism

`simplify_records(records)` canonicalizes each `ExampleRecord.openui` before the
corruption pipeline tokenizes/masks it (`simplify_target` = D2 `canonicalize`
with pass-through on any failure — never drops a record). Placeholders and all
other fields are preserved; canonicalization only normalizes binder names,
statement order, and style. Reported stats include `distinct_canonical_targets`
— the equivalence-class count the reverse model must learn (lower = a smaller,
cleaner learning target).

Verified: two alpha-renamed copies of one layout collapse to a single canonical
target (`distinct_canonical_targets == 1`), which is exactly the forward-process
property — the denoiser is no longer asked to model surface variation it should
ignore.

## Verification

- `tests/test_data/test_diffusion_simplify.py`: canonical + valid + idempotent
  target; alpha-variants collapse to one target; unparseable passes through;
  placeholders preserved; corpus collapse stats. 5 tests green.

## Downstream

The forward-vs-post-hoc-vs-none comparison is an X-row on
`scripts/run_grammar_matrix.py` (grammar matrix), gated on meaningful parse +
canonical exact-match — a GPU-host run, not claimed here. This change only
supplies the forward-simplification data transform and proves the collapse
property.

## Honesty

Data-transform + unit evidence only. No checkpoint, no scoreboard, no ship claim.
