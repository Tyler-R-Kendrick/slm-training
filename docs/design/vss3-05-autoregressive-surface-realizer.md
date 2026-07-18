# VSS3-05: Constrained autoregressive surface realizer with deterministic fallback

**Issue:** SLM-73

**Status:** wiring / fixture evidence. No trained production checkpoint, no
full-corpus train/eval run, and no `--ship-gates` claim. Deterministic surface
realization remains the default and baseline.

## What was added

A small, standalone causal byte-autoregressor for surface-only slots, kept
outside the main TwoTower/diffusion program generator so it cannot accidentally
become a second generator path.

`src/slm_training/models/surface_autoregressor.py`:

- `SurfaceByteVocab` — printable-ASCII byte vocabulary (`<pad>`, `<bos>`,
  `<eos>`, `<unk>`, `B:xx`). Any lowercase identifier or decorative ASCII text
  can be spelled token-by-token without a corpus-bound tokenizer.
- `IdentifierConstraint` and `DecorativeConstraint` — constrained next-token
  sets for OpenUI-style internal identifiers and decorative text. Enforce
  grammar, reserved-word avoidance, peer uniqueness, and byte budgets at decode
  time.
- Small causal decoder (`CausalSelfAttention`, `SurfaceTransformerBlock`,
  `SurfaceContextEncoder`, `SurfaceAutoregressor`) with prompt context encoding,
  cross-attention, weight tying, and constrained greedy/top-k generation.
- `train_surface_autoregressor` — tiny fixture trainer for (prompt, target)
  pairs, used by tests and future harnesses.
- Save/load and `from_records` for API symmetry.

`src/slm_training/dsl/neural_surface_realizer.py`:

- `NeuralSurfaceRealizerConfig` — runtime knobs (`model`, `model_path`,
  `device`, `max_bytes`, `temperature`, `top_k`, `seed`, `fallback_to_deterministic`).
- `NeuralSurfaceRealizer` — a `SurfaceRealizer` backed by the AR model.
  - Only accepts slots already classified as `SURFACE_ONLY`.
  - Only supports `INTERNAL_IDENTIFIER` and `DECORATIVE_TEXT` kinds.
  - Generates each slot independently under its own constraint mask.
  - Falls back to `DeterministicSurfaceRealizer` per slot on dead end, invalid
    proposal, model error, or when no model is configured.
  - Rejects non-`SURFACE_ONLY` slots before the model is invoked.

`src/slm_training/harnesses/model_build/config.py`:

- Default-off AR fields for API symmetry and future harness wiring:
  `surface_realizer`, `surface_ar_enabled`, `surface_ar_d_model`,
  `surface_ar_n_layers`, `surface_ar_n_heads`, `surface_ar_max_bytes`,
  `surface_ar_temperature`, `surface_ar_top_k`, `surface_ar_fallback`,
  `surface_ar_verify_retry`. Deterministic realization remains the default.

`src/slm_training/dsl/schema.py`:

- Added `"surface_realization"` to `TASK_TOKENS` so derived training records can
  carry the correct task label.

`src/slm_training/data/progspec/surface_rows.py`:

- `derive_surface_realization_records(spec, *, realizer, opaque_bindings,
  include_authorities)` — derives one `ExampleRecord` per realized surface slot
  from a verified `ProgramSpec`, preserving `split` and `split_group_id`.
  Defaults to the deterministic baseline when `realizer` is `None` and only
  emits authorities in `include_authorities` (`SURFACE_ONLY` and
  `OPAQUE_USER_VALUE` by default).

Regression tests:

- `tests/test_models/test_surface_autoregressor.py` — byte vocab round-trip,
  identifier/decorative constraints, fixture overfit, save/load, `from_records`
  (torch-only; skipped where torch is unavailable).
- `tests/test_dsl/test_neural_surface_realizer.py` — neural realizer plugged into
  `realize_surface_and_verify`, trained-identifier verification, no-model
  deterministic fallback, dead-end fallback, disabled fallback error,
  authority/kind rejection. The no-model and guard paths are torch-free; the
  model-backed paths are torch-only.
- `tests/test_data/test_surface_rows.py` — `derive_surface_realization_records`
  emits one record per surface slot, respects `include_authorities`, inherits
  split/group (torch-free), and can use a trained neural realizer (torch-only).

## What is intentionally not added

- The AR realizer is **not** registered as a full `ModelPlugin` in
  `factory.py`. That would create a second program generator, which this issue
  forbids.
- The `ModelBuildConfig` fields are wiring placeholders; no train/eval harness
  automatically instantiates or trains the AR realizer yet.
- No production checkpoint, no full-corpus run, and no ship-gate claim.

## Design boundaries preserved

- `DeterministicSurfaceRealizer` remains the default.
- Autoregression is allowed only for slots the pack classifier marks as
  `SURFACE_ONLY`.
- Each slot is generated under a hard constraint mask; the final program is
  still canonicalized and re-verified by `pack.oracle`.
- Fallback is per-slot and transparent in assignment provenance.
- Fixture-only; meaningful-parse and full ship gates remain the production bar.

## End-to-end example

Input (solved semantic IR, placeholders preserved):

```text
root = Stack([title], "column")
title = TextContent(":hero.title")
```

A tiny fixture-trained model maps the binder slot prompt
`kind=internal_identifier authority=surface_only slot_id=openui:binder:title
symbol=title max=64` to the identifier `title`. `realize_surface_and_verify`
substitutes `title` for the semantic symbol, splices the opaque content
`:user.title`, canonicalizes, and verifies:

```text
root = Stack([v0], "column")
v0 = TextContent(":user.title")
```

The assignment carries the model-chosen value and provenance:

```text
SurfaceAssignment(
    slot_id='openui:binder:title',
    value='title',
    provenance='autoregressive',
)
```

If the model is absent or fails, the same slot falls back to deterministic
`v0` with provenance `autoregressive_fallback:<reason>:deterministic:canonical_name`.

## Caveats

- The model is a scratch causal decoder; real capacity and corpus training are
  future matrix work.
- `DecorativeConstraint` accepts all printable ASCII, but OpenUI V1 decorative
  text slots are currently classified as `OPAQUE_USER_VALUE`, so the AR path for
  decorative text is wired but not exercised end-to-end by the default pack.
- Constrained generation is greedy/top-k; sampling temperature > 0 is supported
  but not ship-gated.
- No checkpoint, no full `--ship-gates` run, and no production claim.
