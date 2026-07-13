# Cactus kernel sketches (not vendored)

These files are **reference sketches** for a future Cactus / NEON transpile
target. They are **not** linked into the PyTorch train path.

## Intent

- `force_emit_sketch.hpp` — DFA singleton emit (`=` `(` `)` `[` `]` `,`) without
  calling the transformer when the OpenUI InteractiveParser accepts set is a
  singleton structural terminal.
- `maskgit_admit_sketch.hpp` — hole-admissibility check inspired by
  [constrained-diffusion.ai](https://constrained-diffusion.ai/): reject a
  MaskGIT fill if CFG ∩ completion language is empty.

## Boundary

Training and local eval stay in PyTorch (`slm_training.grammar_fastpath`).
Export via `slm_training.cactus.export_checkpoint_bundle`, then transpile
offline with cactus-compute. Do not fold these headers into `models/`.
