# Cactus kernel sketches (not vendored)

These files are **reference sketches** for a future Cactus / NEON transpile
target. They are **not** linked into the PyTorch train path.

## Intent

- `force_emit_sketch.hpp` — DFA singleton emit (`=` `(` `)` `[` `]` `,`) without
  calling the transformer when the OpenUI InteractiveParser accepts set is a
  singleton structural terminal (Leviathan-adjacent force path; see
  [research-lineage.md](../../../../docs/design/research-lineage.md)).
- `maskgit_admit_sketch.hpp` — hole-admissibility check adapted from
  Mündler et al. 2025 ([arXiv:2508.10111](https://arxiv.org/abs/2508.10111),
  [constrained-diffusion.ai](https://constrained-diffusion.ai/)): reject a
  MaskGIT fill if CFG ∩ completion language is empty.

## Boundary

Training and local eval stay in PyTorch (`slm_training.grammar_fastpath`).
Export via `slm_training.cactus.export_checkpoint_bundle`, then transpile
offline with cactus-compute. Do not fold these headers into `models/`.
