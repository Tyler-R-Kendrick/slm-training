# E46 champion root-target feedback — 2026-07-15

E46 combines lexer output, symbol tables, factorized embeddings, structural masking, template filling, schema conditioning, and the curriculum path.

The 16-step diagnostic point passed smoke: parse 1.0, placeholder fidelity 1.0, structural similarity 0.6489, reward 0.969, and no decode timeouts. This is a smoke-only result; broader suites were not finalized because the matrix supervisor exited during evaluation.

The intended 128-step run completed training with held-out weighted NLL 6.601378989556057, but direct constrained smoke regressed to parse 0.0, structural similarity 0.0, reward 0.0, three decode timeouts, and 6.67s p50. This is strong evidence that loss-based checkpoint selection is misaligned with generation quality. No checkpoint is promoted.

The matrix supervisor now preserves non-interrupt experiment exits as failed results and finalizes progress instead of leaving quality_matrix_progress.json stuck at running.
