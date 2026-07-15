# Tuned accumulation smoke check (2026-07-15)

The `grad_accum=2, lr=6e-4` candidate improved bounded held-out NLL, but its
two-step, one-decode-step smoke check produced **0.0** parse rate, placeholder
fidelity, structural similarity, and reward. It is not promotable. The result
is retained as a failure boundary: future work needs a longer training budget
and generated-quality validation before making any recipe change.
