# E47 constrained candidate-width diagnostic — 2026-07-15

The 1,024-step E47 checkpoint was evaluated with grammar LTR primary/repair
and `grammar_top_k=512` instead of the normal candidate width. This tests
whether the `root` → `=` dead end is caused by the valid assignment token
falling outside the candidate set.

The result remained 0/3 parse, 0 structural similarity, 0 fidelity, and 0
reward. Latency p50 rose to 28,408 ms. Candidate widening therefore does not
recover the missing assignment transition and is rejected as an intervention.

The evidence points back to model conditioning/learning: the decoder's target
transition is valid, but the model does not assign useful probability to a
complete continuation after `root`.

This is scratch smoke evidence, not a ship claim.
