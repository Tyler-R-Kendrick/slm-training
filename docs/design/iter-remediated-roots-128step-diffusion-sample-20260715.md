# Sampled constrained decode diagnostic — 2026-07-15

Using the best checkpoint from the diffusion-mask root-target run, constrained
decode was switched from greedy legal-token choice to sampled legal-token
choice. The corpus, checkpoint, smoke suite (`n=3`), LTR repair, and 64-token
cap were unchanged.

Sampling did not recover generation: parse rate, structural similarity, and
reward remained 0.0 with zero timeouts. p50 latency increased to 7,068 ms.
The sampled decoder path is rejected for this checkpoint. The failure remains
an early/partial constrained sequence problem, so the next change should make
partial-prefix/EOS handling explicit and testable rather than tune sampling.
