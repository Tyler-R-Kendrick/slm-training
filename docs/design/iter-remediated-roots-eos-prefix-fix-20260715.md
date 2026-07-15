# Constrained EOS prefix fix — 2026-07-15

The diffusion-mask checkpoint collapsed constrained outputs to root because incremental grammar acceptance treats incomplete prefixes as UnexpectedEOF, which previously allowed the model's EOS choice to terminate before a complete OpenUI program. The decoder now requires the actual OpenUI parser to certify the prefix before EOS is legal. A focused regression test covers root with EOS as the highest logit.

The existing checkpoint was re-evaluated after this harness change. Smoke remained 0/3 parse, 0 structural similarity, and 0 reward, with zero timeouts and p50 latency 1,819 ms. This is a correctness fix and a rejected model result, not a quality claim; the model still needs an objective/data change.
