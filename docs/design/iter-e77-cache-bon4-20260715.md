# E77 cache path with best-of-4 — 2026-07-15

E77 repeated corrected E76 with `best_of_n=4`, matching the E75 selection
setting. This isolates candidate selection from the cache/eval-config fix.

Quality was unchanged: smoke parse 2/3, held-out parse 3/5, structural
similarity 0.5133/0.4726, and placeholder fidelity 0.0/0.0. Cache telemetry
scaled with the four candidates: hit rates remained 76.7% and 84.4%, with
184/56 and 304/56 hits/misses for smoke/held-out.

Decision: reject E77. Best-of-4 is not the source of the placeholder failure;
the next intervention must target placeholder supervision or curriculum data.

This is scratch smoke/held-out evidence, not a ship claim.
