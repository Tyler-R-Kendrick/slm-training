# E78 slot-aware supervision with corrected cache path — 2026-07-15

E78 restored E75's trust-gate and slot-aware-trust configuration while keeping
the corrected E76 evaluation flags and best-of-4 selection.

The result was exactly the same as E76/E77: smoke parse 2/3, held-out parse
3/5, structural similarity 0.5133/0.4726, and placeholder fidelity 0.0/0.0.
Cache hit rates remained 76.7% and 84.4%.

Decision: reject E78. The trust and slot-aware flags do not affect this
scratch training path. E75's earlier placeholder-fidelity result is not
reproducible from the current harness/data fingerprint and must not be used
as a promotion claim. The next iteration will reconcile checkpoint metadata
and template-fill behavior.

This is scratch smoke/held-out evidence, not a ship claim.
