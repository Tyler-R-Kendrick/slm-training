# E76 successor-cache reuse, corrected evaluation — 2026-07-15

The first E76 result was invalid for the intended harness experiment: the
evaluation config omitted decode-only V7 flags, so cluster decoding ran but
successor speculation did not. `_eval_cfg` now restores cluster verification,
survival gating, successor speculation, fanout, overlap, and E76 repair.

The corrected rerun exercised the cache. Hit rate was 76.7% on smoke and
84.4% on held-out. Parse was 2/3 and 3/5; structural similarity was 0.5133
and 0.4726. Placeholder fidelity remained 0.0 in both suites, so the quality
gate failed despite successful cache telemetry.

Decision: retain the eval-config fix and reject E76 as a model/data promotion.
The next loop should target placeholder supervision while preserving the
corrected V7 evaluation path.

This is scratch smoke/held-out evidence, not a ship claim.
