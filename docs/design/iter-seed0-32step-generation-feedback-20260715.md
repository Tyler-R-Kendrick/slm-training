# Iteration: longer training and generation feedback (2026-07-15)

Seed 0 was trained for **32 steps** with the same scratch TwoTower recipe as
the eight-step controls: batch size `8`, learning rate `6e-4`, effective batch
size `8`, and the unchanged 585-record corpus. The run consumed **44,239 target
tokens**, persisted complete training telemetry, and reached weighted held-out
NLL **8.883**, versus **17.410** at eight steps.

A bounded constrained smoke evaluation (one record, four generation steps, one
attempt) completed in **2,935.05 ms** after the probe-policy fix. It improved
partial generation evidence to structural similarity **0.1125** and placeholder
validity **0.1333**, but parse rate and reward remained **0**.

A decode-only comparison using LTR-primary plus constrained repair was neutral:
parse **0**, structural similarity **0.1125**, placeholder validity **0.1333**,
and reward **0** at **2,956.42 ms**. The lever does not explain the remaining
quality failure, so it is not promoted as a recipe change.

Decision: longer training is promising enough to test at a larger bounded
budget, but generation correctness still gates promotion. These are scratch
diagnostics, not ship claims.
