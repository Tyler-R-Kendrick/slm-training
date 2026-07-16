# E101 early LTR position weighting (2026-07-15)

E101 exposed and exercised the existing `ltr_prefix_loss_weight` mechanism,
adding CLI support so the recipe is reproducible. It trained the same 1,417
record visible-contract corpus for 128 CPU steps with baseline
`ltr_loss_weight=0.5` plus `ltr_prefix_loss_weight=1.0` for the first three
positions.

Training completed with loss `7.44854` and persisted telemetry at
`outputs/runs/iter-e101-ltr-prefix-20260715/e101_ltr_prefix/train_telemetry.json`.
Strict smoke evaluation remained invalid: parse/raw syntax `0.0`, structural
similarity `0.2333`, contract precision/recall `1.0/0.75`, placeholder
fidelity `0.75`, and latency `13749.59 ms`. AgentV failed all five checks.

Decision: reject the weighting change. The early-position emphasis worsened
both structure and fidelity; the CLI option is retained because it makes this
controlled experiment reproducible. Next work should improve repair-target
sampling or decode-state alignment, not add more loss weight.
