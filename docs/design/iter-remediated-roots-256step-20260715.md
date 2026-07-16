# Longer root-target training diagnostic — 2026-07-15

The source-controlled `remediated_roots` corpus was trained for 256 steps to
test whether the 64-step parse failure was only under-training.

## Recipe and telemetry

- TwoTower, scratch context, CPU
- 108 records / 94 unique targets
- Batch size 8, effective batch size 8, random mask, LTR loss weight 2.0
- 256 steps, 121,888 target tokens
- Total telemetry time: 71.26 seconds
- Corpus fingerprint: `f8d714f122ac7f091236fd4e562935758de330534cac146abf30af13d0ac98ce`

Held-out weighted NLL reached its best value of 6.450338 at step 128. The
final step-256 value was 7.170944, so the final checkpoint was not selected.

## Constrained feedback

The best weighted-NLL checkpoint was evaluated with explicit grammar-constrained
decoding, LTR-primary repair, a 64-token LTR cap, one attempt, and the smoke
suite (`n=3`). Parse rate, raw syntax validity, structural similarity,
component recall, and reward were all 0.0. There were zero decode timeouts and
p50 latency was 2,116 ms.

## Decision

Reject the checkpoint. Longer training improves held-out loss but does not
produce valid constrained OpenUI. The next loop should inspect or change the
output model/tokenizer/objective and validate that intervention with the same
constrained smoke feedback.
