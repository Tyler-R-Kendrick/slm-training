# Grammar-aware root-target diagnostic — 2026-07-15

This is a matched follow-up to `iter-remediated-roots-20260715`. It uses the
source-controlled `remediated_roots` corpus (108 records, 94 unique targets,
manifest fingerprint `f8d714f122ac7f091236fd4e562935758de330534cac146abf30af13d0ac98ce`)
with the same TwoTower scratch recipe and 64-step budget, while leaving the
grammar-aware training/decode configuration enabled.

## Result

Training telemetry was complete: CPU, batch size 8, effective batch size 8,
64 steps, 30,705 target tokens, and 18.68 seconds total. Held-out weighted
NLL was 7.270489 at step 64 (broad mean NLL 6.437990), identical to the
matched non-grammar diagnostic.

The corrected constrained smoke evaluation covered 3 examples. Parse rate,
raw syntax validity, structural similarity, component recall, and reward were
all 0.0. There were no decode timeouts; p50 latency was 1,950 ms. The
checkpoint is rejected.

## Decision

Grammar configuration alone did not change optimization or generation quality
at this budget. The next intervention should target the model/tokenizer or
training objective, not merely toggle grammar flags. This run remains a
durable negative result with its AgentEvals bundle under `outputs/runs/`.
