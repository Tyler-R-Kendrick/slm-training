# Iteration: 64-step generation feedback (2026-07-15)

Seed 0 continued the unchanged scratch TwoTower recipe to **64 steps** (batch
size `8`, learning rate `6e-4`, effective batch size `8`, same 585-record
corpus). The run consumed **87,055 target tokens**, persisted complete training
telemetry, and reached weighted held-out NLL **7.057**.

The bounded constrained smoke evaluation used one record, eight generation
steps, and one attempt. It completed in **2,972.43 ms** with AgentV and
scoreboard artifacts. Compared with the 32-step checkpoint, partial generation
signals improved:

| steps | weighted NLL | structural similarity | placeholder validity | component recall | parse | reward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8.883 | 0.1125 | 0.1333 | 0.00 | 0.00 | 0.00 |
| 64 | 7.057 | 0.2333 | 0.2667 | 0.25 | 0.00 | 0.00 |

The decoder is now bounded and the partial metrics are moving in the expected
direction, but the output still fails parsing and does not meet ship gates.
Continue the same controlled training trajectory before changing data or loss
weights; this is a scratch diagnostic, not a ship claim.
