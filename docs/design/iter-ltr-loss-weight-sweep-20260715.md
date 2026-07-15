# Iteration: prefix-LM loss weight sweep (2026-07-15)

Two 64-step seed-0 runs tested stronger left-to-right sequence supervision on
the unchanged corpus and architecture. Both used interval loss feedback every
32 steps and bounded constrained evaluation with eight generation steps.

| LTR loss weight | weighted NLL | structural similarity | placeholder validity | component recall | parse | reward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 control | 7.057 | 0.2333 | 0.2667 | 0.25 | 0.00 | 0.00 |
| 2.0 | 7.294 | 0.5375 | 0.2000 | 0.50 | 0.00 | 0.00 |
| 4.0 | 7.486 | 0.5250 | 0.2000 | 0.50 | 0.00 | 0.00 |

Increasing prefix-LM supervision materially improves partial structure and
component recall, but neither setting produces parseable output or reward. The
weight-4 run adds no structural benefit over weight 2 and worsens NLL, so it is
rejected. Weight 2 is the current structure-oriented candidate for a longer
controlled run; no checkpoint is promoted.

All runs persisted interval loss telemetry, checkpoints, scoreboards, and
AgentV artifacts. These remain scratch diagnostics, not ship claims.
