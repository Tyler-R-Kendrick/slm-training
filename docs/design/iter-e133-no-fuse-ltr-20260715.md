# E133 no-fused-LTR training path — 2026-07-15

E133 disables fused LTR loss while holding the judged corpus and schema/slot
recipe fixed. Three smoke prompts provide the feedback set.

| Suite | n | Parse | Structural similarity | Reward | p50 / p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.0 | 0.0 | 0.0 | 7,549 / 15,002 ms |

One prompt timed out at the 15-second bound. Placeholder and component recall
were also zero. The no-fused-LTR path is rejected; fused LTR remains enabled.
Training telemetry and the complete AgentEvals bundle are persisted. This is a
negative diagnostic, not a ship result.
