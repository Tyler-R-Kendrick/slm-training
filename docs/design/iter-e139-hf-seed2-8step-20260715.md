# E139 HF context seed-2 8-step control — 2026-07-15

E139 repeats the E135 8-step HF recipe with seed 2, after E138 seed 1
regressed the same diagnostic signals.

| Checkpoint | Seed | Steps | Parse | Placeholder validity | Structural similarity | p50 latency | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E135 | 0 | 8 | 0.0 | 0.3167 | 0.2422 | 17,943 ms | 1 |
| E138 | 1 | 8 | 0.0 | 0.0000 | 0.1683 | 12,491 ms | 0 |
| E139 | 2 | 8 | 0.0 | 0.0000 | 0.0000 | 20,000 ms | 2 |

All three seeds fail parse and reward gates. E139 adds no positive quality
signal and has two per-record decode timeouts. This is not evidence for
changing the judged corpus or loss weights; the next work should explain the
seed-0 outlier through explicit checkpoint-selection and decoder diagnostics.

Training telemetry, the checkpoint hash, constrained decode statistics, and
an AgentEvals JSONL bundle are persisted in the companion run directory.
