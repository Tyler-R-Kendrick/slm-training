# E136 HF context 32-step control — 2026-07-15

E136 extends the valid cached HF-context control from E135 to 32 steps with
the same 405-record judged corpus and recipe.

| Control | Steps | Parse | Placeholder validity | Structural similarity | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| E135 HF | 8 | 0.0 | 0.3167 | 0.2422 | 17,943 ms |
| E136 HF | 32 | 0.0 | 0.0 | 0.0825 | 4,594 ms |

E136 finished at loss **10.87** with complete local-only telemetry and
AgentEvals evidence, but longer training regressed the early signal. This is
a negative diagnostic; do not promote or treat more HF steps as the next
automatic lever. Checkpoint selection and supervision alignment need study.
