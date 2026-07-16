# E137 HF context 16-step checkpoint selection — 2026-07-15

E137 fills the midpoint between E135's 8-step HF control and E136's 32-step
regression.

| HF checkpoint | Steps | Parse | Placeholder validity | Structural similarity | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| E135 | 8 | 0.0 | 0.3167 | 0.2422 | 17,943 ms |
| E137 | 16 | 0.0 | 0.4000 | 0.2142 | 7,687 ms |
| E136 | 32 | 0.0 | 0.0 | 0.0825 | 4,594 ms |

The trajectory is non-monotonic. E137 has the strongest placeholder-validity
signal, but no checkpoint parses or earns reward, so none is promotable. The
next harness improvement should support explicit early-checkpoint evaluation
and selection rather than assuming the final step is best. Telemetry and
AgentEvals are persisted.
