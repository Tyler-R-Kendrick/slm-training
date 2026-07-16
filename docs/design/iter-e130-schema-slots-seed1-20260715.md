# E130 schema/slot seed-1 control — 2026-07-15

E130 repeats the E127 recipe with seed 1 over the same 405 judged records.
The corpus is dominated by edit trajectories (279) and corruption repairs
(93); their mean placeholder counts are nearly identical (1.97 and 1.96), so
there is no obvious placeholder-sparse family to remove.

| Suite | n | Parse | Placeholder validity | Normalized fidelity | Structural similarity | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 1 | 0.0 | 0.0 | 0.0 | 0.1542 | 0.0 |

The 32-step seed-1 run finished at loss **15.28** with persisted telemetry.
E127's one-example placeholder improvement is not reproducible, so it is not
promotable. The next experiment should use more than one feedback prompt and
test task/source composition rather than further single-example tuning.
