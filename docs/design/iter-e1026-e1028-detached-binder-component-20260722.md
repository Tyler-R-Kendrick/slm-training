# E1026-E1028 — detached binder-component plan

Date: 2026-07-22. CPU scratch work under the repository wall cap.

Model v268 detaches pooled prompt features before the binder-component
auxiliary head. A regression test proves the head receives nonzero gradients
while every base-model gradient exactly matches a no-head control. No data,
target, candidate, or decode rule changes.

E1026 completes 450 fresh E937 steps in 38.82 seconds, uses no parent, and
explicitly disables sync. Checkpoint SHA is `0a5df8be...55d27b`. Final
auxiliary loss is 3.0196 and sampled-batch accuracy remains zero.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1027 | smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5453 | 0.5 | 0.9490 | 0 / 0 |
| E1028 | held_out | 5 | 0.6 | 0.4 | 0.4333 | 0.3050 | 0.2952 | 0.5062 | 2 / 3 |
| E996 baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

Both evals emit AgentEvals JSONL and pinned AgentV bundles (`0/1` each).
E1028 is prediction-identical to E1022 and records zero binder-plan
applications. The head cannot help when generation never reaches a bound
component declaration.

Retain v268 gradient isolation. Reject E1026 and never sync, promote, serve,
resume, or use it as a parent. Diagnose declaration reachability before
training or tuning this head again.
