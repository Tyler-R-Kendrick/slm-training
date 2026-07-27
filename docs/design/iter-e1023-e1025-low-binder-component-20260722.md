# E1023-E1025 — low-weight binder-component plan

Date: 2026-07-22. CPU scratch work under the repository wall cap.

E1023 tests whether reducing the v267 binder-component auxiliary loss from 1
to 0.25 preserves shared prompt features. It completes 450 fresh E937 steps in
49.07 seconds, uses no parent, and explicitly disables sync. Checkpoint SHA is
`b2744cd7...25e020`. Final auxiliary loss is 3.0868 and sampled-batch accuracy
remains zero.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1024 | smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5453 | 0.5 | 0.9490 | 0 / 0 |
| E1025 | held_out | 5 | 0.8 | 0.4 | 0.4733 | 0.3232 | 0.3619 | 0.6396 | 1 / 4 |
| E996 baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

Both evals emit AgentEvals JSONL and pinned AgentV bundles (`0/1` each).
Weight 0.25 improves E1020's held aggregates but leaves Form collapsed, Tabs
empty, one timeout, and zero binder-head applications. No scalar-weight
promotion path remains.

Reject E1023 and never sync, promote, serve, resume, or use it as a parent.
Close the scalar sweep. The next test should isolate binder-component gradients
from shared prompt features rather than lower the weight again.
