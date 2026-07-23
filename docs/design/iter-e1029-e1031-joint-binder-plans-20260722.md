# E1029-E1031 — detached joint binder plans

Date: 2026-07-22. CPU scratch work under the repository wall cap.

E1029 tests whether the detached binder-arity head can make bound declarations
reachable often enough for the detached binder-component head to contribute.
It trains both losses at weight 1 from scratch for 450 E937 steps, uses no
parent, and explicitly disables checkpoint sync. The active E937/E938 boundary
was re-audited before training: all 1,214 primary and alternate targets passed
the role contract, and no `Slider(...email...)` target or test fixture remained.

Training completes in 54.83 seconds over 1,800 examples. Checkpoint SHA is
`41955a28...1b9f2f`. Binder-arity loss moves 4.1620 to 1.2475 and sampled
accuracy reaches 0.625. Binder-component loss moves 3.5737 to 3.0196, but its
sampled accuracy remains zero.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1030 | smoke | 3 | 1.0 | 0.6667 | 1.0 | 0.5291 | 0.5833 | 0.9570 | 0 / 0 |
| E1031 | held_out | 5 | 0.6 | 0.4 | 0.44 | 0.2395 | 0.4 | 0.5106 | 2 / 3 |
| E991 arity-only | held_out | 5 | 0.8 | 0.6 | 0.8 | 0.4748 | 0.8 | 0.7736 | 1 / 3 |
| E996 retained baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

Both evals emit AgentEvals JSONL and pinned AgentV bundles (`0/1` each).
Binder arity applies 15 times and changes two smoke choices; it applies 26
times and changes three held choices. Binder component still applies zero
times in both suites. Held Form and Settings time out, Dual Card collapses to
one `TextContent`, and Tabs emits an empty `Tabs([])` plus unrelated content.

Reject E1029 and never sync, promote, serve, resume, or use it as a parent.
The joint objective does not solve declaration reachability and is worse than
the arity-only diagnostic. Close this combination; the next hypothesis must
target the declaration action directly rather than add another detached
classifier loss.
