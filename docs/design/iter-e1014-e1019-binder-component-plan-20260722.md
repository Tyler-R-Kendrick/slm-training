# E1014-E1019 — binder-indexed component plan

Date: 2026-07-22. CPU scratch work under the repository wall cap.

E1014 replayed the compiler decisions for the Form/Tabs portion of immutable
E937. The initial all-surface replay exceeded the 110-second command cap and is
invalid. A focused replay over the 17 unique Form/Tabs surfaces completed in
19.5 seconds. E937 contains 17 Form and 24 Tabs surface occurrences across its
582 primary/accepted targets. The unique surfaces expose 59 `component_bound`
decisions, and every decision ranks all 32 legal component types. The rejected
binder-topology and binder-arity heads do not supervise those labels.

No target text, fixture, or corpus was added. E1015 enabled only the existing
generalized binder-indexed component-plan loss on E937, but stopped at the
cumulative wall budget after 59/100 steps. It is invalid and was not evaluated,
resumed, synced, or used as a parent. E1016 restarted from scratch with batch
size 2 and completed 80/80 steps in 58.58 seconds. The auxiliary loss fell
3.4402 to 2.7966; final sampled-batch accuracy was 0.1429 over seven 32-way
decisions. Checkpoint SHA is `4f7fb4a9...5f345bf`; sync was explicitly disabled.

E1017 completed before the emitted policy comparison found
`compiler_schema_component_types=false`; it is retained only as a non-comparable
diagnostic. E1018 and E1019 use the complete retained v266 policy plus
binder-component decode weight 1.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1018 | smoke | 3 | 0.6667 | 0.3333 | 0.5556 | 0.4461 | 0.3333 | 0.5913 | 1 / 0 |
| E1019 | held_out | 5 | 0.8 | 0.4 | 0.5000 | 0.2290 | 0.4286 | 0.6572 | 1 / 3 |
| E996 baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

Both matched evals emitted AgentEvals JSONL and pinned AgentV bundles (`0/1`
each). The learned head applied three times and changed one held choice. That
change is a real narrow success: `held_out_tabs_01` emits
`Tabs([TabItem(...)])` with parse, fidelity, and component recall 1.0,
structure 0.62, and reward 0.985. Form still collapses to one-slot
`TextContent`, and aggregate quality regresses substantially.

Reject E1016 and never sync, promote, serve, resume, or use it as a parent.
Retain the causal Tabs signal. The next arm should optimize or precompute the
shared compiler component-bound supervision so a full-corpus run fits the wall
budget; it must not add semantic text, duplicate topology fixtures, or
component-name special cases.
