# E1009-E1013 — topology exposure without new target text

Date: 2026-07-22. CPU scratch work under the repository wall cap.

E1009-E1010 attempted to compose one opaque Form and one opaque Tabs fixture
into immutable E937 without synthesis. Both builds are invalid for comparison:
the `existing+fixture` path re-expanded E937 and strict curation reduced it from
524 rows to 117 (cap 1) or 372 (cap 6). Their quality reports and synthesis
feedback were read; they report 490/236 parent-cap drops, high rejection, and
77-97% redundant expansion in the dominant corruption-repair family. Neither
snapshot was trained.

E1011 instead trained untouched E937 with the existing exposure-targeted
sampler: decision budget 32, per-root/per-template caps 2, and importance cap 4.
It completed 450 fresh CPU steps in 87.11s, wrote local checkpoint SHA
`788a485a...8a159a2`, and explicitly disabled sync.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1012 | smoke | 3 | 1.0 | 0.6667 | 0.7778 | 0.4397 | 0.6667 | 0.8843 | 0 / 2 |
| E1013 | held_out | 5 | 1.0 | 0.0 | 0.4700 | 0.2457 | 0.4952 | 0.7768 | 0 / 6 |
| E996 baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

Both evals emitted AgentEvals JSONL and pinned AgentV bundles (`0/1` each).
Generic rare-action exposure over-broadened generation: held Form, Tabs, and
Settings each collapsed to a one-slot `TextContent` fallback. Reject E1011 and
never sync, promote, serve, resume, or use it as a parent. The next experiment
must use a topology-specific objective rather than duplicate targets, derived
corpus recomposition, or generic rare-action exposure.
