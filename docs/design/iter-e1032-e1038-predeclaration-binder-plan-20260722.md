# E1032-E1038 — pre-declaration binder-component plan

Date: 2026-07-22. CPU scratch evaluation under the repository wall cap.

v269 reuses the trained E1029 binder-component head at the earlier typed
bind-reference-versus-inline-component decision. It scores only completion
sets containing both a bind path and a proper typed subset of inline
components, and otherwise abstains. No training data changed. The evaluated
checkpoint remains the rejected, local-only E1029 scratch checkpoint.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1032 | smoke | 3 | 1.0 | 0.6667 | 1.0 | 0.5291 | 0.5833 | 0.9570 | 0 / 0 |
| E1034-E1038 | five held one-row subsets | 5 | 0.6 | 0.4 | 0.44 | 0.2395 | 0.4 | 0.5106 | 2 / 3 |
| E1031 v268 control | held_out | 5 | 0.6 | 0.4 | 0.44 | 0.2395 | 0.4 | 0.5106 | 2 / 3 |
| E996 retained baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

E1033 attempted the canonical five-row held suite, but its outer command hit
the 110-second cap and is invalid evidence. E1034-E1038 therefore evaluate the
same five rows as completed one-row diagnostic subsets under the identical
checkpoint and decode policy. Their arithmetic means are diagnostic only, not
a canonical full-suite scoreboard or ship evaluation.

The pre-declaration head applies 15 times and changes one choice on smoke.
Across the three non-timeout held rows it applies 26 times but changes no
choice. The reconstructed held metrics and predictions are exactly unchanged
from E1031. Form and Settings time out; Dual Card collapses to one
`TextContent`; Input parses; Tabs emits an empty `Tabs([])` plus unrelated
content. Every completed eval emits AgentEvals JSONL and a pinned AgentV bundle
(`0/6` total).

Reject this decode arm: it proves that the trained head is reachable before
declaration, but it produces no held decision or metric gain. Never promote,
sync, serve, resume, or parent from E1029. Retain the guarded v269 capability
as an inactive-by-default diagnostic hook; do not enable its weight in the
retained policy. The next arm must improve the declaration decision signal
rather than move the same weak classifier to another decision point.
