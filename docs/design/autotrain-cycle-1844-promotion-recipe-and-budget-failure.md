# Autotrain c1844: promotion recipe and budget failure

**Verdict:** no model attribution. Fresh Lean proof passed and both matched
trains completed, but every smoke and held-out document timed out. Audit then
found two independent supervisor defects: the inner experiment retained the
pre-formal 46.7-second budget after the outer allocator reclaimed 73.3 seconds,
and the champion transition had dropped `mixture_sampling_policy=capacity_aware`
from the original c1830 winner.

| Arm | Params | Effective / draws | Unique | Loss | Train s | Smoke complete / timeout | Held complete / timeout | Tokens smoke / held | Forwards smoke / held |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| projected control | 1,608,962 | 26.67 / 40 | 31 | 14.9107 | 12.03 | 0 / 3 | 0 / 5 | 33 / 55 | 5 / 5 |
| projected tail candidate | 1,608,962 | 26.67 / 40 | 31 | 19.4127 | 11.90 | 0 / 3 | 0 / 5 | 33 / 55 | 5 / 5 |

Both arms used CPU scratch TwoTower, batch size 2, one thread, seed 101844,
20 steps, and 1,608,962 trainable parameters. Candidate SHA is
`874e3cd2...0b45`; control SHA is `eab02650...260d`. Both are local explicit
no-sync artifacts and are never reusable, promotable, syncable, or shippable
except as hash-bound train reuse for an exact measurement repair. AgentV
completed with zero execution errors and correctly reported all eight document
dispositions as internal decode timeouts. No quality metric is defined.

Lean freshly proved `metrics.structural_similarity_monotone` with evidence SHA
`f9ca0239...1abb`; formal work was not the blocker. The repair now propagates
the exact symmetric post-formal share into the inner executor while retaining
the preregistered ceiling and repository hard cap.

The recipe audit is more consequential: the queue's old lever projection
omitted mixture sampling, semantic-exhaustive alignment, compiler cache, and
draft-window levers. Thus c1838 and c1840/c1841 measured tail loss under
replacement sampling, not the c1830 capacity-aware tail treatment. They remain
valid measurements of those executed checkpoints but are invalid as
confirmation/promotion evidence for c1830. The repair preserves every
registered screening lever and reopens affected queue entries from immutable
source experiments. The c1844 frozen retry is intentionally blocked because
replaying a known-drifted treatment would not answer the preregistered question.

Machine evidence:
[`autotrain-cycle-1844-promotion-recipe-and-budget-failure.json`](autotrain-cycle-1844-promotion-recipe-and-budget-failure.json).
