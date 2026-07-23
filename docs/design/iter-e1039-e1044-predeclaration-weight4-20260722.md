# E1039-E1044 — pre-declaration weight 4

Date: 2026-07-22. CPU scratch evaluation under the repository wall cap.

E1039 increases only the v269 pre-declaration binder-component decode weight
from 1 to 4 on the rejected E1029 checkpoint. No training or evaluation data
changes. The active boundary remains E937/E938, whose 1,214 primary and
alternate targets have zero role-contract violations.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1039 | smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5257 | 0.5833 | 0.9610 | 0 / 0 |
| E1040-E1044 | five held one-row subsets | 5 | 0.4 | 0.4 | 0.4 | 0.2072 | 0.3333 | 0.3796 | 3 / 1 |
| E1034-E1038 weight-1 control | five held one-row subsets | 5 | 0.6 | 0.4 | 0.44 | 0.2395 | 0.4 | 0.5106 | 2 / 3 |
| E996 retained baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

E1039 changes four declaration choices and lifts smoke strict-v2 from 0.6667
to 1.0, but structure slips slightly and component recall is unchanged.
E1040-E1044 then evaluate the five held rows as completed one-row diagnostic
subsets under the identical policy. Their arithmetic means are diagnostic,
not a canonical full-suite scoreboard or ship evaluation. Form, Dual Card,
and Settings time out. Input remains strict-valid but loses structure, while
Tabs is unchanged. Every run emits AgentEvals JSONL and a pinned AgentV bundle
(`0/6` total).

Reject weight 4 and close the pre-declaration scalar sweep. A smoke-only
strictness gain is not evidence when held parse, fidelity, recall, reward, and
timeouts all regress. Never promote, sync, serve, resume, or parent from
E1029. The next hypothesis must change the declaration supervision or
representation rather than increase this head's decode margin.
