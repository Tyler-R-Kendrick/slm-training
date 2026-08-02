# Autotrain c1798: semantic contrast rejected after complete replay

**Verdict:** reject the semantic-contrast approach, not the TwoTower model
family. Eval v77 completed the exact frozen c1796 control and treatment on all
eight fixture documents with no runtime failure. At identical 1,608,962
parameters, semantic contrast reduced the held-out primary from 0.25042 to
0.16810 (delta -0.08232).

| Reused arm | Params | Smoke structure | Held-out structure | Runtime | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| matched control, contrast 0.0 | 1,608,962 | 0.32333 | 0.25042 | 3/3 + 5/5 complete | retain baseline |
| semantic contrast 0.25 | 1,608,962 | 0.28333 | 0.16810 | 3/3 + 5/5 complete | reject approach |

No training ran and no checkpoint was created. The replay retained the c1796
checkpoint hashes (`8ad84a58...84be0` control and `feb745fc...efb4`
treatment), recipe, manifests, matched 835-pair exposure, seed, and size.
AgentV completed both bundles with zero execution errors. Both arms fail honest
ship gates: this is fixture evidence (smoke n=3, held-out n=5), and neither arm
meets the structural/meaningful-program thresholds.

The runtime repair did what it was meant to do. Each suite executed as one
production batch (`3` smoke rows, `5` held-out rows; configured batch size 16),
preserved row-tagged decode statistics, and reported zero timeouts. The
treatment was faster at held-out p50 (2662.72 ms versus 3322.69 ms), but the
quality loss controls the declared decision.

The run also exposed a reporting defect after the model decision was already
available: recursive outcome flattening spent its metric cap inside verbose
smoke task details and omitted later held-out headlines, so the terminal
`Primary` cell rendered `—`. Campaign harness v97 extracts every suite headline
independently and the storage projection can recover old primaries from the
content-addressed stage payload. The corrected c1798 terminal matrix now shows
0.25042 and 0.16810. This post-run repair changes reporting, not the run result.

Lean is `not_applicable:retry_measurement`: no candidate reached confirmation
or promotion, so no promotion preflight was authorized. The next priority is a
new preregistered, size-matched quality objective; the registered quality-arm
bank is exhausted and must not recycle this rejected approach.

Machine evidence:
[`autotrain-cycle-1798-semantic-contrast-replay-rejected.json`](autotrain-cycle-1798-semantic-contrast-replay-rejected.json).
