# Autotrain local-loop c2: component-plan measurement incomplete

**Verdict:** inconclusive — not model evidence. The `component-plan`
candidate arm trained (22 steps, 1,755,764 params) and completed its smoke
eval (fixture `n=3`, gate reject as expected), but the matched control's
`evaluate_model --ship-gates` stage exceeded the harness wall-time limit
and produced no scoreboard. Without a completed matched control there is no
size-matched comparison; the candidate's `structural_similarity=0.0964` is
recorded for provenance only and is **not** compared against any baseline.

| Arm | Params | Steps | Loss | Train wall | Smoke n | Structure | Eval status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| component-plan | 1,755,764 | 22 | 19.9160 | 8.760 s | 3 | .0964 | complete (gate reject) |
| control | 1,755,764 | 22 | 14.3902 | 4.102 s | — | — | wall-timeout, no scoreboard |

Both CPU scratch arms used `wf_smoke_v2` fixture data and strict
compiler-tree constrained evaluation. AgentV (`@agentv/core`) ran without
execution errors on the candidate; the control never reached the AgentV
stage because `scripts.evaluate_model --ship-gates` itself hit the
repository's harness wall-time cap first. This is a soft infrastructure
timeout, not a model result, and per the continuous-loop law does not stop
the loop.

Lean is `not_applicable:screening`; no promotion claim is made. Neither
checkpoint is synced, reusable, served, promoted, or ship evidence.

Next: the driver queued a `retry_measurement` action
(`frozen_manifest_sha256=e7615c35…f2f5fc1dd`) to replay the identical frozen
control and component-plan arms before any new hypothesis is tested, so the
partial run is never silently treated as a negative result.

Machine evidence:
[`autotrain-cycle-8c0b60dd-c2-component-plan-timeout.json`](autotrain-cycle-8c0b60dd-c2-component-plan-timeout.json).
