# Autotrain local-loop c4: frozen-replay harness fix validated, timeout persists

**Verdict:** the hyphenated-slug fix
(`harness.autoresearch.experiment_campaign` v64,
[`scripts/run_autotrain_continuous.py`](../../scripts/run_autotrain_continuous.py))
works — this cycle's automatic `retry_measurement` replay of the frozen c2
`component-plan`/`control` arms no longer raises
`RuntimeError('unsupported automatic frozen replay arm: plan')`. The
`component-plan` arm reused c2's frozen checkpoint (no retrain) and
reproduced the identical eval result (`structural_similarity=0.0964`,
gate reject). This is an **executable unblock of the crash**, but not yet
a complete measurement: the `control` arm's `scripts.evaluate_model
--ship-gates` stage hit the harness wall-time cap again, with no
scoreboard produced, exactly as it did in c2.

| Arm | Source | Structure | Eval status |
| --- | --- | ---: | --- |
| component-plan | reused c2 checkpoint | .0964 | complete (gate reject) |
| control | frozen c2 recipe | — | wall-timeout, no scoreboard (2nd occurrence) |

Because the same specific blocker (`scripts.evaluate_model` wall-timeout on
the control arm) has now recurred on two consecutive cycles (c2, c4), the
next automatic replay should be preceded by a look at whether the harness's
Node/AgentV cold-start cost reliably fits inside the continuous cycle's
`max_wall_minutes` budget in this environment, rather than a third blind
retry — per the loop law, three identical failures with no new information
is the threshold for reporting a hard block.

Lean is `not_applicable:screening`; no promotion claim is made. No new
checkpoint was written this cycle (`checkpoint_documentation_required=false`
per the driver's handoff), so MODEL_CARD/README are unchanged.

Machine evidence:
[`autotrain-cycle-8c0b60dd-c4-replay-harness-fix-validation.json`](autotrain-cycle-8c0b60dd-c4-replay-harness-fix-validation.json).
Harness fix: [`autotrain-cycle-8c0b60dd-c2-component-plan-timeout.md`](autotrain-cycle-8c0b60dd-c2-component-plan-timeout.md)
(the cycle that first exposed the bug).
