# Autotrain c1789: binder-component-plan incomplete screen

**Verdict:** inconclusive. The matched control completed only two of three smoke
documents and recorded one typed decode timeout. The candidate completed all
three, so its apparent meaningful-program and structural gains cannot be
attributed to binder-component-plan supervision from this run.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| binder-component-plan | 1,897,922; 23 steps; loss 13.69051; 7.27 s | 3/3 complete; parse 1; meaning .3333; structure .29667; binder F1 .8222; recall .1667; fidelity .7222; reward .87767; p50 4,155.15 ms | incomplete comparison; no promotion |
| matched control | 1,897,922; 23 steps; loss 10.48466; 3.85 s | 2/3 complete; one timeout; parse 1; meaning 0; structure .12000; binder F1 1; recall 0; fidelity 1; reward .973; p50 3,169.52 ms | invalid baseline measurement |

Both arms wrote local explicit no-sync checkpoints and AgentV bundles, and both
fail ship gates. The control's typed timeout makes the comparison non-scoreable.
Even as a diagnostic, the candidate is 31.1% slower at p50, trains 88.9% longer,
has 30.6% higher loss, and lowers reported binder F1 by .1778; none of its
reported gains are promotion evidence.

The governed next action is one exact frozen replay under aggregate manifest
digest `b0c287c80dcc7a97b41f46de2bcd5b0c29c9e4b7c947ba472e59ce5ef2699855`.
That replay must preserve arms, seed, steps, endpoints, gates, and stopping rule.
If the timeout reproduces, the loop should open a typed model-build harness
repair; if it does not, the complete pair can be judged under the existing
quality, binder-F1, and latency constraints.

Lean is `not_applicable:incomplete_measurement`: there is no confirmed champion
and no formal promotion target.

Machine evidence:
[`autotrain-cycle-1789-binder-component-plan-incomplete.json`](autotrain-cycle-1789-binder-component-plan-incomplete.json).
