# Autotrain c1790: binder-component-plan runtime rejection

**Verdict:** reject the current binder-component-plan approach. The exact
frozen replay reused both c1789 checkpoints and reproduced the control-only
typed decode timeout: candidate 3/3 complete, control 2/3 complete. This is a
reproduced runtime effect, but it is not a quality comparison or a promotable
unblock.

| Arm | Checkpoint | Smoke | Decision |
| --- | --- | --- | --- |
| binder-component-plan | reused `d60d6493...893f` | 3/3 complete; parse 1; meaning .3333; structure .29667; binder F1 .8222; recall .1667; fidelity .7222; reward .87767; p50 4,267.55 ms | reject: gates fail; absolute quality low; latency high |
| matched control | reused `36ef2b04...ac45` | 2/3 complete; one timeout; parse 1; meaning 0; structure .12000; binder F1 1; recall 0; fidelity 1; reward .973; p50 3,320.72 ms | reproduced invalid baseline measurement |

The candidate remains 28.5% slower at p50 and its binder F1 remains .1778
lower. Because the control is incomplete, candidate-control quality deltas are
still non-attributable. Both AgentV bundles completed with zero execution
errors and both arms fail ship gates. No new checkpoint was trained in this
cycle; the reused c1789 checkpoints remain local provenance only.

The run exposed two orchestration defects. First, complete non-positive frozen
replays were not added to lineage exhaustion, so successor steering tried to
reopen the already rejected `binder-arity` arm from c1788. Second, the terminal
matrix rendered an unbounded historical argparse trace inside one cell, burying
current results. A live selector audit then found that c1786's complete receipt
still carried its pre-policy `positive=true`, which could recycle the rejected
`steps` arm. Campaign harness v89 closes completed retry arms, reclassifies
complete historical evidence under the current promotion policy, and bounds
every terminal matrix cell. With all earlier arms closed, c1791 preregisters a
size-matched placeholder-fidelity objective (loss weight 0.5 → 1.5) instead of
recycling a rejected approach.

Lean is `not_applicable:retry_measurement`: there is no confirmed champion and
no formal promotion target.

Machine evidence:
[`autotrain-cycle-1790-binder-component-plan-runtime-rejection.json`](autotrain-cycle-1790-binder-component-plan-runtime-rejection.json).
