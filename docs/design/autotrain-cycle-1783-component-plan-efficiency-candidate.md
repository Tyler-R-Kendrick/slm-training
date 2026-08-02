# Autotrain c1783: component-plan fixture efficiency candidate

**Verdict:** queue one fresh-seed confirmation; do not promote. Component-plan
supervision and its size-matched control tie exactly on every smoke quality
metric, while the candidate lowers p50 latency by 6.01% and raises MPR/ms by
6.39%, clearing the preregistered 5% screening minimum. The sample is only
three fixture documents and all ship gates fail.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| component plan | 1,755,764; 21 steps; loss 12.02760; 6.20 s | n=3; parse 1; meaning .3333; structure .17417; binder .6333; fidelity .5278; reward .76533; p50 1,059.20 ms | fresh confirmation |
| matched control | 1,755,764; 21 steps; loss 8.62998; 2.60 s | identical quality; p50 1,126.88 ms | baseline |

Both AgentV bundles completed with zero execution errors. The repaired evaluator
also proves treatment integrity: training and evaluation agree on compact canvas
off, grammar completion bounds off, and component-plan decode weight 1 for the
candidate versus 0 for the control. That repairs c1781/c1782's attribution
failure; it does not turn this tiny CPU fixture into ship evidence.

This cycle is a new seed-101783 screening run, not the promised content-bound
c1782 replay: both manifests have null `replay_of_manifest_sha256`. The handoff
harness is therefore repaired in `harness.autoresearch.experiment_campaign/v86`
to preserve an exact retry action alongside its repair action. c1783 remains a
valid screening comparison on its own terms, but it cannot satisfy c1782's
replay obligation retroactively.

Both explicit no-sync scratch checkpoints are provenance-only and must not be
reused, promoted, synced, or shipped. Lean is `not_applicable:screening`; formal
promotion preflight remains locked until fresh confirmation identifies a
champion. The next run should use the exact size-matched component-plan and
control recipes on a fresh seed, then either retire the lever or open the Lean
preflight based on complete confirmation evidence.

Machine evidence:
[`autotrain-cycle-1783-component-plan-efficiency-candidate.json`](autotrain-cycle-1783-component-plan-efficiency-candidate.json).
