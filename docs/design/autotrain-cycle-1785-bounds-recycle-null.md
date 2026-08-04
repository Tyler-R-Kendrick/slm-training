# Autotrain c1785: recycled bounds arm is null

**Verdict:** reject. Grammar completion bounds and the size-matched control tie
exactly on every smoke quality metric. Bounds lowers p50 latency by only 1.03%,
below the 5% efficiency screen, while increasing training wall time by 23.5%.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| bounds | 1,608,962; 20 steps; loss 13.53649; 2.89 s | n=3; parse 1; meaning 0; structure .05750; binder/fidelity/reward 0; p50 1,135.36 ms | reject |
| matched control | 1,608,962; 20 steps; loss 13.53649; 2.34 s | identical quality; p50 1,147.14 ms | baseline |

Both AgentV bundles are complete with zero execution errors and ship gates fail.
This CPU fixture is not ship evidence. Both explicit no-sync checkpoints are
provenance-only and must not be reused, promoted, synced, or shipped. Lean is
`not_applicable:screening`; there is no champion or formal proof target.

The more important signal is orchestration debt: bounds was already rejected
and then rejected on fresh confirmation in c1771/c1772, but the bounded
"recent exhaustion" window aged that evidence out and selected the same
approach again. The canonical repair makes approach closure lineage-wide,
fails closed when every registered arm is exhausted, and registers two new
size-matched binder-quality objectives (`binder-arity` and
`binder-component-plan`) rather than recycling runtime knobs. These changes
are `harness.autoresearch.experiment_campaign/v87` and `model.twotower/v292`.

The next run should select the first non-exhausted binder-quality objective and
measure binder F1 plus certified structural quality against a capacity-matched
control. The c1785 handoff's proposed combined bounds/canvas diagnostic is
superseded because that approach is also already closed in the loop lineage.

Machine evidence:
[`autotrain-cycle-1785-bounds-recycle-null.json`](autotrain-cycle-1785-bounds-recycle-null.json).
