# Autotrain c1786: steps quality signal rejected on latency

**Verdict:** reject. Doubling training from 21 to 42 steps lowers fixture loss
36.9% and raises smoke structural similarity from .13527 to .41973, but it
does not move parse, meaningful-program rate, binder F1, component recall, or
placeholder fidelity. Decode p50 rises 154.1% from 1,105.90 to 2,810.05 ms.
The apparent scalar-quality win therefore violates the bounded latency tradeoff
and is not a champion.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| doubled steps | 1,608,962; 42 steps; loss 8.81509; 4.91 s | n=3; parse 1; meaning .3333; structure .41973; binder .6333; recall .1667; fidelity .5278; p50 2,810.05 ms | reject: p50 +154.1% |
| matched control | 1,608,962; 21 steps; loss 13.96397; 3.14 s | n=3; parse 1; meaning .3333; structure .13527; binder .6333; recall .1667; fidelity .5278; p50 1,105.90 ms | baseline |

Both AgentV bundles completed with zero execution errors and both arms fail
ship gates. This CPU smoke fixture is not generalization or ship evidence.
Both explicit no-sync checkpoints are provenance-only and must not be reused,
promoted, synced, or shipped. Lean is `not_applicable:screening`: there is no
confirmed champion and no formal promotion target.

The run exposed two harness defects. First, a high-confidence predecessor
priority could override the permanent lineage-exhaustion set, so the already
tested `steps` arm displaced the selected `binder-arity` successor. Second,
the structural-primary classifier did not apply the latency budget already
used by meaningful-quality wins. The repair separates transient funnel skips
from permanently closed approaches, applies the latency budget to every
quality-primary win, and revalidates durable champion-queue entries under the
current policy. This is
`harness.autoresearch.experiment_campaign/v88`.

The next run should execute the lineage-selected, size-matched `binder-arity`
train-and-decode objective. It should prioritize binder-reference F1 and
certified structural similarity while retaining parse and the latency budget;
scalar loss or extra steps remain diagnostics, not promotion proxies.

Machine evidence:
[`autotrain-cycle-1786-steps-latency-rejection.json`](autotrain-cycle-1786-steps-latency-rejection.json).
