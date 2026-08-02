# Autotrain c1776: component-inventory needs fresh confirmation

**Verdict:** fixture candidate; do not promote. Component-inventory supervision
raises held-out meaningful-program rate from 0 to .2 and structural similarity
from .06024 to .10690. It simultaneously lowers held-out binder F1 from .6648
to .4371 and raises p50 latency by 17.1%; smoke meaning and structure are exact
ties.

| Arm | Params / train | Smoke | Held-out | Decision |
| --- | --- | --- | --- | --- |
| component-inventory (train/decode weights 1) | 1,682,363; 20 steps; loss 14.25116; 2.70 s | n=3; meaning .3333; structure .13527; binder .6333; reward .76533; p50 1,381.80 ms | n=5; meaning .2; structure .10690; binder .4371; reward .69400; p50 1,494.22 ms | fresh confirmation only |
| matched control (weights 0) | 1,682,363; 20 steps; loss 13.07904; 2.76 s | n=3; meaning .3333; structure .13527; binder .7333; reward .52967; p50 1,207.83 ms | n=5; meaning 0; structure .06024; binder .6648; reward .13140; p50 1,276.43 ms | baseline |

Both arms complete every record and emit canonical AgentV bundles with zero
execution errors. The held-out semantic and reward gains pass the screening
tradeoff policy, but fixture volume is insufficient, all ship gates fail, and
the binder/latency regressions are material. The queue therefore requires the
exact size-matched recipes on a fresh seed before any promotion decision.

These local CPU scratch checkpoints are explicit no-sync evidence. The model
card and README record them for provenance; they are not reusable, promoted,
synced, or ship-ready. Lean is `not_applicable:no_champion`. The result matrix
now makes the next action explicit: fresh confirmation first, and formal
promotion preflight remains locked until confirmation establishes a champion.

Machine evidence:
[`autotrain-cycle-1776-component-inventory-candidate.json`](autotrain-cycle-1776-component-inventory-candidate.json).
