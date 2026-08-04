# Autotrain c1778: batch size one is a runtime null

**Verdict:** reject. Batch size one and the batch-two control tie exactly on
all smoke quality metrics, while batch size one is 7.82% slower at p50 and has
worse final training loss.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| batch 1 | 1,608,962; 22 steps; loss 11.11431; 2.47 s | n=3; parse 1; meaning .3333; structure .17417; binder .6333; fidelity .5278; reward .76533; p50 1,169.59 ms | reject |
| batch 2 control | 1,608,962; 22 steps; loss 9.51297; 2.39 s | identical quality; p50 1,084.79 ms | baseline |

Both AgentV bundles are complete with zero execution errors. Gates fail, and
this local CPU fixture is not ship evidence. Both explicit no-sync scratch
checkpoints are provenance-only and must not be reused, promoted, synced, or
shipped. Lean is `not_applicable:screening`; no champion exists.

Confirmation-family exhaustion is now included in the cooldown set, so the
matrix no longer recycles the rejected component-inventory family. The next
ranked size-matched quality hypothesis is `binder-topology`.

Machine evidence:
[`autotrain-cycle-1778-batch1-runtime-null.json`](autotrain-cycle-1778-batch1-runtime-null.json).
