# Autotrain c1773: literal-close runtime unblock is incomplete

**Verdict:** candidate runtime unblock observed; quality comparison remains
incomplete. The tail-supervised arm completed all three smoke records, while
the size-matched control timed out on all three along an open nested component
trajectory.
One exact frozen replay is required before attributing the runtime difference.

| Arm | Params / train | Smoke | Decode work | Decision |
| --- | --- | --- | --- | --- |
| literal-close (`ltr_tail_loss_weight=2`) | 1,608,962; 20 steps; loss 25.90953; 2.29 s | 3/3 complete; parse 1; meaning 0; structure .09640; binder F1/fidelity/reward 0; p50 1,097.20 ms | 12 forwards; 31,111 unique states; 857 witness expansions; 32,115 parser forks | executable but poor absolute quality |
| matched control (tail weight 0) | 1,608,962; 20 steps; loss 15.72188; 2.49 s | 0/3 complete; three typed timeouts; quality unavailable; p50 including incomplete 9,260.14 ms | 273 forwards; 87,166 unique states; 7,492 witness expansions; 92,458 parser forks | incomplete |

The candidate/control asymmetry supports termination supervision as a runtime
hypothesis, not a quality win. Strict completeness forbids delta attribution
when the control has no quality metrics, and the candidate's absolute meaning,
binder, fidelity, and reward scores are all zero.

Campaign orchestration v79 now inspects both arms. A completed candidate paired
with a control-only finalized model timeout receives one exact frozen replay
without a false `repair_harness` obligation. If reproduced, the runtime unblock
is recorded as model behavior and the arm is retired or advanced by its honest
quality outcome; it cannot enter an unbounded repair loop.

Both AgentV records bundles completed with zero execution errors. Gates fail,
and these local CPU scratch checkpoints are explicit no-sync fixture evidence,
not reusable, promoted, or ship-ready. Lean is `not_applicable:screening`; no
champion exists.

Machine evidence:
[`autotrain-cycle-1773-literal-close-runtime-unblock.json`](autotrain-cycle-1773-literal-close-runtime-unblock.json).
