# Autotrain c1770: component-edge measurement is incomplete

**Verdict:** no model-quality conclusion. Both size-matched arms trained, but
all three smoke documents in each arm ended as typed decode timeouts. Strict
measurement completeness therefore leaves parse, meaning, structure, binder
F1, and the primary metric unavailable.

| Arm | Params / train | Smoke result | Decode work | Decision |
| --- | --- | --- | --- | --- |
| component-edge | 1,766,987; 20 steps; loss 14.19694; 3.33 s | n=3; completed 0; timeouts 3; p50 including incomplete 8,815.11 ms | 203 forwards; 35,832 unique states; 37,985 transition misses; 21,071 witness expansions; 43,548 parser forks | incomplete |
| matched control | 1,766,987; 20 steps; loss 12.96926; 3.37 s | n=3; completed 0; timeouts 3; p50 including incomplete 8,858.58 ms | 202 forwards; 35,822 unique states; 37,965 transition misses; 20,984 witness expansions; 43,518 parser forks | incomplete |

The two arms have effectively identical runtime amplification. Their bounded
selection traces repeatedly extend a numeric `Slider` literal while legal
`LIT_END` remains far below digit bytes. This is observed model behavior, not
evidence that the timeout or cache policy should be widened. The next run must
leave constrained decoding and timeout policy unchanged and use the already
registered size-matched `literal-close` arm (`ltr_tail_loss_weight=2` versus
0) to test termination-aware tail supervision on a fresh seed.

The supervisor previously labeled this as `repair_harness` and said to add a
tail-weighted signal, even though that typed lever, CLI plumbing, and arm were
already present. Campaign orchestration v76 corrects the action to
`next_experiment`; actual harness failures still require repair receipts.

Both AgentV bundles completed with zero execution errors, but their assertions
failed because quality was unavailable, smoke n=3 is below ship volume, and
the remaining suites were not run. These local CPU scratch checkpoints are
explicit no-sync fixture artifacts. They are not reusable, promoted, or ship
candidates. Lean is `not_applicable:no_champion`; formal proof remains required
if a future fully measured candidate reaches promotion.

Machine evidence:
[`autotrain-cycle-1770-component-edge-incomplete.json`](autotrain-cycle-1770-component-edge-incomplete.json).
