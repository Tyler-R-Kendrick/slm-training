# Autotrain c1777: component-inventory confirmation rejects c1776

**Verdict:** reject. Fresh-seed confirmation produces an exact smoke quality
tie: both arms have parse 1, meaning 0, structure .05750, and zero binder F1,
placeholder fidelity, and reward. The component-inventory candidate is also
6.61% slower at p50.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| component-inventory confirmation (weights 1) | 1,682,363; 20 steps; loss 17.31651; 2.90 s | n=3; parse 1; meaning 0; structure .05750; binder/fidelity/reward 0; p50 1,217.79 ms | reject |
| matched control (weights 0) | 1,682,363; 20 steps; loss 16.13611; 2.50 s | n=3; parse 1; meaning 0; structure .05750; binder/fidelity/reward 0; p50 1,142.29 ms | baseline |

Both arms complete every record and emit canonical AgentV bundles with zero
execution errors. The c1776 held-out meaning/structure signal does not
reproduce on this fresh seed, so the champion fingerprint is exhausted. Lower
training loss is not used as a substitute for certified program quality.

These local CPU scratch checkpoints are explicit no-sync evidence. They are
recorded for provenance but are rejected and must not be reused, promoted,
synced, or shipped. Lean is `not_applicable:confirmation`: the candidate is
rejected before formal promotion, while promotion formal preflight remains
mandatory for any future confirmed champion.

The next registered executable priority is `batch1` as a runtime diagnostic;
the model lane should preregister a new quality-targeted objective rather than
recycle the exhausted component-inventory family.

Machine evidence:
[`autotrain-cycle-1777-component-inventory-confirmation-rejection.json`](autotrain-cycle-1777-component-inventory-confirmation-rejection.json).
