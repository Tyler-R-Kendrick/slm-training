# Autotrain c1772: bounds confirmation rejected

**Verdict:** reject the c1771 bounds efficiency candidate. Fresh-seed
confirmation produced exact candidate/control prediction and quality ties on
smoke and held-out. The candidate is 2.68% faster on smoke but 1.88% slower on
the held-out primary, so the screening speed signal does not reproduce.

| Suite | Bounds confirmation | Matched control | Decision |
| --- | --- | --- | --- |
| smoke n=3 | parse 1; meaning .3333; structure .46417; binder F1 .63333; p50 2,815.53 ms | exact quality tie; p50 2,893.13 ms | quality tie; speed below 5% |
| held-out n=5 | parse 1; meaning 0; structure .30788; binder F1 .43714; p50 2,931.34 ms | exact quality tie; p50 2,877.23 ms | primary tie; candidate slower |

Both arms are size-matched at 1,608,962 trainable parameters, trained for 21
steps, and have identical final loss 11.79216. AgentV completed both two-suite
bundles with zero execution errors. Gates fail on evidence volume, meaningful
program rate, component recall, AST/canonical equality, and missing ship
suites. Training loss is therefore diagnostic only and supplies no promotion
evidence.

The champion fingerprint is exhausted after this confirmation. No promotion
cycle opens, so Lean is `not_applicable:confirmation`; this is the correct
formal boundary, not omission of prover work. Any future confirmed candidate
still requires the Lean/formal promotion preflight.

Campaign orchestration v78 also preserves a high-confidence observed
model-build successor across a bounded champion confirmation/promotion
interruption until that arm actually executes. This prevents c1770's
literal-close diagnosis from being lost behind the rejected c1771 champion.

These are local CPU scratch checkpoints with explicit no-sync policy. Neither
is reusable, promoted, or ship-ready.

Machine evidence:
[`autotrain-cycle-1772-bounds-confirmation-rejection.json`](autotrain-cycle-1772-bounds-confirmation-rejection.json).
