# Autotrain c1724: frozen replay measures complete, honest ship-gate reject

**Verdict:** fixture/scratch measurement now **complete** (unlike c1723) and
**non-positive**. This cycle consumed the pending `retry_measurement` action
from
[c1723](autotrain-cycle-1723-agentv-checkout-gap.md)
by replaying the identical frozen control/bounds arms now that `npm ci` has
installed the AgentV SDK. `scoreboard.json` published cleanly for both arms
(`runner.execution_errors: 0`), confirming the harness gap is resolved. Both
arms still honestly fail ship gates at smoke scale — expected at `n=3` — so no
promotion or stacked PR is authorized.

## Result matrix

| Arm | Smoke result | Ship gates | Disposition |
| --- | --- | --- | --- |
| control | n=3; parse 1.0; meaningful 0.0; binder F1 0.6333; structure 0.0575; p50 2,882.67 ms | fail (insufficient_n, quality thresholds, missing held_out/adversarial/ood/rico_held suites) | Expected fixture-scale reject |
| bounds | n=3; parse 1.0; meaningful 0.0; binder F1 0.6333; structure 0.0575; p50 2,784.73 ms | fail (same reasons) | Expected fixture-scale reject |

`primary_metric_null_or_worse: smoke.binder_reference_f1 control=0.6333
candidate=0.6333 improvement=0.0`. As flagged in c1723, control and `bounds`
carried an identical `config_sha256` again this cycle (this is a frozen replay
of the c1723 manifests by design), so the zero delta is the expected replay
result, not new evidence of a null lever.

## Ship-gate failures (both arms, identical)

```
smoke:insufficient_n actual=3 need>=20
smoke:meaningful_program_rate actual=0.0 need>=0.66
smoke:structural_similarity actual=0.0575 need>=0.35
smoke:component_type_recall actual=0.0 need>=0.35
smoke:ast_beq_rate actual=0.0 need>=0.2
smoke:canonical_beq_rate actual=0.0 need>=0.1
smoke:reward_score actual=0.0 need>=0.3
held_out:missing_suite
adversarial:missing_suite
ood:missing_suite
rico_held:missing_suite
```

This is the expected shape for a `wf_smoke_v2` / smoke-only fixture claim: gate
fails on volume (`insufficient_n`, missing non-smoke suites) are wiring-only,
not a production ship regression.

## Harness repair confirmed

`src/slm_training/evals/agentv.py:_agentv_runtime` now resolves
`node_modules/@agentv/core` after `npm ci`. The identical frozen arm that
crashed in
[c1723](autotrain-cycle-1723-agentv-checkout-gap.md)
completed end to end this cycle with `runner.execution_errors: 0`. No further
harness-family repair is open.

## Next-run priorities

1. Propose a `bounds`/`canvas` candidate whose `config_sha256` genuinely
   differs from control — two cycles in a row have replayed a config-identical
   pair, which cannot produce a metric delta by construction.
2. Keep the matched control every cycle; do not drop it while thrash-screening
   levers.
3. No checkpoint is promoted; `checkpoint_documentation_required` is `false`.

Eval commit: `42e873ea0c11aefe5409b52d02a5a5ef22aeb771`
(`harness.model_build.eval=v73`, `harness.model_build.train=v27`,
`model.twotower=v275`). Machine-readable values are in
[`autotrain-cycle-1724-frozen-replay-honest-reject.json`](autotrain-cycle-1724-frozen-replay-honest-reject.json).
