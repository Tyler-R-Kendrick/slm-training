# Autotrain c1723: AgentV checkout gap on a fresh continuous worktree

**Verdict:** fixture/scratch measurement incomplete, non-positive. This is the
first cycle of a new local continuous loop (`continuous-openui-local`) started
on a fresh checkout. Both 1,608,962-parameter arms trained for 21 steps and the
smoke suite computed real per-document metrics, but the orchestrating
`evaluate_model.py --ship-gates` process crashed while publishing AgentV
ship-gate evidence, so no `scoreboard.json` exists and the driver correctly
recorded this campaign's aggregate metrics as empty. No promotion, ship claim,
or stacked PR is authorized.

## Result matrix

| Arm | Training | Smoke result (raw, from `eval_smoke.json`) | Aggregate scoreboard | Disposition |
| --- | --- | --- | --- | --- |
| control | 21 steps, 101 records, last_loss 22.622 | n=3; parse 1.0; meaningful 0.0; binder F1 0.6333; structure 0.0575; p50 3,258.86 ms | crashed before write | Non-positive; retry pending |
| bounds | 21 steps, 101 records, last_loss 22.622 | n=3; parse 1.0; meaningful 0.0; binder F1 0.6333; structure 0.0575; p50 2,673.03 ms | crashed before write | Non-positive; retry pending |

`primary_metric_null_or_worse: smoke.binder_reference_f1 control=0.6333
candidate=0.6333 improvement=0.0`. The two arms also carry an identical
`config_sha256` (`81fd4757dc398eeca1cfafdc8f8b41feb0a38f26d343d19c118af2b2bba268b8`)
in this cycle's manifests, so even a clean evaluation would have measured zero
lever delta — the hypothesizer proposed no actual knob difference between
control and the `bounds` candidate this cycle.

## Harness gap found and repaired

`evaluate_model.py --ship-gates` unconditionally calls
`publish_model_evaluation` → `publish_agentv_evaluation`, which resolves the
pinned AgentV SDK from `node_modules/@agentv/core`
(`src/slm_training/evals/agentv.py:_agentv_runtime`). This checkout's worktree
had never run `npm ci`, so the SDK was absent and both arms failed identically:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

Repair: ran `npm ci` at the repo root (`package.json` already pins
`@agentv/core@4.42.4`; no source change was required). `node_modules/` is
gitignored, so this repair leaves no tracked diff — it is an environment-setup
step, not a harness code fix, and is not routed through
`improve-openui-harnesses`. Any future continuous-loop worker starting from a
fresh clone of this repository needs the same `npm ci` before its first
`--ship-gates` evaluation.

## Next-run priorities

1. Consume the pending `retry_measurement` action against frozen manifest
   `e0a794d1b80bcfee1caaaa77b3e03b113007c75f9aa18422efb48a09f489efd3`: replay
   the identical control/bounds arms now that the AgentV SDK resolves, and
   check whether the raw smoke metrics (parse 1.0, binder F1 0.6333, meaningful
   0.0) hold once the scoreboard actually publishes.
2. After a complete replay, have the hypothesizer propose a `bounds` arm whose
   `config_sha256` actually differs from control — this cycle's candidate was a
   config-identical no-op, which trivially cannot produce a metric delta.
3. No checkpoint is promoted from this cycle; `checkpoint_documentation_required`
   is `false` and no `MODEL_CARD.md` / README update is needed.

Eval commit: `7c2929811c710542e824ca7eb96f6f862695e13c`
(`harness.model_build.eval=v73`, `harness.model_build.train=v27`,
`model.twotower=v275`). Machine-readable values and artifact boundaries are in
[`autotrain-cycle-1723-agentv-checkout-gap.json`](autotrain-cycle-1723-agentv-checkout-gap.json).
