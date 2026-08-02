# Autotrain c1725: rotated lever, honest reject, checkpoint-documenting cycle

**Verdict:** fixture/scratch measurement complete, non-positive, no promotion.
This is the first cycle of the `continuous-openui-local` loop where the
candidate arm (`both`) actually carries a different `config_sha256` from
control — c1723/c1724 replayed a config-identical pair, so this is the first
real lever test in this loop. `checkpoint_documentation_required` is `true`,
so both new checkpoints are recorded in `docs/MODEL_CARD.md` and the README
summary in this same commit.

## Result matrix

| Arm | Training | Smoke result | Ship gates | Disposition |
| --- | --- | --- | --- | --- |
| control | 20 steps, loss 15.298, SHA `bb603c2c…3fdfa` | n=3; parse 1.0; meaningful 0.0; binder F1 0.7222; structure 0.1725; p50 7,284.5 ms | fail (insufficient_n, quality thresholds, missing non-smoke suites) | Expected fixture-scale reject |
| both | 20 steps, loss 15.298, SHA `33c62bd3…9ce9be` | n=3; parse 1.0; meaningful 0.0; binder F1 0.7222; structure 0.1725; p50 7,051.83 ms | fail (same reasons) | Expected fixture-scale reject |

`primary_metric_null_or_worse: smoke.binder_reference_f1 control=0.7222
candidate=0.7222 improvement=0.0`. Unlike c1723/c1724, control
(`595a7cb6…8236`) and `both` (`534d4581…6313`) have **distinct** config
hashes this cycle, so the rotated lever genuinely ran — it simply landed on
identical smoke-scale metrics at `n=3`. That is underpowered evidence of an
inert lever, not confirmation of one.

## Next-run priorities

1. Re-test the same rotated lever at a size that can actually separate control
   from candidate (a higher-n suite, or repeated seeds) before concluding it
   is inert.
2. Keep the matched control every cycle.
3. This cycle's checkpoints are fixture screening artifacts only —
   `not promotable or ship`. See the roster rows added to
   [`docs/MODEL_CARD.md`](../MODEL_CARD.md) and the README summary in this
   commit.

Eval commit: `988a6c86e5cbf9e04a4f8b556fee2ea3d6bf47f6`
(`harness.model_build.eval=v73`, `harness.model_build.train=v27`,
`model.twotower=v275`). Machine-readable values are in
[`autotrain-cycle-1725-rotated-lever-null-delta.json`](autotrain-cycle-1725-rotated-lever-null-delta.json).
