# Autotrain c1727: positive primary-metric win (fixture scale), harness fix confirmed

**Verdict: positive result** by the `sdlc` autotrain-iteration-delivery gate —
primary metric win plus an efficiency win, at fixture (`n=3`) scale. This is
**not** a ship claim (ship gates still fail; `held`, `adversarial`, `ood`, and
`rico_held` suites are still missing), but it clears the bar for a stacked
delivery layer once documented, per the repository's non-negotiable delivery
law.

This cycle also **confirms a harness fix**: it re-evaluates the exact same
frozen checkpoints from
[c1726](autotrain-cycle-1726-held-out-wall-timeout.md)
(which hit a `MAX_RUN_MINUTES=3` wall timeout on both arms) under
`df143648f504a64eac784f0e6feffe89e804ebfc` — `fix(autotrain): allocate
complete arm budgets (#1287)`, merged upstream between cycles. Both arms now
complete cleanly with no timeout.

## Result matrix

| Arm | Checkpoint (reused, not retrained) | Smoke result | `held_out.structural_similarity` (primary) |
| --- | --- | --- | --- |
| control (21 steps) | `c1c6df79…b4fff` | parse 1.0; meaningful 0.3333; binder F1 **0.9524**; structure 0.4167; p50 18,603.47 ms | 0.4167 |
| steps (42 steps) | `319d4cd9…cf4e0` | parse 1.0; meaningful 0.3333; binder F1 0.8222; structure **0.51**; p50 11,953.61 ms | **0.51** |

## Why this is positive

- `primary_metric_win: held_out.structural_similarity 0.4167 -> 0.51` (
  `improvement=0.0933`, direction `increase`) — the 42-step candidate beats
  the 21-step control on the cycle's declared primary metric.
- `efficiency_win: mpr_per_ms 1.79e-05 -> 2.79e-05` — the win did not cost a
  latency budget; p50 latency for the winning arm is *lower* (11,953.61 ms vs
  18,603.47 ms), the opposite of the original hypothesis ("doubling steps
  only raises cost").
- Per `sdlc` policy, a fixture `insufficient_n` alone is not positive, but a
  genuine primary-metric win with a non-negative efficiency tradeoff is —
  this is the first cycle in the `continuous-openui-local` loop to clear that
  bar.

## What this is not

- `binder_reference_f1` regressed 0.9524 → 0.8222 for the winning arm. That
  metric is not this cycle's declared primary endpoint, so it does not change
  the classification, but it means the win is not unambiguous across every
  quality axis and should not be oversold.
- Ship gates still fail (`n=3` insufficient, `held_out`/`adversarial`/`ood`/
  `rico_held` suites still missing). `climb_state` is `inconclusive` and
  `ship_state` is `blocked` — no promotion.
- One more `retry_measurement` (1/2) remains pending against a third frozen
  arm in this manifest family; this is not yet a fully confirmed champion.

## Delivery

Per `sdlc` autotrain-iteration-delivery, this cycle documents the win here and
then delivers the code/docs delta (this document plus the c1723–c1726 harness
diagnosis chain that led to it) as this loop's first stacked layer.

Eval commit: `8f4b6097916636c3ef2d0b8b01a439917d49ea0f`
(`harness.model_build.eval=v73`, `harness.model_build.train=v27`,
`model.twotower=v275`). Machine-readable values are in
[`autotrain-cycle-1727-steps-structural-similarity-win.json`](autotrain-cycle-1727-steps-structural-similarity-win.json).
