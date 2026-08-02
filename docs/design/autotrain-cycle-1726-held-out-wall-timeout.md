# Autotrain c1726: held_out wall timeout, best-so-far smoke signal unconfirmed

**Verdict:** fixture/scratch measurement incomplete (soft failure — a wall
timeout, not a hard block). This promotion-role cycle asked for `smoke,
held_out` suites with declared primary metric `held_out.structural_similarity`.
Both arms hit the repository-wide `MAX_RUN_MINUTES=3` wall cap before
`held_out` finished, so **no promotion or ship claim is authorized** and the
loop continues per the soft-failure rule (timeouts never stop the loop).

## Result matrix

| Arm | Training | Smoke suite | held_out suite | Disposition |
| --- | --- | --- | --- | --- |
| control | 21 steps, loss 8.852 | timed out before any metric captured | timed out | Wall timeout; zero metrics |
| steps (candidate, 2x steps) | 42 steps, loss 8.088 | **completed**: parse 1.0, meaningful **0.3333** (first non-zero in this loop), binder F1 **0.8222**, structure **0.51**, p50 11,662.32 ms | timed out | Smoke result is the best in this loop so far, but the cycle's declared primary metric (`held_out.structural_similarity`) never materialized |

The hypothesis under test was "doubling steps without levers only raises cost
and does not improve unit decode latency" — the candidate's smoke numbers are
suggestive of the opposite (quality went up, not just cost), but this is not
evidence of a positive result: the primary endpoint is `held_out.*`, not
`smoke.*`, and it is unavailable. Per `sdlc` autotrain-iteration-delivery,
"it ran [smoke only]" without the declared primary metric or unblocking proof
is explicitly **not positive**.

## Next-run priorities

1. Consume the pending `retry_measurement` action against frozen manifest
   `6acffe8d5cc4e13e9bafd106f085334030254022383a82a72cc942722eb53acf` — replay
   the identical arms. If `held_out` keeps timing out at the 3-minute wall
   cap with two suites, the next hypothesis should split `smoke` and
   `held_out` into separate wall-capped cycles rather than widening the cap
   (the cap is a repository-wide invariant, not a per-experiment knob).
2. If a future cycle DOES get a complete `held_out` measurement for the
   `steps` candidate, revisit this cycle's smoke numbers (meaningful 0.3333,
   binder F1 0.8222) — they are the first non-zero meaningful-program smoke
   result in this loop and worth a deliberate follow-up, not just a
   by-product of an unrelated promotion attempt.
3. Both new checkpoints are screening artifacts only —
   `checkpoint_documentation_required` is `true`; roster rows are added to
   `docs/MODEL_CARD.md` and the README summary in this same commit, both
   explicitly marked not promotable/ship pending a complete measurement.

Eval commit: `412737c1a5efd8dd9d5d24df1e5a1c9c5bb555d1`
(`harness.model_build.eval=v73`, `harness.model_build.train=v27`,
`model.twotower=v275`). Machine-readable values are in
[`autotrain-cycle-1726-held-out-wall-timeout.json`](autotrain-cycle-1726-held-out-wall-timeout.json).
