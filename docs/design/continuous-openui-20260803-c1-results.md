# Continuous autotrain: 2026-08-03 cycle 1 (non-positive)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Base commit:** `e1f5e4f0` (current `main` tip; also the base of open PRs
[#1367](https://github.com/Tyler-R-Kendrick/slm-training/pull/1367) and
[#1368](https://github.com/Tyler-R-Kendrick/slm-training/pull/1368))

| Arm | Params | parse | MPR | structural_similarity | binder F1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 1.0 | 0.0 | .05750 | .6333 | 1200.65 |
| bounds | 1,608,962 | 1.0 | 0.0 | .05750 | .6333 | 1021.45 |

**Verdict: non-positive.** The `bounds` knob is size-matched against the
control (`wf_smoke_v2`, `steps=20`, `n=3`) and produces an *exact* tie on the
declared primary (`smoke.structural_similarity`) and on every other quality
metric; only p50 latency moved, which is not the primary and is not a
reliable signal at `n=3`. Ship gates fail as expected
(`insufficient_n`, missing `held_out`/`adversarial`/`ood`/`rico_held`
suites) — fixture screening only, not a ship claim.

Per `sdlc` autotrain-iteration-delivery: **no stacked PR** for this cycle —
docs-only local commit. Checkpoints are scratch (`sync_checkpoints=false`),
never reusable/promotable/syncable/shippable; no MODEL_CARD change.

## Coordination note

Two other sessions have open PRs against this exact base commit:

- [#1367](https://github.com/Tyler-R-Kendrick/slm-training/pull/1367) —
  fresh-seed confirmation of the `compiler-decision-token` c1822 candidate;
  2 of 3 seeds null. Recommends **not** re-running more single-seed `n=3`
  confirmations of that candidate.
- [#1368](https://github.com/Tyler-R-Kendrick/slm-training/pull/1368) —
  cycles c1823-c1830, new compiler-decision/capacity-exposure objectives and
  arms; c1830 is a bounded fixture-positive still pending fresh-seed
  confirmation.

This cycle intentionally screens a distinct hypothesis (`bounds`) already in
the existing screening-arm bank rather than duplicating either open PR's
territory.

## Next priorities (ranked by the driver)

1. `component-plan` quality hypothesis (size-matched, distinct from `bounds`).
2. Keep the matched control as the baseline every cycle.
3. Do not re-select the now-exhausted `bounds` arm without a new hypothesis.

Machine evidence:
[`continuous-openui-20260803-c1-results.json`](continuous-openui-20260803-c1-results.json).
