# Continuous autotrain: 2026-08-03 (session j48f8u) cycle 6 — component-inventory exact tie, non-positive

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c6`
**Integration commit:** `e9b3a0cc` (this session's cycle-5 docs commit, on
top of `main` tip `a3a2c861`)

**Verdict:** `component-inventory` ties its size-matched control exactly on
every smoke quality metric at seed 100006. Fixture screening only — not a
ship or promotion claim.

| Arm | last_loss | structural_similarity | binder_reference_f1 | reward_score | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 13.720 | .0964 | .82222 | .82367 | 1417.29 |
| component-inventory | 14.913 | .0964 | .82222 | .82367 | 1464.53 |

Both arms tie exactly on every quality metric despite
`component-inventory`'s higher training loss and slightly slower p50. This
is the sixth cycle this session and the **fourth exact-tie result** (c1,
c4, c5, c6), reinforcing the pattern from
[c3](continuous-openui-local-j48f8u-c3-results.md)/[c4](continuous-openui-local-j48f8u-c4-results.md)/[c5](continuous-openui-local-j48f8u-c5-results.md)
that training-loss differences on this fixture do not translate into
certified structural-quality differences at eval time.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` were not run.

## SDLC Phase A

**Non-positive** (null primary-metric delta,
`smoke.structural_similarity` control=candidate=`.0964`). No stacked PR
layer for this cycle — local commit and docs only.

## Next priorities

1. Test the distinct size-matched `binder-topology` quality hypothesis next
   rather than re-running the now-exhausted `component-inventory` knob.
2. Keep the matched control as the size-matched baseline every cycle.
3. Rotate thrash recommendation across the lever bank rather than a single
   knob family.

Machine evidence:
[`continuous-openui-local-j48f8u-c6-results.json`](continuous-openui-local-j48f8u-c6-results.json).
