# Continuous autotrain: 2026-08-03 (session j48f8u) cycle 5 — component-edge exact tie, non-positive

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c5`
**Integration commit:** `a3a2c861` (this session's cycle-4 docs commit, on
top of `main` tip `0b57e1e8`)

**Verdict:** `component-edge` ties its size-matched control exactly on every
smoke quality metric at seed 100005. Fixture screening only — not a ship or
promotion claim.

| Arm | last_loss | structural_similarity | binder_reference_f1 | placeholder_fidelity | reward_score | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 12.999 | .04307 | .82222 | .72222 | .31033 | 6389.24 |
| component-edge | 13.980 | .04307 | .82222 | .72222 | .31033 | 6971.96 |

Both arms tie exactly on every quality metric despite `component-edge`'s
higher training loss and 583ms slower p50 — another instance of this
session's recurring pattern (see [c3](continuous-openui-local-j48f8u-c3-results.md),
[c4](continuous-openui-local-j48f8u-c4-results.md)) that training loss does
not track certified structural quality on this fixture.

**Seed note:** 100005 is the same seed flagged in
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md)
as causing a dual-arm decode timeout for a different, larger candidate
family (1,755,760 params, `-confirm` suffix) in a prior session. This
cycle's smaller `component-edge` arm (1,766,990 params) completed cleanly
with no timeout — that timeout pathology is specific to the other
candidate/params combination, not this seed generally.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` were not run.

## SDLC Phase A

**Non-positive** (null primary-metric delta,
`smoke.structural_similarity` control=candidate=`.04307`). No stacked PR
layer for this cycle — local commit and docs only.

## Next priorities

1. Test the distinct size-matched `component-inventory` quality hypothesis
   next rather than re-running the now-exhausted `component-edge` knob.
2. Keep the matched control as the size-matched baseline every cycle.
3. Rotate thrash recommendation across the lever bank rather than a single
   knob family.

Machine evidence:
[`continuous-openui-local-j48f8u-c5-results.json`](continuous-openui-local-j48f8u-c5-results.json).
