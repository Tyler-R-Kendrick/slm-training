# Continuous autotrain: 2026-08-04 (session ixpohr) cycle 1 — bounds/control exact tie, byte-identical reproduction (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `eba6db30` (`origin/main` tip at cycle start)

**Verdict:** `bounds` ties its size-matched control on every guarded smoke
quality metric at seed `100001` — the same exact tie, on byte-identical
checkpoints, as the prior 2026-08-03 session `j48f8u` measurement of this
same recipe. Fixture screening only — not a ship or promotion claim.

| Arm | Seed | structural_similarity | binder_reference_f1 | meaningful_program_rate | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .0575 | .6333 | 0 | 4839.67 |
| bounds | 100001 | .0575 | .6333 | 0 | 4587.41 |

Primary metric `smoke.structural_similarity` improvement is `0.0`
(`0.0575 -> 0.0575`) — an exact tie, not a win.

## Deterministic reproduction

Both checkpoint SHA256 digests are byte-identical to the prior session's
independent run of the same `c1` recipe
([`continuous-openui-local-j48f8u-c1-results.md`](continuous-openui-local-j48f8u-c1-results.md)):

- control: `d2f2dc4b...c557e44b`
- bounds: `eb81529a...b224a2f`

This confirms the fixture harness is fully deterministic for this arm pair
at this seed — this cycle adds no new signal beyond re-confirming that
determinism. The only measured delta is p50 latency (`4839.67ms` control vs
`4587.41ms` bounds); per the SDLC quality-aware tradeoff rule, a latency-only
delta with no accompanying quality or held-out movement is **not** a metric
win.

Ship gates fail as expected: `insufficient_n` (n=3, need 20); `held_out`,
`adversarial`, `ood`, and `rico_held` suites are not run at fixture scale.

## SDLC Phase A

**Not positive** (`fixture_insufficient_n_alone` +
`primary_metric_null_or_worse`). No stack layer opened for this cycle; local
commit and docs only, per `sdlc` autotrain-iteration-delivery.

## Next priorities

1. Run the driver's rank-1 successor hypothesis — the distinct size-matched
   `component-plan` arm — rather than repeating this exhausted bounds-only
   arm.
2. Keep the matched control (seed `100001`, `wf_smoke_v2`) as the baseline
   for the next comparative cycle.
3. Soft ship-gate fixture fails never stop the continuous loop; proceed to
   cycle 2 without pausing.

Machine evidence:
[`continuous-openui-local-ixpohr-c1-results.json`](continuous-openui-local-ixpohr-c1-results.json).
