# Continuous autotrain: 2026-08-03 (scheduled session, branch dpur3s) cycle 1 — bounds arm null, screening-exhausted

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `620ef79b` (`origin/main` tip at cycle start — post multi-seed
thrash-arm-close fix, PR #1386).

**Verdict:** the `bounds` arm ties its size-matched control byte-for-byte on
the declared primary metric — a null delta. Fixture screening only, not a
ship or promotion claim.

| Arm | Seed | structural_similarity | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | .63333 | 4840.99 |
| bounds | 100001 | .05750 | .63333 | 4720.04 |

Primary delta `0.0`. `meaningful_program_rate` stays 0 on both arms.
Ship gates fail as expected: `insufficient_n` (n=3, need 20);
`held_out`/`adversarial`/`ood`/`rico_held` suites not run this cycle.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone` + null primary delta). No
new stacked layer this cycle — local commit and docs only, per
`autotrain-iteration-delivery`.

## Next priorities

1. The `bounds` lever is now screening-exhausted for this recipe at this
   commit; run the distinct size-matched `component-plan` hypothesis next
   (driver rank 1, confidence 0.90). This hypothesis has reproduced a
   `smoke.structural_similarity` win at seed 100002 across four independent
   prior sessions:
   - [`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
     (merged as PR #1369)
   - [`autotrain-cycle-c4-component-plan-efficiency-win.md`](autotrain-cycle-c4-component-plan-efficiency-win.md)
   - [`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)
     (merged as part of PR #1376)
   - [`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md)
     (merged as part of PR #1378)
2. Keep the matched control as the size-matched baseline every cycle
   (rank 2, confidence 0.70).
3. Do not speculatively attempt the blocked fresh-seed `-confirm`/
   `-fresh-confirmation` frozen-replay path or the seed-`100005` dual-arm
   decode timeout; that needs a dedicated `improve-openui-harnesses` session
   (see [`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)).

Machine evidence:
[`continuous-openui-local-dpur3s-c1-results.json`](continuous-openui-local-dpur3s-c1-results.json).
