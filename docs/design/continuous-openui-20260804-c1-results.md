# Continuous autotrain: 2026-08-04 cycle 1 (non-positive, repeated null)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `8551a47f` (`main` tip at the start of this container's
session, itself a merge of many prior autotrain PRs including #1373)

| Arm | Params | parse | MPR | structural_similarity | binder F1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 1.0 | 0.0 | .05750 | .6333 | 4930.61 |
| bounds | 1,608,962 | 1.0 | 0.0 | .05750 | .6333 | 4687.02 |

**Verdict: non-positive (repeated null).** At the default seed (100001), the
`bounds` knob again ties exactly with its matched control on every quality
metric. This is the same null already recorded on 2026-08-03 in
[`continuous-openui-20260803-c1-results.md`](continuous-openui-20260803-c1-results.md),
[`continuous-openui-20260803-c1-container2-results.md`](continuous-openui-20260803-c1-container2-results.md),
and independently in `autotrain-cycle-20260803-c1-bounds-screen.md` from a
concurrent session — at least the fourth independent confirmation that
`bounds` has no measurable structural effect at this fixture scale and
default seed. Ship gates fail as expected (`insufficient_n`, missing
`held_out`/`adversarial`/`ood`/`rico_held` suites).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `fixture_insufficient_n_alone`).
No stack layer opened; local commit only, per `sdlc` autotrain-iteration-delivery.

## Next priorities

1. The driver's default next hypothesis is `component-plan` again, but that
   candidate was already **fresh-seed-confirmation REJECTED** on 2026-08-03
   ([`continuous-openui-20260803-c3-results.md`](continuous-openui-20260803-c3-results.md),
   seed 100003: exact primary tie + `binder_reference_f1` regression). A
   fresh container has no local memory of that rejection, so the next cycle
   should skip re-screening `component-plan` and advance to the
   already-queued `component-inventory` hypothesis instead, to avoid
   re-litigating a settled result.
2. `bounds` is now repeatedly exhausted across 4+ independent measurements at
   the default seed; future cycles should not re-select it without a
   materially different recipe.

Machine evidence:
[`continuous-openui-20260804-c1-results.json`](continuous-openui-20260804-c1-results.json).
