# Continuous autotrain: 2026-08-03 cycle 1, second container (non-positive, corroborating)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `318492c5` (current `main` tip, post [#1370](https://github.com/Tyler-R-Kendrick/slm-training/pull/1370))

**Why this file exists.** This is a fresh, ephemeral scheduled-task container
picking up the `continuous-openui-local` autotrain loop. Its local
`outputs/autoresearch/` state (where the driver's cycle counter and campaign
ledger live) is not shared with the earlier container that produced
[`continuous-openui-20260803-c1-results.md`](continuous-openui-20260803-c1-results.md)
and
[`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
(both merged via #1370). With no local state, the supervised driver restarted
its daily cycle counter at `c1`, and `campaign_id` is derived only from
`loop_id` (`sha256(loop_id)[:8]` + day + cycle), so it re-derived the exact
same campaign-id string as the earlier container's `c1` even though this is a
distinct run at a newer upstream commit. To avoid overwriting the original
evidence file, this cycle's results are recorded under a disambiguated
filename.

| Arm | Params | parse | MPR | structural_similarity | binder F1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 1.0 | 0.0 | .05750 | .6333 | 1152.14 |
| bounds | 1,608,962 | 1.0 | 0.0 | .05750 | .6333 | 1155.13 |

**Verdict: non-positive (corroborates the original c1).** The `bounds` knob is
size-matched against the control (`wf_smoke_v2`, `steps=20`, `n=3`) and again
produces an exact tie on the declared primary (`smoke.structural_similarity`)
and every quality metric, matching the original c1 measurement bit-for-bit
except for p50 latency (timing noise, not the primary). Ship gates fail as
expected (`insufficient_n`, missing `held_out`/`adversarial`/`ood`/`rico_held`
suites) — fixture screening only, not a ship claim.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `fixture_insufficient_n_alone`).
No stack layer opened; local commit only, per `sdlc` autotrain-iteration-delivery.

## Next priorities (ranked by the driver)

1. `component-plan` is the next ranked hypothesis
   (`c20260803-continuous-openui-local-8c0b60dd-c1-component-plan`) — already
   independently confirmed positive once in cycle c2 (this loop, earlier
   container) and still pending fresh-seed confirmation before promotion.
2. Keep the matched control as the size-matched baseline every cycle.
3. Do not re-select the now-twice-exhausted `bounds` arm without a new
   preregistered hypothesis.

Machine evidence:
[`continuous-openui-20260803-c1-container2-results.json`](continuous-openui-20260803-c1-container2-results.json).
