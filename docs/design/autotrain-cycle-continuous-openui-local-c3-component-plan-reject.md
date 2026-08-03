# Autotrain `continuous-openui-local` c3: component-plan quality regression, reject

**Verdict:** fixture measurement complete, **rejected** — the size-matched
`component-plan` candidate (1,755,764 params, same as control) regresses on
every quality axis rather than improving.

## Result matrix

| Arm | n | parse | meaningful | structure (primary) | binder F1 | p50 ms | Ship gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3 | 1.0 | 0.3333 | 0.2308 | 0.7333 | 9760.00 | fail (fixture n, quality) |
| component-plan | 3 | 1.0 | 0.0 | 0.1725 | 0.6333 | 9273.63 | fail (fixture n, quality) |

Primary metric `smoke.structural_similarity` moves the **wrong** direction
(`0.2308 -> 0.1725`, Δ −0.0583) and `binder_reference_f1` fails its
non-regression check (`0.7333 -> 0.6333`). The candidate is 5.1% faster
p50, but a latency win never offsets a quality regression under the
quality-aware tradeoff policy.

## SDLC Phase A classification

`SDLC_PHASE_A NON_POSITIVE` — `non_regression_fail:binder_reference_f1` +
`primary_metric_null_or_worse` (structural_similarity worse) +
`fixture_insufficient_n` on both arms. **No stack layer opened** — local
commit + docs only, per the autotrain-iteration-delivery positive-result gate.

## Next-run priorities (ranked, from the typed handoff)

1. **model** — the `component-plan` arm is exhausted (rejected); test the
   distinct size-matched `component-edge` quality hypothesis next.
2. **evaluation** — keep the matched control every cycle.
3. **model** — rotate thrash recommendation across the lever bank rather than
   repeating `bounds`/`component-plan` only.

No checkpoint was created or promoted, so no `MODEL_CARD.md` / README update is
required. Machine-readable values are in
[`autotrain-cycle-continuous-openui-local-c3-component-plan-reject.json`](autotrain-cycle-continuous-openui-local-c3-component-plan-reject.json).
