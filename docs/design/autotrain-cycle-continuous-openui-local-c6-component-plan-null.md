# Autotrain `continuous-openui-local` c6: fresh-seed component-plan, quality-tie null

**Verdict:** fixture measurement complete, **rejected** (quality-null). A
fresh-seed `component-plan` hypothesis (distinct arm from the [c3
rejection](autotrain-cycle-continuous-openui-local-c3-component-plan-reject.md))
ties the matched control on every quality metric.

## Result matrix

| Arm | structure (primary) | binder F1 | meaningful | p50 ms | Ship gates |
| --- | ---: | ---: | ---: | ---: | --- |
| control | 0.0964 | 0.8222 | 0.0 | 3767.44 | fail (fixture n, quality) |
| component-plan | 0.0964 | 0.8222 | 0.0 | 3648.99 | fail (fixture n, quality) |

Both arms size-matched at 1,755,764 params. `smoke.structural_similarity`
improvement is exactly `0.0`. Candidate is 3.1% faster p50 with no quality
offset — not sufficient for a positive classification under the
quality-aware tradeoff policy.

## SDLC Phase A classification

`SDLC_PHASE_A NON_POSITIVE` — `fixture_insufficient_n` on both arms +
`primary_metric_null_or_worse` (improvement 0.0). **No stack layer opened.**

## Next-run priorities

1. **model** — this arm pair is exhausted (quality-tie null); test the
   distinct size-matched `component-edge` quality hypothesis next.
2. **evaluation** — keep the matched control every cycle.

No checkpoint was created or promoted, so no `MODEL_CARD.md` / README update is
required. Machine-readable values are in
[`autotrain-cycle-continuous-openui-local-c6-component-plan-null.json`](autotrain-cycle-continuous-openui-local-c6-component-plan-null.json).
