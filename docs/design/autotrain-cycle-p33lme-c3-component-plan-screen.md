# Autotrain continuous-openui-p33lme c3: component-plan aux head regresses the matched control

**Outcome:** non-positive model measurement (fixture `insufficient_n`, `n=3`) —
the size-matched `component-plan` structural-aux-head candidate is **worse**
than the matched control on the primary metric and on latency; rejected,
never reuse, promote, sync, or ship.

## What happened

This is the queued next model hypothesis from cycle 2
(`c20260802-continuous-openui-p33lme-489d3aa7-c2-component-plan`), executed as
this loop's cycle 3, campaign
`continuous-loop-20260802-continuous-openui-p33lme-489d3aa7-c3`, integration
commit `165a1e4351e140b49849c2c4ead2a5c981bd9c84`. Both arms trained (CPU
scratch TwoTower, `wf_smoke_v2`, lexer output, 20 steps, batch 2, seed
100003) with equal trainable parameter counts (1,755,764 each — size-matched
per repository law: capability is never bought with parameters), then
evaluated `smoke` (`n=3`, strict compiler-tree policy) to a complete AgentV
bundle (`execution_errors=0`), followed by an honest `--ship-gates`
rejection.

| Arm | latency p50 (ms) | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| control (`...-c3-control`) | 3057.79 | 1.0 | 0.3333 | 0.2308 | 0.7333 |
| candidate (`...-c3-component-plan`) | 3257.47 | 1.0 | 0.0 | 0.1725 | 0.6333 |

Gate failures: `smoke:insufficient_n actual=3 need>=20`,
`smoke:meaningful_program_rate`, `smoke:structural_similarity`,
`smoke:component_type_recall`, `smoke:ast_beq_rate`,
`smoke:canonical_beq_rate` all below threshold, plus
`held_out`/`adversarial`/`ood`/`rico_held` missing (screening cycle runs only
`smoke`). `expected_gate_rejection=true` — honest fixture-scale behavior, not
a harness failure.

## Why this is non-positive (not merely a null tie)

Unlike prior component-head screens in this repository's history that landed
on exact quality ties (e.g. c1732, c1730 component-plan screens), this arm is
a genuine **regression** against its matched control:

- primary metric `smoke.structural_similarity`: control 0.2308 → candidate
  0.1725, `improvement=-0.0583` (direction `increase` not met).
- `meaningful_program_rate`: control 0.3333 → candidate 0.0 (candidate
  produced zero meaningful programs on this 3-doc fixture).
- `binder_reference_f1`: control 0.7333 → candidate 0.6333
  (`non_regression_fail`).
- latency: control 3057.79 ms → candidate 3257.47 ms, **6.53% slower** — a
  quality loss paired with a latency loss, not a quality/latency tradeoff.

Per `autotrain-iteration-delivery.md`'s positive-result gate, none of the
three positive criteria are met: no primary-metric win (moved the wrong
direction), no ship-quality win (gates reject on `n` and thresholds as
expected at fixture scale, but the underlying quality numbers themselves
regressed), and no executable-unblocking (both arms already completed
end-to-end; nothing was broken and then fixed). The driver's own
`_classify_metric_tradeoff` / `classify_positive_metrics` correctly reports
`positive=false`, `stack_action=no_stack_layer_non_positive`. No new stack
layer is opened; this cycle stays local commits + docs only.

## Checkpoints

Both arms' scratch checkpoints
(`runs/c20260802-continuous-openui-p33lme-489d3aa7-c3-{control,component-plan}/checkpoints/last.pt`)
are local-only (`outputs/autoresearch/.../runs/`, explicit no-sync) and are
**rejected — never reused, promoted, synced, or shipped**. Recorded in
`docs/MODEL_CARD.md` and the README model-card summary per the model-card
duty (`checkpoint_documentation_required=true` in this cycle's handoff).

## Next hypotheses

Per this cycle's rank-1 `NextRunPriorityV1`: the component-plan aux head is
exhausted (rejected, not reusable). Run the distinct size-matched
`component-edge` quality hypothesis
(`c20260802-continuous-openui-p33lme-489d3aa7-c3-component-edge`) as the next
model-hypothesis cycle (c4), keeping the matched control as baseline.
Not executed in this delivery — reserved for the next scheduled iteration.

Machine-readable evidence is in
[`autotrain-cycle-p33lme-c3-component-plan-screen.json`](autotrain-cycle-p33lme-c3-component-plan-screen.json).
