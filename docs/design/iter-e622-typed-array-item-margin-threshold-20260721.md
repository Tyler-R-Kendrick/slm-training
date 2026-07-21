# E622: `semantic_plan_typed_array_item_margin_decode_weight` is a minimum-effective-dose threshold, not a scaling lever

**Run date:** 2026-07-21

**Machine-readable result:** [`iter-e622-typed-array-item-margin-threshold-20260721.json`](iter-e622-typed-array-item-margin-threshold-20260721.json)

## Why this experiment

E621 swept `semantic_plan_typed_array_nonempty_margin_decode_weight` (0/2/6)
across 3 seeds and found a true null: it's fully subsumed by its sibling
`semantic_plan_typed_array_item_margin_decode_weight` (held fixed at the
standard recipe's 2.0), because the code's own target-selection branches on
the sibling, not the swept weight, whenever the sibling is active. E621 left
two concrete next steps: (a) sweep the sibling itself — the lever that
actually drives target selection — or (b) zero both margins together as the
true control before sweeping, so the two levers aren't confounded via
`max(margin, typed_margin)`. This iteration does both at once: it sweeps
`semantic_plan_typed_array_item_margin_decode_weight` itself, holding
`semantic_plan_typed_array_nonempty_margin_decode_weight` FIXED AT 0 (the
true ablated-sibling control) throughout, rather than at the previously-used
2.0.

## Method

No new training. Reused E621's 3 already-committed, unchanged 800-step
scratch `twotower` checkpoints (seeds 0/1/2:
`outputs/runs/e621-margin-sweep-seed{0,1,2}-scratch800-20260720`). Replayed
the same E619/E620/E621 standard eval recipe (identical fixed weights) with
`semantic_plan_typed_array_nonempty_margin_decode_weight=0` throughout and
`semantic_plan_typed_array_item_margin_decode_weight` swept at 0, 1, 2, 4.

Ran 8 new real OOD `n=4` eval runs (seed1/seed2 @ dose 0; seed0/1/2 @ dose 1;
seed0/1/2 @ dose 4) and reused 4 already-committed E621 runs unmodified for
the remaining data points: `e621-seed0-bothzero-r1` (seed0, dose 0 — both
weights already 0) and `e621-seed{0,1,2}-margin0-r1` (dose 2 — nonempty
already 0, item already fixed at the standard 2). 12 seed×dose combinations ×
4 OOD records = 48 real predictions total.

A same-config sanity check (a fresh seed0/dose-0 rerun, `e622-seed0-item0-r1`)
reproduced `e621-seed0-bothzero-r1`'s `placeholder_fidelity` (0.4667),
`placeholder_validity` (0.58), `structural_similarity` (0.48855),
`reward_score` (0.62675), and `meaningful_program_rate` (0.5) byte-for-byte,
confirming it's valid to reuse E621's committed runs as data points rather
than re-running everything from scratch.

## Result: a real, code-confirmed threshold effect — not a scaling effect

Comparing `prediction_sha256` across all 4 doses within each seed/record: **9
of the 12 seed×record groups (`ood_auth_01`, `ood_dashboard_01`,
`ood_modal_01` × 3 seeds — 36 of 48 predictions) are byte-identical across
every dose from 0 to 4.** The remaining 3 groups (`ood_gallery_01` × 3 seeds)
show exactly 2 distinct values each: a unique value at `item_margin=0`, and
one shared value across `item_margin ∈ {1, 2, 4}` — byte-identical among
those three. Only 3 of 48 predictions (the dose-0 gallery ones) ever differ
from their group's positive-dose value:

```
item_margin=0:      root = Stack([v0], "column")
                     v0 = ImageGallery([])

item_margin ∈{1,2,4}: root = Stack([v0], "column")
                       v0 = ImageGallery([{src: ":ood.gallery.img", alt: ":ood.gallery.alt"}])
```

This exactly matches E621's instrumentation trace of the code (`margin=0`
means `max(margin, typed_margin) <= 0.0` when `typed_margin` is also 0, so
`_semantic_plan_typed_array_nonempty_bias` returns `None` and the array
closes empty; any `typed_margin > 0.0` enters the schema-derived
`_minimal_schema_id` target-selection branch, which is margin-independent
once entered). The metrics move only through this one record:

| seed | item_margin | meaningful v1 | strict v2 | fidelity | validity | structure | reward | ast node/edge F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0.5000 | 0.0000 | 0.4667 | 0.5800 | 0.4886 | 0.6267 | 0.5437 / 0.3750 |
| 0 | 1/2/4 (identical) | 0.5000 | 0.0000 | 0.5500 | 0.7300 | 0.4886 | 0.8140 | 0.5437 / 0.3750 |
| 1 | 0 | 0.5000 | 0.0000 | 0.5500 | 0.6300 | 0.5248 | 0.6502 | 0.5913 / 0.3500 |
| 1 | 1/2/4 (identical) | 0.5000 | 0.0000 | 0.6333 | 0.7800 | 0.5248 | 0.8375 | 0.5913 / 0.3500 |
| 2 | 0 | 0.2500 | 0.0000 | 0.5000 | 0.5000 | 0.4368 | 0.2432 | 0.5123 / 0.3625 |
| 2 | 1/2/4 (identical) | 0.2500 | 0.0000 | 0.6250 | 0.6750 | 0.4368 | 0.4430 | 0.5123 / 0.3625 |

`meaningful_program_v1` (binary) does not flip for the gallery record in
either state at this checkpoint scale, so the pooled `meaningful_v1` rate
(5/12 = 0.4167, Wilson 95% CI [0.1933, 0.6805]) is identical at every dose.
`structural_similarity`, AST node/edge F1, and `component_type_recall` are
also unchanged at every dose — matching E621's earlier ablation exactly. The
effect lives entirely in `placeholder_fidelity` / `placeholder_validity` /
`reward_score` for the one affected record.

Using the SLM-183/H19 statistical protocol
(`slm_training.evals.power_protocol`) on the pooled 3-seed, 4-record paired
data: the paired-bootstrap `placeholder_fidelity` delta between
`item_margin=1` and `item_margin=0` is `+0.0972` (95% CI `[0.0, 0.1944]` —
positive point estimate touching zero at the lower bound with only n=12
pairs, consistent with a real but per-record-concentrated effect). Critically,
the delta between `item_margin=2` and `item_margin=1`, and between
`item_margin=4` and `item_margin=2`, are both **degenerate `[0.0, 0.0]`** —
every one of the 12 paired records is byte-identical between those dose
pairs.

## Decision: a real finding, but a threshold, not a scaling lever

`semantic_plan_typed_array_item_margin_decode_weight` only needs to be
**positive** to fully flip the one decode site it governs
(`ImageGallery`'s typed-array-open bias) at this checkpoint scale.
`item_margin=1` is exactly as effective as `item_margin=2` (the current
standard recipe's default) or `item_margin=4` — there is zero marginal
benefit from the recipe's standing default of 2 over the untested, cheaper
value of 1 in this diagnostic. There is a real, measurable difference between
0 (off) and any positive dose: fidelity +0.083 to +0.125, validity +0.15 to
+0.175, reward +0.187 to +0.200 across the 3 seeds — all concentrated in the
single record this lever governs.

This completes E621's causal chain end-to-end: E621 read the code and
instrumented the one firing site as a hypothesis; E622 now dose-sweeps the
correct lever with a clean, unconfounded (`nonempty_margin` truly zeroed)
control and confirms both the mechanism and its saturation point with real
matched evals, not just code reading. No checkpoint was promoted or synced.
No code changed
(`python -m scripts.verify_version_stamps --check`: 0 components touched).
**Not a ship claim.**

## Honest caveats

- `n=4`-per-arm, 3-seed diagnostic replay against `ood` only, not a full
  held-out/`rico_held` confirmatory suite.
- The entire effect is driven by exactly one of the four OOD records
  (`ood_gallery_01`) at this checkpoint scale; the other three records show
  zero sensitivity to this lever across the whole 0–4 dose range tested. A
  checkpoint/prompt mix with more typed-array components could reveal
  additional firing sites and a different saturation point.
- Only doses `{0, 1, 2, 4}` were tested. Whether a fractional dose between 0
  and 1 (e.g. 0.5) would show a partial/graded effect, or whether the
  threshold is a hard step at any positive value, is not directly tested —
  but the code path (`if typed_margin > 0.0:`) is a hard boolean gate, so a
  step function exactly at 0 is the code-predicted behavior and this result
  is consistent with, not merely correlated with, that reading.
- `binding_aware_meaningful_v2_rate_strict` (strict v2) stays 0.0 in every
  arm across all 3 seeds and all 4 doses — required-slot coverage remains a
  fully open problem this iteration does not touch.
- The negative ICC on `meaningful_v1` by seed is the same small-sample
  estimation artifact E621 documented (within-seed variance exceeding
  between-seed variance at n=3 clusters), not evidence of a real negative
  seed-cluster correlation.

## Next step

E620's still-open findings (`ood_dashboard_01`'s missing components/slots,
`ood_auth_01`'s `Input.name` role swap, `ood_modal_01`'s undertrained
structure) remain completely unaddressed by both E621 and E622 — none of the
margin-weight sweeps in this lineage (E621's nonempty-margin sweep, E622's
item-margin sweep) ever change those three records at all; they were
byte-identical across every dose and weight combination tested in both
iterations. This makes a root-cause trace of one of those three records the
strongest remaining candidate for the next iteration, since two consecutive
margin-lever sweeps have now shown those failures are untouched by this
whole family of decode-time biases. Separately, since `item_margin=1` is now
shown exactly as effective as the standard recipe's default of 2 in this
diagnostic, a future iteration could test whether shrinking the standard
default from 2 to 1 has any effect on other suites/records where headroom
(not just threshold-crossing) might matter, before treating margin=1 as a
safe simplification.

Raw evidence: [JSON](iter-e622-typed-array-item-margin-threshold-20260721.json).
