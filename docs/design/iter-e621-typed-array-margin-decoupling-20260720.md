# E621: the typed-array nonempty margin is subsumed by its item-margin sibling

**Run date:** 2026-07-20

**Machine-readable result:** [`iter-e621-typed-array-margin-decoupling-20260720.json`](iter-e621-typed-array-margin-decoupling-20260720.json)

## Why this experiment

E620 and the still-open, unmerged PR #625 ("E626") both flagged the same
concrete next step for this lineage: a powered, multi-seed replay sweeping a
decode-time margin lever, rather than another single-seed point estimate.
PR #625 pursues that by adding a brand-new lever
(`required_slot_margin_decode_weight`) on a different, unmerged branch. This
iteration takes the same "powered multi-seed margin sweep" idea but applies it
to a lever that already exists on `main`/this branch —
`semantic_plan_typed_array_nonempty_margin_decode_weight`, the E612-authored
weight specifically meant to stop `ImageGallery`'s typed array from closing
empty — so the result doesn't depend on any unmerged code.

## Method

Trained 3 fresh 800-step CPU scratch `twotower` checkpoints (seeds 0, 1, 2;
same corpus/architecture as E619/E620: `e530_visible_semantic_roles_r2`).
Seed 0's loss (4.068013) reproduces E620's seed-0 loss (4.068010) to 4 decimal
places — the checkpoint SHA differs because this is a fresh training
invocation in a fresh sandbox (E620/E625/E626 all note the same: loss match,
not byte-identical weight files, is this lineage's "faithful replay"
standard).

For each seed, ran the full E619/E620 standard eval recipe (`schema_role_slot_decode_weight=8`
plus every fixed `semantic_plan_*`/`schema_*` weight E619/E620 record) 3
times, varying only `semantic_plan_typed_array_nonempty_margin_decode_weight`
at 0, 2, and 6 — matching E626's own dose points for easy comparison, without
depending on E626's unmerged code. 9 real matched OOD `n=4` eval runs, 36
predictions total.

## Result: a true null, not underpowered noise

All 36 predictions are **byte-identical** across margin ∈ {0, 2, 6} for every
seed and every record. Every headline metric is therefore identical across
doses within each seed:

| seed | margin | meaningful v1 | strict v2 | fidelity | validity | structure | recall | reward | AST node/edge F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0/2/6 (identical) | 0.5000 | 0.0000 | 0.5500 | 0.7300 | 0.4886 | 0.4792 | 0.8140 | 0.5437 / 0.3750 |
| 1 | 0/2/6 (identical) | 0.5000 | 0.0000 | 0.6333 | 0.7800 | 0.5248 | 0.4792 | 0.8375 | 0.5913 / 0.3500 |
| 2 | 0/2/6 (identical) | 0.2500 | 0.0000 | 0.6250 | 0.6750 | 0.4368 | 0.5000 | 0.4430 | 0.5123 / 0.3625 |

No decode timeouts or fallbacks in any of the 9 runs.

Using the committed SLM-183/H19 statistical protocol
(`slm_training.evals.power_protocol`) on the pooled 3-seed data (its first use
against real multi-seed eval records, rather than a synthetic fixture or a
single-seed `analyze-existing` file): pooled `meaningful_program_rate` is
5/12 = 0.4167 (Wilson 95% CI [0.1933, 0.6805]) at **every** margin value, and
the paired bootstrap CI on the margin=6 vs margin=0 `placeholder_fidelity`
delta is a degenerate `[0.0, 0.0]` — not because n=12 pairs is too small to
detect an effect, but because every pair is literally identical. Real seed
variance does exist and dominates: pooled `meaningful_program_rate` masks
0.50/0.50/0.25 by seed, and mean fidelity is 0.55/0.6333/0.625 by seed — seed
is a much larger variance source than this margin weight in this recipe.

## Root cause (code-level instrumentation)

Added a temporary, uncommitted monkeypatch around
`TwoTowerModel._semantic_plan_typed_array_nonempty_bias` to count calls and
log the decode context on every non-`None` return (same "instrument the real
checkpoint, don't guess" method as E616-E619). Across all 3 dose levels on the
seed-0 checkpoint, the function is called 191 times during OOD generation but
only returns a non-`None` bias **once**, for `ImageGallery`'s item array,
pushing the object-open token `{`.

Reading `_semantic_plan_typed_array_nonempty_bias` (`src/slm_training/models/twotower.py`,
~line 5032) explains why raising the swept weight 0→6 had zero effect even
though it *did* fire: the function's own target-selection branches on
`semantic_plan_typed_array_item_margin_decode_weight` (`typed_margin`), not on
the swept `semantic_plan_typed_array_nonempty_margin_decode_weight` (`margin`):

```python
if typed_margin > 0.0:
    typed_target_id = state._minimal_schema_id(dict(schemas[0]))
    ...
    targets = [candidate_ids.index(typed_target_id)]   # <- single, margin-independent
else:
    targets = [position for position, token_id in enumerate(candidate_ids)
               if token_id != close_id]
...
bias[target] = max(0.0, scores.max() + max(margin, typed_margin) - scores[target])
```

`typed_margin` is fixed at 2.0 in every E617-E620 recipe. Whenever it is
`> 0`, `targets` collapses to a single schema-derived candidate regardless of
`margin`'s value — `margin` only contributes to the shared magnitude ceiling
`max(margin, typed_margin)`. At the one place this bias fired, a magnitude of
2 (`typed_margin` alone) was already enough to flip the local argmax toward
the object-open token; raising the ceiling to 6 cannot change *which*
candidate wins once it's already winning at 2.

## Confirmation: it's the sibling weight doing the real work

Ran one more real eval (`e621-seed0-bothzero-r1`) with **both**
`semantic_plan_typed_array_nonempty_margin_decode_weight` and
`semantic_plan_typed_array_item_margin_decode_weight` set to `0` together —
the true full ablation, unlike the swept arms which held `item_margin` fixed
at 2. `ood_gallery_01` reverts to `ImageGallery([])` — E610/E612/E616's
already-rejected "closes empty" failure mode — and `placeholder_fidelity`
drops 0.55→0.4667, `placeholder_validity` 0.73→0.58, `reward_score`
0.814→0.6267, while `structural_similarity`/AST F1/`component_type_recall`
are unchanged. This confirms `semantic_plan_typed_array_item_margin_decode_weight`,
not the weight this iteration swept, is what actually keeps the array
nonempty in the standard recipe.

## Decision

Reject sweeping `semantic_plan_typed_array_nonempty_margin_decode_weight` in
isolation as a route to a dose-response finding in the standard recipe: as
long as `semantic_plan_typed_array_item_margin_decode_weight` stays active
(true throughout E617-E620), the swept weight's own target-selection code
path never executes, and its only remaining contribution — a shared magnitude
ceiling — is already saturated at this checkpoint scale. This is a genuine
implementation coupling between two weights whose names suggest independence,
not a training-depth or gating bug (contrast with E617's silent no-op, which
was a missing flag, not a co-active sibling). No checkpoint was promoted or
synced. **Not a ship claim.**

## Honest caveats

- This is an `n=4`-per-arm, 3-seed diagnostic replay against `ood`, not a
  full held-out/`rico_held` confirmatory suite.
- `binding_aware_meaningful_v2_rate_strict` (strict v2) stays 0.0 in every
  arm across all 3 seeds and both margin values and the full-ablation arm —
  required-slot coverage is still an open problem this iteration does not
  touch.
- The negative ICC reported in the JSON (`-0.2222` on 3 clusters of 4) is a
  small-sample estimation artifact (within-seed variance exceeding
  between-seed variance at this n), not evidence of a real negative
  seed-cluster correlation.
- This does not evaluate PR #625/"E626"'s new `required_slot_margin_decode_weight`
  lever, which is unmerged and not present on this branch; it remains
  independent, promising work.

## Next step

To actually observe a dose-response margin effect here, either (a) sweep
`semantic_plan_typed_array_item_margin_decode_weight` itself — the lever that
actually drives target selection in this code path — or (b) zero both
`nonempty_margin` and `item_margin` together as the true control arm before
sweeping `nonempty_margin` upward, so the two are not confounded via `max()`.
Separately, E620's still-open findings (`ood_dashboard_01`'s missing
components/slots, `ood_auth_01`'s `Input.name` role swap, `ood_modal_01`'s
undertrained structure) remain unaddressed by this iteration.

Raw evidence: [JSON](iter-e621-typed-array-margin-decoupling-20260720.json).
