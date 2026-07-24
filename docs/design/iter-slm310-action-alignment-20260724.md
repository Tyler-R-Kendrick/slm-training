# SLM-310 (LAR2-03): action alignment — corruption sampler × STOP-slot accounting

**Verdicts: corruption=ADD-balanced `adopted`, STOP-slot corrected `adopted`** (fixture-scale matched cells; not a ship claim)

## Preregistered thresholds (locked before results)

- T1: ADD training-target share gain >= 0.1 (both STOP arms)
- T2: ADD-balanced never reduces valid-final rate
- T3: corrected STOP accounting consumes <= legacy STOP budget and never reduces valid-final rate
- T4: lever isolation is structural (single-lever config diffs)
- rule: per lever: adopted iff its thresholds hold in both cells of the matched pair, else rejected; levers are never combined

## Cells (levers isolated)

| cell | corruption | STOP accounting | dead-candidate rate | applicable-ADD recall | ADD target share | verifier calls | STOP budget | valid-final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cell_a | default | legacy | 0.9792 | 0.1034 | 0.1772 | 11092 | 58 | 1.0000 |
| cell_b | ADD-balanced | legacy | 0.9799 | 0.1622 | 0.3275 | 14629 | 74 | 1.0000 |
| cell_c | default | corrected | 0.9792 | 0.1034 | 0.1772 | 11092 | 58 | 1.0000 |
| cell_d | ADD-balanced | corrected | 0.9799 | 0.1622 | 0.3275 | 14630 | 73 | 1.0000 |

## Threshold checks

- T1 gains: {'cell_b_minus_cell_a': 0.15020724214678874, 'cell_d_minus_cell_c': 0.15020724214678874} → ok=True
- T2 valid-final: {'cell_b_vs_cell_a': True, 'cell_d_vs_cell_c': True} → ok=True
- T3 STOP accounting: {'cell_c_vs_cell_a': {'stop_budget_legacy': 58, 'stop_budget_corrected': 58, 'budget_ok': True, 'valid_final_ok': True}, 'cell_d_vs_cell_b': {'stop_budget_legacy': 74, 'stop_budget_corrected': 73, 'budget_ok': True, 'valid_final_ok': True}} → ok=True
- T4 isolation: {'b_minus_a': ['corruption'], 'c_minus_a': ['stop'], 'd_minus_b': ['stop'], 'd_minus_c': ['corruption']} → ok=True

## Baseline distribution audit (cell A)

- dead-candidate rate 0.9792, applicable-ADD recall 0.1034, calibration MAD 0.1547
- top rejection reasons: {'duplicate_state': 1883, 'invalid_result': 206, 'leaf_container_mismatch': 25, 'no_op': 6, 'not_bindable': 2628, 'not_canonical_subtree': 5328, 'not_container': 90, 'not_leaf_component': 540, 'not_removable': 35, 'parent_or_comp_precondition': 141, 'target_not_canonical': 36}
- loss-reweighting prediction: True (predict reweighting iff applicable-ADD recall < 0.5 or |ADD visited share - ADD target share| > 0.2)
- decision: not_added_preregistered_cells_unchanged — The audit's preregistered rule predicted reweighting=True. The matched 2x2 was preregistered without a loss arm; adding one post-hoc would break lever isolation, so it is deferred to a follow-up preregistered cell rather than combined here.

## Honesty

Fixture-scale matched cells; decode-demand metrics come from SLM-310 per-proposal telemetry (visited = enumerated candidates the decode loop actually considered). A fixture verdict is wiring/distribution evidence, not a production ship claim. Negative results are retained per lever, never combined.
