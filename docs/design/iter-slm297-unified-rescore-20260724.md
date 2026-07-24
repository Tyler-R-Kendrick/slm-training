# SLM-297: Unified final-program re-score of the SLM-155 fixture campaign

**Run date:** 2026-07-24

**Claim class:** wiring / fixture only. Deterministic CPU re-score; no model training beyond the shared 20-step fixture scorer, no ship-gate claim.

**Machine-readable result:** [`iter-slm297-unified-rescore-20260724.json`](iter-slm297-unified-rescore-20260724.json)

## Correction note (append-only)

The SLM-155 (SPV3-02) 2026-07-20 comparison scored AR arms with an `accepted` process label (chosen action ∈ accepted_action_ids; no program was ever built) and X22/hybrid arms with a `recovered` label (conflict-slice structural repair flag). Those labels are incommensurate. This re-score retains every arm's final program and scores all arms under one metric (`meaningful_program_v1`, gold-free lane). The 2026-07-20 iter docs are left untouched; this document is the corrected comparison.

## Pivot disposition

PIVOT CONFIRMED: the SLM-155 headline was an incommensurate-label artifact. The old AR 'accepted' rate measured action-id membership while the old X22 'recovered' rate measured structural slice repair; under the unified final-program metric the AR-vs-X22 gap moves from +0.750 to +1.000.

| Family | n | Old process rate | New final-program rate |
| --- | --- | --- | --- |
| ar | 240 | 1.000 | 1.000 |
| x22 | 192 | 0.250 | 0.000 |
| hybrid | 144 | 0.000 | 1.000 |

## New final-program rate per arm (Wilson 95%)

| Arm | Valid | n | Estimate | Low | High |
| --- | --- | --- | --- | --- | --- |
| AR-G | 48 | 48 | 1.000 | 0.926 | 1.000 |
| AR-B | 48 | 48 | 1.000 | 0.926 | 1.000 |
| AR-C | 48 | 48 | 1.000 | 0.926 | 1.000 |
| X-M | 0 | 48 | 0.000 | 0.000 | 0.074 |
| X-P | 0 | 48 | 0.000 | 0.000 | 0.074 |
| X-C | 0 | 48 | 0.000 | 0.000 | 0.074 |
| H-1 | 48 | 48 | 1.000 | 0.926 | 1.000 |
| H-K | 48 | 48 | 1.000 | 0.926 | 1.000 |
| H-C | 48 | 48 | 1.000 | 0.926 | 1.000 |
| gold_ar | 48 | 48 | 1.000 | 0.926 | 1.000 |
| gold_x22 | 0 | 48 | 0.000 | 0.000 | 0.074 |
| oracle_selector | 48 | 48 | 1.000 | 0.926 | 1.000 |

## Old × new transition tables (per arm)

| Arm | Old label | New valid | New invalid |
| --- | --- | --- | --- |
| AR-G | accepted | 48 | 0 |
| AR-B | accepted | 48 | 0 |
| AR-C | accepted | 48 | 0 |
| X-M | not_recovered | 0 | 48 |
| X-P | not_recovered | 0 | 48 |
| X-C | not_recovered | 0 | 48 |
| H-1 | not_recovered | 48 | 0 |
| H-K | not_recovered | 48 | 0 |
| H-C | not_recovered | 48 | 0 |
| gold_ar | accepted | 48 | 0 |
| gold_x22 | recovered | 0 | 48 |
| oracle_selector | accepted | 48 | 0 |

## Paired AR-vs-hybrid final-program validity (by record id + seed)

| Pair | Both valid | AR valid, hybrid invalid | AR invalid, hybrid valid | Both invalid | Unpaired |
| --- | --- | --- | --- | --- | --- |
| AR-G vs H-1 | 48 | 0 | 0 | 0 | 0 |
| AR-G vs H-K | 48 | 0 | 0 | 0 | 0 |
| AR-G vs H-C | 48 | 0 | 0 | 0 | 0 |
| AR-B vs H-1 | 48 | 0 | 0 | 0 | 0 |
| AR-B vs H-K | 48 | 0 | 0 | 0 | 0 |
| AR-B vs H-C | 48 | 0 | 0 | 0 | 0 |
| AR-C vs H-1 | 48 | 0 | 0 | 0 | 0 |
| AR-C vs H-K | 48 | 0 | 0 | 0 | 0 |
| AR-C vs H-C | 48 | 0 | 0 | 0 | 0 |
| gold_ar vs H-1 | 48 | 0 | 0 | 0 | 0 |
| gold_ar vs H-K | 48 | 0 | 0 | 0 | 0 |
| gold_ar vs H-C | 48 | 0 | 0 | 0 | 0 |
| oracle_selector vs H-1 | 48 | 0 | 0 | 0 | 0 |
| oracle_selector vs H-K | 48 | 0 | 0 | 0 | 0 |
| oracle_selector vs H-C | 48 | 0 | 0 | 0 | 0 |

## Reason-code prevalence per arm

| Arm | Reason code | Count |
| --- | --- | --- |
| AR-G | component_recall_unobservable | 48 |
| AR-B | component_recall_unobservable | 48 |
| AR-C | component_recall_unobservable | 48 |
| X-M | component_recall_unobservable | 48 |
| X-M | empty_root_stack | 48 |
| X-M | no_content_components | 48 |
| X-M | no_placeholders | 48 |
| X-P | component_recall_unobservable | 48 |
| X-P | empty_root_stack | 48 |
| X-P | no_content_components | 48 |
| X-P | no_placeholders | 48 |
| X-C | component_recall_unobservable | 48 |
| X-C | empty_root_stack | 48 |
| X-C | no_content_components | 48 |
| X-C | no_placeholders | 48 |
| H-1 | component_recall_unobservable | 48 |
| H-K | component_recall_unobservable | 48 |
| H-C | component_recall_unobservable | 48 |
| gold_ar | component_recall_unobservable | 48 |
| gold_x22 | component_recall_unobservable | 48 |
| gold_x22 | empty_root_stack | 48 |
| gold_x22 | no_content_components | 48 |
| gold_x22 | no_placeholders | 48 |
| oracle_selector | component_recall_unobservable | 48 |

## Exact command

```bash
python -m scripts.run_slm297_unified_rescore
```
