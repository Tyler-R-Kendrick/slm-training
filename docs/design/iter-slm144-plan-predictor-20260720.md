# SLM-144 / SPV1-01: Archetype + role-set predictor plan

**Claim class:** wiring / fixture only  
**Run date:** 2026-07-20  
**Machine-readable result:** [`iter-slm144-plan-predictor-20260720.json`](iter-slm144-plan-predictor-20260720.json)

This is a plan-only manifest. The fixture corpus and arm definitions are wired; run `--mode fixture` to execute the CPU train/eval matrix.

## Manifest

Hypothesis: A tiny CPU predictor can learn archetype and role-set factors from a component-family count vector, and learned heads outperform frequency baselines on a controlled fixture corpus.

| Arm | Archetype | Role set |
| --- | --- | --- |
| baseline_none | none | none |
| frequency_prior | frequency | frequency |
| serialized_inventory | none | serialized |
| set_matching | none | predicted |
| gold_archetype | gold | predicted |
| gold_role_set | predicted | gold |
| gold_both | gold | gold |

## Exact command

```bash
python -m scripts.run_slm144_plan_predictor_fixture --mode plan-only
```
