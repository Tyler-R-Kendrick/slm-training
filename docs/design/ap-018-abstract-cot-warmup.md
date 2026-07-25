# AP-018 (SLM-306): bottleneck warm-up records from SemanticPlanV1 and verbal plans

Date: 2026-07-25
Scope: data build only -- no training run, no checkpoint, no ship claim.

## Decision

Build privileged-plan warm-up examples for the Abstract-CoT causal pilot
(AP2) while forcing the eventual generation head to depend only on the
prompt plus abstract codebook tokens, never directly on the privileged
plan. This issue builds the data and its typed bottleneck contract
(`SegmentLayoutV1`); AP-019 implements the model-side masked attention
that enforces it.

## Source corpus

- input: `src/slm_training/resources/data/train/openui_verified_v1/records.jsonl`
- eligible rows: 1682
- accepted rows (valid plan derived): 1682
- acceptance rate: 1.0000
- emitted rows (committed artifact, balanced by binder bucket): 200

## Exclusion ledger (nothing dropped silently)

| Reason | Count |
| --- | ---: |
| (none) | 0 |

## Binder/reference complexity balance

| Bucket (binding count) | Accepted rows |
| --- | ---: |
| `0` | 9 |
| `1` | 691 |
| `2` | 596 |
| `3+` | 386 |

## Acceptance criteria

- Every eligible row received a valid plan or a typed exclusion reason: 1682 == 1682 + 0 (yes).
- >=90% of eligible verified rows received a valid plan: yes (1.0000).
- Iteration-0 abstract spans are deterministic (record-id + fixed seed keyed RNG) and leak no plan/target content -- see `tests/test_data/test_abstract_cot_warmup.py::test_iteration0_abstract_span_is_deterministic_across_runs`.

## Reproduction

```bash
python -m scripts.publish_abstract_cot_warmup
pytest -q tests/test_data/test_abstract_cot_warmup.py
```
