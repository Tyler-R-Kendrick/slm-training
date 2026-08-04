# continuous-openui-local c5 — component-plan candidate-only decode timeout

| Field | Value |
| --- | --- |
| Cycle | 5 |
| Status | measurement incomplete (infrastructure) |
| Positive | no |

## Evidence

| Arm | structure | decode_timeouts | completed | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | 0.043 | 0 | 3/3 | 1194 (p95 10824) |
| component-plan | 0.017 | **2** | 1/3 | 11127 |

`generate_batch_size=1` was applied (`decode_chunk_n=3`). Candidate still hits the
**12s** per-record ceiling; control p95 is already 10.8s under the same budget.

## Repair (model_build / climb policy)

Recalibrated thrash timing in `policy.v1.json`:

- `screening_decode_timeout_seconds`: 12 → **18**
- `min_train_floor_seconds`: 20 → **12**
- `eval_overhead_seconds`: 8 → **6**
- Fitted under 70s arm wall: **≈17.3s** per record

Component `harness.autoresearch.experiment_campaign` **v181 → v182**.

## Next

`retry_measurement` of the frozen component-plan arm after this repair receipt.
