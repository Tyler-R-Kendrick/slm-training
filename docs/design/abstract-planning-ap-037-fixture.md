# Abstract planning result — `ap-037-fixture`

- Schema: `AbstractPlanningResultV1`
- Claim class: **diagnostic**
- Campaign manifest sha256: `00601c6d1c08dbe7ca22a3402f2c9e93d9178d9634018e0c6db29baef3215f41`
- Source commit: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` (dirty: False)
- Data snapshot sha256: `eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee`
- Checkpoint: `ap037_fixture.pt` sha256 `ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff`

## Decode-path metrics

| Path | n | meaning-v2 | binder F1 | parse |
| --- | --- | --- | --- | --- |
| raw | 24 | 0.3125 | 0.4583 | 0.6250 |
| constrained | 24 | 0.5417 | 0.6667 | 1.0000 |
| repaired | 24 | 0.5833 | 0.7083 | 1.0000 |

## Plan controls

- **oracle** (n=24): binder_reference_f1=0.9583
- **random** (n=24): binder_reference_f1=0.2083
- **empty** (n=24): binder_reference_f1=0.1667
- **shuffled** (n=24): binder_reference_f1=0.2292

## Causal interventions

- `shuffle-steps` — Shuffle plan step order (binder_reference_f1 delta -0.4375)

## Latency / compute (seconds)

| plan | generation | verification | total | p95 |
| --- | --- | --- | --- | --- |
| 3.2500 | 41.5000 | 6.7500 | 51.5000 | 58.2000 |

## Verifier gates

| Gate | Observed | Threshold | Passed |
| --- | --- | --- | --- |
| meaning-improves | 0.2084 | 0.0100 | **True** |

- AgentV: `outputs/ap037/agentv.jsonl` (n=24, calibration `agentv-cal-v3`)

- Human audit: `outputs/ap037/human_audit.md` (n=12, calibration `human-cal-v1`)
