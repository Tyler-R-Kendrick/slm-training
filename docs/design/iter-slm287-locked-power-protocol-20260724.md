# SLM-287: locked five-seed × two-config local baseline

> **Historical policy artifact.** These measured rows were produced before
> mandatory constrained generation and retain their original
> `raw`/`constrained`/`repaired` labels as provenance. The live runner now uses
> `constrained_native` and `constrained_compiler` only. Its changed campaign
> digest requires a new preregistered run; the numbers below must not be
> relabeled or treated as evidence for the new policy.

**Claim class:** completed local trained diagnostic; not a ship or promotion
result.

- locked manifest: `b4ad49cf1b73ad50528709daaad53dbf4846036c9dea787f1c2017c16e0a2d48`
- locked test records: `226`
- cells: `10` (five seeds × two local configurations)
- primary paired delta (design-on minus design-off): `0.000000`
- paired 95% bootstrap CI: `[0.000000, 0.000000]`
- no best seed was selected; human ratings are not a gate.

Each of the ten cells trained a CPU float32 scratch Choice TwoTower checkpoint
from its independently seeded initialization on the strict 97-record
`slm230_symbol_only_v1` snapshot, with a frozen 5,000-token budget (53 steps
for seed 0).  Each was then evaluated over every locked record by 16 isolated
shards (160 total); every shard emitted an AgentV bundle.  The campaign lock is
`652658eb…7ef4bcf` and the protocol lock is `ee6df865…db14664`.

| Variant | Meaning-v2 | Binder/reference F1 | Latency (ms) | Compute proxy | Peak RSS (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.000000 | 0.000000 | 350.57 | 0.0 | 273,093,836.8 |
| constrained | 0.000000 | 0.000000 | 571.34 | 0.0 | 273,093,836.8 |
| repaired | 0.000000 | 0.000000 | 554.65 | 0.0 | 273,093,836.8 |

The paired design-on minus design-off primary delta is 0.000000 (target
cluster bootstrap 95% CI [0.000000, 0.000000]; exact McNemar p=1.0,
0/0 discordant pairs over 1,130 seed-record pairs).  All quality endpoints are
zero, so this is negative evidence, not a best-seed selection or promotion.

The separately locked power analysis uses **absolute probability points**,
not log odds, because the observed base rate is zero.  Its numeric MDE is
**0.0200 absolute probability points** at 80% simulated power (100
simulations, five seeds, 226 targets).  This is a sensitivity statement about
this local diagnostic sample, not an observed gain.  The paired latency deltas
(design-on minus design-off) are raw +26.14 ms (95% CI [15.78, 37.05]),
constrained +29.43 ms ([14.63, 45.75]), and repaired +16.75 ms ([3.36, 28.15]);
quality remains zero in every variant.

The originally selected 480-record E297 snapshot was rejected before training
because free-form strings violate the active `symbol_only/v2` output contract;
no E297 checkpoint or result was emitted.  Replacing it with the valid current
contract corpus did not weaken the gate.  All checkpoint paths are local and
explicitly no-sync; this completed diagnostic is **not ship-grade** and makes
no production or promotion claim.

## Exact command

```bash
python -m scripts.run_flow_power_protocol --mode locked-merge --shard-count 16
```
