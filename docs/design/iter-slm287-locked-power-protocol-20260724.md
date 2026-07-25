# SLM-287: locked five-seed × two-config local baseline

**Claim class:** completed local trained diagnostic; not a ship or promotion result.

- locked manifest: `b4ad49cf1b73ad50528709daaad53dbf4846036c9dea787f1c2017c16e0a2d48`
- locked test records: `226`
- cells: `10` (five seeds × two local configurations)
- primary paired delta (design-on minus design-off): `0.000000`
- paired 95% bootstrap CI: `[0.000000, 0.000000]`
- no best seed was selected; human ratings are not a gate.

Each cell trained a CPU float32 scratch Choice TwoTower checkpoint on the
strict 97-record `slm230_symbol_only_v1` snapshot to its frozen 5,000-token
budget (53 steps for the seed-0 control), then evaluated all 226 locked rows.
All quality endpoints remained zero, with zero observed seed/target variance,
so the preregistered log-odds power curve has no finite MDE. This is valid
negative evidence for the trained local baseline, but it does **not** satisfy
the AP-007 numeric-MDE acceptance criterion.

The originally selected 480-record E297 snapshot was rejected before training
because it contains free-form strings forbidden by the active `symbol_only/v2`
output contract; no E297 checkpoint or result was emitted. The current-contract
SLM-230 snapshot replaced it without weakening that gate. A separately locked
absolute-probability MDE analysis is required before this issue can close.

All cells used the canonical evaluator's raw, constrained, and repaired
variants over the full 226-record locked holdout. This is local CPU scratch
evidence only, not a ship or promotion result.

## Exact command

```bash
python -m scripts.run_flow_power_protocol --mode locked-merge --shard-count 16
```
