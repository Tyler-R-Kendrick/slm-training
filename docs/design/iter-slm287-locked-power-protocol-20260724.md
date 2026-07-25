# SLM-287: locked five-seed × two-config local baseline

**Claim class:** completed local diagnostic; not a ship or promotion result.

- locked manifest: `b4ad49cf1b73ad50528709daaad53dbf4846036c9dea787f1c2017c16e0a2d48`
- locked test records: `226`
- cells: `10` (five seeds × two local configurations)
- primary paired delta (design-on minus design-off): `0.000000`
- paired 95% bootstrap CI: `[0.000000, 0.000000]`
- no best seed was selected; human ratings are not a gate.

The frozen recipe intentionally used zero updates. Consequently all quality
endpoints were zero, observed seed and target variance were zero, and the
simulated power curve could not produce a finite numeric MDE. This is valid
negative evidence for the zero-update control, but it does **not** satisfy the
AP-007 numeric-MDE acceptance criterion. A bounded local trained baseline is
required before this issue can close.

All cells used the canonical evaluator's raw, constrained, and repaired
variants over all 226 locked records. The run is local CPU float32 scratch
evidence only; it is neither a full-HF ship evaluation nor a promotion result.

## Exact command

```bash
python -m scripts.run_flow_power_protocol --mode locked-merge --shard-count 64
```
