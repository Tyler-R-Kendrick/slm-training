# SLM-287: locked five-seed × two-config baseline protocol

**Claim class:** diagnostic protocol; no model result is claimed.

- locked manifest: `b4ad49cf1b73ad50528709daaad53dbf4846036c9dea787f1c2017c16e0a2d48`
- locked test records: `226`
- seeds: `[0, 1, 2, 3, 4]`
- local configurations: `['scratch_design_off', 'scratch_design_on']`
- primary metric: `binding_aware_meaningful_v2`
- human rating: optional audit only, never a gate.

The protocol fails closed: no numeric conclusion is emitted until all
ten cells score the exact frozen record set under raw, constrained,
and repaired decode with matching repeated-initialization hashes.

## Exact command

```bash
python -m scripts.run_flow_power_protocol --mode locked-plan
```
