# SLM-226: AbsoluteSpectralTargetGateV1 (slm226-absolute-spectral-gate-20260723)

**Verdict:** `descriptive_only`

**Report hash:** `6eff650522e6ec503520e39ed5a094cc468ec2707e9cfeb59b563262f0aa6dda`

**Status / claim:** `scratch_measured` / `descriptive_diagnostic` (`scratch_cpu_no_durable_checkpoint`)

## Null finite-size boundary

| Width / role / shape | Draws | Mean alpha | SD | 95% null interval | z(alpha=2) |
| --- | ---: | ---: | ---: | --- | ---: |
| 128 / `ctx_proj` / `128x128` | 200 | 2.273991 | 0.056007 | [2.164217, 2.383764] | -4.892 |
| 256 / `ctx_proj` / `256x128` | 200 | 3.468501 | 0.082900 | [3.306017, 3.630986] | -17.714 |
| 512 / `ctx_proj` / `512x128` | 200 | 4.973008 | 0.119165 | [4.739445, 5.206571] | -24.949 |

Raw alpha is shape-dependent and is never interpreted without this same-shape null. In particular, proximity to alpha=2 is not an authorization signal.

## Scratch trained probes

| Width / shape | Seed | Steps / tokens | Init alpha | Final alpha | Init/final null distance | Final MSE |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| 128 / `128x128` | 0 | 8 / 8192 | 2.240742 | 2.218807 | 0.008873 / 0.015398 | 192.871704 |
| 128 / `128x128` | 1 | 8 / 8192 | 2.228654 | 2.199444 | 0.007745 / 0.011941 | 184.286148 |
| 128 / `128x128` | 2 | 8 / 8192 | 2.284239 | 2.283831 | 0.015528 / 0.010050 | 203.801834 |
| 256 / `256x128` | 0 | 8 / 8192 | 3.440722 | 3.472975 | 0.006901 / 0.008479 | 186.127502 |
| 256 / `256x128` | 1 | 8 / 8192 | 3.480529 | 3.441879 | 0.006724 / 0.010236 | 188.949707 |
| 256 / `256x128` | 2 | 8 / 8192 | 3.501391 | 3.463631 | 0.009989 / 0.015833 | 178.595947 |
| 512 / `512x128` | 0 | 8 / 8192 | 4.877429 | 4.767513 | 0.005388 / 0.008932 | 181.185898 |
| 512 / `512x128` | 1 | 8 / 8192 | 4.986848 | 4.968425 | 0.004905 / 0.007873 | 201.537262 |
| 512 / `512x128` | 2 | 8 / 8192 | 5.013047 | 5.060272 | 0.006660 / 0.006068 | 186.342682 |

The probes are deterministic CPU linear-role diagnostics, not full TwoTower checkpoints or quality evidence. They cannot establish an absolute optimum or minimum production width.

## Gate rationale

- same-shape random nulls place raw alpha near the proposed target at finite width
- scratch probes are descriptive and do not represent a durable serving checkpoint family
- SLM-221 found no reproducible singular-value-shape causal effect
- no provenance-resolvable durable checkpoint family is available
- SemanticFloorGateV1 is inconclusive; semantic outcome use is blocked

## Disposition

Only null-calibrated spectral diagnostics remain allowed. `ww_pgd`, `trace_log`, and `alpha_target` are blocked for every role and shape; the guard helper fails closed on this gate.

**Recipe:** CPU, one PyTorch thread; Gaussian 128x128, 256x128, and 512x128 nulls (200 draws per shape); Pareto/spiked controls; three deterministic seeds; AdamW, 8 steps, no reusable checkpoint.

No canonical model evaluation or AgentV run was performed because this was a spectral profile, not a model-quality evaluation. No checkpoint was written or promoted.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src /home/codex/repos/slm-training/.venv/bin/python -m scripts.run_absolute_spectral_gate --check
```
