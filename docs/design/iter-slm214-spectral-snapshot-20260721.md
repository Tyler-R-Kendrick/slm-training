# SLM-214 (NCS0-01): SpectralSnapshotV1 fixture (slm214-spectral-snapshot-20260721)

**Matrix set:** `slm214_spectral_snapshot`
**Version:** `ncs0-01-v1`
**Status:** fixture
**Claim class:** wiring
**Disposition:** fixture_ok — All eligible matrices produced native spectral statistics with null calibration.

## Honest caveats

- Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.
- Statistics are native PyTorch SVD; optional WeightWatcher parity is not exercised.
- Null calibration uses a small default draw count so the harness stays CPU-only; publication-quality cells should increase --null-draws and chunk via --max-matrices.
- The toy fixture model is not a trained TwoTower; role classification is validated against the canonical path conventions used by the adapter target map.
- Matrices with fewer than 8 singular values are marked ineligible for fitted alpha.

## Summary

- Matrices inspected: 6
- Eligible for alpha: 6
- Ineligible: 0
- Total elapsed: 4.4 ms

## Per-matrix snapshots

| matrix | role | shape | eligible | hill α | null α (mean±sd) | α z | rand-ESD dist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| token_embed.weight | token_embedding | 16×32 | eligible | 4.079 | 3.971±0.814 | 0.133 | 0.0915 |
| denoiser.layers.0.self_attn.in_proj_weight | unknown | 96×32 | eligible | 4.651 | 4.537±0.068 | 1.670 | 7.1129 |
| denoiser.layers.0.self_attn.out_proj.weight | self_attn_out | 32×32 | eligible | 2.641 | 2.354±0.133 | 2.168 | 8.8608 |
| denoiser.layers.0.linear1.weight | mlp_in | 128×32 | eligible | 5.233 | 5.273±0.289 | -0.139 | 8.6766 |
| denoiser.layers.0.linear2.weight | mlp_out | 32×128 | eligible | 5.215 | 5.214±0.586 | 0.003 | 18.4992 |
| action_head.weight | action_head | 8×32 | eligible | 11.164 | 6.280±1.094 | 4.464 | 9.1204 |

## No-go for promotion

This report is wiring/fixture evidence only. No checkpoint, GPU train, or ship gate is claimed.
