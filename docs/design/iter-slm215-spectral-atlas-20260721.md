# SLM-215 (NCS0-02): SpectralAtlasV1 fixture (slm215-spectral-atlas-20260723)

**Matrix set:** `slm215_spectral_atlas`
**Version:** `ncs0-02-v1`
**Status:** fixture
**Claim class:** wiring
**Disposition:** fixture_signal — Observed Spearman correlation 0.876 between alpha_z and parse_rate.
**Atlas hash:** `f68f2da944a99531...`
**Semantic floor gate:** `6a9bf662bcc3f2a698504f0972a1d1160484343f9f049c77808b435bfe739c0a` (inconclusive; `docs/design/semantic-floor-gate-v1.json`)

## Honest caveats

- Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.
- This harness uses existing SpectralSnapshotV1 reports and synthetic or local outcome fixtures; it does not resolve the full historical checkpoint history.
- Real checkpoint provenance resolution and SemanticFloorGateV1 scoping are prerequisites for production-quality claims; unresolved checkpoints are explicitly recorded as 'unresolvable_local_history' where applicable.
- Cross-family holdouts and permutation controls are implemented on the fixture rows; the small fixture size limits statistical power.
- No causal conclusion is drawn; the atlas is retrospective and correlation-only.

## Summary

- Rows: 24
- Runs: 4
- Families: 2
- Source reports: 1
- Unresolved local history: 0

## Signal evaluation

```json
{
  "leave_one_family_out": [
    {
      "held_out_family": "family_0",
      "test_corr": 0.916083916083916,
      "train_corr": 0.8671328671328671
    },
    {
      "held_out_family": "family_1",
      "test_corr": 0.8671328671328671,
      "train_corr": 0.916083916083916
    }
  ],
  "permutation_null_mean": 0.012260869565217394,
  "spearman_alpha_z_vs_parse": 0.8756521739130435,
  "status": "evaluated"
}
```

## Per-role summaries

| role | n | mean α z | mean rand-ESD | mean parse |
| --- | --- | --- | --- | --- |
| mlp_in | 4 | 0.344 | 0.487 | 0.450 |
| mlp_out | 4 | 0.282 | 0.430 | 0.451 |
| self_attn_k | 4 | 0.339 | 0.458 | 0.434 |
| self_attn_out | 4 | 0.545 | 0.461 | 0.453 |
| self_attn_q | 4 | 0.181 | 0.413 | 0.408 |
| self_attn_v | 4 | -0.698 | 0.532 | 0.305 |

## No-go for promotion

This report is wiring/fixture evidence only. No checkpoint promotion, GPU train, or ship gate is claimed.
