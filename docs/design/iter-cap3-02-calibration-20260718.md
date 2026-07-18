# CAP3-02: Grammar-stratified calibration and low-bit adaptation for the local scorer

**Issue:** SLM-91  
**Date:** 2026-07-18  
**Status:** wiring evidence / fixture dry-run

## What was run

```bash
python -m scripts.calibrate_quantization \
  --checkpoint toy \
  --trace-dir outputs/runs/calibration-toy/traces \
  --target local_action_head \
  --format ternary \
  --strategy hybrid_coverage_margin \
  --samples 64 \
  --qat-steps 4 \
  --out-dir outputs/runs/calibration-toy \
  --synthetic-traces 256 \
  --dry-run
```

This is a CPU fixture run with synthetic CAP1-02 grammar-decision traces. It is
intended to wire the calibration harness, sampling strategies, and QAT loop;
it is **not** a ship-grade claim.

## Recipe

- Device: CPU
- Checkpoint: toy `LocalFlatHead`
- Trace source: 256 synthetic `grammar_decision` records
- Calibration format: `ternary` (`group_size=128`)
- Sampling strategy: `hybrid_coverage_margin`
- Calibration samples: 64
- Adaptation mode: short QAT reconstruction (4 steps, SGD)

## Observations

- The manifest schema `cap3-02.v1` records source trace IDs, checkpoint/teacher
  IDs, state-signature version, sample count, inclusion/exclusion rules,
  coverage fields, raw production-frequency weights, bin edges, and split
  hashes. The no-test-leakage assertion passed (no overlap between calibration
  and supplied test split hashes).
- `hybrid_coverage_margin` selected 64 samples covering 8 unique synthetic
  states, 40 state-action pairs, 3 scope signatures, and 2 template signatures.
- Short QAT on the local head reduced KL loss from 0.138 to 0.067 over 4 steps,
  confirming that the STE wrapper lets gradients update the shadow action
  embeddings.
- The physical-cost ledger for the toy model shows 64 ternary weights as 32 B
  of packed weights plus 2 B of FP16 scale and 2 B of bias; the unquantized
  bias/head path is excluded by policy.

## Honest caveats

- Fixture data only; no real grammar-constrained decode rollout was used.
- No `--ship-gates`, meaningful-parse metric, or held-out eval was run.
- No checkpoint was written to the HF bucket; this run is wiring evidence.
- Ship-grade calibration requires GPU-backed training, local E224+ checkpoints,
  and full ship-gate evaluation.

## Artifacts

- `outputs/runs/calibration-toy/calibration_manifest.json`
- `outputs/runs/calibration-toy/selected_samples.jsonl`
- `outputs/runs/calibration-toy/ledger.json`
- `outputs/runs/calibration-toy/metrics.json`
- `docs/design/iter-cap3-02-calibration-20260718.json`
