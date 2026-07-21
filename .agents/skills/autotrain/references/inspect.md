# `slm inspect` — SpectralSnapshotV1 diagnostics

CPU-only weight-matrix spectral inspection for the SLM-214 NCS0-01 backend.

## Commands

```bash
# Schema reference
slm inspect --describe

# Inspect the built-in toy fixture model
slm inspect spectral --output-dir outputs/runs/slm214-spectral-snapshot-test

# Inspect a real checkpoint (CPU)
slm inspect spectral --checkpoint <path-or-hf-id> --null-draws 50 --roles all

# Generate a null-cache summary for a shape
slm inspect spectral-null --shape 128x128 --initializer gaussian --draws 200

# Compare two report JSONs
slm inspect spectral-compare --left a.json --right b.json
```

## What it produces

- `outputs/runs/slm214-spectral-snapshot-<YYYYMMDD>/slm214_spectral_report.json`
- `docs/design/iter-slm214-spectral-snapshot-<YYYYMMDD>.{json,md}` (fixture mode)

## Claim class

Wiring / fixture only. No model-quality, promotion, or GPU-train claim.
