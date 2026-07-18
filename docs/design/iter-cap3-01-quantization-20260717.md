# CAP3-01 (SLM-90): reference low-bit quantizers and physical-cost ledger

Date: 2026-07-17 · Track: CAP3 · Linear: SLM-90

## What shipped

A disabled-by-default reference quantization substrate that lets downstream
issues compare binary, ternary, four-level, INT4, and INT8 realizations with an
honest physical-cost ledger.  No optimized kernels, no QAT campaign, and no
quality/speed winner are claimed.

- `src/slm_training/models/quantization/`
  - `formats.py` — versioned `QuantFormat` descriptor plus all required reference
    formats (FP16/BF16 controls, INT8/INT4, binary, ternary, symmetric four-level,
    learned four-level-with-zero, binary-plus-mask, residual plane descriptors).
  - `observers.py` — deterministic symmetric/asymmetric min-max observers with
    per-tensor and last-dim groupwise scaling.
  - `fake_quant.py` — nearest-level reference forward with optional STE; supports
    per-tensor and groupwise scaling.
  - `cost.py` — `TensorCost`, `FormatCostReport`, `PhysicalCostLedger`, plus
    CAP0-04 `EstimatedEvidence` generation.
  - `diagnostics.py` — per-tensor MSE, max error, cosine similarity, symbol
    entropy, zero rate, scale stats, and level occupancy.
  - `convert.py` — disabled-by-default `convert_twotower`, parameter-group
    selection, tied-weight rejection, and reversible conversion helpers.
  - `__init__.py` — public API.
- Config integration:
  - `ModelBuildConfig.quant_format` and `TwoTowerConfig.quant_format` (None = off).
  - `src/slm_training/harnesses/model_build/factory.py` pipes the flag and calls
    `TwoTowerModel.apply_quant_format` at load time when set.
  - `TwoTowerModel.apply_quant_format` dispatches reference conversion for
    supported format ids.
- CLI: `scripts/inspect_quantization.py` dry-run ledger over a toy model fixture
  or an existing checkpoint; writes `format_ledger.json` and a `docs/design/`
  mirror.  `--write-converted` is required to emit a converted checkpoint.
- Tests: `tests/test_models/test_quantization.py` (21 tests) covering level/scale
  math for every format, learned-four-level/ternary containment, binary-plus-mask
  storage, groupwise scale overhead, packing padding, tied-weight rejection,
  reversibility, whole-model ledger, empirical entropy vs physical bytes, missing
  kernel reporting, and CAP0-04 evidence validation.

## Kernel capability registry

`slm_training.models.quantization.formats.KERNEL_REGISTRY` reports which paths
exist for each format.  Missing CPU/CUDA kernels are explicit system limitations,
not format failures; reference-path latency is not a deployment claim.

| format | reference | cpu opt | cuda | packed | notes |
| --- | --- | --- | --- | --- | --- |
| fp16 | ✓ | ✓ | ✓ | ✓ | control |
| bf16 | ✓ | ✓ | ✓ | ✓ | control |
| int8 | ✓ | ✓ | ✓ | ✓ | reference path only claimed here |
| int4 | ✓ | ✗ | ✓ | ✓ | reference path on CPU |
| binary | ✓ | ✗ | ✗ | ✓ | reference path |
| ternary | ✓ | ✗ | ✗ | ✓ | zero skipping is semantic, not a kernel |
| symmetric_four_level | ✓ | ✗ | ✗ | ✓ | reference path |
| learned_four_level_zero | ✓ | ✗ | ✗ | ✓ | reference path |
| binary_plus_mask | ✓ | ✗ | ✗ | ✓ | sign + explicit zero mask |
| residual_*_plane | ✗ | ✗ | ✗ | ✗ | descriptor only; execution in CAP4 |

## Fixture dry-run ledger

Command:

```bash
python -m scripts.inspect_quantization \
  --formats binary,ternary,learned4zero,int4,int8 \
  --group-size 128 \
  --out outputs/runs/quantization \
  --docs-out docs/design/quantization-results.json
```

Model: toy TransformerEncoder-like fixture (`d_model=64`, `vocab=32`, 2 layers,
4-dim FFN).  Values are analytical/system diagnostics, not quality claims.

| format | ideal bits | physical weight bytes | scale/zp bytes | total bytes | resident bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| binary | 100,352 | 12,544 | 3,968 | 35,968 | 237,072 |
| ternary | 159,054 | 25,088 | 3,968 | 48,512 | 251,184 |
| learned4zero | 200,704 | 25,088 | 3,968 | 48,512 | 251,184 |
| int4 | 392,064 | 50,176 | 3,968 | 73,600 | 279,408 |
| int8 | 802,249 | 100,352 | 3,968 | 123,776 | 335,856 |

Key ledger invariants verified:

- Ternary and learned-four-level-with-zero occupy the same **2-bit physical
  slots** but report different ideal bits (`log2(3)` vs `2.0`).
- Physical bytes include per-group packing padding and separate scale overhead;
  they are never replaced by empirical entropy.
- Embeddings, norms, and heads are excluded from quantization and counted as
  unquantized bytes.
- Whole-checkpoint bytes include metadata and alignment overhead.

JSON mirror: [quantization-results.json](quantization-results.json).

## Verification

- `tests/test_models/test_quantization.py`: 21 passed.
- `.githooks/check-changed`: 320 passed, ruff clean.
- `python -m scripts.repo_policy`: ok.
- `git diff --check`: clean.

## Tradeoffs and caveats

- **Reference-only**: all execution goes through fake-quantized PyTorch tensors.
  No bit-packed kernel is invoked, so latencies here are not deployment claims.
- **No QAT/calibration**: STE hook exists but no training loop uses it yet;
  weight-space MSE is reported, not task loss.
- **Groupwise scaling** is implemented on the last tensor dimension only, which
  matches Linear weight layout `(out_features, in_features)`.
- **Tied weights**: conversion refuses to quantize shared Linear storage by
  default; callers must untie or pass `fail_on_tied=False` and accept duplicated
  storage in the converted model.
- **No checkpoint promoted**: MODEL_CARD is intentionally not updated (fixture
  wiring only).
