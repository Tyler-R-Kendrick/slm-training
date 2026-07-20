# SLM-123 — TwoTower removable low-rank delta adapter (wiring)

## What
Repository-owned, removable LoRA-style low-rank delta backend for selected
TwoTower denoiser projections. The parent checkpoint and context tower stay
frozen; only small `A`/`B` factors are trainable.

## Files
- `src/slm_training/models/adapters/spec.py` — versioned, torch-free `TwoTowerAdapterSpec`.
- `src/slm_training/models/adapters/low_rank.py` — `LowRankAdapter` wrapper (`y = W x + scale * B(A(dropout(x)))`).
- `src/slm_training/models/adapters/twotower_adapter.py` — deterministic target resolution and attachment.
- `src/slm_training/models/twotower.py` — `attach_adapter`, `enable_adapter`, `disable_adapter`, `adapter_parameters`, `save_adapter`, `load_adapter`, `merge_adapter_copy`, `active_adapter_identity`.
- `src/slm_training/harnesses/model_build/config.py` — `adapter_spec`, `adapter_trainable` fields.
- `src/slm_training/harnesses/model_build/factory.py` — `_maybe_load_adapter` integration in `build_model`.
- `scripts/train_model.py` — `--adapter-spec` / `--adapter-frozen` CLI flags.
- `tests/test_models/test_twotower_adapter.py` — torch-backed model tests.
- `tests/test_harnesses/model_build/test_adapter_factory.py` — factory/config round-trip tests.
- `docs/design/iter-slm123-twotower-adapter-20260719.json` — fixture evidence JSON.

## Supported targets
Canonical denoiser linear-projection names (see `TARGET_MODULE_PATHS`):

- `attn_q`, `attn_k`, `attn_v`, `attn_out`
- `cross_attn_q`, `cross_attn_k`, `cross_attn_v`, `cross_attn_out`
- `mlp_in`, `mlp_out`

`include_output_head=True` is rejected for now (output head is not a block-level
linear target). Context-tower adaptation is off by default.

## Adapter artifact layout
```text
adapter/
  adapter_config.json      # spec dict
  adapter_model.pt         # A/B tensors
  adapter_manifest.json    # module map, parameter count, bytes, base identity
```

## Commands
```bash
# Train with a removable adapter (normal use)
python -m scripts.train_model \
  --train-dir outputs/data/train/v1 \
  --adapter-spec outputs/runs/slm123_adapter_evidence/adapter \
  --steps 32

# Load adapter frozen (adapter-only inference, no adapter gradients)
python -m scripts.train_model \
  --train-dir outputs/data/train/v1 \
  --adapter-spec outputs/runs/slm123_adapter_evidence/adapter \
  --adapter-frozen \
  --steps 32
```

## Verification
- `pytest tests/test_models/test_twotower_adapter.py -q` → 14 passed
- `pytest tests/test_harnesses/model_build/test_adapter_factory.py -q` → 3 passed
- `python -m scripts.verify_version_stamps --check` → ok

## Honest caveats
This is **wiring-only evidence**. A real adapter-quality claim requires:

- a trained adapter checkpoint with binding-aware meaningful v2 metrics,
- parent/adapter bit-exact resume and merge-parity tests on the target device,
- and durable HF bucket provenance per SLM-103.

No promoted checkpoint or ship gate is made from this fixture.
