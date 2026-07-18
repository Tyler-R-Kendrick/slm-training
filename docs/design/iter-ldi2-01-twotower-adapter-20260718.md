# LDI2-01 — removable low-rank adapter backend for TwoTower

Date: 2026-07-18
Status: **adapter primitive, deterministic attachment, adapter-only training, one-way
merge, and a removable save/load directory (fail-closed on identity) landed with tests.
The base-`.pt` checkpoint interplay, the config/CLI round-trip, and the `artifact_identity`
base-vs-merged distinction are the remaining follow-on. No training run, no promoted
checkpoint, no quality claim.**

## Why this exists

LDI2 needs a small, removable, lineage-safe actuator on selected TwoTower denoiser
projections so a local intervention can be trained and measured against the untouched
full-update parent — without importing Hugging Face PEFT. The backend is standard
low-rank delta (not DoRA/PiSSA/AdaLoRA, which are not implemented here).

## What landed

- `models/adapters/spec.py` — `TwoTowerAdapterSpec`: torch-free, frozen, versioned,
  validated config that round-trips through `to_dict`/`from_dict` (rejects unknown
  fields), bound to a base by `base_compatibility_fingerprint` / `base_checkpoint_sha` /
  `tokenizer_sha`.
- `models/adapters/low_rank.py` — `LowRankAdapter`: wraps a frozen `nn.Linear` with
  `y = W x + (alpha/rank)·B(A(dropout(x)))`, `B` zero-initialized (a fresh adapter is
  output-identical to the parent bit-for-bit), enable/disable, and a one-way
  `merged_linear()` on a copy that never mutates the parent.
- `models/adapters/twotower_adapter.py` — deterministic target resolution mapping spec
  target names (`attn_q/k/v/out`, `cross_attn_*`, `mlp_in/out`) onto the denoiser block
  linears, failing closed on unknown targets, non-linear matches, and out-of-range
  layers; attaches wrappers in place under a forked RNG (attachment never shifts the
  training RNG). The context tower is never adapted.
- `TwoTowerModel` methods: `attach_adapter` (base-fingerprint fail-closed + freeze every
  non-adapter parameter so the existing `trainable_parameters()` yields only adapter
  tensors), `enable_adapter` / `disable_adapter`, `adapter_parameters`, `has_adapter`,
  `active_adapter_identity` (content digest of adapter tensors), and `merge_adapter_copy`
  (wrapper-free copy equal to the adapter-enabled map; original untouched).

## Acceptance covered by tests

Fresh enabled adapter equals the parent bit-for-bit at every adapted site; disabled
adapter equals the parent after weights change; only adapter tensors receive gradients;
requested targets resolve deterministically and missing/unsupported/out-of-range targets
raise actionable errors; base-fingerprint mismatch fails closed; attachment does not
shift training RNG; adapter-only `trainable_parameters`; merged copy matches the enabled
output and leaves the original adapter intact. Existing `test_twotower.py` unchanged.

The removable adapter directory landed: `save_adapter(path, provenance=...)` writes
`adapter_config.json` (the spec), `adapter_model.pt` (only the lora tensors — the base
checkpoint is never duplicated), and `adapter_manifest.json` (resolved module map,
parameter names/shapes, trainable parameter count, adapter bytes, base/tokenizer
identity). `load_adapter(path, trainable=...)` fails closed on a tokenizer or base
compatibility-fingerprint mismatch before copying any tensor, then attaches and loads.

## Honest remaining scope

- The base-`.pt` checkpoint interplay is **not** a one-line allow-list: attaching an
  adapter renames the wrapped base weight key (`…q_proj.weight` → `…q_proj.base.weight`,
  plus `lora_A`/`lora_B`), so `_state_dict_for_checkpoint` / `_load_checkpoint_state`
  need to strip the wrapper structure (not just allow-list `adapter.` keys) for a
  base checkpoint to round-trip through an adapter-capable model. This needs its own
  careful pass.
- Config/CLI round-trip (`ModelBuildConfig` → attach) and the `artifact_identity` /
  `compatibility_fingerprint` base-vs-merged distinction.
- These are the next commits. This iteration runs no training and makes no quality claim.
