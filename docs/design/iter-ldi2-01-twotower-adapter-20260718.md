# LDI2-01 — removable low-rank adapter backend for TwoTower

Date: 2026-07-18
Status: **adapter primitive, deterministic attachment, adapter-only training, and
one-way merge landed with tests. Save/load-directory, the checkpoint allow-list, config/
CLI round-trip, and the `artifact_identity` distinction are the remaining follow-on.
No training run, no promoted checkpoint, no quality claim.**

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

## Honest remaining scope

- `save_adapter(path, provenance=...)` / `load_adapter(path)` for the removable adapter
  directory (`adapter_config.json` + adapter tensors `.pt` + `adapter_manifest.json` with
  the resolved module map, shapes, parameter count, and bytes), fail-closed on base
  fingerprint / tokenizer / module-map mismatch.
- The `.pt` checkpoint interplay: `_state_dict_for_checkpoint` dropping `adapter.` keys
  and `_load_checkpoint_state` allow-listing them, so historical checkpoints load into an
  adapter-capable model and vice-versa.
- Config/CLI round-trip (`ModelBuildConfig` → attach) and the `artifact_identity` /
  `compatibility_fingerprint` base-vs-adapter-vs-merged distinction.
- These are the next commits. This iteration runs no training and makes no quality claim.
