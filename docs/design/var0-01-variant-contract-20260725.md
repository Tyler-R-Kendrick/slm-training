# VAR0-01: VariantContractV1 — pack-, kernel-, and variant-owned facts

**Issue:** SLM-422 (VAR0-01). **Claim class:** `contract`. This document produces
no quality, capability, or promotion claim — it is a survey of which facts are
currently mis-bucketed, not a certification that they have been fixed.

## What this is

`src/slm_training/dsl/variants.py` defines `VariantContractV1` and registers
three existing model variants: `repl_operators` (the REPL/operator surface),
`tree_edit_diffusion` (the Kapur-style tree-edit diffusion baseline), and
`twotower_prompt_ast` (the TwoTower prompt→AST denoiser). The registry is
derived from each variant's live sources — never hand-authored — and pinned
against `src/slm_training/resources/variant_registry.json` by
`scripts/verify_decode_invariants.py::check_variant_contracts`, mirroring the
existing `check_ops_vocab` gate.

## The three-bucket classification

Every fact a variant might hardcode falls into exactly one bucket
(`FACT_CLASSIFICATION` in `variants.py` enforces this is total):

| Bucket | Facts | Must be read from |
| --- | --- | --- |
| pack-owned | component inventory, per-component property names, property value domains, arity | `src/slm_training/dsl/pack.py` (see caveat below) |
| kernel-owned | op identity, op family, token ids | `src/slm_training/dsl/ops_vocab.py::OPS_VOCAB` |
| variant-owned | action encoding, edit budget, decode loop, search, seed selection | local to the variant |

## A caveat this survey surfaced: two DSL-pack contracts exist

The issue's repository anchors point at `src/slm_training/dsl/packs/` (plural
directory: `types.py`, `openui.py`, `toy_layout.py`, `arith_sketch.py`) as the
pack authority. That module does exist and defines `DSLPack` (all-caps DSL),
but it is **not** what `src/slm_training/dsl/operators/registry.py` actually
imports for pack authority — that file imports `DslPack` (mixed case) from
`src/slm_training/dsl/pack.py` (singular file, 693 lines), which independently
registers `"openui"`, `"toy-layout"`, and other packs via `register_pack`/
`get_pack`/`list_packs`.

`docs/design/dsl-pack-contract.md` describes the field-for-field contract that
matches `dsl/pack.py`'s `DslPack` (fields `backend`, `oracle`,
`corpus_generator`, `scope_extractor`, `placeholder_policy`), while
`dsl/packs/types.py`'s `DSLPack` has a different field set (`grammar`,
`canonicalize`, `canonical_equal`, `validity_oracle`, `corpus_generator`,
`scope_check`, `placeholders`) and no consumer under `src/` other than its own
pack instances. The doc itself notes (around its own lines 93-97) that a
`dsl/packs/*` file layout was "deferred" — yet the files already exist,
suggesting `dsl/packs/` is a newer, partially-landed replacement that has not
displaced `dsl/pack.py` as the live authority.

**Disposition: `UNRESOLVED`.** This issue's non-goals explicitly forbid
unifying or refactoring pack/variant surfaces ("no model, checkpoint,
training, or decode-path change"). `variants.py`'s live checks are therefore
written against `dsl/pack.py` (what `operators/registry.py` actually imports),
and `_imports_module(source_path, "dsl.pack")` matches both `dsl.pack` and
`dsl.packs` import spellings so a future consolidation does not silently
invalidate the gate. Reconciling the two pack contracts is out of scope here
and is not currently owned by any VAR-series issue; it should be triaged
separately.

## Per-variant classification

### `repl_operators` — `src/slm_training/dsl/operators/registry.py`

* **Component inventory / pack-owned facts:** sourced through `DslPack`
  (`registry.py:20` imports `slm_training.dsl.pack.DslPack`; all execution
  routes through `validate_with_pack_authority()`). `inventory_source="pack"`,
  verified live: the gate fails closed if this claim is made while the source
  file stops importing `dsl.pack`.
* **Kernel-owned facts:** this variant's legal action ids are exactly the
  operators that back `OPS_VOCAB` (`ops_vocab.py::_live_operator_ids` /
  `_history_op_ids` read from `dsl/operators/local.py`, `topology.py`,
  `conversation.py` — the same registries `registry.py` exposes). Its
  `action_alphabet_fingerprint` is a content hash over these live operator ids
  (`repl_operators.legal_actions`).
* **Variant-owned facts:** which operators are exposed as *tokens* to a
  decoder (`kernel_ops`) is empty for every variant today — confirmed by
  direct grep: `OPS_VOCAB` has no consumer under `src/` outside
  `ops_vocab.py` itself and its test. VAR2-01 ("give OPS_VOCAB a first
  consumer") is the follow-up that would populate this.

### `tree_edit_diffusion` — `src/slm_training/models/tree_edit_diffusion.py`

* **Component inventory / pack-owned facts — recorded violation:**
  `LEAF_COMPONENTS` (line 72) and `CONTAINER_COMPONENTS` (line 73) are
  module-level Python tuples, not read from any pack. The comment at lines
  69-71 ("Derived from the fixed grammar rather than hardcoded beyond this
  split") is not currently true: the file imports neither `dsl.pack` nor
  `dsl.packs`. `CONTAINER_RESTS` (line 78) and `V05_TEMPLATES` (line 89) are
  likewise module constants. `inventory_source="module_constant"` records
  this honestly; the gate does **not** fail on it (an honest violation is not
  a false claim), and this issue makes no behavior change here — VAR0-03 is
  the follow-up that migrates this variant to read from the pack.
* **Action alphabet / variant-owned:** the 11 `ACTION_*` constants
  (`ACTION_STOP` through `ACTION_BIND_PLACEHOLDER`, `N_ACTIONS = 11`,
  lines 49-64) are this variant's own bounded edit-action space — correctly
  variant-owned per the classification table (`action_encoding`). The
  registry's `action_alphabet_fingerprint` is a content hash over the live
  `ACTION_*` constant names, parsed statically from source (never imported —
  this module imports `torch`, and the registry must stay importable in the
  dependency-light decode-invariants CI job).
* **Kernel-owned facts:** `kernel_ops=()` — this variant does not touch
  `OPS_VOCAB` at all (its edit actions predate and are disjoint from the
  reserved-ops kernel).

### `twotower_prompt_ast` — `src/slm_training/models/twotower.py`

* **Action alphabet:** TwoTower decodes complete programs via masked
  denoising directly over structural tokens (`models/grammar.py` imports
  `STRUCTURAL_TOKENS` from `dsl/openui_tokens.py`; TwoTower imports
  `models/grammar.py`). Its `action_alphabet_fingerprint` hashes the live
  `STRUCTURAL_TOKENS` frozenset.
* **Component inventory / pack-owned facts — `UNRESOLVED`, recorded as
  `inventory_source="n/a"`:** `STRUCTURAL_TOKENS` is a shared kernel-ish
  module (`dsl/openui_tokens.py`), not the `DslPack` registry (`dsl/pack.py`)
  and not a variant-local hardcoded tuple either. Forcing it into "pack" would
  overclaim (the `openui` `DslPack` instance does not expose this field
  itself; the grammar backend does); forcing it into "module_constant" would
  understate that it is shared, not variant-local. Recording `"n/a"` is the
  honest answer per this issue's own escape hatch rather than picking a
  bucket that isn't true.
* **Seed policy:** `TwoTowerConfig.mask_min`/`mask_max`/`gen_steps`
  (`twotower.py:234-236`) describe an iterative masked-denoising schedule;
  `seed_policy_id="twotower_prompt_ast.full_mask_seed"` names that starting
  state without asserting anything about training or quality.
* **Kernel-owned facts:** `kernel_ops=()`, same reasoning as the other two
  variants — no production consumer of `OPS_VOCAB` exists yet.

## What is out of scope here (by design)

* `LEAF_COMPONENTS` / `CONTAINER_COMPONENTS` / `CONTAINER_RESTS` are not
  changed to read from a pack — that is VAR0-03.
* The tree-edit action alphabet and the operator alphabet are not unified —
  distinct surface alphabets may be intentional; this issue records that, it
  does not collapse it.
* No model, checkpoint, training, or decode-path changed. No quality claim is
  made anywhere in this document or in `variants.py`.
* The `dsl.pack` vs `dsl.packs` duplication is recorded as `UNRESOLVED` and
  left for a separate issue.

## Verification

```
python -m scripts.verify_decode_invariants   # now reports a variant_contracts block
python -m pytest -q tests/test_dsl/test_variants.py tests/test_dsl/test_ops_vocab.py
python -m scripts.verify_version_stamps --check --base <merge-base>
```
