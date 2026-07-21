# E623 — close train_model.py's semantic_plan/schema decode-weight CLI gap, then exercise it for real

Date: 2026-07-20
Status: completed, positive real end-to-end result, code retained, checkpoints
rejected, not promotable

E622 found that a real 40-step `slm sft train` run's own periodic eval
genuinely populates `self._slot_contracts` (the E617 gate) live during
training, but every weighted decode lever behind that gate stayed `0.0`
because `scripts/train_model.py`'s CLI never exposed
`schema_role_slot_decode_weight`/`semantic_plan_*_decode_weight`/related
flags at all — only `scripts/evaluate_model.py` did. E622 reported this as a
CLI surface gap and explicitly deferred fixing it. This iteration verifies
that finding, closes the gap, and reruns the same recipe with the
previously-inert levers turned on.

## Step 1: verify E622's finding

```
grep -c semantic_plan scripts/train_model.py scripts/evaluate_model.py
# scripts/train_model.py:0
# scripts/evaluate_model.py:18
```

Confirmed real. Tracing `scripts/evaluate_model.py`'s `add_argument`/config
wiring (lines 179-314, 615-660) against
`src/slm_training/harnesses/model_build/config.py` (`ModelBuildConfig`, all
target fields already exist there, defaulting to `None`) and
`src/slm_training/harnesses/model_build/factory.py` (already reads every one
of these `ModelBuildConfig` fields generically at lines 143-144/505-509 and
copies them into `TwoTowerConfig`) showed the gap is exactly and only in
`scripts/train_model.py`'s argparse surface — no config or factory-side gap
existed. `scripts/train_model.py` also never exposed
`--semantic-role-contract-in-context` (`evaluate_model.py:391`), a
prerequisite: `TwoTowerModel._generate_batch_once` raises `ValueError` if
`semantic_role_decode_weight > 0` without it (`twotower.py:8164-8173`).

## Step 2: close the gap

Added 21 flags to `scripts/train_model.py`, mirroring `evaluate_model.py`'s
exact flag names/help text and each field's already-existing
`ModelBuildConfig`/`TwoTowerConfig` plumbing:

- `--semantic-role-contract-in-context` (placed next to the existing
  `--slot-contract-constrained-decode` context-flag group)
- `--semantic-role-decode-weight`, `--semantic-role-schema-candidates`,
  `--slot-coverage-close-decode-weight`, `--schema-value-decode-weight`,
  `--schema-opaque-decode-weight`, `--schema-enum-close-decode-weight`,
  `--schema-opaque-close-decode-weight`, `--schema-role-slot-decode-weight`
- all nine `--semantic-plan-*-decode-weight` flags
- `--visible-reference-decode-weight`

(placed between the existing `--slot-component-content-arity` and
`--component-edge-loss-weight` flags, matching `evaluate_model.py`'s
relative ordering). Unlike `evaluate_model.py` (whose float defaults are
`None`, meaning "preserve the loaded checkpoint's own value unless
overridden"), `train_model.py` always builds a fresh config, so these use
`default=0.0` to match every sibling decode-weight flag `train_model.py`
already had (`component_inventory_decode_weight` etc. at
`train_model.py:461-465`); `TwoTowerConfig` treats `None` and `0.0`
identically (`getattr(self.config, name, 0.0) or 0.0` throughout
`twotower.py`), so this is a cosmetic difference only.

Added `tests/test_scripts/test_train_model.py::
test_train_cli_wires_semantic_plan_and_schema_decode_weights`, mirroring the
file's existing `test_train_cli_wires_honest_slot_contract`/
`test_train_cli_wires_action_alias_args` pattern: passes every new flag with
a nonzero value plus a few left at the default, asserts the built
`ModelBuildConfig` carries each value through untouched.

## Step 3: exercise it for real (paired control/treatment)

Reran E622's exact recipe as a fresh **control** (same corpus, seed,
architecture, step count) to get a byte-verified paired baseline in this
session rather than reusing old numbers by reference, then a **treatment**
run adding the now-wired flags at E617's own treatment values (the same
values E617 found produced a real positive standalone-eval effect):
`schema_role_slot_decode_weight=8.0`, `semantic_role_decode_weight=8.0` (with
`--semantic-role-contract-in-context`), `slot_coverage_close_decode_weight=2.0`,
`schema_value_decode_weight=4.0`, `schema_opaque_close_decode_weight=4.0`,
`semantic_plan_decode_weight=4.0`, `semantic_plan_margin_decode_weight=2.0`,
`semantic_plan_binding_decode_weight=1.0`, `semantic_plan_root_decode_weight=8.0`,
`semantic_plan_root_margin_decode_weight=2.0`,
`semantic_plan_repeated_array_close_margin_decode_weight=2.0`,
`semantic_plan_repeated_slot_margin_decode_weight=2.0`,
`semantic_plan_typed_array_nonempty_margin_decode_weight=2.0`,
`semantic_plan_typed_array_item_margin_decode_weight=2.0`. All other levers
(`schema_opaque_decode_weight`, `schema_enum_close_decode_weight`,
`semantic_plan_seed_decode_weight`, `semantic_plan_inline_decode_weight`,
`visible_reference_decode_weight`, `semantic_role_schema_candidates`) stayed
at `0.0`/off, matching E617's own recipe exactly.

```bash
python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/e530_visible_semantic_roles_r2_20260719 \
  --model twotower --device cpu --context-backend scratch --output-tokenizer choice \
  --steps 40 --batch-size 1 --seed 0 \
  --test-dir src/slm_training/resources/data/eval/remediated --eval-every 40 --eval-suites ood \
  --honest-slot-contract --slot-contract-constrained-decode \
  --no-sync-checkpoints --run-id e623-control-scratch40-20260720
```

```bash
python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/e530_visible_semantic_roles_r2_20260719 \
  --model twotower --device cpu --context-backend scratch --output-tokenizer choice \
  --steps 40 --batch-size 1 --seed 0 \
  --test-dir src/slm_training/resources/data/eval/remediated --eval-every 40 --eval-suites ood \
  --honest-slot-contract --slot-contract-constrained-decode \
  --semantic-role-contract-in-context \
  --semantic-role-decode-weight 8.0 --schema-role-slot-decode-weight 8.0 \
  --slot-coverage-close-decode-weight 2.0 --schema-value-decode-weight 4.0 \
  --schema-opaque-close-decode-weight 4.0 --semantic-plan-decode-weight 4.0 \
  --semantic-plan-margin-decode-weight 2.0 --semantic-plan-binding-decode-weight 1.0 \
  --semantic-plan-root-decode-weight 8.0 --semantic-plan-root-margin-decode-weight 2.0 \
  --semantic-plan-repeated-array-close-margin-decode-weight 2.0 \
  --semantic-plan-repeated-slot-margin-decode-weight 2.0 \
  --semantic-plan-typed-array-nonempty-margin-decode-weight 2.0 \
  --semantic-plan-typed-array-item-margin-decode-weight 2.0 \
  --no-sync-checkpoints --run-id e623-treatment-scratch40-20260720
```

Both ran under `timeout 165` (real 49.5s / 16.2s per `train_summary.json`'s
own `elapsed_wall_seconds`), comfortably inside the 3-minute cap.

## Result: byte-verified reproduction, then a real, large decode-time delta

**Control** reproduced E622 exactly — `last.pt` sha256
`6b7aaf2bdd4deaa2b9f54f8ff2fae75d36c8dc847e2d49ba11f5c4f9037bf00a`, byte-identical
to E622's own checkpoint — and its `ood`-suite eval numbers
(`parse_rate` 1.0, `meaningful_program_rate` 0.0, `structural_similarity`
0.190625, `placeholder_fidelity` 0.0, `reward_score` 0.0, `ship_score`
0.03465909090909091) match E622's report exactly, confirming CPU determinism
and a true paired rerun rather than a coincidence.

**Training loss is identical between control and treatment**
(`last_loss` 15.396740913391113 in both) — expected and important to state
plainly: every flag added this iteration is decode-only (only read inside
`_generate_batch_once`'s generation path), never inside the teacher-forced
forward/backward loss computation, so no training dynamic changed. The two
runs' underlying trained weights are the same; only the checkpoints' saved
config differs (hence different `last.pt` sha256:
`3dec6e98e8295e1a877ae01e1dde55fd86c5ca38ccff6b19e7f28d22911844c8` for
treatment).

**The `ood`-suite periodic eval (n=4) differs sharply**, and
`scoreboard.json`'s `evaluation_policy` block confirms every treatment weight
landed exactly as passed (`schema_role_slot_decode_weight: 8.0`,
`semantic_plan_root_decode_weight: 8.0`, etc.) — this is the CLI gap now
closed and genuinely exercised inside a live `slm sft train` process's own
periodic eval, not a standalone `evaluate_model.py` replay:

| metric | control | treatment | delta |
| --- | --- | --- | --- |
| `parse_rate` | 1.0 | 1.0 | 0.0 |
| `meaningful_program_rate` | 0.0 | 0.75 | +0.75 |
| `placeholder_fidelity` | 0.0 | 0.5667 | +0.5667 |
| `structural_similarity` | 0.190625 | 0.6439 | +0.4533 |
| `reward_score` | 0.0 | 0.67475 | +0.67475 |
| `ship_score` | 0.03465909090909091 | 0.657201515151515 | +0.6225 |

This delta is much larger than E617's own standalone-eval finding
(`placeholder_fidelity` +0.042 on an already-trained checkpoint from a longer
lineage). The most plausible reading, consistent with E617/E620's mechanism
(compiler-legal candidate biasing at decode time): a 40-step scratch
checkpoint has almost no learned structure of its own, so its unbiased
decode is close to degenerate (control: `meaningful_program_rate` 0.0,
`reward_score` 0.0), and these biases mechanically steer even a
near-untrained network's logits toward compiler-legal, prompt-plan-compatible
continuations — a floor effect that a better-trained checkpoint (E617's) has
much less room to show. This is a real, honest, decode-time-only measurement,
not a training-quality claim.

## Honest caveats

- `n=4` (`ood` suite), a single fixed seed, one 40-step scratch checkpoint
  pair — not a confirmatory or statistically powered comparison (H19's
  protocol from this same matrix would be the right tool for that).
- Both checkpoints remain far too undertrained to be promotable or synced
  (`--no-sync-checkpoints`); this is diagnostic evidence of CLI wiring and
  decode-time mechanism, not a ship claim.
- Training loss is unchanged by design (decode-only levers) — do not read
  the identical `last_loss` as "the flags didn't do anything"; the eval
  table above is the correct place to look.
- This reruns E617's already-vetted bias mechanisms through a newly-wired
  CLI path; it does not re-audit their honesty (E617-E621 already did that
  work) — it only confirms the training-time CLI gap is closed and that the
  effect reproduces end-to-end from a real `slm sft train` invocation.

## Verification

```bash
python -m pytest tests/test_scripts/test_train_model.py -q
python -m scripts.verify_version_stamps --check
python -m scripts.repo_policy
```

Evidence: [JSON](iter-e623-decode-weight-cli-gap-closed-scratch40-20260720.json).
