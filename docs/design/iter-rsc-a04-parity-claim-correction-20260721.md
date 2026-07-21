# RSC-A04: correct recursive parameter-parity claims and replace the misleading parity test (SLM-240)

Run id: `iter_rsc_a04_parity_claim_correction`
Status: **documentation_and_test_correction** (no quality claim made from this patch)
Date: 2026-07-21

## What this is

`src/slm_training/models/recursive_denoiser.py`'s module docstring claimed that
with `recursive_steps=1` and `recursive_transition_layers` equal to the old
stacked layer count, `SharedRecursiveDenoiserTower` "has the same parameter
count and layer names as `DenoiserTower`" — in the very same paragraph that
acknowledges the new z-state path (`z_latent`/`ctx_proj`). Those two claims
directly contradict each other: an added parameter path cannot leave the
parameter count unchanged. A repo-wide grep for this wording (and variants)
found exactly **one** occurrence — this docstring; no design doc, model card,
README, or generated fixture prose repeated it.

The companion test `test_recursive_steps_one_parity_with_denoiser_tower`
compounded the problem: its own docstring said "matches the ... contract",
but the test body only asserted shape equality and finiteness, with an
inline comment already noting the outputs differ. It proved interface
compatibility, not parameter/behavioral parity — the name overclaimed what
the test actually checked.

This is a pure documentation/testing/reporting correction. **No architecture
win/loss is inferred, no parameter-matched control is implemented, and
SLM-138's original wiring-only verdict is unchanged** — see the annotations
in `docs/design/quality-experiment-matrix.md` (V18 section) and
`docs/design/research-lineage.md` (SLM-138 row).

## The exact delta, reproduced from a formula

For the SLM-138 fixture's own denoiser config (`d_model=32`,
`max_len=256`, from `TwoTowerConfig.max_target_len`'s default):

```
z_latent   = max_len * d_model         = 256 * 32 = 8192
ctx_proj   = d_model * d_model + d_model = 32*32 + 32 = 1056
delta      = z_latent + ctx_proj       = 9248
```

`recursive_zstate_parameter_delta(d_model=32, max_len=256)` returns exactly
`9248`. The already-committed fixture reports `stacked_params=64994`,
`recursive_params=74242` — `74242 - 64994 = 9248` (`+14.23%`), matching the
formula exactly (real re-run confirmed below, not re-derived by hand). This
holds regardless of `vocab_size`/`n_layers`/`recursive_steps`, because the
shared transition-block parameters are identical between the two
architectures whenever `recursive_transition_layers == n_layers` and cancel
out of the subtraction; the entire delta is exactly `z_latent` + `ctx_proj`.
`tests/test_models/test_recursive_denoiser.py::test_recursive_parameter_delta_formula_matches_constructed_towers`
verifies this across five `(d_model, max_len)` pairs against real constructed
towers, including this exact fixture config.

## What changed

1. **`recursive_denoiser.py` module docstring** — rewritten to state
   precisely: shared-transition parameters do not scale with
   `recursive_steps` (compute changes, parameter count does not); `R=1`
   preserves the public interface and compatible tensor shapes, **not**
   output equivalence; V1 always adds `z_latent`/`ctx_proj` relative to
   `DenoiserTower`, with the exact formula above; checkpoint layer-name
   compatibility applies only to the mapped/common transition layers — the
   z-state keys require explicit initialization/migration (already true of
   `migrate_to_shared_recursive_denoiser`, whose `initialized_keys` list
   already covers `z_latent`/`ctx_proj.weight`/`ctx_proj.bias`); no
   parameter-efficiency or quality claim exists until a matched control
   campaign runs — a separate, later, out-of-scope issue.

2. **`ArchitectureComparisonReportV1`** (new, in `recursive_denoiser.py`) —
   a frozen dataclass with independently named fields per the issue's
   requirement, built only via `compare_denoiser_architectures(stacked,
   recursive, *, noisy_ids, context, pad_id)`, which measures everything
   from real constructed modules and a real forward/backward pass:

   - `interface_compatible`, `output_shape_compatible` — measured from a
     real forward pass on both towers with the same synthetic batch.
   - `parameter_count_total`, `parameter_count_denoiser` (transition layers
     only — architecture-independent, verified equal for matched configs),
     `active_parameter_count` (elements receiving nonzero gradient from one
     concrete forward pass — a genuinely different, smaller number than
     `parameter_count_total` for embedding-style tables indexed by a short
     sequence).
   - `checkpoint_bytes` — real `torch.save` byte counts.
   - `common_parameter_names_and_shapes` /
     `architecture_specific_parameter_names_and_shapes` — real
     `named_parameters()` name/shape comparison (deduplicated by tensor
     identity, so the internal `_f_layers`/`_g_layers` aliasing slices never
     pollute the architecture-specific set).
   - `parameter_count_delta`, `parameter_count_delta_pct`,
     `parameter_count_delta_matches_formula` — cross-checked in
     `__post_init__` against `parameter_count_total` and
     `recursive_zstate_parameter_delta`.
   - `block_evaluations_per_forward` — verified against real
     `TransformerBlock` forward-hook invocation counts (not merely the
     `recursive_steps * recursive_transition_layers` formula).
   - `estimated_forward_flops` — an explicitly-labeled analytic estimate,
     never a profiler measurement or latency claim.
   - `behaviorally_equivalent_under_declared_degeneracy` — measured via
     `torch.allclose` on real outputs; `False` today (no true-degeneracy
     mode exists).

   **Deliberately no `parity` field anywhere** — `__post_init__` raises if
   one is ever present in `as_dict()`.

3. **Tests** (`tests/test_models/test_recursive_denoiser.py`, 70 total, all
   passing) — renamed
   `test_recursive_steps_one_parity_with_denoiser_tower` to
   `test_recursive_r1_preserves_denoiser_interface_and_finite_shapes` and
   added: `test_recursive_r1_output_not_behaviorally_equivalent_to_stacked`,
   `test_recursive_parameter_delta_formula_matches_constructed_towers`
   (parametrized), `test_recursive_parameter_count_independent_of_recursive_steps`,
   `test_transition_layer_names_and_shapes_map_onto_stacked_1to1`,
   `test_architecture_comparison_report_block_evaluations_match_real_hook_counts`,
   `test_architecture_comparison_report_consistent_with_measured_counts`, four
   `ArchitectureComparisonReportV1.__post_init__` rejection tests, and
   `test_fixture_architecture_comparison_delta_reproduced_from_formula`.
   Strengthened `test_checkpoint_migration_to_shared_recursive` to assert the
   exact new z-state key set is a subset of `initialized_keys`.

4. **Fixture reporting**
   (`scripts/run_slm138_recursive_denoiser_fixture.py`) — a new
   `_architecture_comparison` helper builds a real
   `ArchitectureComparisonReportV1` per run (reusing the deterministic
   `shape_probe_inputs`/`shape_probe_context` RNG namespaces from SLM-239 —
   harmless by `isolated_draw`'s `fork_rng` guarantee) and embeds it under
   `architecture_comparison` in the JSON report. The rendered markdown now
   shows an "Architecture comparison (SLM-240 / RSC-A04)" section
   (`interface-compatible: true`, `parameter-matched: false`, `parameter
   delta: +9248 (+14.23%)`, `behavioral parity: not claimed`, `claim class:
   wiring`) and an objective-decomposition warning immediately above the raw
   stacked/recursive losses, referencing SLM-238's `recursive_depth_aux_mode`
   work so the two numbers are never read as a bare quality comparison.

## Real captured report instance (2026-07-21, this fixture config)

```json
{
  "contract_version": "ArchitectureComparisonReportV1",
  "claim_class": "wiring",
  "d_model": 32,
  "max_len": 256,
  "recursive_steps": 2,
  "recursive_transition_layers": 2,
  "interface_compatible": true,
  "output_shape_compatible": true,
  "parameter_count_total": {"stacked": 43040, "recursive": 52288},
  "parameter_count_denoiser": {"stacked": 33792, "recursive": 33792},
  "active_parameter_count": {"stacked": 35035, "recursive": 36283},
  "checkpoint_bytes": {"stacked": 188419, "recursive": 233018},
  "parameter_count_delta": 9248,
  "parameter_count_delta_pct": 21.486988847583643,
  "parameter_count_delta_matches_formula": true,
  "block_evaluations_per_forward": {"stacked": 2, "recursive": 4},
  "behaviorally_equivalent_under_declared_degeneracy": false
}
```

Full instance (including `common_parameter_names_and_shapes` and
`architecture_specific_parameter_names_and_shapes`) is in the sibling JSON's
`architecture_comparison_report_instance.report` key. Note:
`parameter_count_total`/`parameter_count_denoiser`/`active_parameter_count`
here are **tower-only** (no context tower), so their percentages differ from
the fixture's own top-level `stacked_params=64994`/`recursive_params=74242`
(whole `TwoTowerModel`, including the shared context tower) — the two totals
differ by exactly the (identical, cancels out) context-tower parameter count,
so `parameter_count_delta` (9248) and the whole-model delta are identical;
only the percentage denominator differs. The fixture's rendered markdown
normalizes the percentage against the whole-model total (`+14.23%`) to match
how this number is normally cited.

## Acceptance criteria check

- Repository search finds no unsupported same-parameter/parity wording for
  SLM-138: confirmed via repo-wide grep (see above); the only remaining
  occurrences of that phrasing are in this doc and the code comments that
  explicitly *retract* it.
- Each parity dimension is independently tested and reported: see
  `ArchitectureComparisonReportV1`'s field list and the new tests above.
- The 9,248 fixture delta is reproduced from the formula, not hard-coded:
  `recursive_zstate_parameter_delta(d_model=32, max_len=256) == 9248`,
  cross-checked against real constructed towers and the live fixture in three
  independent tests.
- Existing checkpoint migration remains functional: unchanged behavior,
  `test_checkpoint_migration_to_shared_recursive` still passes (now with a
  stronger assertion).
- No architecture win/loss is inferred: `claim_class` is always `"wiring"`
  (enforced in `__post_init__`); the fixture's loss numbers carry an explicit
  objective-decomposition warning.

## Non-goals (explicitly out of scope, unchanged)

No parameter-matched architecture implementation (a separate, later issue);
no training-objective change; no quality experiment. SLM-233's implementation
is untouched — this issue only leaves a note/link on it per the parent
instructions.
