# RSC-A02: final-depth double-counting semantics + independent depth-loss coefficient (SLM-238)

Run id: `iter_rsc_a02_final_depth_double_counting_semantics`
Status: **semantics + bounded calibration** (no quality or LOTUS-transfer claim)
Date: 2026-07-21

## What this is

SLM-237 (RSC-A01) repaired the recursive deep-supervision objective's
*weighting* math. This issue, SLM-238 (RSC-A02), repairs its *semantics*:
whether the final recursion depth should count once (via the primary
reconstruction term) or twice (primary + auxiliary) was never a stated
choice.

**Verified ambiguity:** `SharedRecursiveDenoiserTower.recursive_outputs`
(`src/slm_training/models/recursive_denoiser.py`) sets
`final_logits = depth_logits[-1]` and `result["logits"] = final_logits` — the
identical tensor, not a recomputation. So `rec_out["logits"] ==
rec_out["depth_logits"][-1]` exactly. The primary mask/reconstruction loss
consumes `logits`; the historical all-depth auxiliary term (SLM-237's fixed
weighted-mean) consumes every `depth_logits[d]` including `d = R-1`. Both
differentiate through the same forward computation for the final depth, and
prior to this issue no config field, telemetry field, or documentation named
that double path.

## Implementation

### Versioned objective mode

`TwoTowerConfig` gains two fields:

- `recursive_depth_aux_mode: str | None = None`
- `recursive_depth_aux_weight: float = 1.0`

Four mode values (`RECURSIVE_DEPTH_AUX_MODES` in `twotower.py`):

| Mode | Semantics |
| --- | --- |
| `off` | Primary final reconstruction only; requires an empty `recursive_depth_supervision_weights` tuple. |
| `intermediate_only` | Weights must cover exactly depths `0..R-2`; the final depth is structurally never indexed by the auxiliary loop. |
| `all_depths` | Weights must cover exactly depths `0..R-1`; the final depth is *intentionally* counted in both the primary and auxiliary terms. |
| `legacy_all_depths` | Same length rule/math as `all_depths`; a reproduction-only label used only by the deterministic migration path below — never the recommended choice for a new config. |

`recursive_depth_aux_weight` is an overall coefficient scaling the whole
auxiliary term (independent of the *relative* per-depth
`recursive_depth_supervision_weights`); it must be finite and `>= 0`, and is
always separately logged.

**The validator was extended, not forked.** SLM-237's
`validate_recursive_depth_supervision`/`ValidatedDepthSupervision` in
`twotower.py` gained `mode`/`aux_weight` parameters and an eligible-
depth-range check; no second parser exists.

### Backward compatibility and migration

`resolve_recursive_depth_aux_mode(mode, weights)`: `mode=None` (the true
dataclass default) resolves deterministically — `"off"` when
`recursive_depth_supervision_weights` is empty, `"legacy_all_depths"` when
non-empty — reproducing SLM-237's weighted-mean-over-every-depth objective
byte-for-byte (with the neutral `aux_weight=1.0` default). This is why **all
26 pre-existing SLM-237 tests pass unmodified** (one telemetry-presence
assertion was updated deliberately — see below, not a regression).

`migrate_recursive_depth_aux_config(raw_cfg)` makes this explicit and
persisted at `TwoTowerModel.from_checkpoint` load time (rather than
perpetually relying on the `None`-resolution forever): a checkpoint/config
missing the key gets `"off"` (no legacy weights) or `"legacy_all_depths"`
(legacy weights present) written in. Idempotent and non-mutating; both this
and `resolve_recursive_depth_aux_mode` have dedicated tests.

### Objective decomposition telemetry

Always populated (explicit `0.0`/mode string when the term is off/disabled,
never omitted):

- `primary_final_reconstruction_loss`
- `recursive_intermediate_aux_loss`
- `recursive_final_depth_aux_contribution`
- `recursive_depth_aux_weight`
- `recursive_depth_aux_mode`
- `combined_training_loss`

By construction: `combined_training_loss == primary_final_reconstruction_loss
+ recursive_depth_supervision_loss` exactly, and
`recursive_intermediate_aux_loss + recursive_final_depth_aux_contribution ==
recursive_depth_supervision_loss` exactly. Verified both in unit tests and
live in the bounded factorial below (max diff < 1e-4, floating-point
roundoff only).

## Tests

`tests/test_models/test_recursive_denoiser.py` — 41 passed (26 pre-existing
+ 15 new), covering every property SLM-238 requires:

1. `intermediate_only` never reads or differentiates through the final depth
   (`test_intermediate_only_never_reads_final_depth_aux_path`).
2. `all_depths` includes the final contribution exactly once
   (`test_all_depths_includes_final_contribution_exactly_once`).
3. Primary final loss is numerically identical across modes for fixed
   logits/targets (`test_primary_final_loss_identical_across_modes`).
4. `aux_weight=0` is exact primary-only with an explicit zero telemetry
   record (`test_aux_weight_zero_is_primary_only_with_explicit_zero_telemetry`).
5. Invalid mode/weight-length/aux_weight combinations raise
   (`test_invalid_mode_weight_combinations_raise` [parametrized] +
   `test_invalid_aux_weight_raises`).
6. R=1 `intermediate_only` reduces to primary-only on an empty tuple, raises
   on a non-empty one (documented contract)
   (`test_r1_intermediate_only_reduces_to_primary_only`).
7. Checkpoint/config round-trip preserves mode+coefficient; migration is
   deterministic and idempotent
   (`test_checkpoint_roundtrip_preserves_mode_and_coefficient`,
   `test_migrate_recursive_depth_aux_config_deterministic`,
   `test_resolve_recursive_depth_aux_mode_backward_compatible`).
8. Generated decomposition sums reproduce the scalar training loss exactly
   (`test_generated_decomposition_sums_reproduce_scalar_loss_exactly`).
9. The required `RecursiveObjectiveContractV2` schema builds validly from
   consistent metrics and rejects an inconsistent one
   (`test_recursive_objective_contract_v2_validates_sum_identities`).

One existing SLM-237 test
(`test_empty_tuple_valid_on_every_architecture_no_aux_term`) was updated —
not removed — to assert `recursive_depth_supervision_loss == 0.0` explicitly
present rather than absent, since that field being an explicit zero (never
omitted) is precisely SLM-238's required test #4 telemetry-contract change.

```
python -m pytest tests/test_models/test_recursive_denoiser.py -q
# 41 passed
```

## Bounded 5-arm factorial

`scripts/run_rsc_a02_depth_aux_mode_factorial.py` — deterministic 2-record
synthetic fixture (HERO/CTA) + bounded 6-record real-corpus smoke
(`train_seeds.jsonl` prefix), 3 training steps per arm, `recursive_steps=3`.
**Calibration/semantics only — no quality or LOTUS-transfer claim, no
promotion, no GPU campaign.** Full result:
`docs/design/iter-rsc-a02-depth-aux-mode-factorial-20260721.json` / `.md`.

Preregistered `lambda = 0.3` (chosen a priori, not tuned post-hoc): the final
depth already gets full weight once via the primary term, so a partial
(0.3x) extra credit in `all_depths` mode avoids letting the auxiliary term
dominate or swamp the primary gradient signal, while staying large enough to
matter; also a simple, reproducible round number for a first pass.

Fixture recipe, final training step, real numbers:

| Arm | Mode | aux_weight | primary | intermediate_aux | final_aux_contribution | combined |
| --- | --- | --- | --- | --- | --- | --- |
| A | off | 0.0 | 23.041454 | 0.0 | 0.0 | 23.041454 |
| B | intermediate_only | 1.0 | 23.048304 | 26.997311 | 0.0 | 50.045614 |
| C | all_depths | 1.0 | 23.044403 | 17.999525 | 7.681468 | 48.725395 |
| D | intermediate_only | 0.3 | 23.043634 | 8.100473 | 0.0 | 31.144108 |
| E | all_depths | 0.3 | 23.043045 | 5.400579 | 2.304305 | 30.747929 |

Observations (real, from this run — not fabricated):

- `combined_training_loss` reproduces the live `training_loss()` return value
  exactly (diff < 1e-4) in every arm.
- `recursive_final_depth_aux_contribution` is exactly `0.0` in both
  `intermediate_only` arms (B, D) and structurally absent from the per-depth
  loop's keys — never merely zero-weighted.
- First/last-depth contribution ratio in the `all_depths` arms is stable
  under the lambda rescale (C: 1.2522, E: 1.2526), as expected since
  `aux_weight` scales the whole term uniformly.
- Per-depth gradient-norm/cosine (measured on a freshly-initialized synthetic
  isolated-tower batch — same pattern this repo's own unit tests already
  use) is **identical across all 5 arms** (cosine vs. final depth: 0.973 /
  0.996 / 1.0 for depths 0/1/2). This is expected and correct, not a bug:
  `recursive_depth_aux_mode` only rescales the *aggregated training loss*
  after the forward pass, never the tower's forward mechanics — so this
  diagnostic is an architecture-level property (recursion depths become
  progressively more aligned with the final depth's gradient direction),
  invariant to the aux-mode choice. It functions as a useful sanity check
  that mode selection has zero side effect on the model's forward path.
- Params/FLOPs are structurally unchanged across arms: every arm builds from
  an identical `TwoTowerConfig` module-construction surface; only the
  loss-aggregation fields differ.

**Bounded safety smoke (syntax/structure/strict-semantic) status:** vacuous
in this sandbox. The official `@openuidev/lang-core` Node bridge failed with
a pre-existing, unrelated `NODE_OPTIONS` incompatibility
(`--import tsx is not allowed in NODE_OPTIONS`), confirmed present
identically on the pre-SLM-238 tree via `git stash` — not introduced by this
change. The factorial script detects this the same way
`scripts/run_perf_matrix.py` does (`_quality_pipeline_ok()`) and reports
`bridge_healthy=false` explicitly per arm rather than fabricating pass/fail
numbers. This smoke was always a non-gating safety diagnostic; its
unavailability here does not affect any RSC-A02 acceptance criterion, all of
which are verified directly by the decomposition numbers and test suite
above.

## Recommendation for SLM-233

**Default-candidate: `intermediate_only`.** The final recursion depth's
output *is* the model's actual prediction; supervising it a second time
through the auxiliary term (`all_depths`) rewards/penalizes the identical
forward computation twice per step for no architecturally distinct reason,
and inflates its effective learning-rate contribution relative to the
intermediate depths the auxiliary term exists to help train.
`intermediate_only` cleanly separates concerns — final-depth supervision
comes solely from the primary reconstruction loss (as it should, since that
*is* the output), while a coefficient-controlled auxiliary term (starting
point `recursive_depth_aux_weight=0.3`) supervises only the genuinely
intermediate representations, matching the standard deep-supervision
rationale used elsewhere in the literature (auxiliary heads on intermediate
representations, not a second vote on the final one).

`all_depths`/`legacy_all_depths` remain fully-supported, explicit controls —
this bounded factorial's scale cannot rule out that the seemingly-redundant
double count helps in practice (e.g. by acting as an implicit
tower-strength-weighted boost on the final depth); that empirical question
belongs to SLM-233, not this doc.

**For SLM-233's future control matrix:** treat `recursive_depth_aux_mode` as
a first-class axis crossed with `recursive_steps` and
`recursive_depth_aux_weight` (a small preregistered lambda grid, e.g.
`{0.1, 0.3, 0.5, 1.0}`), evaluated under this repo's full honest ship gates
(not this bounded smoke) on real held-out OpenUI corpora at a scale
sufficient to move ship-gate metrics outside noise. Include `off` as the
byte-identical baseline, `intermediate_only` at the recommended lambda as the
primary candidate, and `all_depths`/`legacy_all_depths` as an explicit
alternative-hypothesis arm — specifically to test whether double-counting
the final depth is a net positive despite this doc's a priori architectural
objection, rather than assuming the recommendation above without evidence.

## SLM-138 documentation annotation

`docs/design/quality-experiment-matrix.md`'s `## RSC-A01` section (covering
E303/V18's deep-supervision objective) now carries a note that the
pre-existing all-depth objective double-counts the final recursion depth —
an ambiguity neither the original SLM-138 wiring-only landing nor SLM-237's
correctness fix named — and that SLM-238 makes this an explicit, versioned,
tested choice. **SLM-138's original wiring-only verdict and SLM-237's
correctness-fix verdict are unchanged** — this is an annotation, not a
re-verdict.

## Version bumps

- `model.twotower`: v63 -> v65 (v64 added the config fields/validator
  extension/telemetry; v65 added the required `RecursiveObjectiveContractV2`
  schema).
- `model.recursive_denoiser`: v2 -> v4 (v3 added the 14 new tests and the
  new factorial script; v4 added the schema's own test and regenerated the
  factorial's committed docs after an accidental plan-only overwrite).

## Non-goals honored

- No new latent slots or recurrence architecture.
- No routing through a separate auxiliary decoder.
- No production default change outside the versioned recursive path
  (`recursive_depth_aux_mode` defaults to the backward-compatible `None`,
  never a new always-on behavior).
- No adoption decision — that is SLM-233's job; this doc only recommends a
  default-candidate for that future campaign.
- No quality or LOTUS-transfer claim from the bounded factorial —
  calibration/semantics only.

## Deferred to follow-up

- SLM-233's full GPU control-matrix campaign (out of scope by design).
- A bridge-healthy rerun of the syntax/structure/strict-semantic safety
  smoke — this session's sandbox has a pre-existing, unrelated Node bridge
  `NODE_OPTIONS` incompatibility (confirmed present on the pre-SLM-238 tree
  too, so not introduced here).
