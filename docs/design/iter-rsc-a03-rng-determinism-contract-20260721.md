# RSC-A03: bit-reproducible recursive fixtures + RNG determinism contract (SLM-239)

Run id: `iter_rsc_a03_rng_determinism_contract`
Status: **correctness/infrastructure fix** (no model-quality claim made from this work)
Date: 2026-07-21

## What this is

Kimi independently reran the SLM-138 fixture
(`scripts/run_slm138_recursive_denoiser_fixture.py`) and reproduced the
headline totals, but per-depth values moved in the low decimal places when
global RNG consumption order changed. Root cause: the fixture built forward-
shape probes (`torch.randint`/`torch.randn`) using the *global* torch RNG,
between model construction and the `training_loss` call whose internal
corruption sampling (`TwoTowerModel._mask_targets`) also reads the global
RNG, unconditionally, without a reseed of its own. Any harmless probe
inserted or reordered between those two points shifted the corruption draws
seen by `training_loss` -- a fixture used for regression evidence must not
depend on that incidental order. The checked historical artifact
(`docs/design/iter-slm138-recursive-denoiser-20260720.json`) also recorded
`code_dirty=true`, so it was never claim-grade evidence in the first place.

This issue does **not** touch SLM-237/238's objective semantics
(`validate_recursive_depth_supervision`, `recursive_depth_aux_mode`,
`RecursiveObjectiveContractV2` in `src/slm_training/models/twotower.py` are
untouched). It is RNG plumbing around the fixture only.

## A second RNG source, found during this work

`_mask_targets` reads **two** independent RNG sources, not one:

1. The global torch RNG (`torch.rand`/`torch.randint`).
2. A persistent per-instance `random.Random(config.seed)`
   (`TwoTowerModel.self._rng`), used for the "ensure at least one
   predictable token per row" fallback and mixed-pattern statement-span
   selection.

Restoring only the global torch RNG state between two `training_loss` calls
on the same model reproduces a **different** loss the second time --
verified directly in
`test_rsc_a03_restoring_only_torch_rng_is_insufficient_without_model`. The
new `RngCheckpoint`/`seed_training_corruption` helpers
(`src/slm_training/models/rng_contract.py`) capture and restore both sources
together whenever a `model` is passed.

## RNG namespace contract (`FixtureRngContractV1`)

Six disjoint, fixed-forever-once-shipped namespaces, each with a
deterministic offset from a fixture's `base_seed`
(`src/slm_training/models/rng_contract.py::NAMESPACE_OFFSETS`):

| Namespace | Offset | How it is used |
| --- | --- | --- |
| `model_initialization` | 0 | Unchanged: `TwoTowerModel.__init__` seeds `torch.manual_seed(config.seed)` itself. |
| `shape_probe_inputs` | 10000 | Synthetic token-id probe tensors, drawn via `isolated_draw` (`torch.random.fork_rng`) -- provably leaves the outer RNG stream untouched. |
| `shape_probe_context` | 20000 | Synthetic context-float probe tensors, same isolation. |
| `training_corruption` | 30000 | Explicitly (re)seeds both RNG sources (global torch + `model._rng`) immediately before each `training_loss` call. |
| `training_batch_order` | 40000 | Reserved for future batch shuffling; declared but not exercised by this fixture's fixed 1-2 record batch. |
| `control_only` | 50000 | Any other incidental draw (e.g. the fixture's optional extra-harmless-probe test hook). |

`isolated_draw` runs a callable inside `torch.random.fork_rng(devices=[])`
under a namespace-derived seed, so the outer global stream is byte-identical
whether the probe runs, runs twice, or never runs -- this is what makes
probe insertion/reordering provably harmless rather than incidentally
harmless.

## Fixture execution refactor (six phases)

`scripts/run_slm138_recursive_denoiser_fixture.py::_run_fixture` now
separates: (1) construction, (2) deterministic forward-shape probes, (3)
deterministic pre-update objective decomposition (explicit
`training_corruption` seed + `RngCheckpoint` capture immediately before each
`training_loss` call), (4) one optimizer step, (5) deterministic post-update
verification (the *same* `RngCheckpoint` restored -- never a second
`training_loss` call with an implicitly-advanced corruption RNG), (6)
checkpoint round-trip.

## Clean-tree evidence gate

`_run_fixture` always runs (even on a dirty tree, for local debugging) and
honestly reports `evidence_gate.code_dirty`/`comparable`/`claim_grade`
(sourced from `build_version_stamp`'s existing `git status --porcelain`
check, plus a `diff_hash` when dirty). The **persistence** boundary is in
`main()`'s `--mode fixture` path: a dirty (or unknowable) tree without
`--allow-dirty` writes only the local `outputs/runs/` debug artifact and
refuses to write `docs/design/`; `--allow-dirty` writes the artifact anyway,
still marked `comparable=false`/`claim_grade=false` -- it can never launder
into claim-grade evidence.

## FixtureDeterminismReportV1 -- measured verdict

`--mode determinism` runs the fixture twice (run A/run B, identical config),
once with shape probes reordered (`recursive_first`), and once with an extra
harmless probe inserted, then compares every field (excluding
`version_stamp.stamped_at` and the `rng_contract` echo, which legitimately
records which config a given permutation run used) plus the full canonical-
JSON digest.

**Measured verdict: `bit_exact`.** Run A and run B produce byte-identical
JSON (including across separate process invocations -- verified manually
during this work, not just in-process). The probe-order permutation and the
extra-harmless-probe permutation reproduce identical `losses`,
`post_update_verification`, `deep_supervision_metrics`, and
`forward_shapes` to run A (their only difference from run A is the
`rng_contract` config echo itself, which is expected). A distinct
`training_corruption_seed` (999999 vs the default-derived 30000) changes
only `losses`/`post_update_verification`/`deep_supervision_metrics`/
`rng_contract` and nothing else (`namespace_isolation_ok: true`,
`different_training_corruption_seed_unexpected_changes: {}`).

This `bit_exact` verdict was measured on CPU only (the repository's fixture
platform); no GPU determinism claim is made (non-goal, per the issue).

The `determinism_report` embedded in the sibling JSON was regenerated from
committed commit `4411710e0261a9c779b073ec8853b2b3c12118f8` with a clean
working tree throughout (`code_dirty=false` before and after the
`--mode determinism` run, verified via `git status --porcelain` -- not a
separate `git clone`, but the exact committed revision with zero
uncommitted changes at generation time). See
`docs/design/iter-slm138-recursive-denoiser-20260721.json`'s
`rng_contract`/`evidence_gate`/`post_update_verification` fields for a single
concrete fixture run (also regenerated clean, `code_dirty=false`), and the
sibling `iter-rsc-a03-rng-determinism-contract-20260721.json`'s
`determinism_report` key for the full comparison this section summarizes.

## Provenance persisted per run

`config_hash`, `recursive_config_hash`, `data_record_hash`, `tokenizer_hash`
(all stable SHA-256 over the canonicalized dataclass/dict), `code_commit`,
`code_dirty`, `diff_hash` (when dirty), and the full `rng_namespace_seeds`
map, alongside the existing `version_stamp` envelope.

## Tests

`tests/test_models/test_recursive_denoiser.py` -- 17 new tests (55 total, all
passing), covering: namespace-seed derivation/fail-closed on unknown
namespace, `isolated_draw`'s outer-stream-untouched guarantee, two-run byte-
identical JSON, probe-order permutation invariance, extra-probe permutation
invariance, restored-checkpoint repeated evaluation (plus the torch-only-
restore-is-insufficient counter-example that documents the second RNG
source), differing-corruption-seed field isolation, fixture exit-RNG-state
independence from the caller's entry state, the clean-tree gate (pure
function + an end-to-end forced-dirty-stamp case via monkeypatch),
version-stamp-registry match, checkpoint round-trip state-dict digest
identity, and the full `FixtureDeterminismReportV1` verdict.

```
python -m pytest tests/test_models/test_recursive_denoiser.py -q
# 55 passed
```

## Version bumps

- `model.recursive_denoiser`: v4 -> v5 (owns the fixture script, the new
  `rng_contract.py`, the test file, and the dated design docs -- all
  touched).

## What is done vs explicitly deferred

Done: RNG namespace contract + `RngCheckpoint`, the six-phase fixture
refactor, the clean-tree evidence gate, `FixtureDeterminismReportV1` from
real repeated executions and two real call-order/insertion permutations, 17
new deterministic tests, a checked-in clean-tree fixture artifact, and this
doc.

Deferred (explicitly, not silently): a from-scratch-clean-checkout
regeneration in an isolated worktree (this sandbox commits to the same
checked-out branch, so "clean" here means "committed, then regenerated from
that exact commit with no further working-tree changes" -- verified via
`code_dirty=false`/`code_commit` in the regenerated artifact, not via a
separate `git clone`). A GPU determinism claim (non-goal). A broader sweep
of *many* call-order permutations beyond the two exercised here (reordered
probes, inserted extra probe) -- the two exercised are the concrete
permutations Kimi's report and the issue text call out; additional
permutations (e.g. probing with different tensor shapes, additional extra
probes at different program points) would strengthen the evidence further
but were not required to close the acceptance criteria measured here.
`training_batch_order` is declared in the RNG contract but not exercised --
this fixture uses a single fixed batch with no shuffling, so there is
nothing to test yet; a future batched-shuffling fixture should exercise it
before relying on the namespace.
