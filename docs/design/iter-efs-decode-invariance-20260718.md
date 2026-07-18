# EFS0-02 — Decode-invariance factorial audit (harness; audit deferred)

Machine-readable results: [iter-efs-decode-invariance-20260718.json](iter-efs-decode-invariance-20260718.json).

## What this is (and is not)

This iteration delivers the **harness** for the ten-checkpoint decode-invariance
factorial audit (Linear SLM-104 / EFS0-02) and validates it with unit tests and
fixture wiring. It is **infrastructure + wiring evidence, not a ship claim and
not a decoder-sensitivity finding**. The actual ten-checkpoint factorial run and
its verdict are **deferred** — see the blocker below.

## Hypothesis and falsifier

* **Hypothesis:** at least one durable frontier/diagnostic checkpoint changes
  parse, meaningful-program, placeholder/binding, or structural outcomes under a
  corrected/current decoder on **byte-identical weights**, invalidating a prior
  training/architecture attribution (the E288 lesson).
* **Falsifier:** all compatible decode paths agree within the preregistered
  equivalence bands (`abs metric delta <= 0.01`, `paired disagreement rate <=
  0.01`) on every checkpoint and suite.

## Delivered harness

* **`decode_path.py`** — typed `DecodePathSpec` registry for the three required
  paths (`checkpoint_declared` historical control, `current_native`,
  `current_exact_or_compiler`) as declarative lever-bundles + compatibility
  predicates. `current_exact_or_compiler` **preserves each checkpoint's target
  representation** (choice codec → exact `ChoiceDecodeState` pushdown; surface /
  lexer → compiler-tree greedy) and never coerces a surface checkpoint into a
  choice codec. Deterministic per-path fingerprints.
* **`checkpoint_path_manifest.py`** — builds + validates the versioned
  checkpoint × decode-path compatibility manifest keyed on the SLM-103
  `CheckpointReferenceV1` (verified hash). Every audited cell must be eval-only
  over a verified checkpoint hash; `frontier`/`ship_candidate` cells must resolve
  durably. Reports whether ≥6 complete three-path blocks exist.
* **`decode_invariance.py`** — paired disagreement classifier (surface-only /
  syntax-placeholder / semantic-binding / empty-vs-populated / timeout-fallback /
  exact-choice-derivation) + preregistered equivalence bands + an honest
  `decoder_sensitive` verdict.
* **`scripts/build_decode_invariance_manifest.py`** — `--list` describes the
  paths without any checkpoint (the EFS "describe/list before run"); the build
  path emits + validates the manifest, honestly *deferred* when no durable
  checkpoints are supplied.

## Verification

`.venv` (`.[dev]`, CPU torch):

* 23 unit tests pass — registry attributes/compatibility/determinism, manifest
  block-counting + fail-closed validation, disagreement classification, the CLI.
* **E288 decoder-defect regression** (the key invariant): fed the known scenario
  — byte-identical weights, `checkpoint_declared` parse 0 (19/19 empty) vs
  `current_exact_or_compiler` parse 1.0 (valid-but-trivial) — the harness
  classifies all 19 as `empty_vs_populated` and returns `decoder_sensitive =
  True`. So the audit *would* catch a decoder-path-induced change.
* `ruff` clean; `build_decode_invariance_manifest --list` and the deferred
  manifest build succeed.

## Audit status: DEFERRED

**Blocker:** there are **zero durable checkpoints** resolvable in this
environment. The SLM-103 backfill records **227 of 231** model-card checkpoint
rows as `unresolved_local` (gitignored `outputs/…`, absent from a clone); the
only tracked checkpoint is the `playground_demo` fixture (compositional codec,
which exercises only the native greedy-LTR path). The ten-checkpoint factorial
run, its paired generation bundle, and the decoder-sensitivity verdict therefore
require a **GPU host with synced E224+ frontier/diagnostic checkpoints** (via the
SLM-103 `sync_checkpoints --claim-class frontier` path). Fixture-only cells are
diagnostic and cannot decide invariance.

## Limitations and next steps

* No model was run; no decoder-sensitivity conclusion is drawn; no historical
  result document is corrected yet (there is nothing to correct from a fixture).
* Matrix-row execution: concrete eval-only rows are generated **per checkpoint**
  from the compatibility manifest at audit time (each cell = one frozen
  checkpoint × path; `current_exact_or_compiler` levers are codec-adaptive), so
  they are produced on the credentialed host alongside the run rather than
  statically registered here.
* On a GPU host: sync ≥6 frontier/diagnostic checkpoints, build the manifest
  (`build_decode_invariance_manifest --reference-dir …`), generate the eval-only
  factorial via the frozen-checkpoint `run_quality_matrix` path, feed paired
  outcomes to `pair_disagreement_summary`, and publish the verdict + any linked
  corrections.

## Decision

Harness delivered and validated as wiring evidence. Audit deferred to a GPU host
with durable checkpoints. **Meaningful-parse remains the primary metric; no ship
gate is weakened and no checkpoint is promoted.**
