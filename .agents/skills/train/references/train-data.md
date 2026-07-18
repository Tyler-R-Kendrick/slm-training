# Train-data phase

Versioned, immutable, quality-gated training corpora. Owner:
`src/slm_training/harnesses/train_data/`.

## Prerequisites

- None for fixture builds; RICO sources may download via HF (cache respected).
- Never mutate a published version — bump the version instead.

## Commands

```bash
# High-quality versioned corpus (default: all sources + quality synthesizer)
python -m scripts.build_train_data --source all --version v1 --synthesizer quality

# Fast fixture-only rebuild (CI/scratch)
python -m scripts.build_train_data --source fixture --version v0 --synthesizer quality

# Immutable Git publish of a selected snapshot
python -m scripts.publish_train_data --version v1

# Resolve/verify canonical data roots instead of memorizing paths
python -m scripts.data_store list
python -m scripts.data_store resolve train v1
python -m scripts.data_store verify train v1
```

Aux synthesis helpers (same phase): `python -m scripts.generate_progspecs`,
`scripts.synthesize_pack`, `scripts.build_solver_supervision`,
`scripts.verify_data_synthesis`.

## Key flags

`--rico-limit`, `--programspec-count`, `--min-quality-score`,
`--mixture-manifest`, `--publish` / `--publish-root`, `--frontier-artifacts`.

## Outputs

`outputs/data/train/<version>/` with `manifest.json` + structural fingerprints;
sibling typed roots hold preference/trajectory/ProgramSpec/mixture data.

## Gates & invariants

- Source/license governance and structural hashes preserved.
- Committed tiny fixtures live in `src/slm_training/resources/`, never a new root.
- The train manifest is required downstream for test-data disjointness.

## Close out

- Data builds feeding a documented run follow the iron law: update matching
  `docs/design/` JSON + markdown (`documenting-experiment-results`).
- Docs: `docs/design/data-synthesis.md`. Checks:
  `pytest -q tests/test_harnesses/train_data tests/test_data`.
- Changing the harness itself → `improve-openui-harnesses`.
