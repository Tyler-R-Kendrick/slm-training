# Train-data phase

Versioned, immutable, quality-gated training corpora. Owner:
`src/slm_training/harnesses/train_data/`.

## Prerequisites

- None for fixture builds; RICO sources may download via HF (cache respected).
- Never mutate a published version — bump the version instead.

## Commands

```bash
# High-quality versioned corpus (default: all sources + quality synthesizer)
slm data build-train --source all --version v1 --synthesizer quality

# Fast fixture-only rebuild (CI/scratch)
slm data build-train --source fixture --version v0 --synthesizer quality

# Immutable Git publish of a selected snapshot
slm data publish-train --version v1

# Resolve/verify canonical data roots instead of memorizing paths
slm data store list
slm data store resolve train v1
slm data store verify train v1
```

Every `slm` command equals `python -m scripts.<module> …` (`slm list` shows the
mapping). Aux synthesis helpers stay direct:
`python -m scripts.generate_progspecs`, `scripts.synthesize_pack`,
`scripts.build_solver_supervision`, `scripts.verify_data_synthesis`.

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

- Shared duties: [contracts.md](contracts.md).
- Docs: `docs/design/data-synthesis.md`. Checks:
  `pytest -q tests/test_harnesses/train_data tests/test_data`.
- Changing the harness itself → `improve-openui-harnesses`.
