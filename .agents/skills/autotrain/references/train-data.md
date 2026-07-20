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
mapping). Aux synthesis helpers stay direct: `python -m scripts.generate_progspecs`,
`python -m scripts.synthesize_pack`, `python -m scripts.build_solver_supervision`,
`python -m scripts.verify_data_synthesis`.

## Key flags

`--profile strict|permissive` (strict default: fuzzy + semantic dedup, tier
floor, n-gram decontamination, exposure caps; explicit flags override),
`--rico-limit`, `--programspec-count`, `--min-quality-score`,
`--dedup-against <ids>` (exclude pairs already in committed corpora),
`--difficulty-from <run>/record_nll.jsonl` (Superfiltering curation weight;
produce it with `slm sft train … --emit-record-nll`),
`--mixture-manifest`, `--publish` / `--publish-root`, `--frontier-artifacts`.

## Outputs

`outputs/data/train/<version>/` with `manifest.json` + structural fingerprints
**plus the quality loop artifacts**: `quality_report.json` (fitness / garbage /
redundancy / decontamination / warnings), `rejected.jsonl` (every drop with
stage + reason), `synthesis_feedback.json` (per-family and per-synthesizer
yields, recommendations, experiment candidates). Builds register a lineage
DataSnapshot; sibling-typed roots hold preference/trajectory/ProgramSpec/
mixture data. Rejected-ledger preference negatives:
`python -m scripts.mine_rejected_preferences --dataset <version>`.

## Gates & invariants

- Source/license governance and structural hashes preserved.
- Committed tiny fixtures live in `src/slm_training/resources/`, never a new root.
- The train manifest is required downstream for test-data disjointness.
- Nothing is dropped silently; gates are never relaxed to raise yield.

## Close out

- **REQUIRED after every build**: read `quality_report.json`, `rejected.jsonl`,
  **and** `synthesis_feedback.json`, and act on them per the `synthesis-feedback`
  skill — fix the synthesis harness / producers and file the emitted experiment
  candidates; never weaken the quality gates to raise yield. Cross-snapshot
  overlap: `python -m scripts.audit_data_corpora`.
- Shared duties: [contracts.md](contracts.md).
- Docs: `docs/design/data-synthesis.md`. Checks:
  `pytest -q tests/test_harnesses/train_data tests/test_data`.
- Changing the harness itself → `improve-openui-harnesses`.
