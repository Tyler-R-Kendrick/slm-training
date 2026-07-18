# Test-data phase

Held-out suites with structure-disjoint enforcement. Owner:
`src/slm_training/harnesses/test_data/`.

## Prerequisites

- The train manifest of the corpus you must stay disjoint from
  (`outputs/data/train/<version>/manifest.json`).

## Commands

```bash
# Test suites with strict leakage checks against the train manifest
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json

# Expand rico_held with additional HF RICO screens
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json \
  --rico-hf-split test --rico-limit 2600 --target-records 1500
```

## Key flags

`--suites`, `--target-records`, `--rico-hf-split`, `--seed-path`;
`--allow-without-train-manifest` is for isolated experiments only — never for
suites used in claims.

## Outputs

`outputs/data/eval/<version>/` (suites: smoke, held_out, rico_held, adversarial,
ood) with leakage fingerprints.

## Gates & invariants

- Disjointness against the train manifest is mandatory; never fit training data
  to the holdout.
- Keep suite sizes (`n`) explicit in every claim that cites these suites.

## Close out

- Iron law applies when suites feed a documented run
  (`documenting-experiment-results`).
- Checks: `pytest -q tests/test_harnesses/test_data
  tests/test_integration/test_ship_disjoint.py`.
- Changing the harness itself → `improve-openui-harnesses`.
