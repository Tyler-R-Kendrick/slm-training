# A14 source-coverage mutation evidence — 2026-09-08

The new `tests/test_autoresearch/test_autonomy_coverage_mutation.py` exercises the
actual loaded `scripts.check_changed.select_tests` source, mutating only a
disposable function copy. A historical changed-test-only shortcut must fail the
specific `source_owned_nodes_lost` oracle. Unrelated exceptions are not kills.
Baseline and restored selectors pass; a branch witness proves mutant execution.

Local validation: **2 passed, exit 0, 3.75 seconds**; Ruff passed. Actual bounded
pytest collection returned 86 nodes, including 59 source-owned gate nodes. These
are collected identities, not a claim that all 86 tests executed. Collection
rejects timeout, failure, emptiness and truncated output. The sibling JSON binds
actual source/function hashes, node identities, command and outcome.

```sh
PYTHONPATH=src timeout -s INT -k 10 170 python -m pytest -q \
  -o addopts= -o junit_family=xunit1 -m '' -p no:cacheprovider \
  tests/test_autoresearch/test_autonomy_coverage_mutation.py \
  --junitxml=outputs/runs/a14-coverage.xml
```

Validation used the deliberate dirty autonomy candidate, HEAD
`e0eca9f9910244ecc20eb480d363f1852417b599`; publication is isolated on remote base
`a809e81878302baa27a9d6fedd8f0b79ee3af857`. The remote selector interface and bounded
runner were inspected, but this is **not an exact published-tree release receipt**.
The initial test run exposed pytest assertion-rewrite text defeating an exact
error match; the oracle now raises its explicit semantic exception without
loosening the accepted message. No production policy, threshold, evaluator,
checkpoint or service changed. No component bump: test/evidence-only patch.

This addresses SLM-592/A14's changed-test coverage mutation only. It does not fix
the intentionally narrow fast hook, certify all mutation guards, establish OS
isolation, or complete reliable-autonomy release verification. Broader local
mutation-runner hardening is deliberately excluded from this independent patch.
