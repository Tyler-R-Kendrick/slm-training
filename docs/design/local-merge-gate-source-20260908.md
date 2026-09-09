# Local merge-gate source selection — 2026-09-08

A15 / SLM-594 focused prerequisite; not A16 / SLM-595 release completion.

## Change

`python -m scripts.verify_merge_ready --fast --json --source /path/to/candidate`
now checks the explicit candidate directory. Omit `--source` to retain the
repository-root default. Relative paths resolve against the caller's current
directory; paths containing spaces are passed as argv, not shell commands.
The same selection applies without `--fast`. Fast results remain static feedback,
not proof that changed tests ran. No gate, evaluator, threshold, marker, authority,
or hosted workflow was weakened or enabled.

The integration workspace already introduced `--source` but omitted it in the
fast runner call. Remote base `a809e81878302baa27a9d6fedd8f0b79ee3af857` has no
such option. This minimal independent patch supplies the parser option and
existing runner's root argument, without importing the integration workspace's
unfinished controller changes. Component `ci.local_merge_gate` advances v10→v11
on this base; integration must reconcile its independently newer version history.

## Executed evidence

- Integration reproducer: four real-process cases failed before the root fix.
  After the fix, the whole integration gate test file passed 18 tests in 1.15s.
- Exact exported publication source and its gate test file: 20 passed in 1.29s,
  including eight real-process source-selection cases (fast/full ×
  relative/absolute × exit zero/nonzero). Existing default-root tests remain.
  Ruff passed for both files.
- Publication test loading used the exact exported module bytes with the current
  workspace providing existing `scripts.repo_policy` and repository fixtures.
  It is focused local validation, not a full clean-checkout release receipt.
- Invocation: `timeout -s INT -k 10 170` with Python 3.12 and
  `pytest -q --tb=short --show-capture=no -o addopts= -m '' -p no:cacheprovider`.
  A normal checkout rerun is
  `python -m pytest -q -o addopts= -m '' tests/test_scripts/test_merge_ready_gate.py`
  inside that bounded invocation.
- Local XML receipts: `outputs/runs/autonomy-validation/a15-fast-source-first.xml`,
  `a15-fast-source-fixed.xml`, and `a15-publication-export.xml`.
  These local receipts are not included as remote release certificates.

The broader workspace independently passed 25 controller tests. Its combined
scheduling/CLI run had 28 passes and one failure: the registered
`slm experiments learning-comparison` command is absent from autotrain skill
documentation. Those broader changes are excluded here. A frozen-tree release
gate, authenticated independent verification, and integration closeout remain
open. No training, agent job, service activation, or paid CI was run.
