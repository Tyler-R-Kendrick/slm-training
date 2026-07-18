# VSS3-01 (SLM-69): solver supervision corpus — fixture build note

**Date:** 2026-07-18
**Scope:** wiring evidence for the replay-verified solver supervision corpus
builder. **No model or head was trained; no checkpoint, eval, benchmark, or
matrix ran. These fixture rows establish wiring only and do not establish model
quality.**

## What was built

- [`src/slm_training/harnesses/distill/solver_supervision.py`](../../src/slm_training/harnesses/distill/solver_supervision.py)
  — `SolverSupervisionBuilder` + `build_solver_supervision` + `write_corpus`.
  Consumes VSS1-04 solver traces (recorded event stream, no re-solving), replays
  each trace before emitting rows, and produces two discriminated row kinds:
  `support_set` (state × hole domain partitioned into supported / unsupported /
  unknown with certificate + witness digests) and `candidate_cost` (per live
  decision candidate, with observed-or-censored search cost-to-go).
- [`scripts/build_solver_supervision.py`](../../scripts/build_solver_supervision.py)
  — thin CLI (`--trace-root … --output … --verify-replay --manifest`, plus
  `--describe` for a write-nothing dry run).
- [`tests/test_harnesses/distill/test_solver_supervision.py`](../../tests/test_harnesses/distill/test_solver_supervision.py)
  — 14 tests covering the label rules and honesty invariants.

## Fixture build (two synthetic traces: one closure, one search)

Inputs: one exact-closure trace (hole domain `{v1,v2,v3}`: `v1` supported, `v2`
certified-unsupported, `v3` unknown) and one search trace (`{v1,v2}` both
supported, decision chooses `v1`, terminal `solved`). Both are replay-clean with
digest-consistent certificates.

| Metric | Value |
| --- | --- |
| Source traces | 2 |
| `support_set` rows | 2 |
| `candidate_cost` rows | 2 |
| Rejected traces / rows | 0 / 0 |
| Verdicts (supported / unsupported / unknown) | 3 / 1 / 1 |
| Cost rows (observed / censored) | 2 / 0 |

Deterministic artifact content hashes (SHA-256, first 16 hex):

| Artifact | Rows | `content_sha256[:16]` |
| --- | --- | --- |
| `rows/support_set.train.jsonl` | 2 | `aec85da6a08cac07` |
| `rows/candidate_cost.train.jsonl` | 2 | `0bf97f7cf9500278` |
| `manifest.json` | — | `e4990216373b641b` |

Re-running the build over the same traces reproduces identical row ordering and
content hashes (pinned by `test_deterministic_hashes_and_ordering`). The corpus
was built to a scratch directory and is **not** committed or published (default
local / no push).

## Invariants proven by the fixture tests

- Replay-before-emit: a tampered `UNSUPPORTED` certificate (digest ≠ id) rejects
  the whole trace — no hard labels leak.
- `UNKNOWN` never enters the supported/unsupported sets.
- All supported alternatives survive even when a decision chose one of them.
- A local `nogood` is a hard-negative feature, never a global `UNSUPPORTED`
  relabel.
- Cost is `cost_observed=False` (censored) for budget-stopped / truncated /
  nonterminal suffixes, and observed only for a replayable suffix reaching a
  definitive terminal.
- Split lineage is inherited from the root `ProgramSpec`; a state fingerprint
  spanning train and a held-out split is rejected as a leak.
- The trace's final source text never appears in any row.
- Historical non-solver (decode-only) traces are skipped with an explicit reason,
  not crashed.

## Honesty / non-goals

Meaningful-parse and downstream model quality are **not** measured here — there is
no model in this issue. The corpus is a training-target artifact whose hard labels
are trace- and certificate-replayable; using it to train an energy/cost head is
the next issue (VSS3-02 / SLM-70). No ship gate is touched or weakened.

## Verification

```
python -m pytest tests/test_harnesses/distill/test_solver_supervision.py \
    tests/test_harnesses/distill tests/test_data_store.py -q   # 26 passed
python -m ruff check src/slm_training/harnesses/distill/solver_supervision.py \
    scripts/build_solver_supervision.py \
    tests/test_harnesses/distill/test_solver_supervision.py     # clean
python -m scripts.repo_policy                                    # ok
git diff --check                                                 # clean
```

Pre-existing, unrelated: `tests/test_scripts/test_build_train_data_cli.py` and
`tests/test_scripts/test_task_eval_cli.py` fail in this container because the
AgentV Node SDK is not installed (`AgentV SDK is unavailable; run npm ci …`);
these do not reference `solver_supervision` and fail identically with this change
stashed.
