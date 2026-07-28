# Fix: compiler-tree decode no longer swallows the deadline as a fake grammar dead-end

**Honesty:** `fixture_or_scratch`, harness-internal correctness fix with unit-test
coverage. **Not ship** — no train/eval run, no checkpoint, no `--ship-gates`
scoreboard. `is_fix: true`.

## What

Applies items 1 and 2 of the `proposed_fix_sketch` in
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
(itself a follow-up to
[`lever-hard-decode-timeout-wall-measured-results.md`](lever-hard-decode-timeout-wall-measured-results.md),
PR #1167). Four independent sessions (PR #1171's original finding, the
[lexer-cache hero rerun](decode-compiler-tree-lexer-cache-hero-rerun-20260728.md)
in PR #1178, and the scheduled-session rerun in PR #1182) all converged on the
same root cause without anyone applying the fix: `compiler_draft.py`'s
`build_completion_forest` caught a cooperative decode-deadline `TimeoutError`
under a bare `except Exception:` and recorded it as `constrained_dead_ends`
with reason `empty_completion_forest` — indistinguishable from a genuine
grammar dead-end — instead of letting it propagate as a timeout. Separately,
the compiler-tree per-position loop in `_compiler_ltr_decode_one` had zero
`check_decode_deadline()` calls, unlike the LTR-repair and MaskGIT loops
(`model.twotower` v261, PR #1167).

## Change

1. `src/slm_training/dsl/grammar/fastpath/compiler_draft.py`:
   `build_completion_forest` now has `except TimeoutError: raise` before its
   pre-existing bare `except Exception: return CompletionForest((), "none")`
   around the `GrammarCapabilityAdapterV1.completion_domain(request)` call.
2. `src/slm_training/models/twotower.py`: `_compiler_ltr_decode_one`'s
   per-position `while` loop now calls `check_decode_deadline()` at the top of
   every iteration, mirroring the LTR-repair (line ~4906) and MaskGIT (line
   ~13266) loops' existing coverage. `_compiler_ltr_decode_batch` calls
   `_compiler_ltr_decode_one` once per row, so both call sites are covered by
   this one loop edit — no separate change needed there.

Both edits change only *when* an already-armed deadline raises. Neither
changes what the grammar considers a legal continuation (I6) or how a
candidate is selected (I3): a `TimeoutError` only fires when a caller
explicitly armed `set_decode_deadline`; when unarmed, `check_decode_deadline()`
remains a no-op and non-timeout exceptions from `completion_domain` still fall
through to the pre-existing fail-closed empty-forest path (regression-tested).

## Tests

New tests in `tests/test_models/test_compiler_tree_deadline.py` (a new,
dedicated file — kept separate from the existing, much larger
`tests/test_models/test_compiler_decode.py` because that file currently
carries 36 pre-existing, unrelated failures, and the local pre-commit hook
(`.githooks/pre-commit` → `scripts/check_changed.py --changed-tests-only`)
runs a changed test file's *entire* contents with no pre-existing-failure
baseline — touching that file at all would block every future commit
against it):

- `test_build_completion_forest_propagates_deadline_timeout` — monkeypatches
  `GrammarCapabilityAdapterV1.completion_domain` to raise `TimeoutError` and
  asserts `build_completion_forest(..., remaining_tokens=8)` re-raises it
  instead of returning an empty `CompletionForest`.
- `test_build_completion_forest_still_fails_closed_on_other_errors` — same
  monkeypatch shape but raises `ValueError`, asserting the pre-existing
  fail-closed contract (`coverage == "none"`, `paths == ()`) is unchanged for
  non-timeout exceptions.
- `test_compiler_ltr_decode_one_checks_deadline_each_step` — monkeypatches
  `decode_stats.check_decode_deadline` with a counting fake that only raises
  on its *second* call, then calls `model._compiler_ltr_decode_one(...)`
  directly (same call pattern as the existing
  `test_lattice_search_rejects_unknown_mode`) and asserts `TimeoutError`
  propagates with at least 2 calls observed — proving the loop checks the
  deadline every iteration, not only once at loop entry (an earlier revision
  of this test expired a real deadline before the first call, which would
  have passed even with a single entry-time check; addressed per CodeRabbit
  review on PR #1183).

## Checks

- New tests: `3 passed in 1.96s`. `git stash` of only the two production
  files confirms 2 of the 3 are non-vacuous (they fail pre-fix with `DID NOT
  RAISE TimeoutError`); the third (`..._still_fails_closed_on_other_errors`)
  passes both before and after by design, since it asserts the invariant the
  fix must preserve rather than the fix itself.
- `tests/test_harnesses/model_build/test_decode_deadline.py
  tests/test_harnesses/model_build/test_twotower.py -k "compiler or
  deadline"` — 4 passed, 61 deselected.
- `tests/test_models/test_compiler_decode.py -k completion_forest` — 15
  passed, 12 failed; all 12 failures reconfirmed pre-existing via `git
  stash` (`build_completion_forest() got an unexpected keyword argument
  'enforce_schema_component_types'`, the same signature-drift issue PR
  #1177/#1178 already documented — unrelated to this diff).
- `tests/test_models/test_compiler_decode.py` (full file, unmodified this
  session) — `36 failed, 187 passed`; identical 36 pre-existing failures
  reconfirmed via `git stash` (184 passed pre-fix on this branch's HEAD).
  Unrelated to this diff — recorded here only to explain why the new tests
  live in their own file instead.
- `ruff check` (changed files) — clean.
- `python -m scripts.verify_version_stamps --check` — 6 changed files, 1
  component touched: `model.twotower` v261 → v262 (its `versions.json` entry
  already lists both `twotower.py` and `compiler_draft.py`).
- `python -m scripts.repo_policy` — ok.
- `python -m scripts.verify_decode_invariants` — clean.
- `git diff --check` — clean.
- See the matching `.json` for the full static-check and regression-sweep
  record.

## Scope / next steps

Not attempted this session (`proposed_fix_sketch` item 3): re-running the
isolated per-record `exposure12` protocol (PR #1178 / #1182) against this
fix, to see whether `decode_outcome` correctly reclassifies as
`runtime_timeout` or shifts to an on-time completion. That needs a fresh
checkpoint rebuild plus the AgentV/Node DSL bridge (`npm ci`) and three
`MAX_RUN_MINUTES`-bounded eval runs — left for a follow-up scheduled
iteration, matching how PR #1173's lexer-cache fix and PR #1178's rerun were
split across sessions. Also still open: why each `build_completion_forest`
call costs ~7-10s of wall time on average even after the lexer-cache fix
(Node-bridge round-trip vs. candidate-set size vs. cache misses unisolated).
