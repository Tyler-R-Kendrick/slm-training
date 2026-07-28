# Fix: compiler-tree decode no longer swallows a deadline as a fake dead-end

**Honesty:** `fixture_or_scratch`. This is a real code change (`is_fix: true`)
with regression-test coverage, but **not a ship claim** — no
`train_model`/`evaluate_model --ship-gates` run, no checkpoint, no
scoreboard. Same status tier as the finding it closes:
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
(PR #1171), and a sibling to
[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173), which fixed the other half of that finding's root-cause chain
(the lexer-rebuild cost) without touching the deadline-swallow bug itself.

## Task

Apply the finding's "Proposed fix sketch" items 1 and 2:

1. Add `except TimeoutError: raise` before the bare `except Exception:` in
   `build_completion_forest` (`compiler_draft.py:2313`) so a deadline
   `TimeoutError` is never masqueraded as an empty completion forest.
2. Add `check_decode_deadline()` calls inside the per-position loop in
   `_compiler_ltr_decode_one` (`twotower.py`, loop starting ~line 10329) to
   match the LTR-repair/MaskGIT loops' cooperative coverage (both got this
   check in PR #1167, `model.twotower` v261).

Item 3 (re-run the isolated per-record eval protocol after the fix) is
explicitly **not** done this session — see Next steps.

## What changed

`src/slm_training/dsl/grammar/fastpath/compiler_draft.py` —
`build_completion_forest`'s `try`/`except` around
`GrammarCapabilityAdapterV1(get_pack()).completion_domain(request)` gained:

```python
except TimeoutError:
    # A cooperative decode-deadline TimeoutError (see
    # decode_stats.check_decode_deadline) must propagate as a timeout,
    # not be masqueraded as "the grammar has no legal continuation
    # here". See decode-compiler-tree-deadline-swallow-finding.md.
    raise
except Exception:  # noqa: BLE001 - constrained callers fail closed below
    return CompletionForest((), "none")
```

This is the exact idiom already used elsewhere in the codebase for the same
reason — `TwoTowerModel._canonical_valid_openui` and the LTR
`pick_constrained_token` wrapper (`twotower.py:4671`, `:5003`) both already
have `except TimeoutError: raise` ahead of their own bare `except Exception`.
The compiler-tree completion-forest builder was the one caller of a
timeout-reachable code path (the Node DSL bridge parse/validate call inside
`completion_domain`) that didn't yet have it.

`src/slm_training/models/twotower.py` — `_compiler_ltr_decode_one`'s
`while len(prefix) < length and prefix[-1] != self.tokenizer.eos_id:` loop
gained a `check_decode_deadline()` call at its head, placed and commented
identically to the existing calls in the LTR-repair loop (`~line 4906`) and
the MaskGIT loop (`~line 13266`):

```python
while len(prefix) < length and prefix[-1] != self.tokenizer.eos_id:
    # Hard wall: cooperative deadline set by eval_runner (or callers).
    # Without this check, build_completion_forest's bare ``except
    # Exception`` (compiler_draft.py) can swallow a deadline
    # TimeoutError raised inside the Node DSL bridge and masquerade
    # it as an empty completion forest instead of propagating the
    # timeout — see decode-compiler-tree-deadline-swallow-finding.md.
    from slm_training.models.decode_stats import check_decode_deadline

    check_decode_deadline()
    state.remaining_tokens = length - len(prefix)
    ...
```

`_compiler_ltr_decode_batch` calls `_compiler_ltr_decode_one` once per row
inside a plain `for` loop with no wrapping `try`/`except`, so this single
check point covers both the batched and single-row compiler-tree decode
entry points — no separate change was needed there.

### Why this is safe

Both changes are additive fail-fast checks layered on the existing
cooperative-deadline mechanism (`decode_stats.set_decode_deadline` /
`check_decode_deadline`). Neither widens the legal completion set (I6) nor
changes ranking: when the deadline has not elapsed, `check_decode_deadline()`
is a no-op, and the new `except TimeoutError: raise` only changes *which
exception type* propagates out of `build_completion_forest` on a timeout —
every non-timeout exception still falls through to the pre-existing
`except Exception: return CompletionForest((), "none")` fail-closed path
unchanged (proven by the new `test_build_completion_forest_still_fails_closed_on_other_errors`
regression test below).

## Reproduction / test environment

No committed venv existed in this sandbox; built one fresh (gitignored, not
committed):

```bash
python3.12 -m venv .venv-autotrain
.venv-autotrain/bin/pip install --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.6"
.venv-autotrain/bin/pip install --no-deps -e .
.venv-autotrain/bin/pip install \
  "pytest>=8,<9" "pytest-asyncio>=0.23,<2" "ruff>=0.9,<0.16" \
  "numpy>=1.26,<3" "httpx>=0.27,<1" "fastapi>=0.110,<1" \
  "lark>=1.1,<2" "openfeature-sdk>=0.10,<1" "pydantic>=2.7,<3" "PyYAML>=6,<7"
cd src/apps/openui_bridge && NODE_OPTIONS= npm ci
```

`NODE_OPTIONS` is cleared only because this sandbox's ambient
`NODE_OPTIONS="--import tsx" --max-old-space-size=8192` is malformed for a
plain `node` invocation — the same sandbox artifact noted in the lexer-cache
fix doc, unrelated to this change. Before `npm ci`, running
`pytest -x tests/test_models/test_compiler_decode.py` failed immediately on
the *first* torch-training test with `RuntimeError: Install bridge deps: cd
.../openui_bridge && npm ci` — a pre-existing environment-setup gap, not a
regression from this fix (confirmed by re-running the same node IDs green
after `npm ci`).

## Unit tests (added, committed)

`tests/test_dsl/test_grammar_fastpath_deadline.py` (3 tests, new file):

- `test_build_completion_forest_propagates_deadline_timeout` — monkeypatches
  `GrammarCapabilityAdapterV1.completion_domain` to raise `TimeoutError`;
  asserts `build_completion_forest` re-raises it instead of returning an
  empty `CompletionForest`.
- `test_build_completion_forest_still_fails_closed_on_other_errors` — same
  monkeypatch shape but raises `ValueError`; asserts the pre-existing
  fail-closed behavior (`coverage="none"`, `paths=()`) is unchanged — proves
  the fix is scoped to `TimeoutError` specifically.
- `test_compiler_ltr_decode_one_checks_decode_deadline` — builds a real
  `TwoTowerModel` (`compiler_decode_mode="tree"`), arms a 0.01s cooperative
  deadline via `set_decode_deadline` and sleeps past it, then asserts
  `_compiler_ltr_decode_one` raises `TimeoutError` instead of running to
  completion.

```
NODE_OPTIONS= .venv-autotrain/bin/python -m pytest -q \
  tests/test_dsl/test_grammar_fastpath_deadline.py -v
# 3 passed in 3.01s
```

## Regression sweep

| suite | result |
| --- | --- |
| `tests/test_models/test_compiler_decode.py` (targeted node IDs: `test_canonical_valid_openui_propagates_decode_deadline`, `test_compiler_singleton_bypass_requires_complete_coverage`, `test_compiler_tree_batches_only_ambiguous_prefills_with_bound`, `test_compiler_empty_forest_records_bounded_dead_end_trace`) | 5 passed in 3.08s |
| `tests/test_harnesses/model_build/test_decode_deadline.py tests/test_dsl/test_grammar_fastpath_lexer_cache.py` | 10 passed in 9.63s |
| `tests/test_dsl/test_grammar_fastpath_deadline.py -v` | 3 passed in 3.01s |

The full unfiltered `test_compiler_decode.py` (200+ tests, each doing a
torch forward pass and several through the Node DSL bridge) exceeded the
3-minute run cap twice in this sandbox and is not cited as evidence here —
ran the compiler-tree/deadline-relevant node IDs directly instead, which is
sufficient coverage for a two-call-site additive change.

`ruff check` on both changed source files plus the new test file: clean.
`python -m scripts.verify_version_stamps --check`: `ok (vs HEAD; 4 changed
file(s), 1 component(s) touched)` — `model.twotower` bumped v261 → v262 (its
registered paths cover both `twotower.py` and `compiler_draft.py`; no other
watched component's paths include these files).
`python -m scripts.verify_decode_invariants`: exit 0, no change to
`agent_surfaces`/`canonical_defaults`/`strict_policies`/`weakening_levers`.
`python -m scripts.repo_policy`: `ok (tracked + untracked)`.

## Interpretation

Closes proposed-fix-sketch items 1 and 2 from the deadline-swallow finding.
A deadline `TimeoutError` raised while `build_completion_forest` is waiting
on the Node DSL bridge now propagates as a classified timeout — caught by
`eval_runner`'s own `except TimeoutError` at the harness boundary — instead
of being recorded as `constrained_dead_ends += 1` /
`empty_completion_forest`. The compiler-tree per-position loop now fails
closed on an elapsed cooperative deadline exactly like its LTR-repair and
MaskGIT siblings, closing the last decode-loop gap PR #1167 left open.

## Scope note (what this does NOT establish)

- No train/eval run, no checkpoint, no `--ship-gates` scoreboard — this is a
  harness-internal correctness fix with unit-test coverage, not a
  readiness or latency-improvement claim.
- **Not re-measured**: whether the `exposure12` quality-champion hero's
  `decode_outcome` actually reclassifies from `fallback_output` to
  `runtime_timeout` (or a real on-time completion) under this fix. That is
  the finding's own proposed-fix-sketch item 3 and is the most direct
  falsification test of this fix's practical effect — explicitly deferred to
  next steps, not done this session.
- The full `test_compiler_decode.py` file was not run end-to-end inside the
  run cap; targeted node IDs plus the two directly-adjacent regression files
  were used instead (see Regression sweep).

## Next steps

1. Re-run the finding's exact isolated per-record eval protocol
   (`evaluate_model --eval-limit 1 --eval-offset {0,1,2}` on the
   `exposure12` quality-champion checkpoint, `decode-timeout-seconds=30`)
   now that both fix-sketch items are applied, to see whether
   `decode_outcome` reclassifies from `fallback_output`/
   `empty_completion_forest` to `runtime_timeout` (if genuinely out of
   budget) or to a real on-time completion (if the swallow was masking
   otherwise-fast decode). **This is the recommended next lever** — it is
   PR #1171's own deferred next-step #3 and now has both fix-sketch
   prerequisites (this fix + PR #1173's lexer-cache fix) in place.
2. If `compiler_ms` and `decode_outcome` both improve, re-run the seeded
   multi-rep `lever-hard-decode-timeout-wall` protocol to see whether the
   `exposure12` hero's `meaningful_program_rate` recovers under the hard
   30s wall without relaxing it.

## Cleanup note

`.venv-autotrain` (python3.12 venv) and
`src/apps/openui_bridge/node_modules/` created for this session are not
committed (both gitignored).

Captured: 2026-07-28T11:55:00Z
