# Fix: cache the resolved grammar path in `OpenUIIncrementalEngine`; item 1 of the same sketch tested and rejected

**Honesty:** `fixture_or_scratch`. This is a real code change (`is_fix: true`)
with a benchmark-backed unit test, but **not a ship claim** — no
`train_model`/`evaluate_model --ship-gates` run, no checkpoint, no
scoreboard. Same status tier as the finding it partially closes:
[`decode-compiler-tree-witness-search-cost-finding.md`](decode-compiler-tree-witness-search-cost-finding.md)
(PR #1191), itself following on PRs #1171/#1172/#1173/#1189/#1190 in this
stack.

## Task

PR #1191 named two "Proposed fix sketch" items as candidates:

1. Hoist the `@lru_cache`-wrapped `_tail` closure (`pack.py:402`) out of the
   per-candidate loop in `_openui_completion_domain` so all top-level
   candidates at one decode position share one witness cache.
2. Cache the resolved grammar path (module-level, keyed on the raw
   `grammar_path` string) instead of calling `Path(...).resolve()` inside
   every `OpenUIIncrementalEngine.__init__`.

This session implemented **both**, measured each independently, and kept
only the one that measured a real win.

## Item 1: tested, rejected (negative result, not applied)

The sketch's own safety argument was sound — `nodes_left`'s per-candidate
fairness budget is untouched because `lru_cache` skips the decrementing
function body on a hit, so sharing the cache could only ever give a later
candidate a free win, never fewer real chances. But "safe" is not the same
as "helps," and the sketch did not claim a specific magnitude, so this
session measured it before shipping it.

**Implementation** (exactly as sketched): moved the `@lru_cache`-wrapped
`_tail` function to be defined once per `_openui_completion_domain` call,
outside `_tail_from`; `_tail_from` now just resets the shared `nodes_left`
counter to 16 before calling into the shared `_tail`.

**Measurement**: A/B via `git stash` (same process, monkeypatched
`_build_openui_completion_forest` call counter), on 6 real prefixes chosen
to maximize top-level candidate count, including PR #1173's own
`prefix9` (`root = Card([b1,`):

| prefix | n candidates | calls before | calls after |
| --- | ---: | ---: | ---: |
| `root =` | 27 | 271 | 271 |
| `root = Card([b1` | 2 | 23 | 23 |
| `root = Card([b1,` | 12 | 171 | 171 |
| (bos only) | 2 | 10 | 10 |
| `root = Stack([b1, b2` | 2 | 29 | 29 |
| `root = Stack([b1, b2,` | 12 | 518 | 518 |

**Zero reduction on every prefix**, witness content byte-identical in every
case. Why: `initial.paths`' distinct top-level candidates diverge at their
very first token by construction — that is what makes them distinct
candidates — so their subsequent `(current, room)` recursive-search states
in `_tail` never coincide within one `_openui_completion_domain` call.
There is nothing for a shared cache to hit. This is a real, mechanical
property of the witness search's structure, not a fluke of the prefixes
chosen (verified across a 27-candidate case and a 518-call case, not just
small ones).

**Decision**: reverted (`git checkout -- pack.py`) rather than ship a
no-benefit complexity addition — extra state (a `nodes_left`/`_tail`
sharing scheme) that changes nothing about the actual cost this iteration
set out to reduce. Recorded here so a future iteration does not re-attempt
the identical hypothesis without new evidence that a real decode record's
candidate set produces convergent search states (this session's prefixes
did not, but a different grammar shape or a much longer decode might).

## Item 2: applied

`src/slm_training/dsl/grammar/fastpath/engine.py`:

1. New `_resolve_grammar_path(raw_path: str) -> str`, `@lru_cache(maxsize=8)`
   — returns `str(Path(raw_path).resolve())`, computed once per distinct raw
   path string.
2. `OpenUIIncrementalEngine.__init__` now calls `_resolve_grammar_path(str(path))`
   once and reuses the result for both `_load_parser(resolved)` and
   `_load_lexer(resolved)`, instead of calling `path.resolve()` twice per
   `__init__` (once per callee, the exact redundancy PR #1191's cProfile
   flagged: 10,967 `__init__` calls in one profiled decode record, ~2.9s
   `Path.resolve` + ~2.2s `os.path.realpath` cumulative).

### Why this is safe

`Path.resolve()` is a pure function of the raw path string within one
process — no code path here changes cwd or the grammar file's location
mid-run. `_load_parser`/`_load_lexer`'s own `@lru_cache` is keyed on the
*resolved* string exactly as before; caching the resolve step changes
nothing about which cache key either callee receives, so it cannot change
which `Lark`/`BasicLexer` object comes back. Verified directly: a default
`OpenUIIncrementalEngine()` and one constructed with an explicit,
already-resolved `Path` return `is`-identical `_parser`/`_lexer` objects
(`test_equivalent_raw_paths_share_the_same_cached_parser_and_lexer`).

### Reproduction

Same precedent as PR #1173's own fix doc: a targeted micro-benchmark
reproducing the finding's exact profiled call count, not a full
train/eval/checkpoint rerun (see "Scope note" below for why a full
`exposure12` reproduction was not attempted this session).

`bench_resolve.py` (session scratchpad, not committed) times
`N=10967` (PR #1191's own profiled `__init__` call count for one decode
record) constructions of `OpenUIIncrementalEngine()` against the current
(cached) code, vs. an inlined reproduction of the exact pre-fix `__init__`
body (`path.resolve()` called directly, no cache) in the same process.

### Measured: 5.3x-5.6x speedup on `__init__` itself, ~0.63s-0.73s saved per 10,967 inits

| rep | before (s) | after (s) | speedup | saved (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.7979 | 0.1465 | 5.45x | 0.6514 |
| 2 | 0.8759 | 0.1575 | 5.56x | 0.7184 |
| 3 | 0.7760 | 0.1457 | 5.33x | 0.6303 |
| 4 | 0.9013 | 0.1675 | 5.38x | 0.7338 |

Consistent across 4 reps. Smaller in absolute terms than the finding's
~3.26s cumulative cProfile figure for `OpenUIIncrementalEngine.__init__`
(that figure also includes LALR-parser-frontend construction cost beyond
the two explicit `path.resolve()` calls this fix targets specifically), but
a real, repeatable, isolated measurement of the exact syscall this fix
eliminates — roughly **2% of the ~29-34s total wall-clock budget** of the
finding's profiled records. Modest, not transformative: the finding's
dominant identified cost (Lark's `InteractiveParser.accepts()`, 12.6s /
23,530 calls) is untouched by this change.

## Unit test (added, committed)

`tests/test_dsl/test_grammar_fastpath_path_resolve_cache.py` (4 tests):

- `test_resolve_grammar_path_is_cached` — two calls with the same raw path
  return the identical resolved string.
- `test_path_resolve_called_at_most_once_across_many_inits` — monkeypatches
  `Path.resolve` with a counter; constructs 25 engines for the same grammar
  path; asserts `Path.resolve` runs **at most once** total (regression
  guard for the finding's 10,967-calls-for-one-record pathology).
- `test_equivalent_raw_paths_share_the_same_cached_parser_and_lexer` — a
  default-constructed engine and one built with an explicit, pre-resolved
  path share `is`-identical `_parser`/`_lexer` objects (proves the resolver
  cache's key/value split doesn't fragment the downstream parser/lexer
  cache).
- `test_build_completion_forest_unchanged_after_path_cache_fix` — I3/I6
  proof: `build_completion_forest` on PR #1173's own `prefix9` still
  returns `coverage == "complete"`, `n_paths == 12`.

```
python -m pytest tests/test_dsl/test_grammar_fastpath_path_resolve_cache.py -q
# 4 passed in 6.03s
```

## Regression sweep

Each command under the 3-minute run cap (full `.venv` build: Python 3.12 +
`pip install -e ".[dev,grammar]"`):

| suite | result |
| --- | --- |
| `tests/test_dsl/{test_grammar_fastpath_lexer_cache,test_grammar_fastpath_deadline,test_grammar_fastpath_path_resolve_cache}.py` | 13 passed |
| `tests/test_dsl/test_pack.py` (deselect `test_end_to_end_fixture_run_through_pack_interface`, a torch model-build test that exceeds the run cap) | 16 passed, 2 failed |
| `tests/test_dsl/test_grammar_fastpath.py` (unfiltered, 26 tests, torch CPU forward passes) | 19 passed, 0 failed, killed at the 170s cap before completion |

**Both `test_pack.py` failures are pre-existing, confirmed unrelated** via
`git stash` on `engine.py` (reproduce identically on unmodified `HEAD`):
`test_openui_oracle_accepts_source_and_fails_closed` and
`test_verify_stack_verdicts_unchanged`, both `VerificationReport(tier=QUARANTINE)`
— an environment/fixture gap unrelated to this change, which touches only
`engine.py`'s path-resolution caching.

The unfiltered `test_grammar_fastpath.py` run hit the same run-cap wall PR
#1173's fix doc hit on this identical file; a 3-test targeted subset
(`test_engine_force_equal_after_name`,
`test_lexer_root_round_trips_and_is_first_token_legal`,
`test_grammar_state_uses_surface_text_for_lexer_ids`) passed in 2.51s in
isolation, and no failures appeared in the 19 tests that completed before
the cap. No evidence of a regression from either run.

`ruff check` on the changed file and new test file: clean.
`python -m scripts.verify_version_stamps --check` → `ok (vs HEAD; 2 changed
file(s), 0 component(s) touched)` — `fastpath/engine.py` is not itself a
path in any `versions.json` component, identical precedent to PR #1173; the
new test file is untracked-added, not a watched path either.
`python -m scripts.repo_policy` → `ok (tracked + untracked)`.
`python -m scripts.verify_decode_invariants` → exit 0, unchanged.

## Interpretation

Closes PR #1191's proposed-fix-sketch item 2 with a measured, modest win,
and closes item 1 with a documented negative result instead of an
unverified assumption that it would help. Neither change touches the
finding's dominant identified cost — that remains open.

## Scope note (what this does NOT establish)

- No train/eval run, no checkpoint, no `--ship-gates` scoreboard.
- Not re-measured against the full per-record `compiler_ms` telemetry or
  the `exposure12` quality-champion eval protocol from PR #1189/#1190/#1191:
  this session could not locate the exact `slm data build-train`/`sft train`
  invocation used in prior sessions to produce the `lever_exposure12_v1`
  train data and `exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47`
  checkpoint (both live under gitignored `outputs/`, not committed, and
  this session's container starts with an empty `outputs/`). PR #1173's own
  fix doc used the identical isolated-micro-benchmark methodology instead
  of a full pipeline rerun for this same class of caching fix, so that
  precedent is followed here rather than guessing at build commands.
- Item 3 from PR #1191's sketch (memoizing Lark's `InteractiveParser.accepts()`
  per parser-automaton state) — flagged there as "larger, Lark-internals-facing"
  — is not attempted here and remains the dominant open cost.

## Next steps

1. Item 3: investigate whether Lark's `InteractiveParser.accepts()` can be
   memoized per parser-automaton state — the dominant identified cost
   (12.6s / 23,530 calls, ~43% of the profiled record's wall-clock), which
   neither this fix nor the reverted item 1 touches.
2. Pin down (or rebuild via a documented `slm data build-train` /
   `slm sft train` invocation) the `exposure12`/`s16`/`seed47` recipe so a
   future iteration can re-run PR #1191's exact offsets 0/1/2 protocol with
   this fix and the lexer-cache fix (#1173) both in tree, for an honest
   full-record `compiler_ms` delta instead of an isolated micro-benchmark.
3. Do not re-attempt sketch item 1 without new evidence that a real
   decode record's candidate set actually produces convergent `(current,
   room)` states — this session's 6-prefix sweep (2 to 27 candidates,
   10 to 518 forest-build calls) found none.

## Cleanup note

`bench_resolve.py` (session scratchpad, reproduction scaffolding, not a
reusable harness tool) is not committed. No `outputs/` artifacts were
created this session.

Captured: 2026-07-28T13:10:00Z
