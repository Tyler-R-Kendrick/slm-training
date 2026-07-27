# Fix: cache the Lark lexer/scanner per grammar in `OpenUIIncrementalEngine`

**Honesty:** `fixture_or_scratch`. This is a real code change (`is_fix: true`)
with a benchmark-backed unit test, but **not a ship claim** — no
`train_model`/`evaluate_model --ship-gates` run, no checkpoint, no scoreboard.
Same status tier as the finding it closes:
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
(PR #1172), itself following on
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
(PR #1171).

## Task

Apply "Proposed fix sketch" item 1 from the lexer-rebuild finding: cache a
built Lark `Lexer`/`Scanner` per grammar alongside the existing
`@lru_cache`d `_load_parser` (`engine.py:43-53`), and have `_lex_tokens`
(`engine.py:99`) reuse it instead of calling the standalone
`self._parser.lex(prefix)` convenience API, which the finding showed
rebuilds the scanner (`_build_scanner` → `_create_unless`, recompiling
terminal-disambiguation regex) on **every** call — 23,546 times for one
`build_completion_forest` invocation at the `prefix9` case. Then re-run that
diagnostic and confirm call counts / wall time drop with **zero** change to
`n_paths`/`coverage` (I6/I3: a caching change must never become a legality
change).

## What changed

`src/slm_training/dsl/grammar/fastpath/engine.py`:

1. New `_load_lexer(grammar_path: str) -> BasicLexer`, `@lru_cache(maxsize=4)`
   alongside `_load_parser`. It calls `_load_parser(grammar_path)._build_lexer()`
   once per grammar path — the exact same `BasicLexer` construction
   `Lark.lex()` performs internally, just cached.
2. `OpenUIIncrementalEngine.__init__` stores `self._lexer = _load_lexer(...)`.
3. New `_lex(text)` helper: `LexerThread.from_text(self._lexer, text).lex(None)`
   — reimplements what `Lark.lex(text)` does, but against the cached lexer
   instead of building a fresh one. `_lex_tokens` (`engine.py:99`, called by
   both `_full_sync` and `_incremental_sync`, i.e. every `_sync`) now calls
   `self._lex(...)` instead of `self._parser.lex(...)`.

No change to `_full_sync`/`_incremental_sync`/`_sync` control flow, the
`InteractiveParser` feed loop, or any grammar/terminal definition — only
*which lexer object* performs the tokenization.

### Why this is safe (why the token stream can't change)

`Lark.lex(text)` (`lark/lark.py`) is:

```python
lexer = self.lexer if hasattr(self, 'lexer') else self._build_lexer(dont_ignore)
return LexerThread.from_text(lexer, text).lex(None)
```

For a grammar built with `parser="lalr"` (as `_load_parser` does), `Lark`
never sets `self.lexer` (that attribute is only set when `self.options.parser`
is falsy), so every call takes the `_build_lexer()` branch — a fresh
`BasicLexer(lexer_conf)` from the *same* `LexerConf` every time. Our
`_load_lexer` calls that identical `_build_lexer()` method, once, and caches
the resulting `BasicLexer`. Verified directly (`.venv-diag`, same env as the
finding) that reusing it produces a byte-identical token stream:

```python
toks_a = list(parser.lex(prefix))                                    # uncached
toks_b = list(LexerThread.from_text(parser.parser.lexer, prefix).lex(None))  # cached
assert toks_a == toks_b   # True
```

`BasicLexer` itself already lazily caches its `Scanner` on first use
(`self._scanner`, `lexer.py:588-591`); the bug was that the *previous* code
threw away the whole `BasicLexer` (and its `_scanner` cache) after every
call instead of keeping one around. Caching the `BasicLexer` object is
exactly reusing that existing lazy cache instead of discarding it.

## Reproduction

Same environment and prefixes as the prior finding (fresh checkout,
`.venv-diag`/`node_modules` gitignored, not committed):

```bash
python3.12 -m venv .venv-diag
.venv-diag/bin/pip install --quiet -e .
.venv-diag/bin/pip install --quiet "pytest>=8.0,<9" "pytest-asyncio>=0.23,<2" \
  "ruff>=0.9,<0.16" "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu
cd src/apps/openui_bridge && npm ci --silent   # Node DSL bridge, needed for lang_core.parse
```

`before_after.py` (kept under the session scratchpad, not committed) calls
`build_completion_forest(tokenizer, prefix_ids, remaining_tokens=24,
max_path_tokens=8, min_content=0)` directly — no train/eval/checkpoint —
for the same 4 prefixes as the original finding, instrumented with a
`BasicLexer._build_scanner` call counter. Run once against the pre-fix
`engine.py` (`git stash`) and once against the fixed version, each a fresh
process so no `lru_cache` state leaks across the comparison:

```bash
env -u NODE_OPTIONS .venv-diag/bin/python before_after.py   # BEFORE: git stash applied
env -u NODE_OPTIONS .venv-diag/bin/python before_after.py   # AFTER:  git stash pop
```

(`NODE_OPTIONS` is unset only because this sandbox's ambient
`NODE_OPTIONS="--import tsx" --max-old-space-size=8192` is malformed for a
plain `node` invocation — a sandbox artifact unrelated to the fix; the
bridge itself is unaffected by this change.)

## Measured: same coverage/n_paths, ~2-2.1x wall time, scanner built once instead of thousands of times

| prefix | before wall_s | after wall_s | speedup | before `n_paths` | after `n_paths` | before coverage | after coverage | before `_build_scanner` calls | after `_build_scanner` calls |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| cold_empty_prefix | 0.7888 | 0.4142 | 1.90x | 2 | 2 | complete | complete | 1049 | 2 |
| prefix8 (`root = Card([b1`) | 1.2559 | 0.5965 | 2.11x | 2 | 2 | complete | complete | 1558 | 0 |
| prefix9 (`root = Card([b1,`) | 17.0073 | 7.9373 | **2.14x** | 12 | 12 | complete | complete | **23546** | **0** |
| prefix8_repeat | 1.0753 | 0.5371 | 2.00x | 2 | 2 | complete | complete | 1558 | 0 |

Every `coverage`/`n_paths` pair is **identical before and after** — the fix
changed only wall time and call counts, never decode legality (I6/I3
verified). The `prefix9` before-row's `23546` scanner builds reproduces the
prior finding's cProfile count (`lark/lexer.py:592 _build_scanner` called
23,546 times) exactly, confirming this before/after run is measuring the
same phenomenon that finding diagnosed. After the fix, `_build_scanner` is
called **0** additional times for any of the three `build_completion_forest`
calls after the first process-wide warm-up (`cold_empty_prefix`'s `2` =
one `BasicLexer` built internally by `ParsingFrontend` for the LALR parser's
own use, one built by the new `_load_lexer` cache — both built exactly once
per process, never again).

The wall-time win (~2-2.1x) is smaller than the finding's cProfile
`_lex_tokens` share (27.1s / 41.2s profiled ≈ 66%) would suggest for an
*unprofiled* run, because `_lex_tokens`'s absolute uncached cost per call
(0.62ms average, from the finding) is small relative to the recursive
`_tail_from`/`_tail` witness-search overhead and Python-level dispatch that
remain unchanged by this fix — this patch removes the *scanner-rebuild*
tax specifically, not the whole `compiler_ms` budget. `prefix9` still costs
7.9s post-fix; the deadline-swallow behavior from PR #1171 and the
recursive witness-search cost itself remain open (see Next steps).

## Unit test (added, not just this session's script)

`tests/test_dsl/test_grammar_fastpath_lexer_cache.py` (6 tests, committed):

- `test_load_lexer_is_cached_per_grammar_path` — `_load_lexer` returns the
  identical object for the same grammar path.
- `test_engine_builds_scanner_at_most_once_across_many_syncs` — monkeypatches
  `BasicLexer._build_scanner` with a counter; drives an engine through 5
  growing `set_prefix` calls plus a second engine instance for the same
  grammar; asserts the scanner is built **at most once** total (regression
  guard for the finding's 23,546-calls-for-one-query pathology).
- `test_lexer_cache_speeds_up_repeated_lex_calls` — 200 reps of `eng._lex(prefix9)`
  vs. 200 reps of the pre-fix `eng._parser.lex(prefix9)` path (still reachable
  through the cached `Lark` object for a fair same-process A/B); asserts the
  cached path is at least 2x faster (conservative floor under the ~14.6x
  measured locally on repeated single-lex calls in the finding
  follow-up work, robust to slower/loaded CI hosts).
- `test_lex_tokens_byte_identical_to_uncached_lark_lex` — direct token-stream
  equality between `eng._lex(prefix)` and `eng._parser.lex(prefix)` on both
  finding prefixes (I6 proof at the lexer layer).
- `test_build_completion_forest_unchanged_on_finding_prefixes[...]`
  (parametrized x2) — `build_completion_forest` on `prefix8`/`prefix9` still
  returns `coverage == "complete"` and `n_paths == 2`/`12` respectively,
  matching `measured_direct_calls` in the original finding's JSON (I6 proof
  at the decode-output layer).

Sanity check that the tests are not vacuous: ran the identical file against
pre-fix `engine.py` (`git stash`) — 4 of 6 fail with `AttributeError` (the
new `_lex`/`_load_lexer` API doesn't exist yet), and the 2 parametrized
`build_completion_forest` cases pass both before and after (expected — they
assert the invariant the fix is required to preserve, not the fix itself).

```
env -u NODE_OPTIONS .venv-diag/bin/python -m pytest tests/test_dsl/test_grammar_fastpath_lexer_cache.py -v
# 6 passed in 9.48s (post-fix)
```

## Regression sweep (existing suite, unaffected by this change)

Ran with `.venv-diag` (`torch>=2.2,<2.6` CPU wheel added for the
`test_harnesses/model_build` torch-dependent files; `NODE_OPTIONS` unset for
the reason above), each command under the 3-minute cap:

| suite | result |
| --- | --- |
| `tests/test_dsl/{test_pack,test_speculative_rank,test_exact_forced_horizon,test_lattice_search,test_solver_state,test_packs,test_solver_decode}.py` | 102 passed, 1 skipped, 2 failed |
| `tests/test_dsl/test_grammar_fastpath.py -k "lexer or force or completion_forest or engine or pick_constrained"` | 24 passed, 1 failed, 18 deselected |
| `tests/test_harnesses/model_build/{test_lexer_smoke,test_dsl_tokenizer}.py` | 27 passed, 1 failed, 1 deselected |
| `tests/test_harnesses/model_build/{test_grammar_hf,test_v4_levers,test_a3_coverage_remask,test_template_fill,test_v6_remask}.py` | 30 passed, 1 failed, 2 deselected |

**All 5 failures are pre-existing, confirmed unrelated to this change**:

- `test_speculative_rank.py::test_committed_table_matches_its_builder` and
  `::test_committed_table_ranks_real_branch_points_confidently` — stale
  committed `speculative_ngram_v1.json` / low ranking confidence; reproduced
  identically on unmodified `engine.py` (`git stash`).
- `test_grammar_fastpath.py::test_completion_forest_admits_null_for_optional_string_after_slots_exhaust`
  — `build_completion_forest() got an unexpected keyword argument
  'enforce_schema_component_types'`; reproduced identically on unmodified
  `engine.py`. Unrelated to lexing — a `compiler_draft.py` signature/test
  drift.
- `test_lexer_smoke.py::test_surface_identifier_arm_is_prohibited` —
  `pytest.raises(ValueError, ...)` did not raise; reproduced identically on
  unmodified `engine.py`.
- `test_v4_levers.py::test_generate_batch_requests_consumes_harness_slot_contract`
  — prompt-string assertion mismatch in slot-contract formatting
  (`harnesses/model_build/plugin.py`), not re-confirmed in isolation on
  unmodified `engine.py` within the run cap (a single-test pre-fix run did
  not finish inside 150s and was killed — that killed run is not used as
  evidence per the hard-run-cap law). Logically unrelated in scope: this
  fix touches only `OpenUIIncrementalEngine._lex_tokens`'s lexer object, not
  prompt/slot-contract string construction in a different harness module.

`ruff check` on both changed files: clean.
`python -m scripts.verify_version_stamps --check`: `2 changed file(s), 0
component(s) touched` — `fastpath/engine.py` is not itself a path in any
`versions.json` component (only `dsl.grammar_capabilities`/`pack.py`/
`compiler_draft.py` are registered there), so no version bump is required
for this change; the new test file is untracked-added, not a watched path
either.
`python -m scripts.repo_policy`: `ok (tracked + untracked)`.
`git diff --check`: clean (no whitespace errors).

## Interpretation

Confirms the finding's root-cause chain and closes its "Proposed fix
sketch" item 1. The Lark grammar object was already cached
(`_load_parser`); the lexer/scanner built from that grammar was not — every
`_sync` call rebuilt it from scratch via the uncached `Lark.lex()`
convenience path. Caching it the same way removes the rebuild entirely
(`_build_scanner` calls: thousands-per-query → 0 after process warm-up)
without touching tokenization semantics, since the cached `BasicLexer` is
constructed from the identical `LexerConf` the uncached path would have
used.

## Scope note (what this does NOT establish)

- No train/eval run, no checkpoint, no `--ship-gates` scoreboard. This is a
  harness-internal correctness+perf fix with unit-test coverage, not a
  readiness claim.
- The wall-time win (~2-2.1x) is on `build_completion_forest` micro-calls in
  isolation; it was not re-measured against the full per-record
  `compiler_ms` telemetry or the `exposure12` quality-champion eval protocol
  from PR #1171 — that is next steps below, not measured here.
- `_incremental_sync` (`engine.py:149`) still re-lexes the *entire* current
  prefix on every call (proposed fix sketch item 2, not applied this
  session) — only the *lexer construction* is cached, not made incremental
  at the token-stream level. Further wall-time headroom likely remains
  there for long prefixes.
- The `test_v4_levers.py` pre-existing failure's "unrelated" claim rests on
  code-scope reasoning (this fix's diff touches only lexer construction
  inside `OpenUIIncrementalEngine`), not a completed isolated pre-fix rerun,
  because that rerun did not finish inside the run cap and a killed run
  cannot be cited as evidence.

## Next steps

1. Re-run the PR #1171 isolated per-record eval protocol (or the seeded
   multi-rep `lever-hard-decode-timeout-wall` protocol) to see whether
   `compiler_ms` drops enough in aggregate for the `exposure12`
   quality-champion hero to finish inside `decode_timeout_seconds=30`
   without the deadline fix from PR #1171.
2. Apply proposed-fix-sketch item 2: make `_incremental_sync` genuinely
   incremental at the lex layer (only lex newly-appended text) instead of
   re-lexing the full prefix through the now-cached lexer every call.
3. If `compiler_ms` drops substantially in a full per-record eval, re-check
   whether PR #1171's deadline-swallow bug (`TimeoutError` caught by the
   bare `except` in `compiler_draft.py:2313`) is still practically
   reachable, or was mostly masking this lexer-rebuild cost.

## Cleanup note

`.venv-diag`, `src/apps/openui_bridge/node_modules/` (both gitignored) and
the `before_after.py` reproduction script (session scratchpad, not a
reusable harness tool) are not committed.

Captured: 2026-07-27T21:55:59.566231Z
