# Fix: `_incremental_sync` re-lexes only the unsettled tail, not the whole prefix

**Honesty:** `fixture_or_scratch`. This is a real code change (`is_fix: true`)
with a benchmark-backed unit test, but **not a ship claim** — no
`train_model`/`evaluate_model --ship-gates` run, no checkpoint, no
scoreboard. Same status tier as the finding/fix chain it closes:
[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173),
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
(PR #1172),
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
(PR #1171).

## Task

PR #1173's "Scope note" left this explicitly open: *"`_incremental_sync`
(`engine.py:149`) still re-lexes the *entire* current prefix on every call
(proposed fix sketch item 2, not applied this session) — only the *lexer
construction* is cached, not made incremental at the token-stream level.
Further wall-time headroom likely remains there for long prefixes."* This
session measures whether that's true and at what cost, and applies proposed
fix sketch item 2 if the measurement supports it.

## Measured: confirmed — `_incremental_sync` re-lexes the full prefix every call, O(n²) total session cost

Environment: fresh `.venv-diag` (`pip install -e .`, same as prior sessions),
Node bridge built (`cd src/apps/openui_bridge && npm ci --silent`, run with
`NODE_OPTIONS` unset for the same sandbox-artifact reason documented in PR
#1173). No train/eval/checkpoint — `OpenUIIncrementalEngine` exercised
directly via `advance()`/`set_prefix()`.

**Step 1 — instrument `_lex` call sizes** across a growing decode session,
using a realistic full-record OpenUI DSL string (326 chars, pulled from
`src/slm_training/resources/*.jsonl` gold records — the longest found via
`grep -rhoP '"root = [^"]{50,400}"' resources/`), fed in ~140 word/punctuation
chunks via `advance()` (the same incremental-decode call pattern production
code uses):

| session | n `_sync` calls | total chars re-lexed | ideal O(n) chars | amplification |
| --- | ---: | ---: | ---: | ---: |
| `prefix8` (single `set_prefix`) | 1 | 15 | 15 | 1.0x |
| `prefix9` (single `set_prefix`) | 1 | 16 | 16 | 1.0x |
| full 326-char record, 140 chunked `advance()` calls | 161 | 26,469 | 326 | **81.2x** |

The short single-call prefixes (matching PR #1171/#1172/#1173's exact
methodology) show no amplification — one call, one lex, as expected. The
realistic *session* (many small `advance()` calls building up one record, the
actual shape of incremental decode) shows the bug directly: `_lex_tokens` is
called with the **entire current prefix** on every one of the 161 `_sync`
calls, so the same leading characters get re-lexed dozens of times as the
prefix grows — 26,469 characters lexed in total to process a 326-character
session, an 81.2x amplification over the O(n) minimum.

**Step 2 — confirm the growth law is quadratic**, not just a fixed constant
overhead, by scaling a synthetic record (same `Stack([layer_N], ...)` nesting
family) from 326 to 3,580 chars:

| chars | `sync_ms` (before fix) | `sync_ms / chars²` (×1e-6) |
| ---: | ---: | ---: |
| 326 | 46.786 | 440.24 |
| 620 | 101.330 | 263.61 |
| 1212 | 333.084 | 226.75 |
| 2396 | 1166.359 | 203.17 |
| 3580 | 2587.643 | 201.90 |

`sync_ms / chars²` converges to a near-constant ~200-440×1e-6 — the
signature of O(n²) scaling (a constant-per-char² cost), confirming the
per-call full-prefix re-lex compounds across a session instead of being a
one-time cost. Wall time for the lex step alone grows from 47ms at 326 chars
to 2.59s at 3,580 chars (~55x for an 11x length increase).

Root cause, confirmed by reading `_incremental_sync` (`engine.py`, pre-fix):
it already tracks `_fed_token_count` and only *feeds* new tokens to the
`InteractiveParser`, but the *lexing* step above that (`tokens =
self._lex_tokens(prefix)`) always passes the full current `prefix`, then
slices `tokens[self._fed_token_count:]` afterward — the O(prefix) lex work
happens before the slicing, so no lexing work is ever actually saved by the
"incremental" path; only the (cheap, already-fast) `InteractiveParser.feed_token`
calls were incremental.

## Fix applied: track a "settle boundary," re-lex only the unsettled tail

`src/slm_training/dsl/grammar/fastpath/engine.py`:

1. New `self._settle_pos: int` — the character offset (into `self._prefix`)
   where the **last fed token starts**. Initialized to `0` in `__init__` and
   `reset()`.
2. New `_update_settle_pos(tokens, token_index_offset=0)` helper: after any
   successful sync, records `tokens[fed_token_count - 1 - token_index_offset].start_pos`
   as the new `_settle_pos`.
3. `_full_sync` calls `_update_settle_pos(tokens)` after feeding (offset 0 —
   `tokens` already covers the whole prefix from index 0).
4. `_incremental_sync` now lexes only `prefix[self._settle_pos:]` (the
   "unsettled tail": the possibly-still-growing last fed token, plus
   whatever text was newly appended) instead of the whole `prefix`. Token
   `start_pos` values from that tail lex are offset back to absolute prefix
   coordinates (`tok.start_pos += tail_start`) so `_settle_pos` stays
   meaningful across calls. The existing shrink/identity-mismatch guard
   (NAME/COMPONENT gluing, e.g. `"Te"` → `"Text"`) is preserved exactly,
   just computed against the smaller `tail_tokens` list spliced onto the
   already-known `self._fed_tokens[:prev_settled_count]` prefix instead of a
   fresh full relex — any divergence (shrink, unlexable tail, gluing) still
   falls back to `_full_sync(prefix)`, the same safety net as before.

### Why this is safe (why tokens before `_settle_pos` can never change)

`openui.lark` (75 lines) has **no lookbehind or word-boundary assertions**
(`grep -nE '\\b|\(\?<|\(\?!|\(\?='` on the grammar file: no matches) and Lark's
`BasicLexer` is a stateless, regex-based, longest-match-with-priority scanner
— a token's end boundary is decided purely from the characters up to (and
including) that boundary, using either "the regex stopped matching" (a
character earlier in the *already-seen* text mismatched) or "the string
ran out" (only possible for the very last token in whatever text was
lexed). Consequently:

- Every token **before** the last fed one ended because of a *mismatching
  character already present* in the previously-synced prefix — appending
  more text after it cannot retroactively change that decision. These are
  final the moment they're fed.
- Only the **last fed token** could have stopped solely because the prefix
  ran out at that point — it is the one and only token that can still grow
  when more text is appended (the existing NAME/COMPONENT-gluing case).

This is exactly the boundary `_settle_pos` tracks. Re-lexing
`prefix[_settle_pos:]` as a standalone substring reproduces the same token
stream a full lex would produce for that region, because nothing before
`_settle_pos` participates in matching decisions made at or after it (no
lookbehind in the grammar). Any input shape where this reasoning could be
violated (e.g. the recovery path in `_lex_tokens`'s `UnexpectedCharacters`
handler trimming differently against a smaller substring) safely degrades to
`len(keys) < self._fed_token_count` → `_full_sync(prefix)` rather than
silently producing a wrong token stream — same guarantee the pre-fix
shrink-detection already relied on.

### Verification (not just this session's script)

`tests/test_dsl/test_grammar_fastpath_incremental_sync.py` (13 tests,
committed):

- `test_incremental_sync_matches_full_resync_across_sessions` (7 parametrized
  cases: char-by-char `prefix8`/`prefix9`, the full 326-char record chunked
  by word/punctuation, NAME→COMPONENT gluing, number-after-comment gluing,
  runs of trailing whitespace, and a malformed-then-recovered fragment) —
  compares `accepts()`, `_fed_tokens`, and `_prefix` at **every step** of a
  growing session between the real (tail-relex) engine and a shadow engine
  forced to `_full_sync` every call. I6 proof: byte-identical parser state
  throughout, including every edge case the pre-fix shrink/gluing guard was
  designed for.
- `test_incremental_sync_relexes_near_linear_not_quadratic` — regression
  guard: total characters re-lexed across the 326-char record session stays
  under an 8x ceiling over the O(n) ideal (measured 2.8x post-fix vs. 81.2x
  pre-fix).
- `test_incremental_sync_faster_than_always_full_resync_on_long_session` —
  wall-time benchmark; asserts the incremental path beats an always-full-resync
  shadow by at least 1.3x (measured ~2.2x locally), conservative floor for a
  slower/loaded CI host.
- `test_settle_pos_resets_on_reset_and_new_engine` — `_settle_pos` starts at
  0 for a fresh engine and after `reset()`.
- `test_build_completion_forest_unchanged_on_finding_prefixes` (parametrized
  x2, `prefix8`/`prefix9`) — `build_completion_forest` (which drives
  `_incremental_sync` internally via the I3 recursive witness search) still
  returns `coverage == "complete"` and `n_paths == 2`/`12`, matching the
  established finding prefixes. I3/I6 proof at the decode-output layer.
- `test_load_lexer_still_cached_per_grammar_path` — guards against
  accidentally reintroducing a per-call lexer rebuild while reworking
  `_incremental_sync` (PR #1173's fix stays intact).

Sanity check the tests are not vacuous: ran the identical file against
pre-fix `engine.py` (`git stash`) — 2 of 13 fail
(`test_incremental_sync_relexes_near_linear_not_quadratic` with the exact
81.2x amplification measured above, and
`test_settle_pos_resets_on_reset_and_new_engine` with `AttributeError` since
`_settle_pos` doesn't exist pre-fix); the other 11 pass both before and
after — expected, since they assert the invariant the fix must preserve
(parser-state equivalence, decode legality), not the fix's existence.

```
env -u NODE_OPTIONS .venv-diag/bin/python -m pytest tests/test_dsl/test_grammar_fastpath_incremental_sync.py -v
# 13 passed in 6.46s (post-fix)
```

## Measured: post-fix speedup

Re-ran both diagnostics against the fixed `engine.py`:

| session | total chars re-lexed | amplification | `sync_ms` (before → after) | speedup |
| --- | ---: | ---: | ---: | ---: |
| full 326-char record, 140 chunked `advance()` | 920 | 2.8x | 32.06 → 14.79 | 2.17x |

Scaling sweep (same synthetic records as the before-fix table):

| chars | `sync_ms` before | `sync_ms` after | speedup | `sync_ms/chars` after (≈constant ⇒ O(n)) |
| ---: | ---: | ---: | ---: | ---: |
| 326 | 46.786 | 17.845 | 2.62x | 0.0547 |
| 620 | 101.330 | 31.432 | 3.22x | 0.0507 |
| 1212 | 333.084 | 57.899 | 5.75x | 0.0478 |
| 2396 | 1166.359 | 126.984 | 9.19x | 0.0530 |
| 3580 | 2587.643 | 179.885 | **14.38x** | 0.0502 |

Post-fix `sync_ms/chars` is essentially constant (~0.05ms/char) across a
>10x range of prefix lengths — direct evidence of O(n) linear scaling,
replacing the pre-fix O(n²). Speedup grows with prefix length (2.6x at 326
chars → 14.4x at 3,580 chars) exactly as expected for an O(n²)→O(n) fix: the
win compounds with session length, unlike PR #1173's fix (per-call constant
speedup from removing a fixed per-call scanner-rebuild cost).

`build_completion_forest` coverage/n_paths on `prefix8`/`prefix9`: unchanged
(`complete`/2, `complete`/12 — see unit test above). I6/I3 verified.

## Regression sweep (existing suite, unaffected by this change)

Ran with `.venv-diag` (`torch>=2.2,<2.6` CPU wheel, `NODE_OPTIONS` unset),
each command under the 3-minute cap:

| suite | result |
| --- | --- |
| `tests/test_dsl/test_grammar_fastpath_lexer_cache.py tests/test_dsl/test_grammar_fastpath_incremental_sync.py` | 19 passed |
| `tests/test_dsl/{test_pack,test_speculative_rank,test_exact_forced_horizon,test_lattice_search,test_solver_state,test_packs,test_solver_decode}.py` | 102 passed, 3 failed |
| `tests/test_dsl/test_grammar_fastpath.py -k "lexer or force or completion_forest or engine or pick_constrained"` | 24 passed, 1 failed, 18 deselected |
| `tests/test_harnesses/model_build/{test_lexer_smoke,test_dsl_tokenizer}.py` | 27 passed, 1 failed, 1 deselected |
| `tests/test_harnesses/model_build/{test_grammar_hf,test_v4_levers,test_a3_coverage_remask,test_template_fill,test_v6_remask}.py` | 30 passed, 1 failed |

**All 6 failures are pre-existing, confirmed unrelated to this change via
`git stash`** (identical failure on unmodified `engine.py`):

- `test_speculative_rank.py::test_committed_table_matches_its_builder` and
  `::test_committed_table_ranks_real_branch_points_confidently` — stale
  committed `speculative_ngram_v1.json`, same as PR #1173.
- `test_packs.py::test_pack_fixture_loop_generate_train_eval` —
  `require_constrained_generation` rejects the fixture's
  `grammar_constrained=False` test config; reproduced identically with
  `engine.py` reverted. New to this sweep vs. PR #1173's list (different
  test subset run) but confirmed pre-existing, unrelated to lexing.
- `test_grammar_fastpath.py::test_completion_forest_admits_null_for_optional_string_after_slots_exhaust`
  — `build_completion_forest() got an unexpected keyword argument
  'enforce_schema_component_types'`, same signature-drift issue PR #1173
  documented.
- `test_lexer_smoke.py::test_surface_identifier_arm_is_prohibited` — same as
  PR #1173.
- `test_v4_levers.py::test_generate_batch_requests_consumes_harness_slot_contract`
  — prompt-string assertion mismatch in slot-contract formatting. PR #1173
  could not confirm this one in isolation (killed by the run cap); this
  session reproduced it fully in isolation (39.64s, under the cap) with
  `engine.py` reverted — **now confirmed pre-existing**, unrelated to this
  change (touches `harnesses/model_build/plugin.py` prompt formatting, no
  code path through `OpenUIIncrementalEngine`).

`ruff check` on both changed files: clean.
`python -m scripts.verify_version_stamps --check`: `2 changed file(s), 0
component(s) touched` — `fastpath/engine.py` is not itself a path in any
`versions.json` component (same as PR #1173's fix), so no version bump is
required; the new test file is untracked-added, not a watched path either.
`python -m scripts.repo_policy`: `ok (tracked + untracked)`.
`git diff --check`: clean.

## Interpretation

Closes proposed-fix-sketch item 2 from
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
and the explicit open question in PR #1173's scope note. Confirms
`_incremental_sync` re-lexed the whole prefix on every call even after
PR #1173's lexer-construction cache — the *lexing work itself* was never
made incremental, only the lexer *object* was reused. This turned a growing
decode session's total lex cost from O(n) into O(n²) (81.2x amplification
measured on a realistic 326-char record). The fix exploits the grammar's
lack of lookbehind to prove that only the last fed token can still be
"open," so re-lexing needs to touch only the unsettled tail — reducing the
session's total lex cost back to O(n) (2.8x amplification, matching the
small constant overhead of re-deriving each token's tail region a few times
before it settles) and producing a speedup that *grows* with prefix length
(2.6x → 14.4x across the 326–3,580 char range measured), unlike PR #1173's
fix (flat ~2x, a per-call constant-cost removal).

## Scope note (what this does NOT establish)

- No train/eval run, no checkpoint, no `--ship-gates` scoreboard —
  harness-internal correctness+perf fix with unit-test coverage, not a
  readiness claim.
- Not re-measured against the full per-record `compiler_ms` telemetry or the
  `exposure12` quality-champion eval protocol from PR #1171 — still an open
  next step (also left open by PR #1173).
- `probe_chunk` (`engine.py`) has the same full-relex-of-`new_prefix` shape
  as pre-fix `_incremental_sync` did, but was out of this session's scope
  (the task named `_incremental_sync` specifically) — noted as a further
  candidate, not measured or fixed here.
- The realistic-session benchmark uses one real 326-char gold record plus
  synthetic same-family records scaled to 3,580 chars; not a multi-record,
  multi-family statistical sweep.

## Next steps

1. Apply the same tail-relex treatment to `probe_chunk`
   (`engine.py`, the `Q1` copy-probe path), which still re-lexes
   `self._prefix + chunk` from scratch on every call.
2. Re-run the PR #1171 isolated per-record eval protocol (or the seeded
   multi-rep `lever-hard-decode-timeout-wall` protocol) now that both
   PR #1173's and this session's fixes are in place, to see whether
   `compiler_ms` drops enough in aggregate for the `exposure12`
   quality-champion hero to finish inside `decode_timeout_seconds=30`.
3. Extend the scaling sweep to real (not synthetic) long DSL records once a
   corpus of longer gold examples exists, to confirm the O(n) law holds
   outside the `Stack([layer_N], ...)` nesting family used here.

## Cleanup note

`.venv-diag`, `src/apps/openui_bridge/node_modules/` (both gitignored) and
the session's reproduction scripts (`measure_incremental_sync.py`,
`measure_incremental_scaling.py`, `verify_incremental_fix.py` — session
scratchpad, not reusable harness tools) are not committed.

Captured: 2026-07-28T13:46:51Z
