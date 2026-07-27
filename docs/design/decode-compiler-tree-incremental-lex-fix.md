# Fix: suffix-only relex in `_incremental_sync` (`OpenUIIncrementalEngine`)

**Honesty:** `fixture_or_scratch`. This is a real code change (`is_fix: true`)
with a benchmark-backed unit test, but **not a ship claim** — no
`train_model`/`evaluate_model --ship-gates` run, no checkpoint, no scoreboard.
Same status tier as the chain it continues:
[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173) →
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
(PR #1172) →
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
(PR #1171).

## Task

Apply "Next steps" item 2 from the lexer-cache fix doc: make
`_incremental_sync` (`engine.py:182`, pre-change) genuinely incremental at
the **token-stream** level, not just at the lexer-object level. Pre-change,
every `_incremental_sync` call re-lexed the *entire* current prefix from
character 0 through the (now-cached, post-PR-#1173) `BasicLexer` — O(prefix
length) lex work per call, O(n²) total across a generation of length n, even
though the lexer/scanner object itself was no longer being rebuilt.

## What changed

`src/slm_training/dsl/grammar/fastpath/engine.py`:

1. New instance field `self._fed_last_token_start: int | None` — the
   absolute character offset into `self._prefix` where the *last already-fed*
   token starts, or `None` before anything has been fed. Reset in `reset()`.
2. `_full_sync` now also sets `self._fed_last_token_start` after a successful
   feed loop, to the `start_pos` of the last fed `Token`.
3. New shared helper `_feed_delta_tokens(prefix, delta, keys, *, base_offset)`
   — factored out of `_incremental_sync`'s feed loop (identical
   `feed_token`/`UnexpectedToken`/`UnexpectedEOF` shape as before), now also
   updates `self._fed_last_token_start` from whichever delta token was last
   successfully fed, using `base_offset + tok.start_pos` (`base_offset` is 0
   for a from-scratch lex, or the suffix start for a suffix-only lex).
4. `_incremental_sync` now has two branches:
   - **No anchor yet** (`self._fed_token_count == 0` or
     `self._fed_last_token_start is None`): unchanged pre-existing behavior —
     lex the whole (still-short, since nothing fed) prefix from 0.
   - **Fast path** (`self._fed_token_count > 0`): re-lex only
     `prefix[self._fed_last_token_start:]` — the last already-fed token plus
     everything appended after it — instead of the whole prefix. Compare the
     re-lexed suffix's first token against the previously-fed boundary token;
     if unchanged, splice the untouched `self._fed_tokens[:boundary_idx]`
     onto the freshly re-lexed suffix keys and feed only the genuinely new
     tokens (`suffix_tokens[1:]`) via `_feed_delta_tokens`. If the boundary
     token's identity changed (still growing, e.g. `"Te"` → `"Text"`), or the
     suffix fails to lex at all, fall back to the exact pre-existing
     `_full_sync(prefix)` path — never guesses.

No change to `_full_sync`'s or `_sync`'s external contract, `probe_chunk`
(explicitly out of scope per the task), or any grammar/terminal definition.

## Why this is safe

Lark's `BasicLexer` is maximal-munch: at each position it takes the longest
regex match among the grammar's terminal alternatives. A token that already
*ended* at some position `j` — because the lexer found the next token (or
ignored whitespace, or hit end-of-string) starting at `j` — did so using only
characters at positions `< j`, all of which were already present in
`self._prefix` and are **unchanged** by appending more text strictly after
the new, longer `prefix`. Re-lexing from character 0 again cannot produce a
different result for that token: the same greedy match against the same
unchanged characters yields the same match. So every already-fed token
*except possibly the last one* is permanently settled the moment a token
after it was recognized.

The **last** fed token is the one exception: when it was lexed, the lexer had
reached the end of the input text available *at that time* — it could not
prove maximal munch, because there was no way to know whether more matching
characters would be appended next call. That's exactly the NAME/COMPONENT
"gluing" case PR #1173 already detected via a full-prefix relex + key
comparison (`"Te"` + `"xt"` → `"Text"`): a single token whose *value* grows,
not two separate tokens. `self._fed_last_token_start` names that one
uncertain position; re-lexing `prefix[start:]` re-derives that one token (and
whatever comes after it) exactly as a full relex would, without re-deriving
the tokens before it that were never actually in doubt.

Two additional facts specific to this grammar were verified directly (not
assumed) before relying on them:

- **`Token.start_pos`/`end_pos` are populated by the lexer itself**,
  independent of `propagate_positions` (which only controls whether position
  *metadata propagates onto parse-tree nodes* — an entirely different Lark
  option). Confirmed by reading `lark/lexer.py`'s `BasicLexer.next_token`
  (`Token(type_, value, line_ctr.char_pos, ...)` is constructed
  unconditionally) and directly in this repo's `.venv-diag`:
  `eng._lex("root = Card([b1, b2])")` returns tokens with correct
  `start_pos`/`end_pos` for every entry.
- **No terminal in `src/slm_training/dsl/grammars/openui.lark` uses a
  lookbehind or `\b`-style assertion** (`BUILTIN`, `STATE_NAME`, `COMPONENT`,
  `NAME`, `STRING`, `NUMBER`, `BOOL`, `NULL`, `COMMENT`, `_NL` are all
  anchored, no-lookbehind regexes). This matters because
  `LexerThread.from_text(self._lexer, suffix)` lexes `suffix` as a **fresh
  string**, not a zero-copy view with left context into the original
  `prefix` — `BasicLexer.match` calls `mre.match(text.text, pos, text.end)`
  where `text.text` is that fresh string. A lookbehind/`\b` terminal at
  position 0 of the suffix would see no preceding character and could
  mismatch what a full-prefix lex would have produced. Since no current
  terminal does this, slicing is safe for this grammar today; a future
  terminal addition using lookbehind would need to revisit this assumption
  (see Scope note).

On *any* lex failure, ambiguity, or a changed boundary-token identity, the
code falls back to `_full_sync(prefix)` — the exact same full-prefix-relex
path this method always took pre-fix. The fast path is strictly additive:
every case it declines to accelerate degrades to previously-existing,
previously-tested behavior.

## Reproduction

Same `.venv-diag` setup as PR #1173 (fresh per-session venv, gitignored, not
committed):

```bash
python3.12 -m venv .venv-diag
.venv-diag/bin/pip install --quiet -e .
.venv-diag/bin/pip install --quiet "pytest>=8.0,<9" "pytest-asyncio>=0.23,<2" "ruff>=0.9,<0.16"
.venv-diag/bin/pip install --quiet "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu
cd src/apps/openui_bridge && env -u NODE_OPTIONS npm ci --silent
```

(`NODE_OPTIONS` unset for the same sandbox reason as PR #1173: the ambient
`NODE_OPTIONS` is malformed for a plain `node`/`npm` invocation, unrelated to
this fix.)

Two scratch scripts (session scratchpad, not committed) drove the
before/after comparison:

- A long (543-char, 10-line), multi-statement, incrementally-growing valid
  OpenUI DSL program (the finding's own 4 prefixes are too short — 15-17
  chars — to show a quadratic-vs-linear gap), driven one character at a time
  through `set_prefix`, comparing:
  - **new**: the real (fixed) `_incremental_sync`.
  - **old**: a self-contained "forced full relex" oracle — a monkeypatched
    `_incremental_sync` reproducing the exact pre-fix method body (always
    re-lex `prefix` from 0), so no `git stash` was needed for this
    particular comparison. `git stash` *was* used separately to confirm the
    new tests fail meaningfully against the real pre-fix `engine.py` (see
    Unit tests below).
- The finding's own `prefix8`/`prefix9` cases, run through
  `build_completion_forest` with the same before/after oracle technique, to
  measure the fix's effect on the actual hot path the whole PR chain
  originates from.

## Measured

### Long growing-prefix drive (543 chars, char-by-char `set_prefix`)

| metric | before (forced full relex) | after (suffix-lex) | ratio |
| --- | ---: | ---: | ---: |
| total chars lexed | 147,696 | 49,777 | 2.97x fewer |
| wall time | 0.406s | 0.159s | 2.56x |
| `next_terminals()` mismatches across all 543 steps | — | **0** | I3/I6 held |

A synthetic scaling check (1/2/4 repeats of the same block, 543/1086/2172
chars) confirms the O(n) vs O(n²) shape: chars-lexed-per-character-of-prefix
(`chars_lexed / len`) roughly doubles each time the prefix length doubles for
the **old** path (quadratic), consistent with the pre-existing full-relex
cost; the **new** path's total chars lexed grows much closer to linearly
because settled tokens are never re-derived — see caveat below on how much
of that linear growth remains due to one specific access pattern.

### `build_completion_forest` on the finding's own `prefix8`/`prefix9`

| prefix | before wall_s | after wall_s | speedup | before chars lexed | after chars lexed | chars ratio | before `n_paths` | after `n_paths` | before coverage | after coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| prefix8 (`root = Card([b1`) | 0.5474 | 0.3668 | 1.49x | 65,230 | 44,876 | 1.45x | 2 | 2 | complete | complete |
| prefix9 (`root = Card([b1,`) | 5.8577 | 5.3739 | 1.09x | 1,014,590 | 697,043 | 1.46x | 12 | 12 | complete | complete |

`n_paths`/`coverage` are identical before vs after on both prefixes (I6/I3
held). The number of `_lex_tokens` **calls** is unchanged (1,558 / 23,546,
same as measured post-PR-#1173) — this fix does not change how often lexing
is invoked, only how much text each invocation lexes.

## Why the `build_completion_forest` speedup is modest, honestly

The long-growing-prefix synthetic drive shows a strong win (2.4-3x). The
finding's own `prefix8`/`prefix9` cases show a much smaller one (1.09-1.49x),
for two compounding reasons, both confirmed by direct investigation rather
than assumed:

1. **Short absolute prefix/branch lengths.** `prefix9` itself is 17
   characters; `build_completion_forest`'s witness search drafts branches up
   to `max_path_tokens=8` further grammar-tokens deep, but each explored
   branch's total text stays short in absolute terms. The O(n) vs O(n²) gap
   this fix targets only compounds meaningfully as `n` grows — at these
   short lengths, a "full" relex from 0 and a "suffix-only" relex from the
   last token's start aren't that different in absolute character count.
2. **`branch = OpenUIIncrementalEngine(...)` per candidate is unavoidable
   full-lex work this fix does not touch.** `compiler_draft.py`'s witness
   search creates a **brand-new** engine instance per top-level candidate
   token (`branch.set_prefix(prefix_text)` on a fresh, never-`reset()`
   engine `self._ip is None`, always takes `_full_sync`). This one-shot
   per-candidate lex of the full `prefix_text` is inherent to the witness
   search's branch-and-bound structure, not something `_incremental_sync`
   participates in — it happens once per candidate regardless of this fix.
   Only the *within-branch* deepening (`probe_chunk`/`set_prefix`/`advance`
   calls as a single branch's drafted sequence grows past its first token)
   goes through the code this fix changes, and that deepening is bounded by
   `max_path_tokens` (≤8), so its own relex savings are bounded too.

**Important, explicitly scoped limitation:** the suffix-only fast path does
**not** help the case where the *last fed token itself keeps growing one
character at a time* (e.g. `advance("T")`, `advance("Te")`, `advance("Tex")`,
…) — every such call still triggers a full `_full_sync`, identical to
pre-fix behavior, because an already-fed `InteractiveParser` token cannot be
patched in place; only re-deriving it via a full resync (which also rebuilds
`self._ip` from scratch) is currently implemented. This is not a regression
— it is exactly what `_incremental_sync` already did for this case before
this fix — but it means the byte/char-level "literal channel" fallback
described in `dsl_tokenizer.py` (`B:` tokens decode to a single character,
fed one `advance()` call at a time) gets **no** benefit from this fix. The
grammar-vocabulary path (`COMPONENT`/`NAME`/keyword pieces, which the DSL
tokenizer's "lexer-native" vocabulary usually emits as whole tokens per
`advance()` call — see `_token_piece`/`token_surface_piece`) is where this
fix's benefit concentrates, and that is exactly the path
`_forward_binder_component_requirements` and `build_completion_forest`'s
witness search predominantly exercise. See Next steps for the properly-scoped
follow-up (checkpoint/rollback of `InteractiveParser` state) that would close
the growing-boundary-token gap too.

## Unit tests (added, committed)

`tests/test_dsl/test_grammar_fastpath_incremental_lex.py` (7 tests):

- `test_fed_last_token_start_tracks_last_fed_token_offset` — the new offset
  field matches the real `start_pos` of the last fed token across a short
  drive, and resets to `None` on `reset()`.
- `test_incremental_sync_relexes_fewer_characters_than_full_relex_baseline`
  — regression guard: driving the 543-char long program char-by-character
  through the real engine re-lexes strictly (and by ≥1.5x, conservative
  floor under the ~2.97x measured) fewer total characters than a
  self-contained "forced full relex" baseline (monkeypatched
  `_incremental_sync`, not `git stash`, per the task's own guidance for a
  self-contained in-test comparison).
- `test_incremental_sync_matches_forced_full_relex_at_every_step` — I3/I6
  proof: `next_terminals()` compared at **every** one of the 543 steps
  (not just the final one) between the real engine and the forced-full-relex
  baseline; 0 mismatches. This drive exercises both the fast suffix-lex path
  and the still-full-relex growing-boundary-token fallback, since it is
  character-by-character.
- `test_build_completion_forest_unchanged_with_suffix_lex[...]`
  (parametrized on `prefix8`/`prefix9`) — I6 proof at the decode-output
  layer: `build_completion_forest`'s `coverage`/`n_paths` match between the
  real engine and the forced-full-relex baseline, and match the finding's
  original `n_paths` (2 / 12).
- `test_suffix_lex_speedup_floor_on_long_growing_prefix` — wall-time floor
  (1.3x, conservative under the ~2.4-2.56x measured locally) on the same
  long-program drive.
- `test_growing_boundary_token_falls_back_to_full_relex_and_stays_correct` —
  explicit coverage of the documented limitation above: growing a
  `COMPONENT`/`NAME` token one character at a time forces ≥1 additional
  `_full_syncs` (same as pre-fix), and the resulting fed-token-keys/accepts
  state still matches a reference engine that reached the same final text in
  one `set_prefix` call — the fallback path must stay *correct*, not merely
  "not slower than before."

```
env -u NODE_OPTIONS .venv-diag/bin/python -m pytest tests/test_dsl/test_grammar_fastpath_incremental_lex.py -v
# 7 passed in ~14s (post-fix)
```

Sanity check the tests are not vacuous: ran the identical file against the
real pre-fix `engine.py` (`git stash push -- src/slm_training/dsl/grammar/fastpath/engine.py`,
run, then `git stash pop`) — 6 of 7 fail (`AttributeError:
'OpenUIIncrementalEngine' object has no attribute '_feed_delta_tokens'` /
missing `_fed_last_token_start`, since that API doesn't exist pre-fix); the
7th (`test_growing_boundary_token_falls_back_to_full_relex_and_stays_correct`)
passes both before and after — expected, since it asserts an invariant both
versions correctly implement (the fallback-path *correctness*, not this
fix's *optimization*), the same category PR #1173 documented for its own
`build_completion_forest`-invariant tests.

## Regression sweep

Same suites, environment, and 3-minute (`MAX_RUN_MINUTES`) hard cap as
PR #1173's sweep:

| suite | result |
| --- | --- |
| `tests/test_dsl/test_grammar_fastpath_lexer_cache.py` (PR #1173's own tests) | 6 passed, unaffected |
| `tests/test_dsl/{test_pack,test_speculative_rank,test_exact_forced_horizon,test_lattice_search,test_solver_state,test_packs,test_solver_decode}.py` | 102 passed, 3 failed |
| `tests/test_dsl/test_grammar_fastpath.py -k "lexer or force or completion_forest or engine or pick_constrained"` | 24 passed, 1 failed, 18 deselected |
| `tests/test_harnesses/model_build/{test_lexer_smoke,test_dsl_tokenizer}.py` | 27 passed, 1 failed, 1 deselected |
| `tests/test_harnesses/model_build/{test_grammar_hf,test_v4_levers,test_a3_coverage_remask,test_template_fill,test_v6_remask}.py` | 30 passed, 1 failed, 2 deselected |

**All failures are pre-existing, confirmed by rerunning against unmodified
`engine.py`** (`git stash push -- src/slm_training/dsl/grammar/fastpath/engine.py`
/ rerun / `git stash pop`) **this session**:

- `test_speculative_rank.py::test_committed_table_matches_its_builder` and
  `::test_committed_table_ranks_real_branch_points_confidently` — same as
  PR #1173 documented (stale committed `speculative_ngram_v1.json`).
- `test_packs.py::test_pack_fixture_loop_generate_train_eval` — **new to
  this sweep's suite selection** (PR #1173's sweep didn't happen to hit
  this test/failure combination in the same grouping); confirmed identical
  on unmodified `engine.py` (`ValueError: grammar_constrained=False is
  unsafe for OpenUI generation` — a `TwoTowerModel.generate_batch_requests`
  config-validation error, unrelated to lexing).
- `test_grammar_fastpath.py::test_completion_forest_admits_null_for_optional_string_after_slots_exhaust`
  — same as PR #1173 documented (`build_completion_forest()` signature
  drift, unrelated to lexing).
- `test_lexer_smoke.py::test_surface_identifier_arm_is_prohibited` — same as
  PR #1173 documented.
- `test_v4_levers.py::test_generate_batch_requests_consumes_harness_slot_contract`
  — same assertion PR #1173 flagged but could not confirm in isolation
  within the run cap; **this session's isolated rerun completed in 38.81s
  (under the cap) against unmodified `engine.py` and reproduced the
  identical failure**, closing that open caveat from PR #1173's doc.

`ruff check` on both changed files (`engine.py`, the new test file): clean.
`python -m scripts.verify_version_stamps --check`: `2 changed file(s), 0
component(s) touched` — `fastpath/engine.py` is still not a registered path
in any `versions.json` component, so no bump is required; the new test file
is untracked-added, not a watched path either.
`python -m scripts.repo_policy`: `ok (tracked + untracked)`.
`git diff --check`: clean (no whitespace errors).

## Interpretation

Closes "Next steps" item 2 from the lexer-cache fix. The suffix-only relex is
correct (I3/I6 proven at both the `next_terminals()` layer, across every
step of a long incremental drive, and the `build_completion_forest`
decode-output layer) and measurably reduces lex work (2.97x fewer characters
lexed, up to 2.56x wall time, on a long growing prefix; 1.45-1.46x fewer
characters, 1.09-1.49x wall time, on the finding's own short prefixes). The
gap between the strong synthetic result and the modest finding-prefix result
is explained, not hidden: short absolute prefix/branch lengths in the
witness search's bounded (`max_path_tokens≤8`) deepening, plus a large,
unavoidable per-candidate full-lex cost this fix doesn't touch
(`OpenUIIncrementalEngine(...)` constructed fresh per top-level candidate).

## Scope note (what this does NOT establish)

- No train/eval run, no checkpoint, no `--ship-gates` scoreboard — same tier
  as the rest of this PR chain.
- **Does not help the growing-boundary-token case** (single character/byte
  appended to the last fed token repeatedly, e.g. literal-content decode via
  the DSL tokenizer's byte/char fallback channel) — every such call still
  triggers a full `_full_sync`, identical cost to pre-fix. See Next steps.
- Safety of slicing `prefix[safe_offset:]` into a **fresh string** (not a
  zero-copy view with left context) currently relies on this grammar's
  terminals containing no lookbehind/`\b`-style regex assertions — verified
  true today by reading every terminal in `openui.lark`, but not enforced by
  a runtime check. A future terminal added with such an assertion would
  silently reintroduce a correctness risk this fix's I3/I6 tests would not
  necessarily catch (they exercise the *current* grammar only).
- `probe_chunk` (`engine.py`, untouched) still fully re-lexes `new_prefix`
  from 0 on every call — explicitly out of scope per the task ("Next steps"
  item 2 names `_incremental_sync` specifically).
- The `test_packs.py::test_pack_fixture_loop_generate_train_eval` failure is
  newly visible in this session's specific suite grouping/run, but confirmed
  pre-existing (reproduces identically on unmodified `engine.py`); it is not
  a new failure caused by this fix.

## Next steps

1. Close the growing-boundary-token gap: maintain a "pre-boundary"
   `InteractiveParser` checkpoint (`self._ip.copy()`, the same primitive
   `probe_chunk` already uses) taken *before* each token is fed. When the
   boundary token's identity changes, restore that checkpoint and re-feed
   only the corrected boundary token plus whatever follows, instead of a
   full `_full_sync` (which currently also re-derives and re-feeds every
   earlier token from scratch). This is a larger, riskier change (adds a
   `.copy()` to the hot feed path, more bookkeeping state, more edge cases
   around `UnexpectedToken`/`UnexpectedEOF` mid-restore) and deserves its
   own dedicated finding/fix pair rather than folding into this session.
2. Re-measure against the full per-record `compiler_ms` telemetry / the
   `exposure12` quality-champion eval protocol (still not done, same caveat
   PR #1173 carried forward) now that both the lexer-object cache and this
   suffix-lex change are in place.
3. If a future terminal is added to `openui.lark` using a lookbehind/`\b`
   assertion, revisit the "Why this is safe" section's second grammar-fact
   claim before relying on suffix-only lexing remaining correct.

## Cleanup note

`.venv-diag`, `src/apps/openui_bridge/node_modules/` (both gitignored) and
the scratch benchmark/diagnostic scripts (`before_after_incremental.py`,
`scaling_check.py`, `diag_fullsync.py`, `diag2.py`, `forest_bench.py`,
session scratchpad, not reusable harness tools) are not committed.

Captured: 2026-07-27T22:46:44Z
