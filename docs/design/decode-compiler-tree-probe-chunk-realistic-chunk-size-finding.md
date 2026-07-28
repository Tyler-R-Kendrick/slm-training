# Finding: `probe_chunk`'s suffix-anchor fix wins *more*, not less, at realistic chunk sizes

**Honesty:** `fixture_or_scratch`. Finding only — no code change (`is_fix:
false`), no train/eval run, no checkpoint, no `--ship-gates` scoreboard.
Continues the chain:
[`decode-compiler-tree-probe-chunk-suffix-lex-fix.md`](decode-compiler-tree-probe-chunk-suffix-lex-fix.md)
(PR #1176) →
[`decode-compiler-tree-incremental-lex-fix.md`](decode-compiler-tree-incremental-lex-fix.md)
(PR #1174) →
[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173).

## Task

PR #1176's "Next steps" item 1 flagged an open question: its `probe_chunk`
suffix-anchor benchmark drove the engine character-by-character (a synthetic
pattern), while the real call site
(`src/slm_training/dsl/grammar/fastpath/compiler_draft.py`'s
`build_completion_forest` witness search, `branch.probe_chunk(piece)` /
`branch.advance(piece)`) probes with **tokenizer surface pieces** via
`token_surface_piece` — typically several characters at once, not one. PR
#1176 noted this "may shift the anchor-vs-`InteractiveParser.copy()`-overhead
balance measured" in its char-by-char drive. This session re-measures with
the real chunk-size distribution to close that gap.

## What changed

Nothing in `engine.py`. This session adds
`tests/test_dsl/test_grammar_fastpath_probe_chunk_realistic_chunk_size.py`
(4 tests) that:

1. Encode the same 10-line, multi-statement `LONG_PROGRAM` PRs #1174/#1176
   used (one literal adjusted: `Slider`'s first argument is the placeholder
   `":slot_level"` instead of the free-form `"$level"`, since
   `DSLNativeTokenizer.encode` rejects non-placeholder, non-registered
   literal bodies) through the real `DSLNativeTokenizer`.
2. Reproduce `build_completion_forest`'s exact per-candidate loop shape:
   skip `LIT_STR`/`LIT_NUM`/`LIT_END` marker tokens (the real call site
   never probes literal-frame content directly — it treats those three as
   frame-open/close markers and `continue`s past them), and for every other
   token compute its surface piece via `token_surface_piece`.
3. Drive `probe_chunk`/`advance`/`set_prefix` over that realistic chunk
   stream (152 chunks, average 2.47 chars, max 12 — vs. 543 single-char
   chunks in PR #1176's drive) and compare against the same
   forced-full-relex oracle technique (`_probe_delta_from_anchor`
   monkeypatched to always return `None`) PRs #1174/#1176 used.

## Measured

### Realistic tokenizer-piece chunk drive (152 chunks, avg 2.47 chars, LONG_PROGRAM)

| metric | before (forced full relex) | after (suffix-anchor) | ratio |
| --- | ---: | ---: | ---: |
| total chars lexed | 29,539 | 4,054 | **7.29x fewer** |
| wall time (median of 5 runs) | 0.0929s | 0.0594s | **1.56x** |
| probe result mismatches across all 152 steps | — | **0** | I3/I6 held |

### Compared to PR #1176's char-by-char drive (543 chunks, 1 char each, same LONG_PROGRAM text)

| metric | char-by-char (PR #1176) | realistic pieces (this finding) |
| --- | ---: | ---: |
| chars-lexed reduction | 1.98x | **7.29x** |
| wall-time speedup | 1.50x | 1.56x |

## Interpretation

The open question is answered: at the real call site's actual chunk
granularity, the suffix-anchor fix's **chars-lexed reduction is far larger**
(7.29x vs. 1.98x), not smaller as PR #1176 speculated might happen. The
mechanism is the inverse of what narrowed PR #1176's own margin relative to
PR #1174's `_incremental_sync` fix: fewer, larger probe calls mean each call
re-lexes a *larger absolute delta* on the full-relex baseline (which
re-lexes the *entire* growing prefix every time) while the anchor path's
suffix re-lex stays anchored to the same one boundary token regardless of
chunk size — so the gap between "re-lex everything" and "re-lex from the
last fed token" widens, not narrows, as chunks get bigger and calls get
fewer.

**Wall-time speedup barely moves** (1.56x vs. 1.50x) despite the much larger
chars-lexed win, for the same reason PR #1176 already documented:
`InteractiveParser.copy()` and the delta-feed onto that copy are a fixed
per-call cost neither this fix nor chunk size touches, and with only 152
calls (vs. 543) that fixed cost is a *larger* fraction of each call's total
time, diluting the relative wall-time benefit of the (now much bigger)
reduction in lex work. Both directions of this tradeoff are measured and
reported, not assumed.

## Unit tests (added, committed)

`tests/test_dsl/test_grammar_fastpath_probe_chunk_realistic_chunk_size.py`
(4 tests):

- `test_realistic_chunks_are_multi_character_not_single_character` — sanity
  check on this finding's premise (average chunk length 2.47 chars, floor
  asserted at >1.5).
- `test_realistic_chunks_relex_fewer_characters_than_full_relex_baseline` —
  regression guard, floor at 4x (conservative under the ~7.29x measured).
- `test_realistic_chunk_probe_matches_forced_full_relex_at_every_step` —
  I3/I6 proof: 0 mismatches across all 152 realistic-chunk steps.
- `test_realistic_chunk_speedup_floor_on_long_program` — wall-time floor
  1.2x (matches PR #1176's own floor; conservative under the ~1.56x
  measured).

```
.venv/bin/python -m pytest tests/test_dsl/test_grammar_fastpath_probe_chunk_realistic_chunk_size.py -v
# 4 passed in ~1.2s
```

Non-vacuous check: ran the identical file against `engine.py` checked out at
commit `20b658f` (the state before PRs #1174/#1176's suffix-anchor fixes,
via `git show 20b658f:...engine.py` swapped in and restored after) — 3 of 4
fail with `AttributeError: type object 'OpenUIIncrementalEngine' has no
attribute '_probe_delta_from_anchor'` (the 4th, the chunk-shape sanity
check, is independent of the fix and passes either way, matching PR #1176's
own file's convention that no comparison test is expected to pass both
before and after).

## Regression sweep

Same 3-minute (`MAX_RUN_MINUTES`) hard-cap discipline as the rest of this
chain.

| suite | result |
| --- | --- |
| `tests/test_dsl/test_grammar_fastpath_probe_chunk_realistic_chunk_size.py` (this finding's own tests) | 4 passed |
| `tests/test_dsl/{test_grammar_fastpath_probe_chunk_suffix_lex,test_grammar_fastpath_incremental_lex,test_grammar_fastpath_lexer_cache}.py` | 22 passed, unaffected |
| `tests/test_dsl/{test_pack,test_packs}.py` | 20 passed, 6 failed |
| `tests/test_dsl/test_grammar_fastpath.py` (excluding the 2 pre-existing slow smoke tests + 1 pre-existing hang) | 39 passed, 1 failed, 3 deselected |

All 6 + 1 failures are the exact same pre-existing failures PRs #1174/#1176
already documented (`ValueError: generated ProgramSpec failed F2 at G2` in
`test_pack.py`/`test_packs.py`; `build_completion_forest()`
`enforce_schema_component_types` signature drift in
`test_grammar_fastpath.py`) — unrelated to this change (which touches no
production code), reconfirmed present at the identical failure sites this
session.

`ruff check` on the new test file: clean.
`python -m scripts.verify_version_stamps --check`: `1 changed file(s), 0
component(s) touched` — a test-only addition, no `versions.json`-registered
component path touched.
`python -m scripts.repo_policy`: `ok (tracked + untracked)`.
`git diff --check`: clean.

## Scope note (what this does NOT establish)

- No production code change — `engine.py`'s suffix-anchor fixes (#1173,
  #1174, #1176) are unmodified; this is a measurement against the code as
  it already stands.
- Does not touch the shared growing-boundary-token gap (`_incremental_sync`
  and `probe_chunk` both still fall back to a full relex when the last-fed
  token keeps growing one character at a time) — still open, per PR #1176's
  next step 2 (`InteractiveParser` checkpoint/rollback).
- Not re-measured against the full per-record `compiler_ms` telemetry or
  the `exposure12` quality-champion eval protocol — still open, carried
  forward from PRs #1171/#1173/#1174/#1176.
- `LONG_PROGRAM`'s realistic chunk stream (152 chunks, 1 candidate sequence)
  is a single straight-line drive, not `build_completion_forest`'s actual
  multi-candidate fan-out (one `OpenUIIncrementalEngine` branch per
  candidate token, most of which are rejected quickly). The *per-call*
  chunk-size finding here transfers directly (`_token_piece`/`probe_chunk`
  call shape is identical per branch), but the *aggregate* wall-clock
  benefit across a full multi-candidate witness search is not measured in
  this session.

## Next steps

1. Measure the aggregate benefit across `build_completion_forest`'s actual
   multi-candidate fan-out (many short-lived branches, most rejected within
   1-2 probes) rather than this session's single straight-line candidate —
   the per-call chunk-size finding here should transfer, but the aggregate
   speedup across a realistic candidate set (dominated by short, quickly
   rejected branches rather than long admitted ones) is a different
   quantity and is unmeasured.
2. Close the shared growing-boundary-token gap for both `probe_chunk` and
   `_incremental_sync` via the `InteractiveParser` checkpoint/rollback
   approach PR #1174's next steps scoped — still open.
3. Re-measure against the full per-record `compiler_ms` telemetry / the
   `exposure12` quality-champion eval protocol now that the lexer-object
   cache (#1173), `_incremental_sync` suffix-lex (#1174), and the
   `probe_chunk` suffix-lex fix (#1176) are all in place — still open,
   carried forward across this entire chain.

## Cleanup note

`.venv` (gitignored) created for this session is not committed; no scratch
diagnostic scripts were written outside the committed test file — the
measurement snippet used inline `python -c` invocations against the
committed test module's own helpers (`LONG_PROGRAM`, `_realistic_chunks`,
`_drive_realistic_chunks`, `_CharCountingEngine`, `_disabled_anchor`), not a
separate uncommitted script.

Captured: 2026-07-28T00:00:00Z
