# Fix: suffix-only relex in `probe_chunk` (`OpenUIIncrementalEngine`)

**Honesty:** `fixture_or_scratch`. This is a real code change (`is_fix: true`)
with a benchmark-backed unit test, but **not a ship claim** — no
`train_model`/`evaluate_model --ship-gates` run, no checkpoint, no scoreboard.
Same status tier as the chain it continues:
[`decode-compiler-tree-incremental-lex-fix.md`](decode-compiler-tree-incremental-lex-fix.md)
(PR #1174) →
[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173) →
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md)
(PR #1172).

## Task

PR #1174's "Scope note" flagged `probe_chunk` (`engine.py:207` at the time)
as explicitly out of scope: "still fully re-lexes `new_prefix` from 0 on
every call". This session closes that gap by applying the identical
suffix-anchor strategy PR #1174 used for `_incremental_sync`, adapted for
`probe_chunk`'s read-only contract (it must never mutate engine state).

## What changed

`src/slm_training/dsl/grammar/fastpath/engine.py`:

1. New method `_probe_delta_from_anchor(self, chunk)`: when
   `self._fed_token_count > 0` and `self._fed_last_token_start is not None`
   (the same anchor `_incremental_sync` uses), re-lexes only
   `self._prefix[safe_offset:] + chunk` instead of the whole
   `self._prefix + chunk`. Compares the re-lexed suffix's first token
   against the previously-fed boundary token (`self._fed_tokens[boundary_idx]`);
   on a match, returns the not-yet-fed delta tokens (`suffix_tokens[1:]`).
   Returns `None` — never guesses — when there's no anchor yet, the suffix
   fails to lex, or the boundary token's identity changed (still growing,
   e.g. `"Te"` → `"Text"`).
2. Deliberately does **not** reconstruct the full fed-token-key list the
   way `_incremental_sync`'s anchor path does: `probe_chunk` only needs the
   delta tokens (to feed a throwaway `InteractiveParser.copy()`), never the
   keys after computing them, so building `self._fed_tokens[:boundary_idx] +
   suffix_keys` would be a pure O(`fed_token_count`) cost with no
   corresponding benefit — reintroducing linear-per-probe overhead the fix
   is meant to remove.
3. `probe_chunk` now calls `_probe_delta_from_anchor(chunk)` first; on
   success uses its delta directly, otherwise falls back to the exact
   pre-existing full-prefix `_lex_tokens(self._prefix + chunk)` path
   (identical `len(tokens) < fed_token_count` / key-prefix-mismatch checks,
   identical `_copy_probe_fallbacks` bookkeeping).

No change to `probe_chunk`'s external contract (`bool | None` semantics,
never mutates `self`), `_incremental_sync`, `_full_sync`, `_sync`, or any
grammar/terminal definition.

## Why this is safe

Identical maximal-munch argument as PR #1174's `_incremental_sync` fix,
restated for `probe_chunk`'s narrower job: every already-fed token except
possibly the last one is permanently settled (its match depended only on
characters that are unchanged when a chunk is appended after the current
prefix). `self._fed_last_token_start` still names that one uncertain
position; re-lexing `self._prefix[safe_offset:] + chunk` re-derives the
boundary token and everything genuinely new after it, exactly as
`self._lex_tokens(self._prefix + chunk)` would, without re-deriving tokens
that were never in doubt. The two grammar facts PR #1174 verified directly
(`Token.start_pos`/`end_pos` populated independent of `propagate_positions`;
no `openui.lark` terminal uses a lookbehind/`\b` assertion) are unchanged by
this session and apply identically here — no new verification was needed,
this fix reuses the same anchor invariant, not a new one.

On any anchor failure — no anchor yet, suffix lex failure, or a changed
boundary-token identity — `probe_chunk` falls back to the exact
pre-existing full-prefix lex, which independently re-derives the correct
answer from scratch. So whenever the anchor path *does* return a value, it
is provably identical to what the full-prefix path would have returned;
the anchor never gets a chance to be "confidently wrong" — it only ever
short-circuits work the full path would have reached the same conclusion
on, or defers to that full path.

`probe_chunk` never mutates `self._prefix`, `self._fed_token_count`,
`self._fed_tokens`, or `self._fed_last_token_start` — the anchor helper
reads that state but writes nothing, matching the pre-existing contract
that repeated `probe_chunk` calls against the same synced prefix are
side-effect-free and idempotent.

## Reproduction

Local `.venv` (Python 3.12.3, this session — no `.venv-diag` naming
convention needed since no prior venv existed in this container):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
```

The realistic drive mirrors `compiler_draft.py`'s actual `probe_chunk`
usage (`branch.probe_chunk(piece)` then `branch.advance(piece)` on a
`True`/`None` result) rather than a synthetic pattern: at each step, probe
the next character against the currently-synced prefix without committing,
record the result, then commit it via `set_prefix` before probing the next
character. Driven against the same 543-char, 10-line, multi-statement valid
OpenUI DSL program `LONG_PROGRAM` PR #1174's tests use, for direct
comparability.

"Before" oracle: a monkeypatched `_probe_delta_from_anchor` that always
returns `None`, forcing every probe through the pre-existing full-prefix
path — self-contained, no `git stash` needed for the perf/I3-I6 comparison
itself (same technique PR #1174 used). `git stash` was used separately to
confirm the new unit tests are not vacuous.

## Measured

### Long growing-prefix drive (543 chars, char-by-char probe-then-commit)

| metric | before (forced full relex) | after (suffix-anchor) | ratio |
| --- | ---: | ---: | ---: |
| total chars lexed | 197,473 | 99,554 | 1.98x fewer |
| wall time (median of 5 runs) | 0.349s | 0.233s | 1.50x |
| probe result mismatches across all 543 steps | — | **0** | I3/I6 held |

## Why the speedup is smaller than `_incremental_sync`'s, honestly

PR #1174 measured 2.97x fewer characters / 2.56x wall time for the
analogous `_incremental_sync` fix on the identical `LONG_PROGRAM` drive.
This fix measures a smaller 1.98x / 1.50x, for two reasons, both confirmed
by direct comparison rather than assumed:

1. **`probe_chunk`'s per-call chunk is a single character in this drive**
   (`program[i]`), the smallest possible unit — so the suffix re-lexed on
   each call (`self._prefix[safe_offset:] + chunk`) already starts from a
   position close to the end of the current prefix on both the anchor and
   (for a single trailing token) full paths, narrowing the absolute
   character-count gap between them compared to `_incremental_sync`'s
   drive, which re-lexes larger deltas relative to the anchor position at
   comparable points because `set_prefix` is called with the same growing
   substrings.
2. **`InteractiveParser.copy()` and the subsequent delta feed onto the
   copy are fixed per-call costs `_incremental_sync` doesn't pay** (it
   mutates `self._ip` directly instead of snapshotting it). These are
   unaffected by this fix and dilute the relative wall-time win from the
   reduced lex work, since they're a larger fraction of each call's total
   cost than in `_incremental_sync`.

Both reasons are inherent to what `probe_chunk` is for (a cheap, mutation-free
probe primitive used inside `compiler_draft.py`'s witness search) — not
artifacts of an incomplete fix. The gap is real and reported here rather
than only showing the more flattering number.

## Unit tests (added, committed)

`tests/test_dsl/test_grammar_fastpath_probe_chunk_suffix_lex.py` (6 tests):

- `test_probe_chunk_relexes_fewer_characters_than_full_relex_baseline` —
  regression guard: probing the 543-char long program char-by-character
  through the real engine re-lexes strictly (and by ≥1.5x, conservative
  floor under the ~1.98x measured) fewer total characters than the
  self-contained forced-full-relex baseline.
- `test_probe_chunk_matches_forced_full_relex_at_every_step` — I3/I6 proof:
  `probe_chunk`'s actual decode-facing return value compared at **every**
  one of the 543 steps (not just the final one) between the real engine and
  the forced-full-relex baseline; 0 mismatches. Exercises both the fast
  anchor path and the still-full-relex growing-boundary-token fallback,
  since the drive is character-by-character.
- `test_probe_chunk_speedup_floor_on_long_growing_prefix` — wall-time floor
  (1.2x, conservative under the ~1.50x measured locally, and deliberately
  tighter than PR #1174's 1.3x floor given the smaller measured margin
  explained above).
- `test_growing_boundary_token_probe_falls_back_and_stays_correct` —
  explicit coverage of the shared limitation: growing a `COMPONENT`/`NAME`
  token one character at a time forces the anchor to decline (falls back to
  full lex, same as pre-fix); probe results must still match an
  anchor-disabled reference engine exactly through that fallback.
- `test_probe_chunk_anchor_unused_before_anything_fed` — before any token
  is fed, `_probe_delta_from_anchor` has no anchor and must return `None`,
  so `probe_chunk` takes the pre-existing full-prefix path unchanged.

```
.venv/bin/python -m pytest tests/test_dsl/test_grammar_fastpath_probe_chunk_suffix_lex.py -v
# 5 passed in ~2s (post-fix)
```

Sanity check the tests are not vacuous: ran the identical file against the
real pre-fix `engine.py` (`git stash push -- src/slm_training/dsl/grammar/fastpath/engine.py`,
run, then `git stash pop`) — all 5 fail with
`AttributeError: 'OpenUIIncrementalEngine' object has no attribute
'_probe_delta_from_anchor'`, since that method doesn't exist pre-fix (unlike
PR #1174's sweep, there is no test in this file that is expected to pass
both before and after, since every test either calls
`_probe_delta_from_anchor` directly or relies on monkeypatching it).

## Regression sweep

Same 3-minute (`MAX_RUN_MINUTES`) hard cap discipline as PR #1174's sweep.

| suite | result |
| --- | --- |
| `tests/test_dsl/test_grammar_fastpath_probe_chunk_suffix_lex.py` (this fix's own tests) | 5 passed |
| `tests/test_dsl/{test_grammar_fastpath_incremental_lex,test_grammar_fastpath_lexer_cache}.py` (PR #1174/#1173's tests) | 13 passed, unaffected |
| `tests/test_dsl/{test_pack,test_packs}.py` | 20 passed, 6 failed |
| `tests/test_dsl/test_grammar_fastpath.py` (excluding 2 pre-existing slow smoke tests + 1 pre-existing hang, see below) | 39 passed, 1 failed |

**All failures are pre-existing, confirmed by rerunning against unmodified
`engine.py`** (`git stash push -- src/slm_training/dsl/grammar/fastpath/engine.py`
/ rerun / `git stash pop`) **this session**:

- `test_pack.py::{test_openui_oracle_accepts_source_and_fails_closed,
  test_verify_stack_verdicts_unchanged,
  test_end_to_end_fixture_run_through_pack_interface}` and
  `test_packs.py::{test_pack_contract_invariants[openui],
  test_openui_pack_placeholder_policy,
  test_pack_fixture_loop_generate_train_eval}` — all 6 fail identically on
  unmodified `engine.py` with `ValueError: generated ProgramSpec failed F2
  at G2` (a `progspec/generate.py` verifier-gate issue, unrelated to
  lexing).
- `test_grammar_fastpath.py::test_completion_forest_admits_null_for_optional_string_after_slots_exhaust`
  — same `build_completion_forest()` signature-drift failure PR #1174
  documented (`enforce_schema_component_types` keyword no longer accepted),
  unrelated to lexing.

**One test in `test_grammar_fastpath.py` hangs past the 3-minute run cap on
unmodified `engine.py` too** —
`test_semantic_guards_run_before_singleton_bypass` — confirmed by running it
in isolation against `git stash`-ed (pre-this-fix) `engine.py`: still times
out at 60s with zero output past `collecting ...`. Per the hard-run-cap rule
("a timed-out, interrupted, or killed run is never evidence"), this hang is
recorded as a pre-existing, unrelated-to-this-fix condition rather than
treated as a pass or a fail; it was deselected (`-k "not
semantic_guards_run_before_singleton_bypass"`) for the rest of this file's
sweep. `test_train_fuse_and_cache_smoke` and `test_cactus_kernel_sketch_files_exist`
were also deselected up front as known-slow smoke tests outside this fix's
scope (same category PR #1174's sweep avoided by scoping its `-k`
selection).

`ruff check` on both changed files (`engine.py`, the new test file): clean.
`python -m scripts.verify_version_stamps --check`: `2 changed file(s), 0
component(s) touched` — `fastpath/engine.py` is still not a registered path
in any `versions.json` component, so no bump is required.
`python -m scripts.repo_policy`: `ok (tracked + untracked)`.
`git diff --check`: clean (no whitespace errors).

## Interpretation

Closes PR #1174's "Scope note" gap: `probe_chunk` no longer fully re-lexes
`self._prefix + chunk` from position 0 on every call on its common path. The
suffix-anchor relex is correct (I3/I6 proven at every step of a long
incremental probe-then-commit drive, 0 mismatches against a forced-full-relex
oracle) and measurably reduces lex work (1.98x fewer characters lexed, 1.50x
wall time on the shared `LONG_PROGRAM` drive) — a real but smaller win than
`_incremental_sync`'s, for reasons specific to `probe_chunk`'s single-char
per-call usage pattern and its `InteractiveParser.copy()` overhead, explained
above rather than hidden.

## Scope note (what this does NOT establish)

- No train/eval run, no checkpoint, no `--ship-gates` scoreboard — same tier
  as the rest of this PR chain.
- **Does not help the growing-boundary-token case**, identical limitation to
  PR #1174's fix and for the identical reason (an already-fed token's
  identity can't be patched in place without a full resync).
- Relies on the same "no lookbehind/`\b` terminal in `openui.lark`" fact
  PR #1174 verified directly — not re-verified here since nothing about the
  grammar changed, but inherits the same caveat: a future terminal using
  such an assertion would need this assumption revisited.
- Does not touch `_incremental_sync`, `_full_sync`, or `_feed_delta_tokens`
  — those are unchanged from PR #1174.
- Not re-measured against full per-record `compiler_ms` telemetry or the
  `exposure12` quality-champion eval protocol — same open item PR #1173 and
  PR #1174 both carried forward.

## Next steps

1. Re-measure `probe_chunk`'s real-world call-site benefit inside
   `compiler_draft.py`'s witness search (where chunks are whole
   lexer-native token pieces via `_token_piece`, not single characters as
   in this session's synthetic drive) — the per-call chunk size there is
   typically larger than one character, which may shift the
   anchor-vs-`InteractiveParser.copy()`-overhead balance measured here.
2. Close the shared growing-boundary-token gap for both `probe_chunk` and
   `_incremental_sync` together via the `InteractiveParser`
   checkpoint/rollback approach PR #1174's Next steps already scoped (a
   larger, riskier change deserving its own finding/fix pair).
3. Re-measure against the full per-record `compiler_ms` telemetry / the
   `exposure12` quality-champion eval protocol now that the lexer-object
   cache (#1173), `_incremental_sync` suffix-lex (#1174), and this
   `probe_chunk` suffix-lex fix are all in place.

## Cleanup note

`.venv` (gitignored) created for this session is not committed; no scratch
diagnostic scripts were written outside the committed test file — the
measurement snippet used inline `python -c` invocations against the
committed test module's own helpers (`LONG_PROGRAM`, `_drive_probe_and_commit`,
`_disabled_anchor`, `_CharCountingEngine`), not a separate uncommitted
script.

Captured: 2026-07-27T23:49:30Z
