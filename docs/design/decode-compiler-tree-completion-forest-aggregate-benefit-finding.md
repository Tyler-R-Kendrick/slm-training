# Finding: the suffix-anchor fixes' real win is ~2.4x/~1.15x in aggregate, not 7.29x/1.56x

**Honesty:** `fixture_or_scratch`. Finding only — no code change (`is_fix:
false`), no train/eval run, no checkpoint, no `--ship-gates` scoreboard.
Continues the chain:
[`decode-compiler-tree-probe-chunk-realistic-chunk-size-finding.md`](decode-compiler-tree-probe-chunk-realistic-chunk-size-finding.md)
(PR #1177) →
[`decode-compiler-tree-probe-chunk-suffix-lex-fix.md`](decode-compiler-tree-probe-chunk-suffix-lex-fix.md)
(PR #1176) →
[`decode-compiler-tree-incremental-lex-fix.md`](decode-compiler-tree-incremental-lex-fix.md)
(PR #1174) →
[`decode-compiler-tree-lexer-cache-fix.md`](decode-compiler-tree-lexer-cache-fix.md)
(PR #1173).

## Task

PR #1177's own "Scope note" and "Next steps" item 1 flagged an explicit,
unmeasured gap: its realistic-chunk-size benchmark drove a single
straight-line candidate (`LONG_PROGRAM`, one token sequence), but the real
`build_completion_forest` call site fans out into **many short-lived
`OpenUIIncrementalEngine` branches per call** — `compiler_draft.py`'s
per-candidate loop constructs a fresh branch engine for every candidate
token, most rejected within 1-2 probes, and `pack.py`'s recursive
`_tail_from` witness search multiplies this further. PR #1177 explicitly
said: "the aggregate speedup across a realistic candidate set ... is a
different quantity and is unmeasured." This session measures it.

## What changed

Nothing in `engine.py`. This session adds
`tests/test_dsl/test_grammar_fastpath_completion_forest_aggregate_benefit.py`
(7 tests) that:

1. Drive the **real** `build_completion_forest` entry point on PRs
   #1171/#1174/#1176's `PREFIX8`/`PREFIX9` fixtures, through `pack.py`'s
   `remaining_tokens`-bounded path — the path that actually exercises the
   recursive multi-candidate `_tail_from` search, not a hand-rolled
   single-candidate drive.
2. Install a per-engine-construction counter
   (`OpenUIIncrementalEngine.__init__`) and a `_lex_tokens` char counter,
   comparing the real suffix-anchor fixes (#1174 `_incremental_sync`, #1176
   `probe_chunk`) against a forced "before" oracle with both
   `_incremental_sync` and `_probe_delta_from_anchor` disabled — the
   identical monkeypatch technique PRs #1174/#1176/#1177 used.
3. Confirm the fan-out premise directly, not by assumption: `PREFIX8`
   constructs 508 branch engines per call; `PREFIX9` constructs 7,661.

## Measured

### PREFIX8 (`"root = Card([b1"`, 2-path forest, 508 branch engines/call)

| metric | before (forced full relex) | after (suffix-anchor) | ratio |
| --- | ---: | ---: | ---: |
| total chars lexed | 65,230 | 27,102 | **2.41x** |

### PREFIX9 (`"root = Card([b1,"`, 12-path forest, 7,661 branch engines/call)

| metric | before (forced full relex) | after (suffix-anchor) | ratio |
| --- | ---: | ---: | ---: |
| total chars lexed | 1,014,590 | 416,458 | **2.44x** |
| total `_lex_tokens` calls (both variants) | 23,546 | 23,546 | — |
| wall time, 3 reps | [5.651, 5.724, 5.800]s | [4.840, 4.865, 5.014]s | **~1.15x** (median) |

`coverage`/`n_paths` are identical before vs. after on both prefixes (I3/I6
held, across the *entire* multi-candidate fan-out — not just the one
straight-line candidate PRs #1174/#1176 already proved this for).

### Compared to PR #1177's single-candidate drive

| metric | PR #1177 (1 candidate, `LONG_PROGRAM`) | this finding (aggregate, `PREFIX9`) |
| --- | ---: | ---: |
| chars-lexed reduction | 7.29x | **2.44x** |
| wall-time speedup | 1.56x | **~1.15x** |

## Interpretation

PR #1177's open question is answered: the aggregate benefit across
`build_completion_forest`'s real multi-candidate fan-out is **real and
reproducible** (consistent ~2.4x chars-lexed reduction on both `PREFIX8`
and `PREFIX9`; ~1.15x wall time on `PREFIX9`, with no overlap across 3 reps
each side), but **far smaller** than the 7.29x/1.56x figures measured on a
single straight-line candidate.

**Root cause, verified not assumed:** `build_completion_forest`'s
per-candidate loop constructs a *brand-new* `OpenUIIncrementalEngine` for
every candidate token (`compiler_draft.py`: `branch =
OpenUIIncrementalEngine(engine.grammar_path)`). Every branch therefore
starts **anchor-less** (`_fed_last_token_start is None`) — its first
`probe_chunk`/`_incremental_sync` call cannot take the suffix-anchor fast
path regardless of the fix, by construction, since there is no prior fed
token to anchor from. With 7,661 branches averaging only ~3.07 lex calls
each before rejection (23,546 total lex calls ÷ 7,661 engines), most of a
branch's short life is spent on calls that get, at best, one or two
suffix-anchored calls *after* its mandatory anchor-less first one. That
dilutes the fix's per-call chars-lexed win from ~7.29x down to ~2.4x in
aggregate, and dilutes `InteractiveParser.copy()`'s already-fixed per-call
overhead (unaffected by either fix, unaffected by chunk size) down to a
~1.15x aggregate wall-time gain — the same overhead-dilution mechanism PRs
#1176/#1177 already documented, now shown to bite harder in aggregate
because most branches are even shorter-lived than a single long candidate
drive.

## Unit tests (added, committed)

`tests/test_dsl/test_grammar_fastpath_completion_forest_aggregate_benefit.py`
(7 tests, parametrized over `PREFIX8`/`PREFIX9` where applicable):

- `test_multi_candidate_fan_out_creates_many_branch_engines` — sanity check
  on this module's premise (>50 branch engines constructed per call; floor
  kept far under the 508/7,661 measured).
- `test_build_completion_forest_result_unchanged_with_both_fixes_disabled`
  — I3/I6 proof across the full fan-out, not just one candidate.
- `test_aggregate_chars_lexed_reduced_but_far_below_single_candidate_ratio`
  — regression guard (floor 1.8x, conservative under the ~2.4x measured)
  *and* an explicit ceiling check (<5x) proving the aggregate ratio really
  is smaller than PR #1177's 7.29x figure, not merely unmeasured.
- `test_aggregate_wall_time_speedup_floor_on_prefix9` — median-of-3 wall
  time floor (1.05x, conservative under the ~1.15-1.2x measured).

```
.venv-diag/bin/python -m pytest tests/test_dsl/test_grammar_fastpath_completion_forest_aggregate_benefit.py -v
# 7 passed in 43.74s
```

Non-vacuous check: ran the identical file against `engine.py` checked out
at commit `20b658f` (the state before PRs #1174/#1176's suffix-anchor
fixes, via `git show 20b658f:...engine.py` swapped in and restored after)
— 1 failed + 6 errored, all with `AttributeError: type object
'OpenUIIncrementalEngine' has no attribute '_probe_delta_from_anchor'`,
proving the tests exercise the real fix rather than a tautology.

## Regression sweep

Same 3-minute (`MAX_RUN_MINUTES`) hard-cap discipline as the rest of this
chain (this module's own suite runs in ~44s, well inside the cap).

| suite | result |
| --- | --- |
| this finding's own tests | 7 passed in 43.74s |
| `test_grammar_fastpath_completion_forest_aggregate_benefit,test_grammar_fastpath_incremental_lex,test_grammar_fastpath_lexer_cache,test_grammar_fastpath_probe_chunk_realistic_chunk_size,test_grammar_fastpath_probe_chunk_suffix_lex` (combined) | 29 passed in 63.18s |

`ruff check` on the new test file: clean (one `RET501` auto-fix applied —
bare `return` instead of `return None` in the disabled-anchor oracle).
`python -m scripts.verify_version_stamps --check`: `1 changed file(s), 0
component(s) touched` — a test-only addition, no `versions.json`-registered
component path touched.
`python -m scripts.repo_policy`: `ok (tracked + untracked)`.
`python -m scripts.verify_decode_invariants`: clean.

## Scope note (what this does NOT establish)

- No production code change — `engine.py`'s suffix-anchor fixes (#1173,
  #1174, #1176) are unmodified; this is a measurement against the code as
  it already stands.
- Does not touch the shared growing-boundary-token gap (`_incremental_sync`
  and `probe_chunk` both still fall back to a full relex when the last-fed
  token keeps growing one character at a time) — still open, per PR
  #1176/#1177's carried-forward next step.
- Not re-measured against the full per-record `compiler_ms` telemetry or
  the `exposure12` quality-champion eval protocol — still open, carried
  forward from PRs #1171/#1173/#1174/#1176/#1177. Notably, PR #1178's
  independent finding (lexer-cache fix alone does not clear the
  `exposure12` hero decode wall — more candidates explored inside the same
  fixed deadline, not faster completion) suggests that scale may see *even
  less* benefit than either this session's aggregate figures or PR #1177's
  single-candidate figures, since candidate count there is bounded by a
  fixed decode-timeout wall rather than by how quickly this fix lets the
  search run to completion.
- Measured on `_build_openui_completion_forest`'s own `PREFIX8`/`PREFIX9`
  fixtures at `remaining_tokens=24` — not the `exposure12` quality-champion
  smoke suite's actual prefixes/token budgets. The aggregate ratio measured
  here (~2.4x/~1.15x) may not transfer exactly to that different scale and
  candidate shape.

## Next steps

1. Consider whether an anchor could be seeded across sibling branches that
   share a common prefix (e.g. cloning a parent branch's
   `_fed_last_token_start`/`_fed_tokens` state before the per-candidate
   divergence point) instead of constructing every candidate's
   `OpenUIIncrementalEngine` from scratch — would need its own correctness
   argument for why a cloned anchor stays valid across the divergent
   suffix. Not attempted this session; this finding only measures the
   current fan-out's aggregate benefit, it does not propose or evaluate a
   sibling-anchor-sharing fix.
2. Close the shared growing-boundary-token gap for both `probe_chunk` and
   `_incremental_sync` via the `InteractiveParser` checkpoint/rollback
   approach PR #1174's next steps scoped — still open.
3. Re-measure against the full per-record `compiler_ms` telemetry / the
   `exposure12` quality-champion eval protocol — still open, carried
   forward across this entire chain. Given PR #1178's independent finding
   about that specific scale, expect a smaller benefit there than either
   this session's or PR #1177's figures, not a larger one.

## Cleanup note

`.venv-diag` (gitignored) created for this session is not committed; no
scratch diagnostic scripts were committed — ad hoc `python -c` invocations
were used only to explore the measurement before writing the committed
test module, and the final numbers reported above were reproduced by
running that committed module directly via `pytest`.

Captured: 2026-07-28T00:00:00Z
