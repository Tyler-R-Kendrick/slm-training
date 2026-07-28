# Hoisting `_tail`'s witness-search cache out of `_openui_completion_domain`'s per-candidate loop: safe, tested, but a measured no-op on `_sync`/`compiler_ms`

**Honesty:** `fixture_or_scratch` — deterministic, grammar-only microbenchmark
on the same two `openui.lark` prefixes (`PREFIX8`/`PREFIX9`) used by PR
#1173's lexer-cache tests and PR #1191's cProfile finding. No model, no
checkpoint, no eval harness involved. **Real, tested code change** (not a
deferred finding by the "unsafe to share" branch) — but the honest headline
is that it does not reduce `_sync`/`compiler_ms` on the prefixes measured,
because the premise in the predecessor finding (redundant same-state
recomputation across sibling candidates) does not hold for this cache's key
shape.

## Task

PR #1191's next-step #1 (also
[`decode-compiler-tree-witness-search-cost-finding.md`](decode-compiler-tree-witness-search-cost-finding.md)'s
named next lever): hoist the `@lru_cache`-wrapped `_tail` closure
(`src/slm_training/dsl/pack.py:402`) out of the per-top-level-candidate loop
in `_openui_completion_domain` so sibling candidates share witness-search
results, with a unit test proving witness content and `nodes_left` fairness
are unchanged, then re-measure `compiler_ms`/`_sync`-call-count.

## What changed

`_openui_completion_domain` used to define a brand-new `lru_cache`-wrapped
`_tail` function (and a fresh `nodes_left = 16` counter) *inside* `_tail_from`
— and `_tail_from` was called once per non-EOS top-level candidate in
`initial.paths`. Every candidate therefore paid for its own private,
zero-carryover memo table.

The fix moves the `@lru_cache`-wrapped `_tail` (and its fairness budget,
`_nodes_left`) to be constructed exactly once per `_openui_completion_domain`
call, before the candidate loop. `_tail_from` now just resets
`_nodes_left[0] = 16` and calls the shared `_tail`. `maxsize` changed from 64
to unbounded (`None`) since the cache's lifetime is now the whole call
instead of one candidate's recursion.

**Why sharing is safe:** `_build_openui_completion_forest`'s result for a
given `(current, room)` key depends only on `request.tokenizer`,
`request.slot_contract`, `request.max_path_tokens`, and
`request.min_content` — every one of those is constant across the entire
`_openui_completion_domain` call, never candidate-specific. `nodes_left` is a
*fairness* budget, not a correctness input, and a cache hit skips the
function body entirely (never touches it) — so widening the cache's scope
can only ever hand a later candidate a free win on an already-proven state,
never cost it a real chance.

## Measured: structurally hoisted and byte-identical, but zero `_sync` reduction

Reused `PREFIX8 = "root = Card([b1"` and `PREFIX9 = "root = Card([b1,"` (the
exact prefixes from PR #1173's lexer-cache tests and PR #1191's cProfile) via
a direct, un-mocked call to `_openui_completion_domain` with the real
`openui.lark` grammar and `DSLNativeTokenizer`.

| metric | PREFIX8 (`"root = Card([b1"`) | PREFIX9 (`"root = Card([b1,"`) |
| --- | ---: | ---: |
| top-level `_tail`-`lru_cache` objects created — before | 2 | 14 |
| top-level `_tail`-`lru_cache` objects created — after | **1** | **1** |
| `_tail` `cache_info()` after fix | `hits=0, misses=22` | `hits=0, misses=4419` |
| candidate count | 2 (both before/after) | 12 (both before/after) |
| `domain.status` / `domain.reason` | `complete` / `""` (unchanged) | `complete` / `"witness_pruned"` (unchanged) |
| candidate `(kind, token_ids, terminal_witness)` | byte-identical before/after | byte-identical before/after |
| `OpenUIIncrementalEngine._sync` call count | 1073 (before) → **1073** (after) | 16082 (before) → **16082** (after) |

The cache-object count collapsing from 2/14 to 1/1 confirms the hoist
happened as specified. The candidate tuples and `_sync` call counts are
identical to the last integer, both measured directly (not inferred) on the
real grammar/tokenizer, before and after the change (`git stash` used to
switch between the two code states in the same process/session).

**The `_tail` cache records zero hits in both scopes** (`hits=0` for both 22
misses on PREFIX8 and 4419 misses on PREFIX9). The reason: the cache key is
`(current, room)` where `current` is the **absolute token-id history since
the start of this completion-domain call** and `room` is the **remaining
budget**, and both change monotonically on every recursive step (`current`
strictly grows, `room` strictly shrinks) — so no two calls, whether within
one candidate's own recursion or across two different top-level candidates
(which start from different `current` prefixes by construction, since each
candidate's own top-level tokens are prepended), can ever land on the same
key. The `lru_cache` was already functionally inert *before* this fix too
(same 0-hit behavior, just spread across more, smaller cache objects) — the
predecessor finding's premise that "the same grammar state can get
re-explored... under a different top-level candidate" (which motivated this
lever) does not hold for this cache's literal-history key shape.

## Tests added

`tests/test_dsl/test_pack_witness_cache.py` (6 tests, all pass; suite runtime
~27s):

1. `test_tail_cache_created_once_per_completion_domain_call` — structural
   proof of the hoist (cache-object count 1, vs 2/14 pre-fix). **Sanity
   checked to fail against pre-fix code**: `git stash`ing the `pack.py`
   change and rerunning this test reproduces exactly the 2/14 counts in the
   table above (`assert 2 == 1`, `assert 14 == 1` failures).
2. `test_completion_domain_candidates_byte_identical_to_pre_hoist` — I3/I6:
   full `(kind, token_ids, terminal_witness)` tuples for every candidate,
   plus `status`/`reason`, match golden values captured from the pre-fix
   code.
3. `test_sync_call_count_unchanged_by_hoist` — the honest benchmark: asserts
   `_sync` call count equals the measured baseline (1073 / 16082) rather than
   claiming a reduction that was not observed; documents *why* in the
   docstring.
4. `test_nodes_left_fairness_unchanged_by_hoist` — a synthetic
   always-novel-state grammar stub (never returns EOS, so no cache hit is
   possible even in principle) proves each of 5 top-level candidates still
   gets its own fresh 16-node budget: total real forest-build calls is
   exactly `1 + 5*16 = 81`. If the hoist had accidentally shared `nodes_left`
   itself (not just the cache), the total would plateau at `1 + 16`
   regardless of candidate count — this test would have caught that failure
   mode.
5-6. Parametrized variants of (1) over both prefixes.

## Why this is a real (tested) fix, not a deferred finding

The task's escape hatch ("if the cache genuinely cannot be safely hoisted...
document why and defer") does not apply here: the hoist **is** safe, proven
by byte-identical witness content and preserved `nodes_left` fairness. What
did not materialize is the *performance* hypothesis from the predecessor
finding doc — this is closer to a null result on a well-scoped experiment
than an unsafe/abandoned approach (I14: closes this specific approach to
`compiler_ms` reduction, not the goal of reducing it).

The refactor still ships because it is (a) strictly safer/simpler code (one
shared cache object instead of a fresh one per candidate, still correct
under I3/I6 as proven above) and (b) closes out PR #1191's named next lever
with a real measurement instead of leaving the untested sketch in place.

## Next lever

The two real cost drivers identified by PR #1191's cProfile remain
unaddressed by any *safe* memoization at this key granularity, since
`(current, room)` never repeats by construction:

1. **Lark's `InteractiveParser.accepts()`** (12.6s / 23,530 calls in PR
   #1191's cProfile) is the dominant cost and is Lark-internal, not
   `_tail`-cache-shaped. A real win here needs either (a) memoizing
   `accepts()` by parser-automaton state (not token history) — the only way
   to get actual cache hits, since automaton state genuinely can repeat
   across different token histories that reach the same LALR state, unlike
   `_tail`'s literal-history key — or (b) reducing how many terminals
   `accepts()` is asked to probe per call.
2. **`OpenUIIncrementalEngine.__init__`**'s per-construction `Path.resolve()`
   / `os.path.realpath()` cost (10,967 calls / ~5.1s combined in that
   cProfile) — proposed-fix-sketch item 2 from the predecessor finding,
   still unapplied and still cheap to verify (cache the resolved grammar
   path once, module-level, keyed on the raw `grammar_path` string).

Recommend (2) next: it is a small, low-risk, easily-testable win (pure
path-string memoization, no witness-search semantics involved) while (1)
needs a genuine per-automaton-state cache key design (bigger, riskier,
Lark-internals-facing) with its own safety proof before attempting.

## Validation

```text
python -m scripts.repo_policy                       # repo-policy: ok
python -m scripts.verify_version_stamps --check      # ok; dsl.operators.registry v5 -> v6
python -m scripts.verify_decode_invariants            # exit 0, unchanged
ruff check src/slm_training/dsl/pack.py tests/test_dsl/test_pack_witness_cache.py   # all checks passed
pytest tests/test_dsl/test_pack_witness_cache.py -q   # 6 passed in 27.4s
pytest tests/test_dsl/test_pack.py tests/test_dsl/test_packs.py \
  tests/test_dsl/test_minimal_witnesses.py tests/test_dsl/test_grammar_capabilities.py \
  tests/test_dsl/test_grammar_fastpath_lexer_cache.py -q   # (NODE_OPTIONS= cleared)
  # 45 passed, 1 pre-existing failure unrelated to this change
  # (test_pack_fixture_loop_generate_train_eval, ValueError on
  # grammar_constrained=False -- reproduces identically on HEAD without this
  # diff applied, confirmed via git stash)
pytest tests/test_dsl/test_operator_registry.py -q    # 8 passed (component-owning test file)
```

## Scope note

- No model, checkpoint, or eval-harness run this session — pure grammar/DSL
  microbenchmark, `fixture_or_scratch` per `honest-ship-eval`.
- `dsl.operators.registry` bumped v5 → v6 (`src/slm_training/dsl/pack.py` is
  in that component's watched paths).
- `outputs/` untouched this session.

Captured: 2026-07-28T13:05:00Z
