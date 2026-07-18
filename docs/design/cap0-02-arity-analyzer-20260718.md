# CAP0-02 (SLM-78): exact arity analyzer — bounded arith-sketch fixture

**Date:** 2026-07-18. **Status:** wiring / mathematical evidence only. This note
adds a deterministic, Torch-free static analyzer and one committed bounded
fixture. It makes **no model, train, eval, quantizer, checkpoint, or ship
claim**. It preregisters exact, replayable counts *for that fixture* under a
frozen frame; it does not touch decode or deployment behavior.

Owner package: [`src/slm_training/dsl/analysis/arity/`](../../src/slm_training/dsl/analysis/arity/).
CLI: [`scripts/analyze_grammar_arity.py`](../../scripts/analyze_grammar_arity.py).
Machine certificate: [`cap0-02-arity-analyzer-20260718.json`](cap0-02-arity-analyzer-20260718.json).

## Honesty boundary (read first)

The external CAP0-01 program review
([calculated-arity-adaptive-precision.md](calculated-arity-adaptive-precision.md),
"Source review and corrections") records source-reported estimates: **130**
bounded ASTs, **351** trie states, **41** minimized states, decision histograms
**162 / 190 / 345**, exact **Hankel rank 40**, **99%-energy rank 32**, the
residual-plane **5 / 35 / 1**, and a raw **86**-state value. That document states
the typing-rule variant, executable, signature frame, and checksums required to
replay those numbers are **absent from the repository**, so they are
**source-reported estimates, not repository certificates**, and the raw 86-state
claim "was not reproduced."

Therefore:

- This analyzer **does not reproduce** 130 / 351 / 41 / 162-190-345 / Hankel /
  residual. It builds a *correct* pipeline on a *different, fully specified*
  committed fixture and reports whatever exact counts that fixture yields.
- The counts below are **repository certificates for the `bounded-expr` fixture
  only** — byte-stable, replayable, frame-relative. They are deliberately not
  equal to the external estimates (400 ≠ 130, 844 ≠ 351, 28 ≠ 41).
- The raw **86** "frontier × scope" value is **retired** from all new
  conclusions. We report our fixture's own frontier × scope count (11) as a
  separate coarse signature, never as a reproduction of 86.
- **No Hankel certificate is claimed.** We compute no prefix/suffix Hankel
  matrix here; the external Hankel rank 40 / energy rank 32 remain source
  estimates. Any future string-serialized prefix/suffix Hankel would be a
  string-serialization diagnostic, **not** a subtree (tree-)Hankel certificate.

Per the CAP0-01 non-equivalence rule, an exact symbolic-state count is **not** a
task-relevant information rate, **not** a neural weight/activation precision, and
**not** a deployed-system optimum. Nothing here crosses those boundaries.

## The committed fixture: `bounded-expr`

Grammar backend: the frozen `arith-sketch` G4 grammar
([`arith_sketch.lark`](../../src/slm_training/dsl/grammars/arith_sketch.lark),
`get_backend("arith-sketch")`), reused verbatim for parsing and — via
`backend.validate` — for type rejection. `grammar_hash` in the certificate is
the sha256 of that grammar file
(`f54c881fce9170dabb2d78b9c894a839e7df50ecb2b392f5a8115af2bbfd9601`).

The fixture is the finite language of **canonical straight-line programs** under
the declared frame `F`:

| Frame member | Value |
| --- | --- |
| `max_ast_nodes` (total nodes, all statements) | 6 |
| `max_ast_depth` | unbounded within the node budget |
| `max_live_bindings` (scope window; also caps statement count to window+1) | 2 |
| operators | `+  -  *  /` (the full G4 set) |
| template classes | `N` (one numeric literal template symbol) |
| result type | `number` |
| capacity dimension `d` | 4 |

Canonicalization (owner: `arity/canonical.py`, reusing `production_codec`
sigils):

- numeric literals collapse to the typed template symbol `("lit", "N")` —
  concrete values are irrelevant to arity;
- identifiers collapse to **de Bruijn** refs `("ref", delta)` against the
  nearest preceding binder (so binder renaming ⇒ identical canonical form, and
  redefinition shadows correctly);
- alpha-equivalent programs share one sha256 fingerprint;
- **liveness**: every non-root binder must be referenced by a later statement;
- **type validity**: the materialized source must pass `backend.validate`. The
  arith-sketch oracle rejects a `root` that resolves to a bare atom, so those
  candidates are removed *before counting* (3 rejected here — a non-vacuous
  gate).

## Tri-state pipeline

`enumerate canonical ASTs → validate/type-reject → prefix trie → acyclic
minimization → branching / completion / K^d capacity`. Enumeration is fully
deterministic (statement count ascending, then node-size tuples, then
`itertools.product` in a fixed order). Minimization is a single **iterative**
reverse-topological (deepest-first) sweep keyed on
`(terminal_status, tuple(sorted (action, child_class)))` — the standard acyclic
Myhill–Nerode / DAWG collapse, no recursion.

## Certified counts (fixture `bounded-expr`, frame above)

Every count traces to the exact signature or bound it is computed under. These
are separate signatures, reported separately (never collapsed into one scalar);
their numeric ordering is whatever the fixture yields, not an assumed hierarchy.

| Count | Value | Signature / bound it traces to |
| --- | --- | --- |
| `canonical_ast_count` | **400** | distinct valid canonical ASTs (fingerprint dedup) after the `.validate` type gate |
| `raw_state_count` | **11** | distinct coarse `(frontier × scope × expected_type)` structural signatures (the local "raw frontier × scope" analog; **not** the retired 86) |
| `trie_state_count` | **844** | distinct action-prefix nodes over the preorder alphabet |
| `minimized_state_count` | **28** | acyclic Myhill–Nerode classes on `(terminal, sorted(action, child_class))` |
| `action_alphabet_size` | **8** | `{ =, o:+, o:-, o:*, o:/, #N, ~1, ~2 }` |
| `scope_signature_count` | **3** | scope windows `{0, 1, 2} = min(completed, max_live)` |
| `max_local_branching` | **6** | max out-degree over minimized states |
| `branching_histogram` | `{0:1, 1:11, 2:5, 3:2, 4:2, 5:4, 6:3}` | out-degree over the 28 minimized states (sums to 28) |
| `completion_counts` | `{0:3, 1:8, 2:7, 3:6, 4:4}` | minimal completion length over the 28 minimized states (sums to 28) |
| `forced_visit_fraction` | **9 / 27** (≈ 0.333) | forced (out-degree 1, non-accepting) ÷ decision states, exact integer ratio |
| `work_counters.validate_rejected` | **3** | type-invalid candidates rejected before counting |

**Capacity (K^d), d = 4, integer arithmetic only:** the least `K` with
`K**4 ≥ 28` is **K = 3** (`2**4 = 16 < 28 ≤ 81 = 3**4`). Reported as
`capacity = {state_count: 28, d: 4, min_k: 3}`, with `d` printed beside the
result. This is a cardinality fact about naming 28 states with a length-4 code;
it asserts nothing about task distortion, precision, or deployed cost. (For the
design-doc comparator quotient `M = 41`, the same integer routine gives `K = 2`
at `d = 6` and `K = 3` at `d = 4` — both preregistered arms have enough names;
we compute it, we do not claim `M = 41` for this fixture.)

## Replay

```bash
python -m scripts.analyze_grammar_arity --fixture bounded-expr \
    --max-ast-nodes 6 --max-live-bindings 2 --dimensions 4 \
    --out outputs/runs/arity/bounded_expr_report.json
```

Writes the scratch JSON and the durable certificate above with identical bytes.
The CLI fails closed (non-zero exit) on incomplete enumeration (safety cap hit),
missing required bounds, an unknown fixture, or missing/stale version metadata.
`schema_version = 1`, `signature_version = 1`, `codec_version = 1`,
`parser_version = 1` are emitted and validated on read.

## Data-contract & version provenance

`ExactArityReport` and `StateSignature` are frozen, JSON-safe, schema-versioned
dataclasses with deterministic `to_dict` / `from_dict` round-trips.
`StateSignature.fingerprint()` hashes only hard state (generation step, grammar
prefix, frontier, scope window, expected type, template state) — never a logit,
score, timestamp, or pid. Stale `schema_version` / `signature_version`, or
missing version metadata, raise `SchemaError` on read.

## Scope note

Wiring and mathematical evidence only. No checkpoint, model card, README, eval,
or ship-gate change is warranted by this issue. Future capacity/precision
experiments must keep the CAP0-01 non-equivalence separations, use the existing
quality/performance matrices, and preserve meaningful-parse as the primary
metric.
