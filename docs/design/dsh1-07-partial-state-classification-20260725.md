# DSH1-07: classify COMPLETE, FRAGMENT, and compiler-reachable PREFIX states

**Issue:** SLM-359 (project DSH1 — CAP0 Grammar-Complete Symbolic Synthesis,
milestone M3 — Symbolic QA views and legal partials)
**Status:** wiring / unit-tested contract. No decode-path integration, training
run, eval, benchmark, checkpoint, or ship claim.

## Decision

Distinguish complete programs, named valid fragments, and incomplete prefixes
with at least one compiler-proven accepting continuation, so partial training
examples align with states the constrained decoder can actually encounter.

## What was added

`PartialStateClassV1` (`src/slm_training/dsl/solver/partial_state.py`) — a
Torch-free, problem-independent frozen dataclass plus classifier:

- `PartialKind`: `COMPLETE`, `FRAGMENT`, `PREFIX`, `INVALID`.
- `PartialStateClassV1`: `kind`, `start_symbol` (`FRAGMENT` only),
  `frontier` (`PREFIX` only — the open next-terminal set), `open_nonterminals`,
  `required_roles` (open binder references), `singleton_positions`,
  `support_coverage` (the `CompletionForest` coverage tier), and
  `min_completion_cost`. Strict `__post_init__` validation enforces the shape
  each kind must have (e.g. a `COMPLETE` state carries no `start_symbol` or
  frontier; a `PREFIX` state requires a non-empty frontier, non-`none`
  coverage, and a certified `min_completion_cost`).
- `classify_partial_state(...)`: decides from facts the caller has already
  established — mirrors the injected-fact style of
  `slm_training.dsl.solver.support` (`ProblemExpander`/`Verifier`) rather than
  parsing or building a forest itself, so the core classifier stays reusable
  outside the OpenUI pack.

`src/slm_training/dsl/solver/openui_partial_state.py` wires the contract to
OpenUI using **two already-canonical, independently tested authorities** — no
new parsing or grammar authority was introduced:

- `classify_openui_source(source, ...)` — tries `validate_output(source, kind)`
  (`slm_training.dsl.parser`, `OutputKind` = `document`/`statement`/
  `expression`/`typed_node`/`lexical`) in order. `"document"` success is
  `COMPLETE`; any other accepted kind is `FRAGMENT(start_symbol=kind)`; no kind
  accepting is `INVALID`. This is the same live fragment parser opaque-region
  splicing (`dsl/opaque_regions.py`) and scope extraction already trust.
- `classify_openui_prefix(tokenizer, prefix_ids, ...)` — builds the real
  `CompletionForest` (`build_completion_forest`) for an in-progress decode
  prefix. `forest.terminals` is the recorded `frontier`/`open_nonterminals`;
  `min_completion_cost` is the shortest drafted continuation length among
  enumerated compiler paths (a lower-bound proxy from the forest, **not** an
  exhaustively certified shortest path to a verified terminal — that stronger
  claim requires the separate `EnumerativeSupportOracle`); `required_roles`
  comes from unresolved binder references
  (`unresolved_binder_reference_pieces`); `singleton_positions` are prefix
  indices outside `gold_compiler_decision_positions` (i.e. positions the
  compiler forced, not a real branch). Refuses (raises `ValueError`) when the
  prefix ends inside an open `LIT_STR`/`LIT_NUM` literal frame
  (`literal_frame_is_open`) — per the DSH1-07 stop rule, no cut may split a
  lexical token or opaque marker, so that position is refused outright rather
  than silently reclassified.

Two small public wrappers were added to
`src/slm_training/dsl/grammar/fastpath/compiler_draft.py` (purely additive,
reusing existing private logic, no behavior change to any existing candidate,
coverage, or decision path):

- `literal_frame_is_open(tokenizer, token_ids)` — public view of the existing
  `_literal_frame_is_open`.
- `unresolved_binder_reference_pieces(tokenizer, prefix_ids)` — deterministic,
  sorted surface pieces for binder references with no declaration yet, derived
  from the existing `_binder_scope`.

`PartialKind`/`PartialStateClassV1`/`classify_partial_state` are re-exported
from `slm_training.dsl.solver` alongside the other core, problem-independent
solver primitives (`state.py`, `support.py`, `closure.py`).

## Acceptance criteria: how this increment satisfies them

- **Every admitted PREFIX replays to at least one verified terminal.** Tested
  by walking a full known-good gold program (`root = Card(":t.x")\n`) token by
  token: every prefix before the trailing EOS classifies `PREFIX` with a
  non-empty frontier, and the assembled full source independently classifies
  `COMPLETE` via the live parser
  (`test_every_prefix_of_a_gold_program_is_prefix_until_eos`). This increment
  certifies "at least one drafted, grammar/schema-admissible continuation
  exists" (a real `CompletionForest` path) rather than the stronger "verified
  by the full G0-G12 gate stack" claim; wiring `classify_openui_prefix` to the
  `EnumerativeSupportOracle` for an exhaustive verified-terminal certificate is
  follow-up work (see Reopen conditions).
- **Every FRAGMENT parses under its declared start rule.** `FRAGMENT` is only
  ever constructed from a `completed_kind` that `validate_output` already
  accepted for that exact text — there is no path that fabricates a
  `start_symbol` without a real parse
  (`test_named_fragment_classifies_under_its_declared_start_rule`).
- **`UNKNOWN`/partial coverage never authorizes destructive pruning.** This
  module does not prune, rank, or remove anything; it only classifies and
  records. A `partial`-coverage forest still yields `PREFIX` (never silently
  upgraded to a stronger guarantee), and `INVALID` is diagnostic, matching the
  existing `SupportVerdict.UNKNOWN` convention in
  `slm_training.dsl.solver.support`.
- **No cut splits a lexical token or marker.** `classify_openui_prefix` raises
  before classifying when `literal_frame_is_open` is true
  (`test_open_literal_frame_refuses_classification`); `classify_partial_state`
  itself refuses (raises) whenever `cut_is_lexical_boundary=False` is passed,
  so this invariant is enforced at both the OpenUI adapter and the core
  contract layer.

## Explicit non-guarantees (stop rule)

- `min_completion_cost` for `PREFIX` is a **lower-bound proxy** (the shortest
  drafted `CompletionForest` path length), not an exhaustive shortest-path
  certificate. Any consumer that needs a certified minimum must additionally
  run `EnumerativeSupportOracle`/`exact_closure`.
- `classify_openui_prefix` never returns `COMPLETE`/`FRAGMENT` — a decode loop
  that stops must re-classify the final assembled source through
  `classify_openui_source` to get an honest terminal verdict.
- No decode path calls this module; nothing here changes constrained decode,
  training data, or any experiment outcome.
- Opaque-region boundaries beyond the framed-literal case (e.g. a cut inside a
  spliced `OpaqueRegion` from `dsl/opaque_regions.py`) are not yet checked by
  `classify_openui_prefix`; only literal-frame openness is enforced today.

## Verified

- `ruff check src/slm_training/dsl/solver/partial_state.py
  src/slm_training/dsl/solver/openui_partial_state.py
  src/slm_training/dsl/solver/__init__.py
  src/slm_training/dsl/grammar/fastpath/compiler_draft.py
  tests/test_dsl/test_partial_state.py
  tests/test_dsl/test_openui_partial_state.py` → all checks passed.
- `pytest tests/test_dsl/test_partial_state.py
  tests/test_dsl/test_openui_partial_state.py
  tests/test_dsl/test_solver_decode.py tests/test_dsl/test_solver_state.py
  tests/test_dsl/test_solver_support.py tests/test_dsl/test_solver_closure.py
  tests/test_dsl/test_solver_replay.py tests/test_dsl/test_solver_controller.py
  tests/test_dsl/test_capsule_solver.py tests/test_dsl/test_topology_solver.py
  -q` → 147 passed.
- `pytest tests/test_dsl/test_grammar_fastpath.py -q` → 2 failed (rest passed).
  Both failures (`test_lexer_literal_bytes_are_not_grammar_admitted`,
  `test_structural_preference_does_not_override_confident_binder`) reproduce
  identically on a clean `git stash` of this same worktree with none of this
  change's diffs applied — a pre-existing local-environment gap, not a
  regression from the two additive `compiler_draft.py` wrappers. A full,
  uninterrupted re-run of the rest of the file (and of
  `tests/test_models/test_compiler_decode.py` /
  `tests/test_models/test_solver_decode_integration.py`) did not finish inside
  the session's run cap — this sandbox's single-threaded CPU torch execution
  is slow enough that the full file takes several minutes; a partial re-run
  excluding the two known-bad tests observed 15/37 further tests, all passing,
  zero new failures. Given the `compiler_draft.py` diff is purely additive (two
  new top-level public functions, two new `__all__` entries, zero bytes
  changed in any existing function body) and the full solver/topology/capsule
  suite passes cleanly, this is recorded as honest partial evidence rather
  than a completed full-file run; see
  `dsh1-07-partial-state-classification-20260725.json` for the exact commands
  and observations.
- `python -m scripts.verify_version_stamps --check` — `model.twotower` and the
  new `dsl.solver.partial_state` component both carry a same-diff bump.

## Reopen conditions

A future DSH1-07 increment should: (1) certify `PREFIX`'s `min_completion_cost`
and continuation existence through `EnumerativeSupportOracle` rather than the
raw forest-path proxy; (2) extend the lexical-boundary check to spliced
`OpaqueRegion` spans, not only framed literals; (3) decide whether
`classify_openui_prefix` should also detect a `FRAGMENT`-shaped prefix (a
sub-tree that already parses under a non-document start rule mid-decode)
instead of requiring a separate `classify_openui_source` call after decode
stops. None of this is required for the bounded facts this increment records.
