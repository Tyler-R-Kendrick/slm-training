# AP-026 (SLM-320): explicit binder/reference precision channel

Status: harness wiring only. **No training run or benchmark was executed as
part of this change** -- this document registers a structural type, its
derivation, and a gated connector plus their regression tests, not an
experiment result. It is not a claim that conditioning decode on a predicted
binder graph improves binder/reference F1 or meaning-v2; establishing that
oracle ceiling and training a predicted channel are follow-on work (see
Deferred, honestly).

## Decision

Keep binder definitions, uses, scopes, and reference edges in a verified
discrete channel rather than forcing the abstract scratchpad to encode exact
identity -- per arXiv:2605.25745 and
[`docs/design/verifier-stack.md`](verifier-stack.md).

## What this adds

`src/slm_training/data/progspec/binder_graph.py`:

- **`BinderGraphV1`** -- definitions, directed reference edges, scopes, and
  unresolved placeholders, reusing the exact `ScopeNode`/`ScopeEdge`/
  `DependencyKind` vocabulary `data/progspec/capsules.py` already produces
  for `CapsuleGraph` (the VSS capsule-solver's own dependency graph), rather
  than inventing a parallel node/edge shape. `capsules.py`'s node/edge walk
  was extracted into a shared `derive_node_edge_walk(spec,
  raise_on_unresolved=...)`: `derive_capsule_graph` still raises `ValueError`
  on a forward/undefined reference (unchanged behavior, still tested by
  `test_forward_reference_raises`); `derive_binder_graph` collects the same
  case as an `UnresolvedPlaceholder` instead, so a program that isn't fully
  resolved yet is still representable as data.
- **`BinderGraphV1.check_g3()`** -- evaluates the same four-part contract
  `docs/design/verifier-stack.md`'s G3 gate names ("one root, resolved
  binders, reachability, no cycles"), directly over the parsed structure
  rather than the flat-text regex walk `data/verify/stack.py`'s
  `Gate.REFERENCES` uses. Reachability is undirected connectivity from the
  root position: `derive_node_edge_walk` directs a `REFERENCE` edge from the
  *referencing* occurrence toward the statement that defines it (not from
  the program's entry point outward), and never emits a `ROOT_OUTPUT` edge
  at all (`"root"` is filtered out of every `definitions` set by
  `data/progspec/scopes.py`, so `definition_to_node["root"]` never exists) --
  both are pre-existing properties of the reused walk, not something this
  change alters, so `check_g3` seeds its connectivity walk from every node at
  `ast_path == ()` (the synthetic document root and the program's own
  `root = ...` statement node both sit there) rather than assuming a single
  directed entry point.
- **`BinderGraphV1.alpha_fingerprint()`** -- a structural fingerprint
  invariant to binder surface spelling, mirroring
  `dsl.scope_env.ScopeEnv.fingerprint`'s alpha-rename convention (surface
  spelling is realization data, not identity). Node identity is canonicalized
  by `ast_path` sort order rather than the raw `node_id` string, since
  `node_id` embeds `spec_id` and would otherwise make the fingerprint
  spec-specific rather than shape-specific.

`src/slm_training/models/binder_precision_channel.py`:

- **`BinderChannelArm`** -- six named arms sharing one resolution path:
  `disabled`, `oracle`/`predicted_soft` (passthrough), and three
  content-perturbed controls at matched node/edge-count architecture --
  `absent` (empty graph), `shuffled` (a full node-id permutation across
  edges, breaking every binder correspondence), `corrupted` (each reference
  edge's target redirected to a different, wrong node -- plausible-shaped
  but wrong, distinct from `shuffled`'s batch-level scramble).
  `resolve_binder_graph(graph, arm=..., generator=...)` is the single
  function every arm resolves through.
- **`confirm_predicted_edges(predicted, verifier=...)`** -- the
  non-negotiable gate the acceptance criteria require: "condition only
  binder/reference-eligible decisions via a gated connector; never
  hard-enforce learned edges without verifier confirmation." A predicted
  reference edge earns no legal authority from being predicted; it is
  confirmed if and only if the exact `(source, target, role)` triple is also
  a reference edge in the deterministic verifier graph for the same program.
  This is decode-invariant I3 ("ranking is a lever; legality is not") applied
  to binder edges specifically, and it is a *filter* over predicted edges,
  never a merge with the verifier's own edges (`confirm_predicted_edges`
  never returns more edges than `verifier.reference_edges()` has).
- **`BinderChannelGate`** (`nn.Module`) -- a single learnable scalar
  initialized at `0.0` (`tanh(0) == 0`, matching
  `AbstractPlanConnector.gate`), scaling caller-supplied per-edge features
  masked to confirmed edges only. An edge with `confirmed_mask == 0`
  contributes exactly zero regardless of the gate's value, so an unconfirmed
  predicted edge cannot leak into the bias even with the gate fully open.

## Deferred, honestly

- **No `DenoiserTower`/`TwoTowerModel` integration.** `BinderChannelGate`
  does not own an embedding table and is not wired into any forward pass.
  Unlike `AbstractPlanConnector` (which had a trained producer --
  `AbstractPlanHead` from AP-023 -- to consume), there is no trained producer
  of predicted `BinderGraphV1` structures yet; wiring a bias hook into the
  decoder before one exists would be untested plumbing pretending to be a
  shipped feature. `BinderChannelGate.bias_for_confirmed_edges` is a
  standalone, directly-testable unit (see
  `tests/test_models/test_binder_precision_channel.py`) that a follow-up
  issue can call from `DenoiserTower.project` once a predicted-graph head
  exists, the same way AP-024 called `resolve_plan_vector`/
  `AbstractPlanConnector.bias_for_vocab` from that same call site.
- **No oracle-ceiling or promotion experiment.** The acceptance criteria's
  "oracle channel establishes a measurable ceiling on binder/reference F1 and
  meaning-v2" and "predicted channel promotes only if it beats
  abstract-plan-only" are evaluation claims requiring an actual training run
  against `data/semantic_contrast/`'s hard-valid contrast pairs (SLM-290,
  merged); this change adds the structural and gating primitives that
  experiment would consume, not the experiment itself.
- **`semantic_plan_predictor.py`/`grammar.py`/`dsl/solver/` are read, not
  modified.** The issue's target-path list names these as context for where
  binder/reference concepts already partially exist
  (`SemanticPlanV1.symbols`/`.bindings` in `data/progspec/semantic_plan.py`,
  `dsl/scope_env.py`'s `ScopeEnv`/`StableSymbolId` alpha-rename machinery,
  `dsl/solver/capsule_solver.py`'s SCC-based capsule solving) -- this change
  does not modify any of them; `BinderGraphV1` deliberately reuses
  `capsules.py`'s existing node/edge vocabulary instead of duplicating it.
- **`Gate.REFERENCES` (`data/verify/stack.py`) is untouched.** `check_g3`
  is a separate, AST-level evaluation of the same four-part contract; it does
  not replace or call into the existing flat-text `_reference_graph` gate,
  and the corpus verifier stack's G3 result is still the one that gates
  Gold/Silver/Bronze/Quarantine tier assignment.

## Acceptance criteria mapping

- "Define `BinderGraphV1` for definitions, uses, directed edges, scopes, and
  unresolved placeholders." -- `data/progspec/binder_graph.py`.
- "Support oracle, predicted-soft, absent, shuffled, and corrupted modes." --
  `BinderChannelArm` / `resolve_binder_graph`.
- "Condition only binder/reference-eligible decisions via a gated connector;
  never hard-enforce learned edges without verifier confirmation." --
  `confirm_predicted_edges` + `BinderChannelGate` (see
  `test_confirm_predicted_edges_drops_edges_the_verifier_does_not_have`,
  `test_gate_never_leaks_unconfirmed_edge_features_even_when_open`).
- "Persist raw predicted graph, verifier-adjusted graph, and final program
  graph separately." -- the shapes exist (`predicted: BinderGraphV1`,
  `confirm_predicted_edges(...) -> tuple[ScopeEdge, ...]` as the
  verifier-adjusted view, `verifier: BinderGraphV1` as the deterministic
  program graph) and are never mutated in place; persisting them to a corpus
  record is deferred with the oracle-ceiling experiment above.
- "Tests cover alpha-renaming, cycles, scopes, unresolved references, and
  causal corruptions." --
  `tests/test_data/test_binder_graph.py::test_alpha_renaming_binder_names_preserves_fingerprint`,
  `::test_reference_cycle_is_detected`, `::test_unreachable_binder_is_reported`,
  `::test_unresolved_reference_is_collected_not_raised`; and
  `tests/test_models/test_binder_precision_channel.py::test_corrupted_arm_redirects_reference_edges_to_a_wrong_target`.
- "Integrate G3/VSS feedback for scoring/refinement while retaining raw
  predictions." -- `BinderGraphV1.check_g3()` is the G3-shaped feedback
  surface; VSS/`capsule_solver.py` integration (beyond reusing its node/edge
  vocabulary) is deferred with the rest of the decoder-wiring work above.
