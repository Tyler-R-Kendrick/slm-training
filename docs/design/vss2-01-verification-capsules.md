# VSS2-01: Dependency-closed verification capsules

**Issue:** SLM-65  
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

A deterministic derived graph over existing `ScopeContract`s in `src/slm_training/data/progspec/capsules.py`:

- `DependencyKind` enum (`DEFINES`, `REFERENCE`, `CONTAINMENT`, `ROOT_OUTPUT`, `EFFECT`, `EXTERNAL`).
- Immutable JSON-safe `ScopeNode`, `ScopeEdge`, `VerificationCapsule`, and `CapsuleGraph` dataclasses.
- `derive_capsule_graph(spec: ProgramSpec) -> CapsuleGraph`:
  1. Reuses `derive_scope_contracts` for stable statement/non-statement scopes.
  2. Builds graph nodes only from `ScopeKind.STATEMENT` contracts plus a synthetic root interface node.
  3. Attaches nested `COMPONENT_CALL` / `CHILD_LIST` contracts as member paths on the nearest containing statement/root.
  4. Walks the typed AST to add `REFERENCE` edges for binder uses, `EXTERNAL` edges for slot/template inputs, and a `ROOT_OUTPUT` edge.
  5. Raises `ValueError` for forward references / undefined binders.
  6. Runs Tarjan SCC to produce dependency-closed verification capsules.

Exports added to `src/slm_training/data/progspec/__init__.py`.

## Verified

- `ruff check` passes.
- `python -m compileall` passes.
- `pytest tests/test_data/test_progspec.py` passes (17 tests, including 7 new capsule tests).
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.

## Design boundaries preserved

- `ScopeSlice` and `extract_scope_slices` remain syntax-oriented and untouched.
- `ScopeContract` / `derive_scope_contracts` remain the stable AST contract; capsules are a separate projection.
- No third scope extractor was created.
- Forward references fail closed; the graph does not weaken legality.

## Caveats

- This is fixture wiring only. The V1 graph boundary is implemented; richer effect/ containment edge semantics and capsule-level solving are follow-ups.
- No model, checkpoint, or ship gate is claimed.

## Persistent typed `ScopeEnv` core

`src/slm_training/dsl/scope_env.py` now provides the Torch-free, pack-neutral
lexical-environment core that a later runtime migration can share. This is a
new fact owner for symbol identity and visibility mechanics; it does not replace
`ScopeContract` AST metadata, capsule dependency graphs, the pack verifier, or
solver hole identity.

### Namespaces and identity

`SymbolNamespace` keeps seven disjoint namespaces:

- external content slots;
- lexical binders;
- state variables;
- queries;
- actions;
- mutations;
- compiler-local declarations.

Each declaration receives a request-local `StableSymbolId` from a monotonic
namespace ordinal, such as `content:0000`. Allocation depends only on declared
order, never on a caller/template spelling, hash of that spelling, or runtime
randomness. IDs are not recycled after scope exit. `ScopeSymbol` carries only
the stable ID, declaration frame, and optional declared type/role facts; it has
no surface-name field.

Caller and template spellings live exclusively in the separately serialized
`SurfaceAliasMap`. Consequently the `ScopeEnv` JSON and fingerprint exclude
aliases, while the alias map has its own fingerprint for authority-side
transport and audit. Renaming `hero.title` to `marketing.heading` with the same
declaration order leaves the environment, stable IDs, and model-safe
fingerprint byte-identical. A future model adapter must accept only the
alias-free symbol view; raw aliases and alias-derived hashes must not enter
semantic scoring.

### Persistent scope operations and policies

`ScopeEnv` is immutable and parent-linked. `enter_scope`, `exit_scope`,
`predeclare`, `declare`, `resolve`, and `visible` return new state or read the
active frame chain without mutating an ancestor. Visibility is
namespace-specific. A nearer declaration hides an outer declaration with the
same alias, while unrelated aliases retain deterministic declaration order.

Policy is explicit at each authority-sensitive operation:

- `ShadowingPolicy.FORBID` rejects a visible outer alias; `ALLOW` permits a
  nested declaration to hide it. Duplicate declarations in one frame always
  fail closed.
- `ForwardReferencePolicy.FORBID` rejects resolution before declaration;
  `ALLOW_PREDECLARED` resolves only an explicit, typed predeclaration. An
  unknown name never becomes an implicit symbol.
- Cycles are deliberately outside this core. `ScopeEnv` does not own reference
  edges and therefore cannot certify a program acyclic. The active pack
  reference verifier and `CapsuleGraph` remain authoritative; a later adapter
  must carry stable IDs onto those edges and apply the pack's declared cycle
  policy without duplicating it in `ScopeEnv`.

Late surface naming follows the same boundary: semantic solving and codec
selection use stable IDs; only a verified result may consult
`SurfaceAliasMap` to restore caller names or choose canonical late names.

### Migration boundary

The core and its focused tests are implemented, but it is **not globally wired
or enabled**. In particular, `GenerationRequest`, lexer `SymbolTable`, compiler
prefix analysis, `ChoiceDecodeState`, the production codec, and TwoTower model
features still use their existing adapters and behavior. No configuration
default, tokenizer layout, checkpoint contract, verifier policy, or ship gate
changes in this step.

Runtime adoption requires separate parity-tested adapters that preserve legacy
serialization while projecting prompts, target placeholders, dynamic symbol
features, compiler candidates, and choice references through opaque stable IDs.
The currently pinned production pack also continues to reject state/query/
mutation/action syntax; representing those namespaces does not enable OpenUI
v0.5 runtime semantics. Focused coverage lives in
`tests/test_dsl/test_scope_env.py` and includes persistence, namespace
separation, stable alias-independent identity, visibility/shadowing,
predeclared forward references, serialization, and fail-closed validation.
