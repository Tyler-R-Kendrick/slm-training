# VSS2-03: Capsule-owned topology edits as finite semantic holes

**Issue:** SLM-67
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

A Torch-free adapter between `GrammarDiffusionModel` topology nodes and finite-domain solver holes, under `src/slm_training/dsl/solver/`:

- `state.py` — immutable `HoleId`, `DomainValue`, `HoleDomain`, `FiniteDomainState`, and `SupportVerdict`.
- `topology_adapter.py` — `TopologyAction`, `TopologyEdit`, `TopologyNodeLike`, `TopologyAdapterConfig`, `legal_topology_productions`, and `derive_topology_holes`.
- The adapter enumerates **complete edit tuples** `(action, production_id, arity, slot_id)` for each active topology node rather than independent logits.
- The adapter imports neither `torch` nor `grammar_diffusion.py`; model-side callers pass `TopologyNode` instances into the Torch-free protocol.
- `__init__.py` exports the public contracts.

## Verified

- `ruff check` passes.
- `python -m compileall` passes.
- `pytest tests/test_dsl/test_topology_adapter.py` passes (6 tests).
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.
- `test_importing_adapter_does_not_import_torch` confirms the adapter stays Torch-free.

## Design boundaries preserved

- The production codec and model decode loop are unchanged.
- `derive_capsule_graph` (VSS2-01) is reused as the upstream capsule owner.
- The adapter is model-independent; only the caller supplies a codec and a topology-node view.

## Caveats

- This is fixture wiring only. Capsule-to-topology correlation (mapping each hole to its owning `VerificationCapsule`) is scaffolded through stable `HoleId` paths; a follow-up will bind topology-node positions to `ScopeNode` / `CapsuleGraph` coordinates.
- No model generation behavior was altered.
- No model, checkpoint, or ship gate is claimed.
