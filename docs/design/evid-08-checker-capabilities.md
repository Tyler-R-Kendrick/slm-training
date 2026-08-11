# Checker capabilities, independence, and trust domains (EVID-08 / SLM-535)

**Status:** Implemented. Production policy for FormalAuthorityV2 / formal loop /
mutation suite — not a third evidence stack.

## Artifacts

| Artifact | Path |
| --- | --- |
| Capability registry | `src/slm_training/resources/formal/checker_capability_registry.v1.json` |
| Fixtures | `src/slm_training/resources/formal/evid08_capability_fixtures.v1.json` |
| Module | `src/slm_training/formal/checker_capability.py` |
| Loop wiring | `src/slm_training/formal/loop.py` |
| Authority gate | `src/slm_training/formal/authority.py` |
| Tests | `tests/test_formal/test_checker_capability.py` |

## Vocabulary

`structural_consistency`, `exact_proposition`, `axiom_audit`, `semantic_replay`,
`certificate_checking`, `encoding_correctness`, `runtime_refinement`,
`provenance_only`, plus binding/toolchain identity capabilities.

## Trust-domain examples

| Domain | Backends |
| --- | --- |
| `python_cpython_formal_structural_family` | `python_structural`, `python_reference` |
| `python_enumerative_replay` | `python_replay` |
| `lean4_kernel` | `lean_kernel` |
| `python_encoding_bridge` | `python_encoding_ref` |

## Policies

- `formal_object_conformance/v1` — redundant structural family may close conformance.
- `formal_semantic_authority/v1` — requires independent verification; shared
  structural family alone fails; skipped ≠ success; unadvertised capabilities
  cannot be satisfied.

Component: `formal.objects` v19.
