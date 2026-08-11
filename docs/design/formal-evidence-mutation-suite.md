# Formal evidence mutation / red-team suite (EVID-11 / SLM-559)

**Status:** release-blocking fixture suite. Not a ship-quality or Lean-kernel
claim — it certifies that the named adversarial mutations fail at the
intended formal-evidence gate and that positive controls still pass.

## Artifacts

| Artifact | Path |
| --- | --- |
| Matrix | `src/slm_training/resources/formal/evid11_mutation_matrix.v1.json` |
| Suite | `src/slm_training/formal/mutation_suite.py` |
| Verify | `scripts/verify_formal_evidence_mutations.py` |
| Results | [`formal-evidence-mutation-results.json`](formal-evidence-mutation-results.json) |
| Tests | `tests/test_formal/test_mutation_suite.py` |

## Run

```bash
PYTHONPATH=src python -m scripts.verify_formal_evidence_mutations --check
PYTHONPATH=src python -m scripts.verify_formal_evidence_mutations --write
```

## Acceptance

- Every required mutation family has ≥1 matrix row.
- Every mutation `rejected=true` at its declared gate/capability.
- Positive controls (`exact_replay`, honest encoding, sealed binding) pass.
- `structural_consistency_alone_confers_authority=false`.
- `python_structural` + `python_reference` share a trust domain and do not
  satisfy the production semantic-authority policy (EVID-08).
- Shared serialization / unadvertised-capability mutations are rejected.

Component: `formal.objects` v19.
Capability registry: `resources/formal/checker_capability_registry.v1.json`.
