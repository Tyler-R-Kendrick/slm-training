# SLM-187 (FFE1-01): topology solver/runtime transition parity fixture (slm187-topology-parity-20260721)

Matrix set: `slm187_topology_parity`

Version: `ffe1-01-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

The topology finite-domain solver state carries enough resolved tree and context information to reconstruct every successor, and the solver's enumerated edit domain contains every edit the runtime can commit for the actions that participate in solver filtering (EXPAND/KEEP).

## Falsifier

An exhaustive fixture over bounded topology states finds a runtime-committable EXPAND or KEEP edit that is missing from the solver domain, or a solver-domain edit whose successor fingerprint differs from the runtime's successor.

## Summary

- Cases: 8
- Parity OK: 3
- Runtime-only tuples (missing from solver): 114
- Solver-only tuples (outside runtime EXPAND/KEEP contract): 255
- Disposition: **parity_gap**

## Parity cases

| Case | Description | Solver domain | Runtime domain | Shared | Runtime-only | Solver-only | Parity | Terminal valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| doc_root | active document root | 6 | 9 | 0 | 9 | 6 | False | False |
| doc_statement | document root -> active statement | 12 | 12 | 1 | 11 | 11 | False | False |
| stmt_component | resolved statement -> active component | 72 | 1 | 0 | 0 | 72 | True | False |
| component_list | component -> active list | 78 | 75 | 64 | 11 | 14 | False | False |
| leaf_slot | leaf node with slot binding | 0 | 2 | 0 | 0 | 0 | True | False |
| fragment_root | fragment output marker root (document fallback) | 6 | 9 | 0 | 9 | 6 | False | False |
| max_depth_leaf | leaf at topology_max_depth boundary | 3 | 76 | 1 | 74 | 2 | False | False |
| sibling_choices | two active sibling expressions | 144 | 2 | 0 | 0 | 144 | True | False |

## Disposition

**parity_gap**

114 runtime EXPAND/KEEP tuple(s) are missing from the solver domain. The hard state or adapter enumeration needs repair before topology intermediates can be used for Markov/CTMC training.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The parity oracle, V2 state carrier, and transition comparison are exercised on deterministic synthetic trees. Real model runtime parity under `topology_verified_solver=True` and full multi-step terminal validation are required before topology intermediates may be used for Markov/CTMC/flow training. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_solver_runtime_audit``.

## Honest caveats

- Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.
- Runtime proposal logic is mirrored torch-free for comparison; small discrepancies with the live grammar_diffusion.py path are reported as parity gaps, not fixed silently.
- DELETE and CONTRACT proposals are intentionally outside the finite edit domain in the current runtime (production_id < 0 bypasses solver filtering); the oracle treats them as a separate contract.
- STOP is a solver-domain structural action that the runtime does not currently emit; this asymmetry is surfaced, not suppressed.
- Multi-step terminal traces are validated with the DSL parser; a parse failure is reported honestly rather than treated as parity success.

## Reproducibility

```bash
python -m scripts.run_slm187_topology_parity_fixture --mode plan-only
python -m scripts.run_slm187_topology_parity_fixture --mode fixture
```
