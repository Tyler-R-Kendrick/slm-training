# Core formal claims (Mathlib-free Lean 4 theories)

**Status:** Implemented under `src/leverproof_lean/` as self-contained axiomatized
theories. No Mathlib dependency. Built and audited by `make -C src/leverproof_lean proofs`.

These theories formalize the structural safety layer of the grammar-constrained
system. They do **not** replace training, evaluation, matched controls, or ship
gates, and they do not prove that a particular PyTorch backend matches the model
— that remains CI + unit tests (`verify_decode_invariants`,
`forwards_count == 0`, exact-closure replay tests).

## Design constraints

| Constraint | Rationale |
| --- | --- |
| No Mathlib | Keep the external API surface inside Lean core (`List`, `Nat`, `Bool`) so proof translation stays in a controlled regime |
| No `sorry` / `admit` / custom `axiom` / `unsafe` | Same hard rule as the metric-oracle package |
| Membership sets via lists | `Subset` / `SetEq` over `List Nat` replace `Finset` |
| Boolean contracts where Python links | Trace validity mirrors the executable checker style without requiring `LawfulBEq` for every type |

## Theories

| Module | Claim family | Established theorems |
| --- | --- | --- |
| `ListSet` | Finite-set algebra | subset/union/inter/diff membership laws |
| `Forest` | Completion-forest certified closure | `close_monotone`, `close_idempotent`, `close_history_preserved`, `close_never_adds_live`, rollback partition, `lossy_history_counterexample` |
| `Trace` | Trace validity | `valid_step_sound`, `valid_trace_all_steps`, `close_step_valid` |
| `StructuralMetrics` | Metric direction | `recall_mono`, `structural_similarity_mono`, `mean_mono`, `extra_component_can_reduce_similarity` |
| `ExactClosure` | VSS exact-closure honesty | pass subset, supported/unknown/failed-replay never remove, iterate monotone, certified bottom, honest fixed point |
| `DecodeInvariants` | I1 / I2 / I6 | singleton bypass ignores ranker, empty domain is dead end, ranked token must be legal, unconstrained paths are illegal |

## What each theory is (and is not)

### Forest

Models irreversible certified removal plus append-only history. Rankers may only
permute live candidates; hard membership is invariant under swap/cons. Lossy
history reconstruction (rebuild from the live set alone) is **refuted** by a
bounded counterexample.

### Trace

A step is valid when `afterRemoved` is membership-equal to
`beforeRemoved ∪ certified` and history is prefix-extended. Multi-step traces
also require adjacent agreement. Certificate replay is an external precondition:
Lean proves what acceptance implies after replay has already passed.

### Structural metrics

Algebraic monotonicity of the declared proxy under explicit component
inequalities. Extra unmatched structure can lower the proxy — it is not free
evidence of quality.

### VSS exact-closure

Only replay-valid `UNSUPPORTED` answers shrink a domain. `SUPPORTED`, `UNKNOWN`,
and failed replay never remove. Pass and multi-pass iteration only shrink the
live set. Certified bottom is the empty live set after universal certificate
removal.

### Decode invariants

Production configs are grammar-constrained and refuse unconstrained fallback. A
complete singleton commits by bypass (I2). An empty legal domain is a constrained
dead end (I6), never a full-vocabulary fallback. Rankers may only select members
of the legal domain.

## Build and audit

```bash
make -C src/leverproof_lean proofs
# or
make -C src/leverproof_lean test
```

`Test/Proofs.lean` prints axiom dependencies for every exported core claim.
Acceptable kernel axioms only: `propext`, `Quot.sound`, and (where classical
search is used elsewhere) `Classical.choice`. Custom `axiom` declarations are
CI-forbidden.

## Relationship to the metric oracle and formal preflights

- **Metric oracle** (`Interval`, `Band`, `Resource`, `Certificate`, `Protocol`):
  experiment-resource certificates for promotion / band classification.
- **Core claims** (this document): structural safety of forest/trace/closure/
  decode/metrics algebra.
- **Ecosystem tier** ([`ecosystem-tier.md`](ecosystem-tier.md)): the growing
  OpenUI library (components, roles, operators) is measured **separately** for
  generation metrics and formal preflight success. Production-core formal
  modules above never take ecosystem inventory size as a pass condition
  (`EcosystemTier.core_success_ignores_library_size`).
- **Multi-prover export** ([`formal-objects-multi-prover.md`](formal-objects-multi-prover.md)):
  Lean claims and VSS `SupportCertificate`s export as `formal_object/v1` and
  are checked by independent backends (`python_structural`,
  `python_reference`, optional `python_replay` / `lean_kernel`). The formal
  loop rejects single-kernel reliance.
- A later Mathlib-backed preflight package (if reintroduced under
  `src/slm_training/formal/lean/`) must stay consistent with these claims or
  deliberately supersede them with a version bump and measured-results note.

## Trust boundary

Lean proves the declared calculation and classification over the abstract model.
Measurement truth, certificate sensor fidelity, JSON I/O, SHA-256, the Lean
runtime, and OS remain trusted. A proved structural claim is a pre-training
no-go filter when refuted; it is never a predicted quality score (I14).
