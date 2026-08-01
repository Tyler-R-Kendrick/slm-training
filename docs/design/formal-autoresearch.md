# Formal preflights for autoresearch

**Status:** Lean 4 proof package, executable trace contract, and tiered campaign
gate implemented. These proofs reduce invalid experiment ideas; they do not replace
training, evaluation, matched controls, or ship gates.

## Decision boundary

Use Lean when a proposed architecture change contains a claim that can be stated
over a mathematical abstraction:

- a metric component is monotone under explicit input inequalities;
- certified parse-forest closure only removes candidates and preserves history;
- rollback cannot resurrect irreversible removals;
- a recurrence update is bounded when its scale and transition satisfy stated
  bounds.

Do not ask Lean to prove an empirical conclusion from those abstractions. Lean
cannot establish that a learned transition is globally contractive, that the
abstraction matches every PyTorch execution, that optimization reaches a useful
solution, or that held-out quality or latency improves. Those remain measured
claims under the normal experiment campaign and honest evaluation gates.
This repository's formal model covers the grammar-constrained symbolic system and
selected recurrence algebra; it is deliberately not a complete model of the neural
network or training dynamics.

For the fixed-point-combinator example, the useful claim is not “a fixed-point
combinator preserves history.” The combinator alone provides no such guarantee.
The formal target is:

1. the state carries an append-only history;
2. each certified closure step extends that history and monotonically removes
   candidates;
3. the fixed-point traversal composes only those valid steps;
4. rollback retains irreversible removals.

`OpenUIProofs.Forest.lossy_history_counterexample` rejects the alternative design
that tries to reconstruct history from only the current live set. This can eliminate
that approach before a training run. It still does not predict whether a particular
history-preserving implementation improves a model metric.

## Proof packages

Autoresearch uses two pinned Lean 4.30.0 projects. The promotion-critical
`metrics.structural_similarity_monotone` template runs the existing
Mathlib-free `src/leverproof_lean/` package via `make test`; that same bounded
run builds the certificate checker used later in promotion. The remaining
templates stay in `src/slm_training/formal/lean/` with Mathlib `v4.30.0`.
Every formal command runs in its own bounded process group; a timeout interrupts
and, after the canonical grace, kills and reaps Lake/Lean descendants before the
typed timeout result is returned.
The same canonical template registry includes the six Semantic Factor Frontier
`AdvisoryResidual` obligations. Their proof digests bind the individual theorem and
declared source set, so a cached SFF artifact is validated by the normal formal
preflight path rather than an unknown-template side path.

| Module | Established claim | Scope |
| --- | --- | --- |
| `LeverProofLean.StructuralMetrics` | recall, structural-similarity components, and pointwise means are monotone under their declared inequalities; extra unmatched structure can lower the proxy | universal over modeled nonnegative, common-scale integers |
| `Forest` | certified closure is monotone/idempotent, never adds live candidates, extends history, and rollback preserves the declared partition | universal over finite modeled states |
| `Trace` | the Boolean JSON trace contract implies every accepted step applies its declared certified-removal set and prefix-preserves history | universal over accepted traces, assuming certificate replay happened first |
| `Recurrence` | delta/LayerScale update norms and winner-margin perturbations obey explicit algebraic bounds | conditional on the scale, contraction, and margin assumptions |

`make -C src/leverproof_lean test` rejects `sorry`, `admit`, custom `axiom`,
`unsafe`, and `native_decide`; `scripts.verify_formal_contracts` rejects the
proof placeholders and custom axioms in the Mathlib package. Both audit their
exported theorems for Lean's `sorryAx`. The repository hard run cap is the
timeout.

### Measured promotion-proof runtime (2026-08-01)

Durable results: [`autotrain-runtime-preflight-results.json`](autotrain-runtime-preflight-results.json).
This is local CPU proof-harness evidence, not a model-quality or ship result.

| Path | Cache | Command scope | Wall | Peak RSS | Result |
| --- | --- | --- | ---: | ---: | --- |
| LeverProof promotion path | cold (`.lake` excluded) | build + proof audit + protocol tests | 10.29 s | 875,944 KiB | pass |
| LeverProof promotion path | warm | build replay + proof audit + protocol tests | 1.02 s | 800,632 KiB | pass |
| Historical Mathlib package | warm | build only; diagnostic, not directly comparable | 4.58 s | 852,408 KiB | pass |

The cold LeverProof path completes well inside the 180-second repository cap
without fetching Mathlib. The comparator is intentionally not used as an exact
speedup claim because it omits the source/proof audits included by `make test`.

## Typed campaign integration

An `ExperimentSpec` may declare `formal_claims`. Each claim names one versioned
template and chooses a policy:

- `required`: execution is blocked unless the immutable preflight has status
  `proved`;
- `advisory`: `conditional`, `refuted`, or `unknown` results are persisted for
  review but do not independently block execution.

Run the preflight before locking the campaign:

```bash
python -m scripts.autoresearch formalize \
  --campaign-id <id> --experiment <experiment.json>
```

The command builds and audits Lean, then writes
`artifacts/formal_preflights/<content-sha>.json`. Copy the emitted
`formal_obligation` objects into `ExperimentCampaignV1.formal_obligations` and add
`{"kind": "formal_preflight"}` to `artifact_requirements`. `run` verifies the
artifact digest, experiment/claim binding, template version, proof-bundle digest,
and all linked Python/Lean source digests before planning or execution. A stale or
missing artifact fails closed.

The available templates are:

| Template | Normal policy | Result |
| --- | --- | --- |
| `metrics.structural_similarity_monotone` | required when the hypothesis exactly matches its inequalities | proved |
| `forest.history_preservation` | required for a closure/traversal change claiming history preservation | proved; linked to `OpenUIProofs.Trace.validTrace/v1` |
| `forest.lossy_history_counterexample` | required when testing the lossy reconstruction claim | refuted with a bounded counterexample |
| `recurrence.layerscale_stability` | advisory until a trained-transition bound is independently established | conditional |

The proposal compiler and matrix hypothesizer are instructed to attach a claim only
when its scope exactly matches one of these identifiers. They must keep recurrence
claims advisory until the missing trained-transition bound exists and must not turn a
formal result into a predicted metric.

The formal artifact records theorem, assumptions, open assumptions, evidence
scope, counterexample when applicable, Lean/Mathlib versions, proof digest, linked
source digests, and build-output digest. A formal refutation is a typed pre-training
no-go for that claim, not evidence that the invariant or broader research goal
should be abandoned (I14).

## Model-to-code linkage

`FormalTraceStepV1` is the JSON-side specification for
`OpenUIProofs.Trace.validTrace`; concrete candidate identities are first projected
to stable, request-local nonnegative ordinals. `check_formal_trace` implements the
same set-union, history-prefix, and adjacent-state checks for captured Python
traces. `formal_trace_from_closure` performs that projection directly from the
canonical exact-closure deductions after their certificate replay. The Lean
soundness theorem establishes what acceptance implies; Python tests exercise the
serialization/checker boundary.

This linkage is intentionally narrow. Source hashes make code drift visible, but a
hash is not a semantic refinement proof, and this structural checker does not replay
support certificates itself. The production exact-closure checker must establish
certificate validity before projecting the step into this trace. When a production
transition changes, update its template/model and add a parity or property test
connecting concrete records to the abstract step before treating the proof as
required evidence.

## Efficient proof ladder

Use the cheapest sound level that can falsify the proposal:

1. **Algebraic metric proof** — seconds; check direction, bounds, aggregation, and
   counterexamples.
2. **Finite-state trace invariant** — model candidate sets, certificates, history,
   and rollback; validate real traces with the executable checker.
3. **Recurrence bound** — prove local update inequalities, then measure or certify
   the missing Lipschitz/contraction assumptions. Keep the result conditional until
   that bridge exists.
4. **Training experiment** — only after the formal preflight leaves a genuinely
   empirical question.

This ordering saves compute by rejecting contradictions and underspecified claims
early. It does not use a toy proof as a surrogate quality score.
