# SLM-192 (FFE3-01): stage-accurate flow-pipeline cost profile (slm192-profile-flow-pipeline-20260721)

Matrix set: `slm192_profile_flow_pipeline`

Version: `ffe3-01-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no ship-gate claim is made.

## Hypothesis

Valid-edit bridge/training/decode/closure/verification stages have separable, measurable CPU cost profiles; the dominant bottleneck on toy fixtures is either candidate enumeration, exact closure, or verifier replay; and the combined per-target bridge+enumeration cost extrapolates to an on-policy epoch budget.

## Falsifier

The cold and warm profiles are identical (no caching benefit), or the top bottleneck is not enumeration/closure/verification, or the per-target cost extrapolates beyond the 30-minute on-policy epoch bound despite the tiny fixture domain.

## Arms

| arm_name | condition | total_ms | n_repeats | top_span | top_span_ms |
| --- | --- | --- | --- | --- | --- |
| no_model_canonical_baseline | cold | 0.813 | 1 | canonicalization | 0.503 |
| no_model_canonical_baseline | warm | 1.301 | 2 | canonicalization | 0.718 |
| bridge_planner_canonical_greedy | cold | 31.049 | 1 | candidate_enumeration | 30.007 |
| bridge_planner_canonical_greedy | warm | 14.162 | 2 | candidate_enumeration | 13.393 |
| x22_edit_enumeration | cold | 93.308 | 1 | candidate_enumeration | 91.813 |
| x22_edit_enumeration | warm | 81.369 | 2 | candidate_enumeration | 78.855 |
| exact_closure_toy | cold | 0.658 | 1 | exact_closure | 0.538 |
| exact_closure_toy | warm | 0.777 | 2 | exact_closure | 0.657 |
| support_oracle_check | cold | 0.123 | 1 | verifier | 0.083 |
| support_oracle_check | warm | 0.235 | 2 | verifier | 0.159 |
| direct_scorer_fixture | cold | 11.021 | 1 | candidate_projection | 9.434 |
| direct_scorer_fixture | warm | 15.326 | 2 | candidate_projection | 13.901 |

## Cost gate

Max on-policy epoch seconds: `1800`

Allowed strategy: **on_policy_viable**

Enumeration bound (enum/closure/verifier > 50% warm wall time): **True**

Bottlenecks:
- candidate_enumeration
- candidate_projection
- state_materialization

## On-policy feasibility

Strategy: **on_policy_viable**

Projected seconds per target: `0.0955`

Projected seconds for 108 targets: `10.32`

Extrapolated five-seeds seconds: `51.59`

Extrapolated DAgger round seconds: `5.16`

Extrapolated confirmation suite seconds: `1.03`

Rationale: Measured warm bridge+enum per target is 0.0955s. 108 targets = 10.3s; five seeds = 51.6s, well under the 1800s epoch bound on this fixture.

## Disposition

**cost_profile_wired**

CPU-only fixture cost profile wired for all declared arms; no model, GPU, checkpoint, or ship claim is made.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The cost spans are measured over deterministic CPU-only operations with synthetic model and verifier signals. Production on-policy training requires real model decode timing, GPU kernel profiles, and checkpoint benchmarks before any ship claim.

## Honest caveats

- Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.
- Direct/flow decode arms are optional or skipped when torch is absent; no learned policy or value head is measured.
- The toy support provider is synthetic (payload['ok'] flag); real support queries require a verifier and problem expander.
- Real model decode cost (forward pass, beam scoring, sampler overhead) is not measured here; extrapolations are from CPU-only canonical operations.
- Only the HERO fixture is exercised; production targets will have different statement counts, domain sizes, and verifier profiles.

## Reproducibility

```bash
python -m scripts.profile_flow_pipeline --describe
python -m scripts.profile_flow_pipeline --fixture
```
