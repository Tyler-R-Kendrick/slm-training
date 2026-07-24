# SLM-194 (FFE3-03): exact-fallback candidate proposals

**Status:** measured CPU fixture screen. **Verdict:** `retain_exact_cached_enumeration`.

## Result

| arm | best k | target recall | acceptable recall | fallback | warm p50 delta | final work avoided |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grammar_partition | all | 1.000 | 1.000 | 0.000 | -0.002 | 0 |
| description_retrieval | all | 1.000 | 1.000 | 0.000 | +0.000 | 0 |
| tiny_mlp | all | 1.000 | 1.000 | 0.000 | +0.000 | 0 |
| low_rank_cross_attention | all | 1.000 | 1.000 | 0.000 | -0.002 | 0 |
| direct_policy_logits | 16 | 1.000 | 1.000 | 0.000 | -0.000 | 0 |
| flow_rate_logits | all | 1.000 | 1.000 | 0.000 | +0.008 | 0 |

No non-oracle learned arm cleared the joint ≥95% target/acceptable
recall and ≥30% warm-p50 gate. Mandatory exact fallback restored every
omitted candidate, so final membership/output parity was exact but no
final projection, verifier, or support work was avoided. The proposal
overhead was therefore not amortized. Retain exact cached enumeration.

The oracle is diagnostic only. The tiny four-row/two-cluster SLM-196
fixture cannot license a production proposal claim, and confirmation was
not touched.

## Safety and provenance

- Exact final parity: `True`.
- UNKNOWN-as-negative: `False`.
- Candidate features contain no final source or future witness text.
- SLM-192 profile SHA: `f45d0ad153a4e9bcc7c65fa75d94fe08e7aab71903892a421595228ba8d79cf9`.
- SLM-193 cache SHA: `adbbf9becd8dc198d784a1961704a592ebaab960ad46900e31e02de09f652768`.
- AgentV: `{'total': 5, 'passed': 5, 'failed': 0, 'executionErrors': 0, 'durationMs': 24, 'meanScore': 1}`.

## Recipe

- Device/backend: `cpu` / `torch fixture + exact compiler corpus`.
- Steps/seed: `32` / `0`.
- k grid: `[1, 2, 4, 8, 16, 'all']`.
- Uncertainty: `{'method': 'SLM-183 cluster_bootstrap_ci', 'cluster_key': 'target_cluster_id', 'resamples': 1000, 'alpha': 0.05, 'power_status': 'underpowered_two_cluster_fixture'}`.
- Wall seconds: `2.851`.
- Checkpoint: none; no promotion.

## Caveats

- The SLM-196 fixture has four rows and two target clusters; confidence intervals are descriptive and underpowered.
- Cheap candidate descriptors come from the exact bridge corpus. The matrix measures projection/verification scheduling, not compiler enumeration avoidance.
- The description arm uses the deterministic SLM-176 fixture-style hash surrogate; no production retrieval checkpoint exists.
- Direct and flow logits are small fixture-trained scorers on the same dynamic interface, not promoted checkpoints.
- Oracle acceptable scores use labels and are diagnostic only.
- Confirmation data was not touched.

## Reproduce

```bash
python -m scripts.run_candidate_proposal_matrix --eval
```
