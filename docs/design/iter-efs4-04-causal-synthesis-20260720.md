# EFS4-04 — Causal diagnosis and explicit architecture disposition

**Date:** 2026-07-20T04:04:04.102502Z
**Campaign:** evidence-first-semantic-slm-campaign
**Manifest hash:** `815b9582843ac61a09afb565a621b1037f0344016bac0d463d8a41454fd4a9fd`
**Generation command:** `python -m scripts.synthesize_efs_campaign --manifest /tmp/pytest-of-codex/pytest-320/test_validate_only_with_existi0/manifest.json --docs-design /tmp/pytest-of-codex/pytest-320/test_validate_only_with_existi0`

## 1. Executive verdict

The Evidence-First Semantic SLM Campaign has reached a wiring/plan-only terminal state. No EFS experimental branch has cleared its activation gate or produced durable frontier evidence. The only honest causal diagnosis is **insufficient valid evidence**; the primary blocker is unresolved measurement provenance and decoder invariance. No architecture branch is promoted. All required disposition items are recorded as ``NOT_RUN_BY_GATE``/``INCONCLUSIVE`` or, for safety infrastructure, ``ADOPT_AS_SAFETY_ONLY`` without a quality claim.

## 2. Campaign execution / completeness table

| Issue | Hypothesis | State | Result refs |
|-------|------------|-------|-------------|
| SLM-103 | efs0-01-checkpoint-provenance | `MISSING` | — |
| SLM-104 | efs0-02-decode-invariance | `MISSING` | — |
| SLM-105 | efs0-03-meaningful-v2 | `MISSING` | — |
| SLM-106 | efs0-04-judge-independence | `MISSING` | — |
| SLM-107 | efs0-05-rejected-lever-readjudication | `MISSING` | — |
| SLM-108 | efs1-01-external-ceiling | `MISSING` | — |
| SLM-109 | efs1-02-exposure-ladder | `MISSING` | — |
| SLM-110 | efs1-03-empty-length-bias | `MISSING` | — |
| SLM-111 | efs2-01-x22-scaling | `MISSING` | — |
| SLM-112 | efs2-02-trigger-telemetry | `MISSING` | — |
| SLM-113 | efs2-03-conflict-slice-repair | `MISSING` | — |
| SLM-115 | efs2-04-verifier-cascade | `MISSING` | — |
| SLM-118 | efs3-01-solver-state-supervision | `MISSING` | — |
| SLM-120 | efs3-02-corruption-curriculum | `MISSING` | — |
| SLM-124 | efs3-03-b3-capacity-v2 | `MISSING` | — |
| SLM-127 | efs3-04-candidate-selector | `MISSING` | — |
| SLM-130 | efs3-05-canonical-ast-dedup | `MISSING` | — |
| SLM-133 | efs3-06-ast-sketch-retrieval | `MISSING` | — |
| SLM-135 | efs4-01-trailed-assumptions | `MISSING` | — |
| SLM-138 | efs4-02-shared-recursive-denoiser | `MISSING` | — |
| SLM-139 | efs4-03-stochastic-recursive-state | `MISSING` | — |

## 3. Measurement validity findings

- **efs0-01-checkpoint-provenance** (SLM-103): `MISSING` — No committed result manifest matched the expected refs.
- **efs0-02-decode-invariance** (SLM-104): `MISSING` — No committed result manifest matched the expected refs.
- **efs0-03-meaningful-v2** (SLM-105): `MISSING` — No committed result manifest matched the expected refs.
- **efs0-04-judge-independence** (SLM-106): `MISSING` — No committed result manifest matched the expected refs.
- **efs0-05-rejected-lever-readjudication** (SLM-107): `MISSING` — No committed result manifest matched the expected refs.

## 4. Causal layer diagnosis

**Primary:** `insufficient_valid_evidence`

Core measurement issues (checkpoint provenance, decoder invariance, semantic metric, judge independence, re-adjudication) are not all POSITIVE. Without durable, invariant, independently measured evidence, no causal diagnosis of training, data, search, or architecture can be asserted.

**Counterfactual evidence:** Counterfactual: if SLM-103/104/105/106/107 were all POSITIVE, the remaining NOT_RUN_BY_GATE states could be attributed to training-exposure or architecture limits rather than to measurement uncertainty.

**Secondary considerations:** measurement_limited

## 5. Semantic-quality and cost Pareto fronts

No frontier Pareto front is available: all rows are either plan-only or diagnostic-only. The existing fixture rows report only wall-second / verifier-call placeholders. Any Pareto claim requires the exposure ladder (SLM-109) and external-ceiling (SLM-108) frontier runs to complete first.

## 6. Architecture disposition table

| Item | Disposition | Supporting | Falsifying | Next action |
|------|-------------|------------|------------|-------------|
| compiler-owned grammar/schema/binding lattice and exact closure | `ADOPT_AS_SAFETY_ONLY` | — | — | Keep as correctness infrastructure; do not claim quality improvement. |
| reduced-product/cross-domain propagation claims | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| reversible decisions, local nogoods, and certificate-backed trailing | `ADOPT_AS_SAFETY_ONLY` | — | — | Keep as correctness infrastructure; do not claim quality improvement. |
| free-form typed-AST/topology diffusion | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| X22/all-valid tree-edit diffusion | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| deterministic shared recursive denoiser/deep supervision | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| request-local recurrent latent persistence | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| triggered PTRM/inference-only low-level noise | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| learned GRAM-style high-level stochastic state | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| candidate AST dedup/semantic mode tracking | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| contract-grounded selector and abstention | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| conflict-slice remasking | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| cheap-to-expensive verifier cascade/cache | `ADOPT_AS_SAFETY_ONLY` | — | — | Keep as correctness infrastructure; do not claim quality improvement. |
| choice representation/capacity conclusion | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| gold/on-policy mixed supervision and nearly-solved curriculum | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| AST-sketch data balancing and choice-native retrieval | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |
| content-floor/length/mask-mass corrections | `NOT_RUN_BY_GATE` | — | — | Run the related EFS branch to a terminal state before disposition. |

## 7. Comparison with LDT/TRM/PTRM/GRAM and agent critiques

The repository contains adapted implementations of lattice-diffusion, tree-edit (X22), triggered PTRM, shared-recursive denoiser, and GRAM-style stochastic-state ideas. None has cleared ship-gates or produced durable frontier evidence. The honest position is that these remain research hypotheses, not reproduced architectures.

## 8. Interaction with VSS and CAP

This synthesis links the EFS branch decisions but does not duplicate VSS or CAP claims. See ``docs/design/verified-scope-solver.md`` and the CAP5 synthesis for their respective dispositions. EFS4-04 treats VSS/CAP outputs as external evidence inputs.

## 9. Champion / promotion decision

**no_promotion**. No checkpoint is promoted.

## 10. Next three experiments or consolidation plan

### 1. Make frontier checkpoint provenance and persistence fail-closed.
- **Why not duplicate:** EFS0-01 is the root dependency of every causal claim; no later branch is interpretable without it.
- **Expected information gain:** Distinguishes measurement-limited from genuinely architecture-limited conclusions.
- **Smallest decisive experiment:** Sync one E228-class checkpoint to hf://buckets/TKendrick/OpenUI and verify hash from a fresh clone.
- **Budget:** <1 GPU-hour + bucket storage
- **Kill criterion:** Hash mismatch or unresolvable checkpoint blocks all dependent syntheses.
- **Dependencies:** SLM-103

### 2. Run the ≥100× E228 exposure ladder to falsify 'just train longer'.
- **Why not duplicate:** EFS1-02 is the only branch designed to separate exposure from representation/objective limits.
- **Expected information gain:** Either reveals an exposure threshold or falsifies the current recipe.
- **Smallest decisive experiment:** Continue seed-0 run to 128× T0 and confirm 1×/threshold/128× on seeds 1–2.
- **Budget:** ~8 GPU-hours
- **Kill criterion:** Semantic metrics flat with tight CI excluding minimum useful delta.
- **Dependencies:** SLM-104, SLM-105, SLM-109

### 3. Re-run the B3 surface-vs-choice capacity ladder with the corrected choice-native decoder.
- **Why not duplicate:** EFS3-03 isolates representation capacity from the prior decoder confound.
- **Expected information gain:** Determines whether externalized syntax shifts the quality-capacity curve.
- **Smallest decisive experiment:** 18-row grid (2 representations × 3 widths × 3 seeds) on frozen recipe.
- **Budget:** ~6 GPU-hours
- **Kill criterion:** Capacity curves overlap within equivalence margin.
- **Dependencies:** SLM-104, SLM-124

## 11. Limitations and exact reproduction command

This report is a wiring-grade synthesis over plan/fixture manifests. It does not run training, download models, or mutate checkpoints. All claims are scoped to the committed result manifests under ``docs/design/``.

```bash
python -m scripts.synthesize_efs_campaign \
  --manifest docs/design/evidence-first-semantic-slm-campaign-v1.json \
  --docs-design docs/design \
  --out-json docs/design/iter-efs4-04-causal-synthesis-$(date +%Y%m%d).json \
  --out-md docs/design/iter-efs4-04-causal-synthesis-$(date +%Y%m%d).md
```

## Unresolved risks

- efs0-01-checkpoint-provenance (MISSING): No committed result manifest matched the expected refs.
- efs0-02-decode-invariance (MISSING): No committed result manifest matched the expected refs.
- efs0-03-meaningful-v2 (MISSING): No committed result manifest matched the expected refs.
- efs0-04-judge-independence (MISSING): No committed result manifest matched the expected refs.
- efs0-05-rejected-lever-readjudication (MISSING): No committed result manifest matched the expected refs.
- efs1-01-external-ceiling (MISSING): No committed result manifest matched the expected refs.
- efs1-02-exposure-ladder (MISSING): No committed result manifest matched the expected refs.
- efs1-03-empty-length-bias (MISSING): No committed result manifest matched the expected refs.
- efs2-01-x22-scaling (MISSING): No committed result manifest matched the expected refs.
- efs2-02-trigger-telemetry (MISSING): No committed result manifest matched the expected refs.
- efs2-03-conflict-slice-repair (MISSING): No committed result manifest matched the expected refs.
- efs2-04-verifier-cascade (MISSING): No committed result manifest matched the expected refs.
- efs3-01-solver-state-supervision (MISSING): No committed result manifest matched the expected refs.
- efs3-02-corruption-curriculum (MISSING): No committed result manifest matched the expected refs.
- efs3-03-b3-capacity-v2 (MISSING): No committed result manifest matched the expected refs.
- efs3-04-candidate-selector (MISSING): No committed result manifest matched the expected refs.
- efs3-05-canonical-ast-dedup (MISSING): No committed result manifest matched the expected refs.
- efs3-06-ast-sketch-retrieval (MISSING): No committed result manifest matched the expected refs.
- efs4-01-trailed-assumptions (MISSING): No committed result manifest matched the expected refs.
- efs4-02-shared-recursive-denoiser (MISSING): No committed result manifest matched the expected refs.
- efs4-03-stochastic-recursive-state (MISSING): No committed result manifest matched the expected refs.
