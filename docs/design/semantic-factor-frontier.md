# Semantic-factor frontier (advisory residual plane)

**Status:** default-off experimental vertical slice (fixture/wiring claim class)  
**Authority:** exact compiler / solver / verifier unchanged  
**Campaign:** `SFF-anti-e237-v1` via `scripts/run_semantic_factor_frontier.py`  
**Related:** [adr-constrained-diffusion-topology-split.md](adr-constrained-diffusion-topology-split.md),
E236/E237/E729 topology failures, HyCE-RAG / SHIFT / Search-R1 research transfer notes below.

## Thesis

`slm-training` learns neither ordinary free symbol completion nor an opaque
reasoning graph. It may learn to **construct, compress, query, and rank a
query-conditioned semantic frontier** whose admissible actions are defined only
by the existing compiler, solver, scope, and verifier.

```text
exact plane (CompletionSession, SemanticState, forests, binders, certificates)
        │ read-only projection
        ▼
advisory evidence plane (role-ported factors + provenance)
        │ residual scores only
        ▼
legal candidates A_compiler(s)  — support unchanged
```

## Non-goals (this delivery)

- RL / PPO / GRPO / Search-R1 policy training  
- Spectral losses or spectral promotion  
- Production topology / hypergraph head  
- Graph pruning of legal support  
- Recurrent semantic inference / faithful SHIFT soft-token loop  
- Native hypergraph libraries (DeepHypergraph, GraphBLAS, …)  
- Baking request-local state into the static completion artifact  

## Contracts

| Type | Owner |
| --- | --- |
| `FactorPortV1` / `AdvisoryEvidenceFactorV1` / `SemanticEvidenceSnapshotV1` | `data/progspec/semantic_evidence.py` |
| Restart diffusion reference operator | `models/semantic_factor_propagation.py` |
| Residual scorer (default-off) | `models/semantic_residual_scorer.py` |
| Anti-E237 harness | `harnesses/experiments/anti_e237_semantic_factor_frontier.py` |

### Authority rule

\[
A_{\text{after}}(s)=A_{\text{compiler}}(s),\qquad
\operatorname{keys}(\Delta_s)\subseteq A_{\text{compiler}}(s).
\]

Complete singleton ⇒ factor calls = propagation = ranker applications = neural
forwards = 0. UNKNOWN is never collapsed to UNSUPPORTED.

## Representation arms (shared interface)

`none`, `direct_factors`, `lossy_pairwise`, `factor_node`, `role_ported_factor`,
`flat_hyperedge`, `role_shuffled`, `factor_shuffled`, `exact_typed_zero_parameter`,
`oracle_diagnostic`.

`factor_node` must round-trip membership exactly (lossless bipartite encoding of
higher-order factors). Role stripping changes features, not membership.

## Propagation

\[
S=B D_e^{-1}B^\top D_v^{-1},\qquad
c^{(t+1)}=\lambda r+(1-\lambda)Sc^{(t)}.
\]

Convergence is proved as a contraction for \(0<\lambda\le 1\) under positive
degrees (Lean + NumPy property tests). Convergence ≠ semantic correctness.
Contradiction uses separate nonnegative channels, never negative stochastic \(S\).

## Causal contract (anti-E237)

Arms: control / train-on·decode-off / train-on·decode-on (+ representation
controls). Kill when applications>0 & choice_changes=0; quality↓ with
choice_changes; singleton work; legal-set mutation; UNKNOWN collapse; leakage.

Fixture runs are **wiring-only**. Promotion needs held_out meaningful ≥
control+0.10 at n≥20 under the accepted ADR bar.

## Research transfer labels

| Source | Transfer |
| --- | --- |
| HyCE-RAG incidence restart | **Adapted** reference operator (not full RAG stack) |
| SHIFT soft tokens | **Surrogate blocked** — ContinuousLatentCodec is not SHIFT; faithful arm not implemented |
| SHIFT gate | **Not implemented** (would be advisory-only if added) |
| Search-R1 | **Adapted** trajectory ownership idea only; no RL |

## External versioned configs (metrics + parameters)

Metric inventories, formula definitions, scorer hyperparameters, and campaign
metadata live in **versioned external JSON** under
`src/slm_training/resources/experiments/semantic_factor_frontier/`. Harness Python
dispatches by formula `type` and arm id; changing the number of metrics,
endpoint list, alphas, or role weights should only edit the resource files.

| File | Schema | Purpose |
| --- | --- | --- |
| `metrics.v1.json` | `sff_metrics/v1` | Aggregates, derived formulas, required fields, campaign endpoints, efficiency gate |
| `scorer_params.v1.json` | `sff_scorer_params/v1` | Arm table (`representation`/`decode_apply`/`alpha`), defaults (restart λ, role multipliers, …) |
| `campaign.v1.json` | `sff_campaign/v1` | Campaign id, seeds, budget, gates; links to the two resources above |

Each file requires `schema` + `version` at the root. Loaders:
`semantic_factor_config.py`. Formula execution: `semantic_factor_metric_engine.py`
(supported types only — new **kinds** of calculation need a one-time engine
extension; new metrics of an existing type are file-only).

Scoreboards bind the exact files used via `config_resources` (repo-relative
`path`, `schema`, `version`, content `sha256`). Missing binds fail
`validate_scoreboard_metrics`.

### How to change parameters without harness edits

1. Edit `scorer_params.v1.json` (new arm, different α, new role weight).
2. Edit `metrics.v1.json` (add aggregate of type `mean`/`ratio`/…, extend
   `required_arm_fields`, add endpoint).
3. Bump the component in `resources/versions.json` (or `no-bump:` when
   behavior-neutral). Schema version (`sff_*/v1`) stays until the shape breaks.
4. Re-run `python -m scripts.run_semantic_factor_frontier`.

## Metrics contract (required on every future run)

Module: `harnesses/experiments/semantic_factor_metrics.py` (required field sets
loaded from `metrics.v1.json`).

Scoreboard kind: **`semantic_factor_frontier_results/v3`** (from metrics resource).

Writers **must** call `validate_scoreboard_metrics(payload)` before durable
write. The CLI exits non-zero if the contract fails. Campaigns **must** declare
endpoints `wall_ms_mean` and `quality_per_ms` (listed in the metrics resource).

### Required per-arm runtime fields

Owned by `metrics.v1.json` → `required_arm_fields`. Today includes:
`wall_ms_total`, `wall_ms_mean`, `wall_ms_p50`, `wall_ms_p95`, `wall_ms_min`,
`wall_ms_max`, `project_ms_*`, `score_ms_*`, `quality_per_ms`,
`wall_ms_mean_vs_control`, `wall_ms_total_vs_control`,
`delta_accuracy_per_extra_ms`, `dominates_control_quality_runtime`,
`delta_accuracy_vs_control`.

### Required campaign `runtime` block

Owned by `metrics.v1.json` → `required_runtime_block`:
`campaign_wall_s`, `campaign_process_s`, `timer` (= `time.perf_counter`),
`decision_path`, `control_wall_ms_mean`.

### Required `config_resources` bind

`metrics` / `scorer_params` / `campaign` each with `path`, `schema`, `version`,
`sha256`.

Runtime is a **decision axis**, not a footnote: a slightly worse accuracy at
~100× lower `wall_ms_mean` can be preferred. Tests in
`tests/test_harnesses/experiments/test_semantic_factor_metrics.py` and
`test_semantic_factor_config.py` pin this.

## Claims (validate / invalidate)

Each campaign run emits `claims[]` via
`harnesses/experiments/semantic_factor_claims.py`. Verdicts are **validated**,
**invalidated**, or **inconclusive** from measured metrics (not narrative).

| ID | Statement family |
| --- | --- |
| C1–C5 | Safety / representation (support, singleton, UNKNOWN, decode-off, lossless) |
| C6–C11 | Causal residual arms (role_ported, factor_node, direct, roles, higher-order, exact typed) |
| C12 | Kill criterion: applications without choice changes |
| C13–C14 | Math: column-stochastic S; soft-token non-injectivity |
| C15–C16 | Forbidden paths unimplemented; promotion bar |
| C17–C19 | Runtime tracked; efficiency gate; exact_typed quality×runtime |

## Run

```bash
python -m scripts.run_semantic_factor_frontier \
  --out-dir outputs/runs/semantic_factor_frontier_measured \
  --docs-json docs/design/semantic-factor-frontier-results.json \
  --seeds 0 1 2
```

Results: [semantic-factor-frontier-results.md](semantic-factor-frontier-results.md).
