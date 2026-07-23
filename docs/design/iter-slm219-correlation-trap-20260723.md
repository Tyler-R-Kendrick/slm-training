# SLM-219: correlation-trap early-warning retrospective

**Verdict:** `inconclusive`

**Report hash:** `0340c57d223df0304bb520da638c56b7af786668620f38ac490ca4574d36fffa`

**Inventory hash:** `5f4bcabae030afb92a66b34fe16d11c154bb26f43094f20b97617b72e4baa774`

## Outcome

The deterministic historical-prefix reproduction produced one transient outcome-only collapse at the 4k-token checkpoint. The three spectral roles are dependent views of the same seed and family, so this run cannot establish that a correlation trap predicts collapse.

Correlation-trap language is **not authorized as an early-stopping rationale**. No recommendation artifact was emitted and no production behavior changed.

## Trajectory inventory

| Source | Endpoints | Resolved step checkpoints | Eligible trajectory | Missing intervals |
| --- | ---: | ---: | --- | --- |
| `docs/design/iter-e501-e396-e500-warm-start-20260719.json` | 4 | 0 | `false` | all pre-final intervals; reports retain endpoints only |
| `docs/design/iter-e502-initialization-prior-retention-20260719.json` | 4 | 0 | `false` | all pre-final intervals; reports retain endpoints only |
| `docs/design/iter-e503-initialized-weight-retention-20260719.json` | 4 | 0 | `false` | all pre-final intervals; reports retain endpoints only |
| `docs/design/iter-e504-parent-corpus-replay-20260719.json` | 5 | 0 | `false` | all pre-final intervals; reports retain endpoints only |

The original E501-E504 artifacts retain endpoints, not time series. SLM-219 therefore reproduced one six-point trajectory with 6 hash-verified deterministic-prefix checkpoints and three preregistered matrix roles.

| Point | Step | Tokens | Structure | Duplicate-subtree rate | MLP trap z | Cross-attn trap z | LM-head trap z |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t0000 | 0 | 0 | 0.2145 | 0.000 | 82.218 | 1.264 | -0.628 |
| t1039 | 22 | 1039 | 0.3369 | 0.000 | 97.429 | 1.150 | -0.489 |
| t2047 | 42 | 2047 | 0.1986 | 0.000 | 73.409 | 1.698 | -0.643 |
| t3008 | 61 | 3008 | 0.3375 | 0.000 | 70.595 | 1.154 | -0.651 |
| t4007 | 80 | 4007 | 0.1137 | 0.667 | 100.359 | 1.821 | -0.662 |
| t5019 | 99 | 5019 | 0.2117 | 0.000 | 138.247 | 2.124 | -0.364 |

## Frozen rules and controls

- Collapse: structure ≤ 0.15 and repetition ≥ 0.33 for 1 snapshot; spectral fields used: none. The threshold was frozen from published E501/E502 parent and collapsed-endpoint evidence before inspecting the reproduced spectra.
- Warning: trap z ≥ 2.0 or parent-relative outlier-energy increase ≥ 0.10 for 2 snapshots; onset is the confirmation step, never the backdated first qualifying step.
- Actual dependent-role evaluation: TP `1`, FP `0`, FN `2`, TN `0`, FPR `undefined` (no independent non-collapse trajectory means no FPR denominator).
- Time-shuffled control: precision `1.000`, recall `0.333`, FPR `undefined`.
- Synthetic contract control: precision `1.000`, recall `0.667`, FPR `0.000`. Synthetic results are not model evidence.
- Independent same-shape null: 32 draws, FPR `0.000` at z ≥ 2.0.

## Native versus WeightWatcher

Pinned `weightwatcher-0.7.5` completed 18 matrix comparisons in the analysis environment. Native and WeightWatcher stable rank agree to maximum absolute error `8.527e-14`. WeightWatcher alpha changed only descriptively because the evidence is limited to one seed/family with a transient collapse:

| Role | Parent alpha | Final alpha | Delta |
| --- | ---: | ---: | ---: |
| `cross_attn_out` | 5.2704 | 5.2781 | +0.0077 |
| `lm_head` | 6.4712 | 6.4553 | -0.0159 |
| `mlp_out` | 3.3397 | 3.3794 | +0.0397 |

## Recipe and durable evaluation evidence

- CPU; historical code `f2ab01f8ae6af6be49db3f294cd166fe034b67a5`; HF context backend; seed 0; batch 2; LR 3e-4; 22/42/61/80/99 optimizer steps; 1,039/2,047/3,008/4,007/5,019 target tokens. Each checkpoint is an independent deterministic prefix rerun from the same parent and uniform sample order, not a resumed-stage claim.
- Matrix set `slm219_correlation_traps`, version `ncs1-03-v2`; three roles; 24 same-shape null draws per checkpoint/role; honesty mode `no-design-md-context`.
- Smoke suite n=3 at every checkpoint. Each evaluation emitted an AgentEvals JSONL spec and pinned AgentV result bundle; all six were honest non-ship 0/1 results with zero execution errors.
- Evidence manifest: `docs/design/iter-slm219-correlation-trap-evidence-20260723.json`. Normalized AgentV artifacts: `docs/design/iter-slm219-correlation-trap-agentv-20260723/`.
- Held-out NLL and gradient norms are explicitly unavailable in the historical telemetry. Train loss is retained as a proxy and RMS drift/update norm are computed from the hash-verified checkpoints; neither is relabeled as held-out evidence.
- Scratch continuation checkpoints were rejected, not promoted, and not synced. They do not alter the serving roster or model card.

Current `main` correctly rejects the old E500 corpus under `symbol_only/v2`. The reproduction used the historical code revision that originally admitted that committed corpus and deterministic prefix reruns; no current gate was weakened.

## Decision

- E501-E504 preserve final endpoints but no resolvable step-indexed pre-collapse checkpoints
- a deterministic six-point prefix reproduction on the historical code revision produced one independently labeled transient collapse at 4k tokens
- the three matrix roles are dependent views of one seed/family, so their warning counts cannot establish held-out precision, recall, or false-positive rate
- time shuffling and the single seed/family do not establish temporal specificity or cross-family generalization
- WeightWatcher stable rank agrees with the native implementation, but its alpha trajectory remains descriptive rather than predictive
- historical telemetry omitted held-out NLL and gradient norms; train loss and RMS drift are retained as explicitly weaker baselines

Verdict: **inconclusive and not supported for use**. The actual trajectory contains one independently labeled collapse, but one seed/family and three dependent matrix roles cannot identify generalizable precision, recall, or false-positive rate. A new multi-seed/family study requires independently labeled trajectories and persisted held-out NLL/gradient telemetry.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_correlation_trap_retrospective --check
```
