# Quality experiment matrix — TwoTower OpenUI

> **Lineage rule:** E/X rows are ablation evidence, never deployable model
> identities. Modern V4+ runs require `--parent <checkpoint>`; use
> `--scratch-control` only for explicitly non-deployable controls. Production
> improvements use [`scripts/model_cycle.py`](../../scripts/model_cycle.py) and
> the [two-track lineage contract](model-lineage.md).

> **Validity notice:** Historical scores below predate strict train/eval isolation. Curriculum runs imported adversarial fixtures and are invalid for model selection; regenerate every result with the repaired harness before comparing models.
> Agentic workers: implement levers below, then run `python -m scripts.run_quality_matrix`.

**Goal:** Clear honest `--ship-gates` by attacking fidelity=0 / parse≈0 with every approach from the quality brief.

**Architecture:** Each row is an isolatable lever (plus a stacked `combo` run). All runs use scratch context on CPU by default; HF is optional when cached.

**Tech stack:** TwoTower, OpenUI grammar, ship_gates, preference composite reward.
**Research map:** [research-lineage.md](research-lineage.md) (MaskGIT, constrained diffusion, DPO/GRPO surrogates);
correction / remask candidates: [research-correction-critics.md](research-correction-critics.md).

---

## Failure baseline (`twotower_v1_ship`)

| Suite | parse | fidelity | struct | Gate |
| --- | --- | --- | --- | --- |
| smoke | 1.0 | **0.0** | 0.68 | fail fidelity |
| held_out | **0.0** | 0.0 | 0.48 | fail parse |
| adversarial | **0.0** | 0.0 | 0.45 | fail parse |
| ood | 0.25 | 0.0 | 0.45 | pass |
| rico_held | **0.0** | ~0 | 0.17 | fail |

## Matrix

| ID | Approach | Primary lever | Expected gate delta | Run id |
| --- | --- | --- | --- | --- |
| E0 | Baseline ship recipe | `v1` train, LTR primary, no repair | establish current | `qx_e0_baseline` |
| E1 | Constrained decode | `grammar_ltr_repair=true` | ↑ held_out/adv parse | `qx_e1_repair` |
| E2 | Curriculum data | stages A→B→C sampling | ↑ rico struct / parse | `qx_e2_curriculum` |
| E3 | Fidelity aux loss | `fidelity_loss_weight>0` | ↑ smoke/held fidelity | `qx_e3_fidelity` |
| E4 | Schema conditioning | `schema_in_context=true` | ↑ component recall / parse | `qx_e4_schema` |
| E5 | Preference + best-of-N | valid-worse pairs + `best_of_n=4` | ↑ reward / parse | `qx_e5_pref_bon` |
| E6 | Retrieval skeletons | `retrieval_k=1` | ↑ rico parse/struct | `qx_e6_retrieve` |
| E7 | Capacity upgrade | larger d_model / layers / gen_len | ↑ rico struct | `qx_e7_capacity` |
| E8 | Combo (all) | E1–E7 stacked | best chance at ship | `qx_e8_combo` |
| E9b | Fidelity anti-leak | Soft curriculum mix + fidelity aux + schema + repair | ↑ fidelity without `:adv.*` smoke leak | `qx_e9b_fidelity_antileak` |
| E10 | GRPO-lite RL | Online group rollouts + structure-only reward | ↑ reward / parse after SFT | `qx_e10_grpo` |

## V2 matrix (root-cause fixes — compositional tokenizer + slot contract)

| ID | Approach | Primary lever | Run id |
| --- | --- | --- | --- |
| E11 | Compositional tokenizer | F1 subtoken placeholders | `qx_e11_compositional_tok` |
| E12 | Slot contract | F2 inventory in context + constrained decode | `qx_e12_slot_contract` |
| E13 | LTR aligned | F4 weighted suffix CE + LTR repair | `qx_e13_ltr_aligned` |
| E14 | Namespace augment | F5 `:acme.*` re-prefix train data | `qx_e14_namespace_aug` |
| E15 | Combo | E12 + E13 + leak-free curriculum + E7 capacity | `qx_e15_combo` |
| E16 | Long train | E15 at 2000+ steps | `qx_e16_long_train` |
| E17 | Decode sweep | Eval-only gen_steps/repair/best-of-N on E15 ckpt | `qx_e17_decode_sweep` |

## V3 matrix (length-safe decode + SOTA diffusion levers)

Root cause after V2: compositional tokenization lengthened programs (fixture max ~160
tokens) while LTR decode still capped at 64–96 → **parse stayed 0 even at 2000 steps**.

Constrained-decode follow-ups (this branch): force-emit must run over real logits (never
zero stand-ins), `placeholder_required` is not a hard error mid-string, and slot-contract
intersections must not drop `.` / whitespace inside quoted placeholders.

| ID | Approach | Primary lever | Run id |
| --- | --- | --- | --- |
| E18 | Length-safe LTR | `grammar_ltr_max_tokens≥192`, stages `(64,128,192,256)` | `qx_e18_length_safe` |
| E19a | MaskGIT-primary | Train/infer match: MaskGIT decode (no LTR-primary) | `qx_e19a_maskgit_primary` |
| E19b | LTR-matched | LTR-primary + strong `ltr_loss_weight` + length-safe | `qx_e19b_ltr_matched` |
| E20 | Template fill | Slot-contract skeleton seed + MaskGIT refine | `qx_e20_template_fill` |
| E21 | MDLM schedule | Continuous-time absorbing mask + `1/t` CE weights | `qx_e21_mdlm_schedule` |
| E22 | Remasking | Confidence remask of weak committed tokens | `qx_e22_remask` |
| E29 | Champion | E18+E20+E21+E22 + slot contract + capacity | `qx_e29_champion` |

```bash
# Diagnostic ceiling (gold-as-prediction must score ~1.0)
python -m scripts.diagnose_eval --train-dir outputs/data/train/v1 \
  --test-dir outputs/data/eval/v1

# Historical V2 matrix (CLI default is v3; prefer v4/v5 for ship claims)
python -m scripts.run_quality_matrix --matrix v2 --steps 800 --device cpu

# Isolated levers
python -m scripts.run_quality_matrix --only E11,E12,E13 --steps 800

# Champion combo + decode sweep (E17 needs E15 checkpoint)
python -m scripts.run_quality_matrix --only E15,E17 --steps 1200 --gen-steps 16

# V3 focused ship path (length-safe → decode match → template → champion)
python -m scripts.run_quality_matrix --matrix v3 --only E18,E19a,E19b,E20,E29 \
  --steps 400 --device cpu --context-backend scratch

# V4 honest contract + decode scaling
python -m scripts.run_quality_matrix --matrix v4 --only E35,E36 \
  --steps 40 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control
```

## Success criteria (honest gates)

Same policy as `docs/design/adversarial-review.md`. Matrix may evaluate `rico_held` with `--rico-limit` for CPU time; full 1500 is the ship claim.

**Honesty rule (E35):** template fill / slot-contract decode must obtain inventory
from the user-visible prompt (or DESIGN.md), not by reading `gold.placeholders`
as a hidden eval channel. Eval may *surface* gold inventory into the prompt via
`ensure_prompt_inventory` (inventory-in-prompt API); `_resolve_slot_contract`
then extracts it.

## Commands

```bash
# Build curriculum corpus (once) — train excludes test fixture structures automatically
python -m scripts.build_train_data --source all --version v1_curriculum \
  --synthesizer quality --curriculum

python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json

# Migrate legacy v1 tokenizer checkpoints (optional; fresh train preferred)
python -m scripts.migrate_checkpoint \
  --checkpoint outputs/runs/legacy/checkpoints/last.pt \
  --train-records outputs/data/train/v1/records.jsonl \
  --output outputs/runs/legacy_v2/checkpoints/last.pt

# Run full matrix (scratch CPU). Use --no-design-md-context when seeding from
# the fixture-demo ship checkpoint (it was trained without DESIGN.md in context).
python -m scripts.run_quality_matrix \
  --device cpu --context-backend scratch --steps 800 \
  --no-design-md-context \
  --seed-checkpoint outputs/runs/twotower_v1_ship/checkpoints/last.pt

# Fidelity-focused + RL stages
python -m scripts.run_quality_matrix --only E9b,E10 --steps 200 --context-backend scratch

# V2 root-cause fixes (default matrix)
python -m scripts.run_quality_matrix --matrix v2 --only E11,E12,E15 --steps 800

# Online RL alone (after an SFT checkpoint)
python -m scripts.train_rl \
  --checkpoint outputs/runs/qx_e9b_fidelity_antileak/checkpoints/last.pt \
  --train-records outputs/data/train/v1/records.jsonl \
  --out-dir outputs/runs/grpo --steps 50 --group-size 4

# Cycle telemetry (train + generate spans)
python -m scripts.bench_telemetry --train-steps 8 --gen-prompts 8
```

## External semantic ceiling (SLM-108 / EFS1-01)

Off-the-shelf 1–7B instruct/code models scored through the same compiler-owned
legal candidate space as the tiny SLM. This is a control experiment, not a
replacement model. The matrix set is registered separately because it evaluates
external weights rather than training a new checkpoint.

| Arm | Model | Decode | Purpose |
| --- | --- | --- | --- |
| A | tiny SLM | constrained | Baseline from existing champion run |
| B | HuggingFaceTB/SmolLM2-135M (1-2B) | constrained | Lower-bound external constrained |
| C | Qwen/Qwen2.5-7B-Instruct (6-7B) | constrained | Upper-range external constrained |
| D | Same as B | unconstrained + postvalidation | Constraint-distortion control |
| E | Same as B | complete-candidate rerank | Diagnostic rerank mode |

```bash
# Fixture wiring run (CPU, no model download)
python -m scripts.run_quality_matrix \
  --matrix-set external-ceiling \
  --mode fixture \
  --run-root outputs/runs/slm108_external_ceiling \
  --checkpoint-reference-uri hf://buckets/TKendrick/OpenUI/checkpoints/<baseline_run_id>/ref.json

# Frontier run (GPU + pinned durable checkpoints required)
python -m scripts.run_external_ceiling \
  --mode frontier \
  --output-dir outputs/runs/slm108_frontier \
  --checkpoint-reference-uri hf://buckets/TKendrick/OpenUI/checkpoints/<baseline_run_id>/ref.json
```

Primary metric: `binding_aware_meaningful_v2_rate_strict`. Fixture runs are
wiring-only and cannot claim ship gates. Frontier execution requires durable
checkpoint provenance from SLM-103 and the EFS0 comparison stack.

## Exposure ladder (SLM-109 / EFS1-02)

Frozen E228 legal-candidate-margin recipe scaled from ~6.4k target tokens to
≥100×. The ladder tests whether the tiny SLM is underexposed or whether the
recipe is mis-specified.

| Multiplier | Target tokens | Purpose |
| --- | --- | --- |
| 1× | 6,401 | Reproduction / resume baseline |
| 4× | 25,604 | First ladder checkpoint |
| 16× | 102,416 | Mid-ladder |
| 64× | 409,664 | Late-ladder |
| 128× | 819,328 | ≥100× threshold |

```bash
# Plan / fixture wiring (CPU, no training)
python -m scripts.run_e228_exposure_ladder --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm109_e228_fixture

# Frontier dispatch (GPU + durable checkpoint required)
python -m scripts.hf_jobs_train \
  --run-id e228-ladder-m4-s0 --steps 3200 \
  --extra-train-args "--resume-from outputs/runs/e228-candidate-margin-matched/checkpoints/last_full_state.pt --target-token-budget 25604"
```

Primary metric: `binding_aware_meaningful_v2_rate_strict` plus AgentV and
independent labels. The recipe is frozen; any hash mismatch is a failed
experiment. Frontier execution requires the E228 checkpoint and SLM-103 bucket
sync.

## Corruption curriculum (SLM-120 / EFS3-02)

Frozen E228 legal-candidate-margin recipe with staged near-solved semantic
corruption shares. Tests whether injecting 5–15% one- and two-error states
improves recovery and fixed-point stability without degrading from-scratch
generation.

| Arm | Near-solved share (S1+S2) | Purpose |
| --- | --- | --- |
| A_control | 0% | Clean reproduction / resume baseline |
| B05 | 5% | Low intervention |
| B10 | 10% | Medium intervention |
| B15 | 15% | High intervention |
| B30 | 30% | Stress / copying-failure control |

Within the near-solved share, S1 and S2 are split 50/50.

```bash
# Plan / fixture wiring (CPU, no training)
python -m scripts.run_corruption_curriculum --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm120_corruption_fixture

# Frontier dispatch (GPU + durable checkpoint required)
python -m scripts.hf_jobs_train \
  --run-id slm120-curriculum-b10-s0 --steps 3200 \
  --extra-train-args "--resume-from outputs/runs/e228-candidate-margin-matched/checkpoints/last_full_state.pt --near-solved-share 0.10"
```

Primary metric: `binding_aware_meaningful_v2_rate_strict`, with separate
S0 stability, S1 recovery, and S2 recovery rates. Fixture runs are wiring-only
and cannot claim ship gates. Frontier execution requires the EFS1-decided base
recipe/checkpoint and SLM-103 bucket sync.

## Causal PEFT FTPO (SLM-121 / LDI1-02)

Frozen E228 legal-candidate-margin recipe with small removable PEFT adapters
trained on exact-state causal decision events with FTPO objectives. Tests
whether adapter-only updates can shift good legal actions above bad legal
actions without changing base weights.

| Objective | Purpose |
| --- | --- |
| `unlikelihood` | Negative control over bad legal actions |
| `ftpo_single` | Exactly one good vs one bad action |
| `ftpo_set` | Weighted good × bad margins |
| `legal_set_mass` | Shift legal-space mass from bad set to good set |

```bash
# Plan / fixture wiring (CPU, no training)
python -m scripts.run_causal_peft_ftpo --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm121_causal_peft_fixture

# Frontier dispatch (GPU + durable checkpoint required)
python -m scripts.hf_jobs_train \
  --run-id slm121-causal-peft-ftpo-single-s0 --steps 3200 \
  --extra-train-args "--resume-from outputs/runs/e228-candidate-margin-matched/checkpoints/last_full_state.pt --adapter-method lora --ftpo-objective ftpo_single"
```

Primary metric: `binding_aware_meaningful_v2_rate_strict` plus
`reference_locality_drift`. Fixture runs are wiring-only and cannot claim ship
gates. Frontier execution requires a causal base checkpoint, an admitted
DecisionEventV2 corpus, and SLM-103 bucket sync.

## TwoTower removable adapter (SLM-123 / LDI2-01)

Repository-owned LoRA-style low-rank delta over selected TwoTower denoiser
projections. Parent weights stay frozen; only `A`/`B` factors are trainable.
The adapter can be disabled (restoring the exact parent map), saved/loaded as a
separate artifact, and merged one-way into a wrapper-free copy.

| Target | Purpose |
| --- | --- |
| `attn_q`, `attn_v` | Attention query/value adaptation |
| `attn_k`, `attn_out` | Attention key/output adaptation |
| `cross_attn_*` | Cross-attention adaptation (when present) |
| `mlp_in`, `mlp_out` | FFN adaptation |

```bash
# Load a saved adapter and train only adapter parameters
python -m scripts.train_model \
  --train-dir outputs/data/train/v1 \
  --adapter-spec outputs/runs/slm123_adapter_evidence/adapter \
  --steps 32

# Load adapter frozen (inference / no adapter gradients)
python -m scripts.train_model \
  --train-dir outputs/data/train/v1 \
  --adapter-spec outputs/runs/slm123_adapter_evidence/adapter \
  --adapter-frozen \
  --steps 32
```

Primary metric: `binding_aware_meaningful_v2_rate_strict`. Fixture runs are
wiring-only and cannot claim ship gates. A real adapter-quality claim requires
parent/adapter merge-parity tests, trained-adapter metrics, and SLM-103 bucket
provenance.

## B3 surface-vs-choice capacity ladder v2 (SLM-124 / EFS3-03)

Rerun the B3 direct capacity experiment after the E288 choice-native decoder
fix. Compare surface-token (`lexer`) and choice-sequence (`choice`) models at
`d_model ∈ {64, 128, 192}` over three seeds each, with matched recipe and
semantic-example exposure.

| Arm | Representation | Widths | Seeds | Decode fingerprint |
| --- | --- | --- | --- | --- |
| surface | `lexer` | 64, 128, 192 | 0, 1, 2 | surface lexer, grammar-constrained, non-LTR |
| choice | `choice` | 64, 128, 192 | 0, 1, 2 | E288 choice-native, forced singleton decisions |

```bash
# Plan / fixture wiring (CPU, no training)
python -m scripts.run_b3_capacity_v2 --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm124_b3_capacity_fixture

# Frontier dispatch (GPU + durable checkpoints required)
python -m scripts.run_scaling_ladder --capacity-arm lexer \
  --train-dir outputs/data/train/v1 --test-dir outputs/data/eval/v1 \
  --widths 64,128,192 --seeds 0,1,2 --steps 3200 --representation lexer

python -m scripts.run_scaling_ladder --capacity-arm choice \
  --train-dir outputs/data/train/v1 --test-dir outputs/data/eval/v1 \
  --widths 64,128,192 --seeds 0,1,2 --steps 3200 --representation choice
```

Primary metric: `binding_aware_meaningful_v2_rate_strict` versus trainable
parameters / checkpoint bytes / `d_model`. Fixture runs are wiring-only and
cannot claim ship gates. Frontier execution requires 18 matched trains, a GPU
host, durable HF bucket sync per SLM-103, and the EFS1 exposure decision from
SLM-109.

## Contract-grounded candidate selector (SLM-127 / EFS3-04)

Given a prompt/contract and a bounded set of generated OpenUI candidates, learn a
small per-candidate utility + contract-success head and calibrate an abstention
threshold on a validation split. The hypothesis is that exposing generator score,
value score, energy score, set size, and a feature count to a tiny MLP yields a
selector with lower regret than single-score baselines, and that a calibrated
abstention can keep the "invalid selected over valid" count at zero.

| Arm | Selection signal | Abstention |
| --- | --- | --- |
| `model_score` | Highest `generator_score` | never |
| `value_score` | Highest `value_score` | never |
| `energy_score` | Highest `energy_score` | never |
| `hard_then_simple` | `semantic_success=True` first, then `generator_score` | never |
| `learned_no_abstain` | `sigmoid(contract_success_logit)` argmax | never |
| `learned_abstain` | `sigmoid(contract_success_logit)` argmax if > calibrated threshold | risk-bounded |

```bash
# Fixture wiring (CPU, no checkpoint training)
python -m scripts.run_candidate_selector --fixture \
  --out outputs/runs/slm127_candidate_selector/report.json

# Evaluate an existing JSONL corpus
python -m scripts.run_candidate_selector --groups groups.jsonl \
  --selector learned_abstain --out report.json
```

Primary metric: selector regret, `invalid_over_valid_count`, and calibration
error on the test split. Fixture runs are wiring-only and cannot claim ship
gates. Frontier execution requires a trained OpenUI checkpoint, a labeled
candidate corpus, and validation labels independent of the training set.

## Configuration glossary — verified-solver decode (VSS1-03)

Experimental, **disabled by default**, and **unmeasured**. These flags gate the
certificate-checked exact-closure pruning of the compiler-tree forest before soft
ranking (`docs/design/verified-scope-solver.md` → "Implemented decode
integration"). They are **not** part of any honest gate, champion recipe, or
matrix row; no row above enables them, and the honest ship policy
(`STRICT_COMPILER_TREE_POLICY`) does not set them. Enabling changes decode
behavior only for the semantic choice/compiler path on a DSL-native pack; an
unsupported tokenizer/pack raises a capability error.

| Flag (`TwoTowerConfig` / `ModelBuildConfig`) | CLI (`evaluate_model.py`) | Default | Meaning |
| --- | --- | --- | --- |
| `verified_solver_decode` | `--verified-solver-decode` | `False` | Master switch. Off ⇒ decode is byte-identical to today. |
| `solver_max_nodes` | `--solver-max-nodes` | `512` | Enumeration node budget per decision (also bounds the token budget). |
| `solver_max_depth` | — (config/checkpoint) | `64` | Search depth budget. |
| `solver_max_backtracks` | — (config/checkpoint) | `64` | Backtrack budget. |
| `solver_max_verifier_calls` | — (config/checkpoint) | `64` | Verifier-call budget. |
| `solver_max_wall_ms` | — (config/checkpoint) | `0` | `0` = no wall timer; deterministic budgets stay authoritative. |
| `solver_unknown_policy` | `--solver-unknown-policy` | `keep_and_rank` | Only supported value: `UNKNOWN` candidates stay live for the soft ranker. |
| `solver_certificate_mode` | `--solver-certificate-mode` | `summary` | `none \| summary \| full` certificate detail. |
| `solver_energy_head` (VSS3-02) | — (config/checkpoint) | `False` | Enable the learned cost-to-go energy scorer. Off ⇒ decode unchanged. |
| `solver_ranker` (VSS3-02) | — (config/checkpoint) | `deterministic` | `deterministic \| model \| energy`. Ranking-only; never alters hard membership. |
| `solver_energy_hidden_dim` | — (config/checkpoint) | `64` | Energy MLP hidden width. |
| `solver_energy_loss_weight` | — (config/checkpoint) | `0.0` | Huber cost-to-go regression weight (observed rows only). |
| `solver_energy_pairwise_weight` | — (config/checkpoint) | `0.0` | Pairwise ranking-loss weight over same-state/hole pairs. |
| `solver_energy_cost_version` | — (config/checkpoint) | `v1` | Versioned work-target coefficients (stored in dataset/checkpoint metadata). |
| `solver_energy_fallback` | — (config/checkpoint) | `deterministic` | Order used when a scorer output is missing/duplicate/NaN/infinite. |

```bash
# Opt-in solver-pruned decode on the honest compiler-tree path (experimental).
python -m scripts.evaluate_model --checkpoint <ckpt> \
  --verified-solver-decode --solver-max-nodes 512 \
  --solver-unknown-policy keep_and_rank --solver-certificate-mode summary
```

## Measured results (CPU, 800 steps, scratch, no DESIGN.md in context)

See [quality-matrix-results.json](quality-matrix-results.json). Headline deltas vs ship memorizer:

| ID | Smoke parse | Adv parse | RICO parse | Notes |
| --- | --- | --- | --- | --- |
| SHIP | 1.0 | 0.0 | 0.0 | fidelity 0 everywhere |
| E1 repair | **1.0** | **0.25** | 0.0 | adversarial gate met; fidelity still 0 |
| E2 curriculum | 0.0 | **0.75** | 0.0 | best stress parse |
| E4 schema | 0.0 | 0.0 | 0.22 | mild RICO lift |
| E7 capacity | 0.0 | 0.0 | **0.88** | best RICO; struct 0.37 |
| E8 combo | 0.0 | 0.0 | 0.0 | underfit at 800 steps on stacked levers |
| **E9b / accel follow-up** | 0.0* | **0.75** | 0.02 | Historical “E9 accel combo” run; matrix table uses **E9b**. *smoke poisoned by curriculum-C; **adv fidelity 0.875** |

\*That historical accel combo emitted `:adv.*` placeholders on smoke prompts — curriculum overfit. Still the first strong fidelity signal.

**None clear honest `--ship-gates`.** Implemented follow-ups (see [phase-abc-results.json](phase-abc-results.json)):

| Stage | Result |
| --- | --- |
| **E9b / Phase A** | Soft mix + fidelity aux: **adv parse 0.50 / fid 0.65 / struct 0.48** (adversarial gates met). Smoke still parse/fid 0. |
| **Phase B** | Soft-corrupt preference: adv fid up to **0.77**, parse down to 0.25 — did not beat SFT champion score. |
| **Phase C** | Stable GRPO-lite (skip zero-reward groups, restore best-reward weights) matched SFT champion; did not clear ship. |
| **HF long-train** | Deferred offline — no SmolLM2 hub cache in this environment. |

Pipeline: `python -m scripts.run_phase_pipeline --seed-checkpoint outputs/runs/qx_e9_accel_combo/checkpoints/last.pt`  
Telemetry: train/eval spans show generate/eval dominating wall time; use `scripts/bench_telemetry.py`.

## V2 measured results (CPU, 200–500 steps, compositional tokenizer)

See [quality-matrix-results.json](quality-matrix-results.json). After F1–F5 fixes:

| ID | Smoke fid | RICO fid | Notes |
| --- | --- | --- | --- |
| E11 | 0.0 | 0.0 | tokenizer-only; 200 steps insufficient for parse |
| **E12** | 0.0 | **0.44** | slot contract — first strong fidelity signal on held RICO |
| E15 | 0.0 | 0.36 | combo capacity + curriculum + contract |
| E16 | 0.0 | 0.41 | 500-step long train |
| E17 | 0.0 | 0.38 | decode sweep on E15 ckpt |

**Ship gates still not cleared** at 200–500 CPU steps; ceiling diagnostic confirms
metrics are achievable (gold-as-prediction = 1.0). Next levers: **V3 length-safe
decode (E18)** — V2 kept `grammar_ltr_max_tokens` at 64–96 while compositional
programs need ≥160 tokens — then E19–E22 / E29 champion; and grammar-native
diffusion (X matrix below).

## V3 notes

- `--matrix v3` covers E18–E29; default CLI remains `v3` for back-compat.
- Prefer `--matrix v4` for honest ship claims (E35/E36).
- `scripts/diagnose_eval.py` reports `length_budget` and exits 2 when p95 exceeds LTR budget.
- Defaults: `grammar_ltr_max_tokens=192`, stages `(64,128,192,256)`.
- Research: MDLM schedule + remasking tagged **Adapted** in [research-lineage.md](research-lineage.md).

## V3 measured results (CPU, scratch, fixture suites)

See [quality-matrix-results.json](quality-matrix-results.json) (primary payload is
**V5**; historical V4 lives under `prior_matrices.v4`; older V3 rows remain in
git history).

| ID | Smoke parse | Smoke fid | Ship gates | Notes |
| --- | --- | --- | --- | --- |
| E18 | 0.0 | 0.0 | fail | length-safe alone underfit at 80 steps |
| **E20** | **1.0** | **1.0** | **pass*** | template fill; used silent gold placeholders |
| **E29** | **0.67** | **0.67** | **pass*** | champion stack at 40 steps |

\*V3 passes used `gold.placeholders` directly for template fill (eval leakage).
E35 is the honest successor.

## V4 measured results (CPU, 40 steps, scratch, rico n=3)

See [quality-matrix-results.json](quality-matrix-results.json).

| ID | Smoke parse | Smoke fid | Ship gates | Notes |
| --- | --- | --- | --- | --- |
| E30 | 0.0 | 0.0 | fail | rollback alone underfit @ 40 steps |
| E31 | 0.33 | 0.0 | fail | trust gate; adv parse 0.75 |
| E32 | 0.0 | 0.0 | fail | visible corrupt alone underfit |
| E33 | 0.33 | 0.0 | fail | combined remask matches E31 |
| **E35** | **0.67** | **0.67** | **pass** | honest inventory-in-prompt |
| **E36** | **0.67** | **0.67** | **pass** | BoN lifts ood parse 0.5→0.75 |

First honest `--ship-gates` clears (no silent gold placeholder channel).
Production claim still needs full `rico_held` (1500) + HF context.

Implementation order = table order (risk ascending). **E30** touches decode only
(`twotower.py`, `parallel_decode.py`). **E31** needs frozen-backbone error
mining. **E32** changes the train corruption graph. **E33** composes E30+E31
signals with existing V3 remask. **E34** waits until cheaper remask policies
saturate. Structural remask targets from V5 (`remask_span=statement`, `<SYM>`
ids) make E30–E34 materially easier to ground on semantic units.

## V4 matrix (critic-guided revision)

Implements the research roadmap in
[research-correction-critics.md](research-correction-critics.md). Tags in
[research-lineage.md](research-lineage.md) are **Adapted** for the subsets
below. Prefer seeding decode-only rows from an E35 (or E29) checkpoint.

| ID | Approach | Primary lever | Run id |
| --- | --- | --- | --- |
| E30 | Suffix-rollback LTR | ReMDM-style revisable window \(W\) behind LTR frontier; remask on grammar / entropy triggers | `qx_e30_suffix_rollback` |
| E31 | BackPlay-lite trust head | Freeze denoiser; train [`FastPathGate`](../../src/slm_training/dsl/grammar/fastpath/gate.py) on model token errors; remask with gate scores | `qx_e31_trust_gate` |
| E32 | Corruption-aware train | `_mask_targets` flips visible tokens → wrong ids; CE recovers gold (GIDD/RemeDi-lite) | `qx_e32_visible_corrupt` |
| E33 | Combined remask policy | Budgeted remask ∝ grammar hard-error + gate + entropy (`select_remask_policy_indices`) | `qx_e33_remask_policy` |
| E35 | Honest slot contract | Inventory-in-prompt API: surface slots into prompt, extract via `inventory_from_prompt` (no silent `gold.placeholders`) | `qx_e35_honest_contract` |
| E36 | Decode scaling | Best-of-N + remask-round scaling on E35 ckpt | `qx_e36_decode_scaling` |
| E34 | Latent falsification MoE | Deferred; runner skips unless `--force-e34` | `qx_e34_latent_critics` |

```bash
# V4 focused ship path (honest contract → decode scaling)
python -m scripts.run_quality_matrix --matrix v4 --only E35,E36 \
  --steps 40 --device cpu --context-backend scratch --no-design-md-context \
  --rico-limit 3 --scratch-control

# Ablations + trust gate (optional seed)
python -m scripts.run_quality_matrix --matrix v4 \
  --only E30,E31,E32,E33,E35,E36 --steps 40 --device cpu \
  --context-backend scratch --no-design-md-context --rico-limit 3 --scratch-control
```

## V5 matrix (DSL-native output tokenizer)

Design: [dsl-native-tokenizer.md](dsl-native-tokenizer.md). Replaces the
output-side string-piece tokenizer with a compiler-derived alphabet while
keeping MaskGIT / remask architecture on **TwoTower**. Stages 3–4 (tree canvas /
full production diffusion) are covered by the parallel **X matrix**
`grammar_diffusion` plug-in below — not by swapping TwoTower encoding alone.

| ID | Approach | Primary lever | Run id |
| --- | --- | --- | --- |
| E40 | Lexer-native only | Fixed terminals + literal channel (no `<SYM>`) | `qx_e40_lexnative` |
| E41 | Symbol table | E40 + `<SYM_i>` / `<BIND_j>` + slot contract | `qx_e41_symtable` |
| E42 | Factorized embeddings | E41 + `E_tok + E_kind` | `qx_e42_factorized` |
| E43 | Exact grammar masks | Eval-only overlay on E41 (kind-authored `allowed_id_set`) | `qx_e43_exact_masks` |
| E44 | Structural mask/remask | E41 + `mask_pattern=mixed` + `remask_span=statement` | `qx_e44_structmask` |
| E45 | Teacher init | E41 + HF teacher-initialized symbol rows (skip without cache) | `qx_e45_teacher_init` |
| E46 | V5 champion | E40+E41+E42+E44 + template fill + MDLM + remask + capacity | `qx_e46_champion` |

```bash
# Length diagnostic (compositional vs lexer)
python -m scripts.diagnose_tokenizer --fixtures

# V5 focused path
python -m scripts.run_quality_matrix --matrix v5 --only E40,E41,E44,E46 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control
```

## V5 measured results (CPU, scratch, 80 steps, fixture suites)

See [quality-matrix-results.json](quality-matrix-results.json) (`matrix_set: v5`).
Tokenizer diagnostic on `src/slm_training/resources/train_seeds.jsonl`: compositional mean 72.6
tokens → lexer+symtable **46.3** (ratio **0.64**); fixed output vocab **296**.

| ID | Smoke parse | Smoke fid | Smoke reward | Notes |
| --- | --- | --- | --- | --- |
| E40 | 0.0 | 0.0 | 0.0 | literal-channel alone underfit at 80 steps |
| E41 | 0.0 | 0.0 | 0.0 | struct≈0.47 — first structural lift without template |
| E42 | 0.0 | 0.0 | 0.0 | factorized alone underfit at 80 steps |
| E43 | 0.0 | 0.0 | 0.0 | matches E41 (eval overlay) |
| E44 | 0.0 | 0.0 | 0.0 | **adv parse 0.25 / fid 0.25** — structural remask signal |
| **E46** | **1.0** | **1.0** | **0.97** | fixture parse/fid/struct clear (held_out 0.6/1.0; adv/ood 1.0/1.0); fails only empty `rico_held` |

**Headline:** the V5 champion (lexer-native + symbol table + factorized +
structural remask + template fill) matches V3 E20/E29 fixture quality while
shrinking target sequences ~36% and fixing the output alphabet. Isolating
levers at 80 steps still underfits without template fill; E44’s adversarial
lift supports statement-span remask before longer trains.

## V6 matrix (CoRe remask + slot-aware trust + honest V5 stack)

Implements SOTA remask / critic updates from
[research-correction-critics.md](research-correction-critics.md) and
[research-lineage.md](research-lineage.md) (CoRe, T2M, RemeDi/BackPlay slot
mining) on top of the V5 alphabet + E35 honesty.

| ID | Approach | Primary lever | Run id |
| --- | --- | --- | --- |
| E50 | CoRe-lite remask | `remask_policy=core` + neighbor-mask support-drop scores | `qx_e50_core_remask` |
| E51 | T2M + statement remask | `remask_to_mask=True` + `remask_span=statement` | `qx_e51_t2m_statement` |
| E52 | Slot-aware trust gate | `slot_aware_trust_gate` + `remask_policy=combined` | `qx_e52_slot_trust` |
| E53 | Honest V5 champion | E46 + E35 + E33 + E50 + slot trust | `qx_e53_honest_v5_champion` |
| E54 | Grammar-diffusion honest | `grammar_diffusion` + `honest_slot_contract` (no gold channel) | `qx_e54_grammar_honest` |
| E55 | Process stage | Preference + GRPO-lite on E53 (skip RL if no variance) | `qx_e55_process` |

```bash
# V6 focused ship path
python -m scripts.run_quality_matrix --matrix v6 --only E50,E53,E55 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control

# Grammar-diffusion honest contract (E54) + frozen fixed-canvas evidence
python -m scripts.run_quality_matrix --matrix v6 --only E54 --steps 80 --scratch-control
```

**Honesty fix (required for E35/E53 fidelity):** `generate_batch_requests`
surfaces `GenerationRequest.slot_contract` into the prompt via
`ensure_prompt_inventory` when `honest_slot_contract=True`, then extracts
inventory from the prompt text (inventory-in-prompt API — not a silent gold
channel).

## V6 measured results

See [quality-matrix-results.json](quality-matrix-results.json) (`matrix_set: v6`).
CPU scratch, 80 train steps, re-eval with inventory-in-prompt fix, `rico_held` n=20.

| ID | Smoke parse | Smoke fid | Ship gates | Notes |
| --- | --- | --- | --- | --- |
| **E50** | **1.0** | **1.0** | **pass** | CoRe remask; rico parse/fid 1.0 (n=20) |
| **E53** | **1.0** | **1.0** | **pass** | Honest V5 champion stack; held_out 0.6/1.0 |
| E54 | 0.0 | 0.0 | fail | grammar_diffusion underfit @ 80 steps |
| **E55** | **1.0** | **1.0** | **pass** | Process stage inherits E53 clear |

**Headline:** after fixing `generate_batch_requests` to surface
`GenerationRequest.slot_contract` into the prompt under `honest_slot_contract`,
E50/E53/E55 clear honest `--ship-gates` (including rico_held n=20). Full 1500
`rico_held` + HF context remains the production claim. Grammar-diffusion (E54/X2)
needs longer train / capacity before competing with the TwoTower V5 stack.

### E121 judged-corpus follow-up (2026-07-16)

E121 reran the E53 stack against the committed `remediated_roots_judged`
corpus (405 records), using CPU scratch, 8 train steps plus the 30-step trust
gate, and no DESIGN.md context. The first invocation exposed a matrix
precedence bug: E53 silently used the stale default curriculum snapshot even
when `--train-dir` was explicit. The runner now maps an explicit train corpus
to curriculum input unless a separate curriculum path is provided.

The bounded five-suite diagnostic (`smoke`, `held_out`, `adversarial`, `ood`,
`rico_held`, capped at n=3) exceeded the CPU wall limit under E53's `best_of_n=4`
decode before producing suite rows. A one-record smoke diagnostic with a
5-second per-record timeout recorded parse/fidelity/structure/reward **0.0**
and one decode timeout at 5,001 ms. This is a negative scratch result, not a
ship claim. The evaluator also received two fixes during this iteration: the
duplicate `--output-tokenizer` CLI option was removed and the
`generate_with_stats` path now returns the required predictions/evidence tuple.
See [iter-e121-judged-corpus-e53-20260715.md](iter-e121-judged-corpus-e53-20260715.md)
and the generated failure scoreboards
`quality-matrix-results-iter-e121*-e53-judged-20260715.json`.

Grammar X results: [grammar-matrix-results.json](grammar-matrix-results.json).

## X matrix (grammar-native diffusion ablations)

X2-X8 are frozen evidence for the retired fixed-canvas implementation. The current
runner preserves those results but refuses to execute their IDs, so a topology run
cannot silently overwrite an architectural baseline. Reproduction requires the
`source_commit` recorded in `grammar-matrix-results.json`.

The three-seed X2 reproduction at source commit `e1c2c0d` is recorded in
[grammar-fixed-canvas-baseline-results.json](grammar-fixed-canvas-baseline-results.json):
80 CPU/scratch steps, seeds 0/1/2, and all five honest suites produced zero parse,
fidelity, structure, and reward. All three AgentV runs executed without SDK errors;
ship gates failed. The local checkpoints were retained only as comparison artifacts.

Staged ablations **X9-X15** add one topology lever at a time. Each experiment runs
**3 seeds** (0/1/2). Successive halving uses the median topology composite across
all seeds for `smoke` → `held_out` → `adversarial`, then evaluates survivors on all
five suites. The normal ship gates remain authoritative.

| ID | Approach | Primary lever | Model | Run id |
| --- | --- | --- | --- | --- |
| X0 | Corrected baseline | twotower + honest DESIGN.md eval | twotower | `gx_x0_baseline` |
| X1 | Data/contract | slot contract in context + inventory decode | twotower | `gx_x1_contract` |
| X2 | Production codec (frozen) | fixed canvas over production+slot heads | retired grammar_diffusion v1 | `gx_x2_codec` |
| X3 | Block objective (frozen) | fixed positional block noise | retired grammar_diffusion v1 | `gx_x3_block_obj` |
| X4 | Confidence schedule (frozen) | fixed-canvas parallel unmask | retired grammar_diffusion v1 | `gx_x4_confidence` |
| X5 | Extendability decode (frozen) | constrained positional posterior | retired grammar_diffusion v1 | `gx_x5_extend` |
| X6 | Grammar curriculum | soft A/B/C mix (anti-leak) | twotower | `gx_x6_curriculum` |
| X7 | Champion combo (frozen) | fixed-canvas X1-X6 stack | retired grammar_diffusion v1 | `gx_x7_champion` |
| X8 | Process optimization (frozen) | fixed-canvas preference/RL stage | retired grammar_diffusion v1 | `gx_x8_process` |
| X9 | Typed topology baseline | tree state + synchronous expansion | grammar_diffusion v2 | `gx_x9_topology_base` |
| X10 | Edit actions | X9 + delete/contract/stop supervision | grammar_diffusion v2 | `gx_x10_actions` |
| X11 | Tree coordinates | X10 + type/parent/depth/sibling embeddings | grammar_diffusion v2 | `gx_x11_tree_embeddings` |
| X12 | Heterogeneous corruption | X11 + node/depth-specific noise | grammar_diffusion v2 | `gx_x12_heterogeneous_noise` |
| X13 | Critic scheduling | X12 + accept/defer/contract calibration | grammar_diffusion v2 | `gx_x13_critic` |
| X14 | Dynamic work buffer | X13 + bounded active nodes/global sync | grammar_diffusion v2 | `gx_x14_buffer` |
| X15 | Topology champion | X14 + curriculum/capacity | grammar_diffusion v2 | `gx_x15_topology_champion` |
| X16 | Scope corpus control | X9 + shared scope derivatives | grammar_diffusion v2 | `gx_x16_scope_corpus` |
| X17 | Scope contracts | X16 + contract embeddings + summary/local-gate heads | grammar_diffusion v2 | `gx_x17_scope_contracts` |
| X18 | Scope noise | X17 + independently noised scopes | grammar_diffusion v2 | `gx_x18_scope_noise` |
| X19 | Local oracle supervision | X18 + local-gate/failure-cone targets | grammar_diffusion v2 | `gx_x19_scope_oracle` |
| X20 | Contract negatives | X19 + boundary and local/global negatives | grammar_diffusion v2 | `gx_x20_scope_negatives` |
| X21 | Scoped topology stack | X20 + X14 actions/buffer/global sync | grammar_diffusion v2 | `gx_x21_scoped_topology` |

`grammar_diffusion` is the harness plug-in for
`GrammarDiffusionModel` format v2 (typed production-tree diffusion). See
[grammar-topology-diffusion.md](grammar-topology-diffusion.md),
`src/slm_training/models/grammar_diffusion.py`, and
`src/slm_training/harnesses/model_build/factory.py`. Format-v1 checkpoints require
the explicit warm-start migrator; the runtime has no legacy alias.

### Commands

```bash
# Full topology matrix with successive halving and two-row confirmation
python -m scripts.run_grammar_matrix \
  --device cpu --context-backend scratch --steps 80 --confirm-steps 200

# Isolated topology levers
python -m scripts.run_grammar_matrix --only X9,X10,X11 --steps 80

# Inspect ScopeDiff rows without data builds, training, or result writes
python -m scripts.run_grammar_matrix --only X16,X17,X18,X19,X20,X21 --describe

# Disable halving (run every experiment×seed to completion)
python -m scripts.run_grammar_matrix --no-halving --only X9,X15 --steps 80

# Or confirm two selected rows directly.
python -m scripts.run_grammar_matrix --no-halving --only X<best>,X<runner-up> \
  --steps 200 --gen-steps 16
```

Screening score:

```text
topology_composite = 0.45 honest_quality
                   + 0.25 AST_topology
                   + 0.20 expansion_and_critic_trace
                   + 0.10 node_pass_efficiency
```

The component definitions, budgets, checkpoint boundary, and evidence rules are in
[grammar-topology-diffusion.md](grammar-topology-diffusion.md). Evidence-complete
negative results count; undocumented or JSON-only runs do not.

X16-X21 were measured on 2026-07-16 with 80-step, three-seed CPU screening and
200-step confirmation of X18/X21. Both confirmation rows failed every unchanged
multi-suite ship decision; see
[grammar-scope-matrix-results.json](grammar-scope-matrix-results.json). Their v1
scope diagnostics are `scope_contract_metrics` overall and grouped by scope
kind/family, but this campaign's generalization suites carry no scope metadata, so
those diagnostics are unavailable rather than inferred. Parser-exit and
identity-level symbol F1 remain unimplemented.

Artifacts: `outputs/runs/grammar_matrix_summary.json`,
[`docs/design/grammar-matrix-results.json`](grammar-matrix-results.json),
[`docs/design/grammar-scope-matrix-results.json`](grammar-scope-matrix-results.json),
`outputs/runs/baseline_reproduction_summary.json`.

### Measured X16-X21 result (2026-07-16 UTC)

```bash
python -m scripts.run_grammar_matrix \
  --only X16,X17,X18,X19,X20,X21 \
  --scope-dir outputs/data/train/scopediff_x16x21_v1 \
  --test-dir outputs/data/eval/scopediff_x16x21_eval_v1 \
  --device cpu --context-backend scratch \
  --steps 80 --seeds 0,1,2 --rico-limit 32 --confirm-steps 200 \
  --confirm-top 2 --gen-steps 8 \
  --docs-output docs/design/grammar-scope-matrix-results.json
```

X21 and X18 survived smoke, held-out, and adversarial halving. At 200 steps,
median parse and placeholder fidelity were 0.0 on smoke n=3, held-out n=5,
adversarial n=4, OOD n=4, and limited RICO n=32 for both rows. X21 retained weak
structural signal (median 0.115 smoke and 0.046 RICO) but failed every ship
decision. Six AgentV bundles ran five domain assertions each with 0/5 passes. The
six local scratch checkpoints were not promoted or synced.

### Measured X9-X15 result (2026-07-15 UTC)

The 80-step, three-seed screen trained all 21 topology candidates and selected
X14 and X9 after smoke, held-out, and adversarial halving. A frozen-vocabulary
evaluation failure was repaired in `4bf964d`; the exact checkpoints were reused,
and held-out production OOV is now measured at 0.0443. The 200-step confirmation
then ran X9 and X14 across seeds 0/1/2 and all five limited suites.

Every confirmation failed the unchanged ship gates. Median held-out parse was
0.0 for both rows; median held-out topology composite was 0.372 for X9 and 0.277
for X14. X9 reached median parse 0.667 on limited `rico_held` n=3 but retained
zero median parse on held-out, adversarial, and OOD, so it is not promotable.
X14's median RICO parse was 0.0. Full recipes, per-seed metrics, evaluator
provenance, AgentV evidence counts, and checkpoint disposition are in
[grammar-matrix-results.json](grammar-matrix-results.json) and
[grammar-topology-diffusion.md](grammar-topology-diffusion.md).

## V7 matrix (speculative denoising)

Implements the speculative-denoising design in
[speculative-denoising.md](speculative-denoising.md) (paper tags:
[research-lineage.md](research-lineage.md) §"Speculative denoising (V7)").
Levers adapt LESS stability signals, DAPD/DAWN attention dependency clusters,
Self-Spec-MD ordered verification, DSpark survival scheduling, and
Saguaro-SSD successor caching onto the honest V5/V6 TwoTower stack.

| ID | Approach | Primary lever | Run id |
| --- | --- | --- | --- |
| E70 | Stability remask (LESS-lite) | `remask_policy=stability` + persistence commit gate | `qx_e70_stability` |
| E71 | Attention clusters (DAPD/DAWN-lite) | `unmask_mode=cluster` + anchor-first ordering | `qx_e71_clusters` |
| E72 | Ordered cluster verification | `cluster_verify` → outcome `(j, repair)` | `qx_e72_cluster_verify` |
| E73 | Survival head (DSpark-lite) | `survival_gate` + cumulative commit budget | `qx_e73_survival` |
| E74 | Successor cache (SSD-lite) | `speculative_successor` + K=2 fanout | `qx_e74_successor` |
| E75 | V7 champion | E53 honest stack + E70–E74 | `qx_e75_v7_champion` |

```bash
# V7 focused path
python -m scripts.run_quality_matrix --matrix v7 --only E70,E72,E74,E75 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control

# Full V7 with rico cap for CPU time
python -m scripts.run_quality_matrix --matrix v7 --steps 80 --device cpu \
  --context-backend scratch --no-design-md-context --rico-limit 20 --scratch-control
```

Beyond parse/fidelity, V7 rows record decode telemetry per eval
(`speculative_stats` in the run summary): denoiser forwards per generate,
successor-cache hit rate, remask volume.

## V7 measured results

See [quality-matrix-results.json](quality-matrix-results.json) (`matrix_set: v7`).
CPU scratch, 80 train steps, `--no-design-md-context`, `rico_held` n=20.
Historical V6/V5 payloads live under `prior_matrices`.

| ID | Smoke parse | Smoke fid | Ship gates | Decode telemetry (smoke) | Notes |
| --- | --- | --- | --- | --- | --- |
| **E70** | **1.0** | **1.0** | **pass** | 16.5 fwd/gen | LESS-lite stability remask; held_out 0.6/1.0 |
| **E71** | **1.0** | **1.0** | **pass** | 16.5 fwd/gen; clusters proposed | Attention clusters; quality matches E70 |
| **E72** | **1.0** | **1.0** | **pass** | 16.5 fwd/gen; 0 cluster rejects | Ordered verify; grammar accepted all proposed clusters on fixtures |
| **E73** | **1.0** | **1.0** | **pass** | 16.5 fwd/gen; fewer clusters via survival budget | Survival head reduces proposed clusters ~10× vs E71 |
| **E74** | **1.0** | **1.0** | **pass** | 16.5 fwd/gen; **successor hit rate 1.0** | SSD-lite cache: 45 speculative batches → 45 hits (0 miss) |
| **E75** | **1.0** | **1.0** | **pass** | 31.5 fwd/gen; speculation skipped | Champion stack; trust-gate remask needs extra forwards → speculation auto-disabled (no wasted misses) |

**Headline:** every V7 row clears honest `--ship-gates` on the fixture suites
(including rico_held n=20), matching the V6 E53 quality floor. E74 realizes
the SSD transfer: successor-cache hit rate 1.0 with forwards/gen unchanged vs
the non-speculative path (~16.5). E73's survival budget materially shrinks
cluster proposals without hurting parse/fidelity. E75's higher forward count
is the cost of the V6 trust-gate remask path (one extra forward per remask
step), not of speculation — speculation correctly abstains when remask is
non-deterministic.

Full 1500 `rico_held` + HF context remains the production claim.

## Overnight honest eval and retrain (2026-07-15)

The committed demo checkpoint was evaluated with the pinned AgentV SDK and
the full ship-gate suite using the remediated test artifact (`smoke n=3`,
`held_out n=5`, `rico_held` capped at 32). Every suite had parse=0.0,
structural similarity=0.0, and placeholder fidelity=0.0; AgentV recorded
5/5 failed cases with no execution errors. No promotion was made.

A 200-step CPU scratch retrain (`overnight_retrain_200`, 857,282 trainable
parameters, scratch context, batch 16, last loss≈6.64) produced the same
all-suite parse=0.0 result. Its checkpoint remains local-only and is not a
champion. Durable scoreboard, gates, AgentEvals JSONL, and telemetry are under
`outputs/runs/overnight_retrain_200_eval/` and
`outputs/runs/overnight_retrain_200/`.

The evaluation harness also had a real correctness bug: omitted
`--grammar-ltr-primary` / `--grammar-ltr-repair` flags defaulted to `False` and
overwrote checkpoint settings. Those flags are now tri-state; omitted means
preserve checkpoint configuration. The fix is covered by
`test_factory_overrides.py`.

An extended 1,000-step continuation (`overnight_retrain_1000`, 857,282
trainable parameters, scratch context, batch 16) reduced training loss to
≈1.12 but produced parse=0.0, placeholder fidelity=0.0, and reward=0.0 at
every smoke checkpoint (steps 200, 400, 600, 800, and 1000). This is a
negative training result, not a promotion candidate; the durable run summary
and telemetry are under `outputs/runs/overnight_retrain_1000/`.

The E53 V6 scratch-control follow-up completed its 80-step base train and
30-step trust-gate fit, but both the full matrix evaluator and a direct
smoke-only evaluation of its lexer checkpoint exceeded the overnight runtime
budget without emitting a scoreboard. This is recorded as an operational
failure, not a quality result or a ship signal; the partial checkpoint remains
under `/tmp/overnight-e53/` for later profiling.

## Iterative training loop (2026-07-15)

The first post-UI-loop V7 E70 reproduction used the remediated corpus, CPU
scratch context, honest slot contract, 20 steps, batch size 4, and `rico_held`
limit 5. The run persisted a checkpoint, `train_summary.json`,
`train_telemetry.json`, per-step metrics, a scoreboard, and three AgentV
JSONL bundles under `outputs/runs/iter_e70_20260715/qx_e70_stability/`.

| Run | Smoke (n=3) | Held-out (n=5) | Outcome |
| --- | --- | --- | --- |
| E70 `qx_e70_stability` | parse 1.00, fidelity 1.00, reward 0.969 | parse 0.60, fidelity 1.00, reward 0.989 | fixture control clears available gates; not a ship claim |

The run's training loss reached 21.01 at step 20. Telemetry identified
`eval_suites` as 94.88% of wall time because this probe evaluated at steps 10
and 20; subsequent probes should evaluate only at the end unless checkpoint
trajectory feedback is specifically needed. The requested `rico_held` suite
was not emitted by this short run and the missing `adversarial`/`ood` suites
remain an explicit limitation, not a pass.

The adjacent E71 attention-cluster reproduction used the same corpus and
20-step recipe with end-only evaluation. It persisted all five available
limited suites (`smoke` n=3, `held_out` n=5, `adversarial` n=4, `ood` n=4,
`rico_held` n=3) under `outputs/runs/iter_e71_20260715/qx_e71_clusters/` and
passed those bounded fixture gates. Metrics matched E70 on smoke and
held-out; adversarial parse was 1.00 with structural similarity 0.8263, OOD
parse was 1.00 with structural similarity 0.5974, and RICO-held parse was
1.00 with structural similarity 0.7618. Decode telemetry proposed clusters
but accepted none, so this is a reproducibility result, not evidence that the
cluster optimization is active or a production ship claim.

The separate E70 40-step continuation attempt produced partial training
artifacts but no complete scoreboard and is retained as an operationally
incomplete run, not included in the comparison.

E72 ordered cluster verification was then trained for the same 20 steps. The
complete bounded result persisted all five suites (`smoke` n=3, `held_out`
n=5, `adversarial` n=4, `ood` n=4, `rico_held` n=3) plus AgentV bundles.
Metrics matched E70/E71, while ordered verification accepted every proposed
cluster (320/320 smoke, 594/594 held-out, 402/402 adversarial, 480/480 OOD,
444/444 RICO-held) with zero rejects. This is a bounded fixture result, not a
production ship claim; the next iteration should test whether the accepted
clusters reduce forwards or wall time before promotion.

E73 survival-budget training exposed a harness issue: the base SFT config was
also enabling survival decoding before the auxiliary head existed. The runner
now keeps `survival_gate=False` during SFT and enables it only after
`_maybe_survival_gate`; the auxiliary fit also honors the existing
`--pref-steps` and `--pref-limit` controls without a hidden 20-step floor.
Two CPU probes still ended before checkpoint finalization (steps 19 and 4),
so E73 has no quality or performance result yet and is not promoted.

E74 successor-cache training exposed the same phase-mixing issue: successor
reuse is decode-only but was enabled in the base SFT configuration. The runner
now keeps `speculative_successor=False` during SFT and enables it only in the
evaluation configuration. The available CPU probes still stopped at step 2
before checkpoint finalization, so E74 has no quality or cache-performance
result and is not promoted.

E75 V7 champion composition was also attempted. A fresh CPU run stopped after
one training step; an overlay seeded from the completed E72 checkpoint reached
the auxiliary trust-gate stage but stopped before producing a scoreboard or
AgentV bundle. E75 therefore remains **incomplete** with no champion or ship
claim. The full composition needs a longer-lived execution host so its
survival/trust/successor telemetry can be compared fairly against E72.

The runner now exposes `--override-gen-steps` so diagnostic evaluations can
override experiment presets such as E74's 16-step decode. The delayed smoke
artifact now shows the overlay is valid (parse 1.00, fidelity 1.00,
structural 0.6489, reward 0.969), but successor telemetry is zero at the
16-step preset. The one-step diagnostic also completes with 18/18 clusters
accepted but, correctly, cannot speculate on the final step. E74 therefore
still has no cache-hit/performance evidence; the next diagnostic is a two-step
decode to force one successor opportunity.

An E72 lower-learning-rate control (`lr=1e-4`, requested 5 steps) produced
losses 656.3, 456.8, and 2460.5 before stopping without a checkpoint or
scoreboard. The persisted run-insights report therefore rejects the simple LR
hypothesis; the next training repair should inspect batch/data outliers and
objective stability instead.

The remediated-corpus profile found 585 records, with `rico+template` making
up 240 (41%) and OpenUI target lengths from 63 to 1341 characters (median
498, p90 884). No single extreme record explains the E72 spike; the next
instrumentation step is to persist batch IDs, source families, and target
lengths alongside each loss row before changing the data mixture.

## P13 data-synthesis verification (CPU scratch, 2026-07-14)

The accepted comparison uses the same E50 experiment and effective decode
recipe on both corpora: CPU scratch, 80 train steps, batch 4, lr `3e-4`, seed
0, honest slot contract, four-step best-of-1 decode, no template fill or
DESIGN.md context, and unchanged gates.

| Suite | n | Fixture fidelity | Integrated fidelity | Delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `held_out` | 5 | 0.08 | 0.12 | +0.04 | bounded signal pass |
| `rico_held` | 5 | 0.0667 | 0.10 | +0.0333 | bounded signal pass |

Both arms have parse/structure/reward 0.0 and fail unchanged ship gates. The
earlier E0/E41/E53 probes were negative or non-attributable and are
superseded; the verifier rejects differing experiment/decode settings and
requires strict gains on both suites. Full evidence and the no-promotion
decision are in [data-synthesis.md](data-synthesis.md) and
[data-synthesis-results.json](data-synthesis-results.json).

## V8 dynamic-symbol and constraint-system matrix (proposed, unrun)

## Minimal output-contract corpus rebuild (2026-07-15)

The committed `remediated_roots_judged` snapshot now replaces 305 legacy
verbose language-contract derivatives with 62 canonical shortest-output rows:
4 lexical, 56 expression, 1 statement, and 1 unavoidable full document. The
rebuild kept 193 unrelated records, admitted all 62 compact rows, and recorded
zero verifier or quality rejects. Fragment correctness/efficiency is diagnostic;
the existing document ship gates are unchanged. See
[iter-minimal-output-contract-20260715.md](iter-minimal-output-contract-20260715.md)
and its [JSON evidence](iter-minimal-output-contract-20260715.json).

## E156–E158 constrained decoder follow-up (CPU diagnostic, 2026-07-16)

The matched judged-corpus follow-up is recorded in
[iter-e156-singleton-fix-20260715.md](iter-e156-singleton-fix-20260715.md).
The singleton legal-token fix and compiler/tree decode both leave parse at 0.0;
64 training steps lower loss to 9.6653 but still leave parse at 0.0. These are
negative diagnostic results, not ship claims or data-promotion evidence.

The later symbolic-tree audit is recorded in
[iter-e159-e165-symbolic-tree-20260715.md](iter-e159-e165-symbolic-tree-20260715.md):
lexer-native output is required, BOS root handling and partial-forest fallback
were repaired, and E165 reports zero seeded unconstrained fallbacks. Parse still
fails, so the remaining lever is semantic model choice within the certified
tree—not undertraining or weakened gates.

E166–E168 further separate syntax from meaningful-program quality:
`syntax_parse_rate=0.6667`, `meaningful_program_rate=0.0`, and zero compiler or
seeded fallbacks in E168. See
[iter-e166-e168-semantic-boundary-20260715.md](iter-e166-e168-semantic-boundary-20260715.md).

E169 removed the literal-specific compiler filters and reran the matched E160
lexer checkpoint: syntax fell to 0.3333 with meaningful parse still 0.0 and
zero seeded unconstrained fallbacks. The negative result shows that Lark
reachability permits partial expression continuations; the next lever is
generated AST/schema completion state, not more steps or case-specific bans.
See [iter-e169-grammar-derived-20260716.md](iter-e169-grammar-derived-20260716.md).

E170 reads the active call frame from Lark parser state instead of source-text
scanning. It is quality-neutral on the matched checkpoint
(`syntax_parse_rate=0.3333`, `meaningful_program_rate=0.0`, zero seeded
fallbacks), so the next step is generated-AST/schema completion state. See
[iter-e170-lark-state-20260716.md](iter-e170-lark-state-20260716.md).

E172 maps generated schema value types to Lark terminal categories. Syntax
validity improves to 0.6667 and compiler fallbacks fall to 1, while meaningful
parse remains 0 and component recall is 0.25. The next lever is semantic
component-role supervision and training-data coverage. See
[iter-e172-schema-types-20260716.md](iter-e172-schema-types-20260716.md).

E173 trained 32 steps with schema and slot-contract context. A bounded probe
was syntactically valid but still selected a lone `TextContent` for the hero
task (`meaningful_program_rate=0.0`); the full smoke invocation did not persist
a scoreboard and is not claimed. The next lever is semantic component-role
coverage/supervision. See [iter-e173-schema-context-20260716.md](iter-e173-schema-context-20260716.md).

E174 tested an unfrozen HF context tower for 8 steps. Loss rose to 39.4253 and
the bounded syntax probe fell to 0.0, so the control is rejected; retain frozen
context while improving semantic data coverage. See
[iter-e174-unfrozen-context-20260716.md](iter-e174-unfrozen-context-20260716.md).

E175 added retrieval k=4 to frozen schema-context training. The bounded probe
had syntax 0.0 and meaningful parse 0.0, so retrieval is rejected; semantic
role coverage/supervision remains the next lever. See
[iter-e175-retrieval-20260716.md](iter-e175-retrieval-20260716.md).

E176 trained on the broader 1,417-record prompt-contract corpus. The bounded
probe remained parse/syntax 0.0 and structure fell to 0.1187, so the broad
corpus is rejected as a replacement. Add targeted judge-gated semantic-role
examples instead. See [iter-e176-broad-corpus-20260716.md](iter-e176-broad-corpus-20260716.md).

E177–E180 separate data admission, schema structure, and semantic selection.
The corrected E177 judge rejects two mismatched pairs and publishes 496 records
with telemetry for future runs, but its matched 32-step train does not improve
the bounded probe. E178 adds generated-schema required/max arity; E179 stops the
legacy repair path from overwriting compiler output; E180 constrains the symbolic
root binding to generated component kinds. E180 raises syntax validity from 0.0
to 1.0 and cuts p50 from 7538.12 ms to 3712.78 ms, while meaningful parse remains
0.0 because component recall is only 0.25. Continue with balanced semantic-role
supervision, not syntax special cases or weakened gates. See
[iter-e177-e180-semantic-compiler-20260716.md](iter-e177-e180-semantic-compiler-20260716.md).

E181–E194 test the next generalized layer. A committed balanced mixture lowers
the matched 32-step loss but leaves quality unchanged. Root score telemetry
falsifies complete-path length bias and identifies learned semantic ranking as
the failure. Component-only compiler-state alignment recovers the `Stack` root;
random alignment across all grammar branches regresses it and is rejected.
Grammar-derived AST completeness, lexer-surface synchronization, nested call
frames, binder scope, and schema-scoped symbols replace tactical output-string
repairs. The best E194 diagnostic still has meaningful parse 0.0 (structure
0.3600, component recall 0.25), so no checkpoint is promoted. Next: stratified
component/binder alignment derived from parser decision kinds. See
[iter-e181-e194-compiler-alignment-20260716.md](iter-e181-e194-compiler-alignment-20260716.md).

E195–E199 stratify compiler alignment by tokenizer/parser-derived decision kind.
E195 is an invalid control because `--train-version` did not load its online
mixture; that resolver is fixed for future runs. The matched E196 train covers
component, binder, structural, symbol, and literal decisions on every eligible
row. E199 restores syntax 1.0 with zero compiler fallbacks after enum restriction
is tied to parser slot progress, but meaningful parse and component recall remain
0.0 because the forward binder receives a primitive declaration. No checkpoint
is promoted. Next: generated schema/AST reference-role propagation. See
[iter-e195-e199-stratified-alignment-20260716.md](iter-e195-e199-stratified-alignment-20260716.md).

E200–E204 generalize the next failure into grammar/schema roles. A canonical
audit found all 1,916 declarations in the 496-record judged corpus have
component-valued RHS ASTs. The decoder now derives declaration, generated
`children`, and content-slot candidate roles from typed Lark tokens, generated
schema properties, and the centralized content contract. E204 improves
component recall to 0.25 and placeholder validity to 0.70 with zero fallback,
but recursively extends legal children until the token cap; syntax and
meaningful parse remain 0.0. E201 is not promotable. Full evidence:
[iter-e200-e204-layout-role-compiler-20260716.md](iter-e200-e204-layout-role-compiler-20260716.md).

E205–E207 split structural alignment by active Lark terminal instead of one
tokenizer-level `struct` bucket. A bounded 64-record audit found 134 list-close
and 69 list-extend decisions; the matched E205 train supervises each grammar
terminal class. E207 adds generated-schema enum completion paths through the
typed literal channel. Syntax reaches 1.0, structure 0.3125, and compiler
fallback falls to zero, but empty bound stacks keep meaningful parse and
component recall at 0.0. E205 is not promotable. Full evidence:
[iter-e205-e207-lark-terminal-alignment-20260716.md](iter-e205-e207-lark-terminal-alignment-20260716.md).

E208–E213 test contextual grammar-decision alignment. The committed corpus has
496/496 populated roots, versus 156 populated and 12 empty bound containers.
Occupancy-only E208 and root/bound-scoped E210 still emit empty roots. E212
derives binder signatures from declaration/reference role, typed scope, and the
active generated-schema slot; root-children reference loss falls from 61.4179
to 1.0285 and E213 recovers a populated root plus normalized fidelity 0.50.
Required `FormControl` input semantics still fail, leaving syntax/meaningful
parse 0.0. No checkpoint is promotable. Full evidence:
[iter-e208-e213-contextual-decisions-20260716.md](iter-e208-e213-contextual-decisions-20260716.md).

E214–E216 test the next generalized data hypothesis. G11 now parses outputs and
checks resolved AST property values against generated schema roles; no component
names are special-cased. It rejects 49/496 E177 candidates and commits
the 447 accepted records as immutable E214 data with synthesis telemetry. The
matched E215 train reaches E216 syntax 1.0 with zero fallback or constrained dead
ends, eliminating E213's invalid `FormControl.input` failure. Meaningful parse
remains 0.0 because component recall is only 0.25, so this is a negative ship
result and E215 is not promotable. A later audit below supersedes the claim that
all 49 rejects were invalid. Full evidence:
[iter-e214-e216-schema-role-judge-20260716.md](iter-e214-e216-schema-role-judge-20260716.md).

Post-audit correction: E214 overfiltered 27 legal optional positional `null`
omissions. E218 moves canonical schema normalization before G11, derives Slider
enum/array shapes and language-contract `anyOf` values from generated schema, and
restores 33 records versus E214. The matched E219/E220 control preserves syntax
1.0 but exactly matches E216's component recall 0.25 and meaningful parse 0.0.
The data fix is retained for future runs; semantic quality did not improve and no
checkpoint is promotable. Full evidence:
[iter-e218-e220-schema-normalization-20260716.md](iter-e218-e220-schema-normalization-20260716.md).

These rows are registered under `--matrix v8`; `--list` prints definitions
without building data or starting a run. They require the lexer-native parent
contract and unchanged honest five-suite gates. No result or champion is claimed.

| ID | Isolated lever | Required diagnostics | Status |
| --- | --- | --- | --- |
| E200 | Current fixed-row DSL symbol control | Standard scoreboard plus active-symbol recall | proposed/unrun |
| E201 | Slot permutation and alpha-renaming augmentation | Rename invariance and permutation robustness | proposed/unrun |
| E202 | Surface-conditioned features for every role | Entity copy and binder rename sensitivity | proposed/unrun |
| E203 | Role-gated features | Binder invariance with entity/state sensitivity | proposed/unrun |
| E204 | E203 plus semantic candidate masks | Candidate recall; false hard-prunes must be zero | proposed/unrun |
| E205 | E204 plus hybrid grammar/attention graph | Graph conflict, accept, and remask rates | proposed/unrun |
| E206 | Complete stack, fixed padded canvas | Quality/steps/tokens control for E207 | proposed/unrun |
| E207 | Complete stack, compact active canvas | Matched quality, active tokens, steps, latency | proposed/unrun |

Preview only:

```bash
python -m scripts.run_quality_matrix --matrix v8 --list
```

Any execution must add AgentEvals/AgentV evidence, result JSON, recipe/suite sizes,
and a matching measured-results update before the run is considered complete.

E226 corrects the E225 evaluation contract without training a new checkpoint.
Telemetry now wraps the production request API, ship evals enforce the visible
slot contract, and compiler completion derives document boundaries from Lark plus
the generated AST. The existing E224 checkpoint reaches syntax parse 1.0 and
contract precision 1.0 on all five suites with zero fallback/full projection.
Meaningful-program rate remains 0/0/0/0/0.3333 and five honest gates fail, so
the next lever is topology/component branch supervision rather than grammar or
sampler repair. Full evidence:
[iter-e226-honest-compiler-policy-20260716.md](iter-e226-honest-compiler-policy-20260716.md).

E227 matched E224 while replacing full-vocabulary compiler alignment CE with
loss over each Lark/compiler forest's legal candidate set. The mechanism
executed (alignment loss 15.3994 to 2.4120), but honest tree evaluation collapsed
to empty layouts: syntax remained 1.0, meaningful-program rate was 0.0 on all
five suites, 12 gates failed, and AgentV passed 0/5. Restricted candidate CE is
insufficient; the next justified lever is a grammar-derived positive margin for
populated-child choices over legal empty-list alternatives. Full evidence:
[iter-e227-candidate-set-alignment-20260716.md](iter-e227-candidate-set-alignment-20260716.md).

E228 adds a configurable grammar-legal positive margin to E227. After a required
latest-main fetch and clean rebase, the matched 32-step run reduced margin
violations from 0.9130 to 0.5636. Honest evaluation retained syntax and contract
precision 1.0, recovered populated layouts, and reduced failed gates to four;
AgentV remains 1/5, so the local checkpoint is diagnostic only. Full evidence:
[iter-e228-candidate-margin-alignment-20260716.md](iter-e228-candidate-margin-alignment-20260716.md).

E229 resumed E228 bit-exactly to total step 64 after a clean latest-main audit.
An initial syntax regression exposed tokenizer-frame leakage, not undertraining:
quote-equivalent lexer tokens were admitted inside `LIT_STR + BYTE* + LIT_END`.
A token-kind-derived frame restriction restored syntax 1.0 on all suites without
literal cases. Corrected evaluation still fails the same four gates and regresses
several quality metrics, falsifying more duration for this recipe. Full evidence:
[iter-e229-margin-continuation-20260716.md](iter-e229-margin-continuation-20260716.md).

E230 replaces the derivative-heavy E218 exposure with a source-controlled,
independently judged 126-root corpus spanning RICO, human-curated, ProgramSpec,
language-contract, renderer, and web producers. Sampler telemetry confirms 30
RICO and 25 human draws in the matched 32-step run. Strict evaluation exactly
matches E228 on four suites and regresses adversarial quality; the same four gates
fail and AgentV remains 1/5. Retain the pipeline/data repair, reject the
checkpoint, and require request-level schema/AST component supervision next.
Full evidence:
[iter-e230-diverse-judged-roots-20260716.md](iter-e230-diverse-judged-roots-20260716.md).

E231 adds grammar-derived request-level component-inventory supervision and
biases only compiler-legal component candidates. The auxiliary target learns
strongly (top-k recall 0.0000 → 0.9167), but a full bias-off ablation has
identical aggregate metrics and component choices. Strict evaluation keeps
syntax 1.0 yet fails six thresholds across four suites; AgentV remains 1/5.
Retain the generalized instrumentation and causal override, reject the checkpoint
and the pooled inventory as a sufficient semantic solution. Full evidence:
[iter-e231-component-inventory-20260716.md](iter-e231-component-inventory-20260716.md).

E232 derives separate root-component and bound-component count targets from the
compiler's grammar-role decisions. The planner learns both targets and causally
improves adversarial recall/fidelity, but only 3/137 applications change a choice;
weight 4 changes 19 choices with no aggregate gain. Syntax remains 1.0, the same
four frontier thresholds fail, and AgentV remains 1/5. Retain the generalized
mechanism, reject the checkpoint and stronger calibration. Full evidence:
[iter-e232-role-component-plan-20260716.md](iter-e232-role-component-plan-20260716.md).

E233 derives parent→child component targets from the official parser's resolved
AST and conditions legal bound-component ranking on the compiler token-role
reference graph. Edge top-k recall learns from 0 to 0.50, but edge-off and
edge-on evaluations are identical across all five suite aggregates; only 1/47
applications changes a choice. Four frontier thresholds still fail and AgentV
remains 1/5. Retain the generalized mechanism and reject the checkpoint. Full
evidence:
[iter-e233-component-edges-20260716.md](iter-e233-component-edges-20260716.md).

E234 replaces global edge BCE with parent-conditioned cross-entropy over each
compiler decision's legal components. Decision accuracy learns from 0 to 0.5714
and edge bias changes 5/53 choices, but edge-on and edge-off suite aggregates
remain identical. Four frontier thresholds still fail and AgentV remains 1/5.
The final sampled batch also has 16 bound decisions without a prefix-known
parent versus 14 aligned rows, motivating binder-level instance topology next.
Retain the generalized objective and reject the checkpoint. Full evidence:
[iter-e234-edge-decision-alignment-20260716.md](iter-e234-edge-decision-alignment-20260716.md).

E235 indexes component targets by the compiler grammar's binder instances, so
all 30 final bound rows receive direct supervision instead of E234's 14 aligned
and 16 unknown-parent split. Binder accuracy learns from 0 to 0.40 and changes
4/16 applied legal choices, but binder-on and binder-off suite aggregates remain
identical. Syntax stays 1.0, nine thresholds across four suites fail, and AgentV
remains 1/5. Retain the generalized binder planner, reject the checkpoint, and
model binder topology/arity next. Full evidence:
[iter-e235-binder-instance-plan-20260716.md](iter-e235-binder-instance-plan-20260716.md).

E236 predicts parent→child binder references at compiler-classified legal
decisions. The direct weight-1 objective does not learn (accuracy 0.5455 →
0.5238), its 38 decode applications change zero choices, and semantic quality
collapses to zero across all suites despite syntax 1.0. Twelve thresholds fail
and AgentV is 0/5; the decode-off ablation is identical. Retain the generalized
mechanism and telemetry, reject the checkpoint, and require normalized/staged
topology learning plus explicit reference arity before retrying. Full evidence:
[iter-e236-binder-topology-20260716.md](iter-e236-binder-topology-20260716.md).

E237 detaches pooled context before the topology head to test shared-feature
interference. Because the HF context tower is already frozen, the change is a
no-op: train diagnostics and every suite aggregate reproduce E236, 38 decode
applications change zero choices, twelve thresholds fail, and AgentV is 0/5.
Retain the defensive detach for future unfrozen runs, reject the hypothesis and
checkpoint, and reformulate topology around reference arity/stop decisions.
Full evidence:
[iter-e237-detached-topology-20260716.md](iter-e237-detached-topology-20260716.md).

E238 adds grammar-derived binder-reference arity and directly scores
continue/stop completion paths. The head learns, but the run is invalidated:
optional-head initialization advanced global Torch RNG and silently changed
matched masking/dropout draws. Strict evaluation failed ten thresholds and
AgentV was 0/5; decode on/off aggregates were identical despite three changed
choices. Auxiliary modules now use isolated stable seeds, and E239 is the
corrected rerun. Full evidence:
[iter-e238-binder-arity-confounded-20260716.md](iter-e238-binder-arity-confounded-20260716.md).

E239 removes every observed optional-head coupling: isolated initialization,
detached auxiliary backward, separate optimizer groups, and per-group clipping.
The matched candidate/control checkpoints have 104/104 bit-exact shared tensors.
The arity target learns and 1,606 applications change 29 legal choices, improving
smoke syntax 0→0.3333 and structure 0.1591→0.2591. Meaningful-program rate is
still 0 on all five suites, however; both on/off settings fail 11 thresholds and
AgentV is 0/5. Retain the generalized mechanism and isolation invariants, reject
the checkpoint, and correct pathological long compiler-tree trajectories before
wider search. Full evidence:
[iter-e239-binder-arity-isolated-20260716.md](iter-e239-binder-arity-isolated-20260716.md).

## V9 lattice-guided recursive compiler search (two fixture-grade runs 2026-07-16)

The research synthesis and implementation boundary are in
[`lattice-recursive-search.md`](lattice-recursive-search.md). These rows keep the
compiler completion forest authoritative, use model scores only to order legal
paths, and compare bounded rollback with selectively triggered PTRM/GRAM-style
trajectory policies. They do not reproduce those papers' training methods.

| ID | Isolated lever | Required diagnostics | Status |
| --- | --- | --- | --- |
| E240 | Corrected greedy compiler-tree control | Standard scoreboard, coverage, fallbacks, calls | measured; fail |
| E241 | Hard/soft lattice plus bounded rollback | Bottoms, rollbacks, nogoods, termination | measured; identical to E240; fail |
| E242 | Stagnation-triggered localized nogoods | Conflict recurrence and false-prune audit | measured; no trigger; fail |
| E243 | Triggered PTRM-style width 4 | Triggers, trajectories, unique valid ASTs, calls | measured; no trigger; fail |
| E244 | Always-on PTRM-style width 4 control | Matched quality/calls against E243 | measured; semantic collapse; fail |
| E245 | GRAM-style semantic diversity width 4 | Unique validated AST fingerprints | measured; no trigger; fail |
| E246 | Full stack width 4 | Quality, validity, abstention, regret, latency | stopped by continuation rule (E228 campaign); fixture-run (scratch campaign) |
| E247 | Full stack width 8 | Width scaling benefit versus verifier/call cost | stopped by continuation rule (E228 campaign); fixture-run (scratch campaign) |

Statuses above are from the E228-checkpoint evaluation-only campaign; a second
fixture-grade campaign the same day ran all eight rows eval-only from a fresh
E240 scratch control (see "V9 measured results" below).

Preview only:

```bash
python -m scripts.run_quality_matrix --matrix v9 --list
```

Listing must not create output artifacts. Any execution requires the full honest
five-suite scoreboard, AgentEvals, AgentV, result JSON, recipe/suite sizes, and a
measured-results update in this document.

### Measured E240-E245 result (2026-07-16 UTC)

The CPU evaluation-only campaign used the unchanged E228 checkpoint (SHA-256
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`), seed
0, honest schema/slot context, and suite sizes 3/5/4/4/3. E240-E243 and E245
were output-identical: syntax was 1.0, but smoke meaningful was 0.333,
held-out and OOD meaningful were 0, and RICO structure was 0.163, leaving four
gates failed. The triggered policies never activated.

E244 always-on width 4 made 76 verifier calls and found only one valid AST per
selected-valid record; meaningful rate and component recall became 0 on all
five suites, structure fell to 0.017-0.057, and median latency rose to
52.4-68.5 seconds. Syntax stayed 1.0 and false hard eliminations remained zero,
but the semantic regression rejects the hypothesis. E246-E247 were not run
because E244 failed the predeclared continuation rule. Full recipe, telemetry,
scoreboards, AgentV evidence, and applicability boundary:
[`lattice-recursive-search.md`](lattice-recursive-search.md) and
[`iter-e240-e245-lattice-search-20260716.json`](iter-e240-e245-lattice-search-20260716.json).

The trace-backed E240 control, including exact suite metrics, aggregate decode
telemetry, checkpoint lineage, and the four gate failures, is recorded separately
in [iter-e240-greedy-tree-control-20260716.md](iter-e240-greedy-tree-control-20260716.md).

### V9 measured results (CPU, fixture-grade, 2026-07-16)

Recipe: E240 trained as an explicit `--scratch-control` row (800 steps, batch 4,
seed 0, `--context-backend scratch`, `--no-design-md-context`, fixture v1 corpus,
108 records); E241–E247 evaluated **eval-only from E240's frozen checkpoint** via
`--eval-checkpoint` (shared lineage, the campaign's matched-checkpoint
requirement). Suites: smoke 3 / held_out 5 / adversarial 4 / ood 4 / rico_held 0
(fixture corpus has no RICO records; full 1500 remains the ship bar). AgentV
published per row. JSON:
[quality-matrix-results-iter-v9-lattice-20260716.json](quality-matrix-results-iter-v9-lattice-20260716.json);
full narrative:
[iter-e240-e247-lattice-campaign-20260716.md](iter-e240-e247-lattice-campaign-20260716.md).

All eight rows fail the honest gates (syntax parse 0.0, meaningful parse 0.0):
the 800-step scratch model emits non-placeholder string literals, which the
strict placeholder policy rejects — a genuine fixture-scale capacity result,
not a broken verifier (per-record errors are real placeholder-policy
violations). Signal lives in the lattice diagnostics: stagnation-triggered rows
(E241–E243, E245–E247) observe the forest (891 lattice states, ~29k ranked
candidates) with **zero bottoms/rollbacks/nogoods/trajectories** and reproduce
E240's greedy outputs byte-for-byte; always-on PTRM (E244) fires 1680 triggers /
6640 trajectories at ~3× decode latency (24.1s vs ~8s per record) and lowers
structural similarity on every suite (e.g. ood 0.413→0.239) — supporting
selective over always-on stochasticity. **Wiring evidence only, not a ship
claim**; the ship-grade campaign against local E224+ frontier checkpoints (GPU,
full suites) is still required, and E241/E242's conflict machinery has not been
exercised outside unit/integration tests because greedy decode never stalls on
this checkpoint.

## LDI campaign index (local decision interventions)

The **LDI** campaign is the local-decision-intervention line of work. Its canonical
architecture/research contract, invariants, named owners, and the 42-source manifest
are in [`local-decision-interventions.md`](local-decision-interventions.md). This
index is a namespace pointer, not a new set of rows: it claims **no** unrun row and
allocates **no** E ID.

Measured record (authoritative): the **V10 exact-state local preference** rows
E248-E254 (below) and the measured **E265-E286 local-preference ledger** recorded in
this matrix and the per-run `iter-e2*.md` docs (broad/guarded FTPO, reference
tethers, balanced sampling). The chain is negative — E249 and E252 are rejected
(local metrics moved, semantic quality regressed), and no LDI intervention has
cleared the unchanged five-suite ship gates or been promoted. Current blocker:
**stable state support does not imply objective/action-partition support**; exact
state identity does not prove the good/bad action partition is verifier-supported.
`DecisionEventV2` action-verdict tables (LDI0-02) target this gap.

**E-ID allocation rule.** New LDI experiments take a globally unique E ID from the
existing allocation process; the `LDI` name is prose/config only and reserves no ID.
As of 2026-07-17 the highest allocated ID is **E291** (B1/B3 tracks; see the
[`README.md`](../../README.md) run ledger), and E248-E291 plus the E263/E264
local-preference rows are consumed. Do not assume "the next number after E286" is
free — the next free ID is **≥ E292**.

## V10 exact-state local preference (E248 control measured)

The full source audit (34 works), source manifest, objective definition, and honesty
boundary are in [`local-decision-interventions.md`](local-decision-interventions.md).
V10 reuses the existing preference harness and append-only decode traces. It does not
introduce an adapter/SAE trainer and does not claim that a local loss produces a
local parameter update.

All rows require one immutable `DecisionEventV1` JSONL, the same parent checkpoint,
split, steps, learning rate, and seed. E252-E254 fail closed unless the training
split contains at least one same-state-verified multi-good or multi-bad event.

| ID | Isolated lever | Required diagnostics | Status |
| --- | --- | --- | --- |
| E248 | Unchanged parent control | Standard five-suite scoreboard | measured; matched control; 4 gates failed |
| E249 | Exact-event CE plus margin | Event win/margin and per-kind recurrence | measured; lexical objective generalized; semantic quality regressed; rejected |
| E250 | Bad-token unlikelihood | Bad probability mass and held-out recurrence | proposed/unrun |
| E251 | Single-pair clipped FTPO | Active weight, chosen/margin win, drift | proposed/unrun |
| E252 | Verifier-backed set FTPO | Set coverage, evidence source, held-out recurrence | measured; local held-out margin improved, but semantic quality collapsed; rejected |
| E253 | E252 plus frozen-reference tether | Non-target MSE, target excess MSE, unchanged decisions | proposed/unrun |
| E254 | E253 plus balanced sampling | Source/kind/rejected-set exposure and all E253 metrics | proposed/unrun |

Preview without artifacts:

```bash
python -m scripts.run_quality_matrix --matrix v10 --list
```

Execution requires a parent and mined events:

```bash
python -m scripts.run_quality_matrix --matrix v10 --only E248,E249,E250,E251 \
  --parent <checkpoint.pt> --decision-events <local_decisions.jsonl>
```

Initial hypothesis constants are margin `epsilon=2`, temperature `tau=1`,
non-target tether `0.4`, target tether `0.05`, and target grace `1.0`. They are
borrowed starting points, not validated TwoTower settings. Any future execution
must update the canonical result JSON and this measured-results ledger, publish
AgentEvals/AgentV, preserve unchanged ship gates, and update the model card for
every checkpoint created.

E248 evaluates the unchanged E228 parent without copying or training a
checkpoint. The corrected strict policy exactly reproduces E240: syntax is 1.0
on all 19 examples, four gates fail, and AgentV passes 1/5. An earlier parse-zero
attempt was invalidated as harness policy drift; V9 and V10 now consume one
shared strict compiler-tree policy. Full evidence:
[iter-e248-local-parent-control-20260716.md](iter-e248-local-parent-control-20260716.md).

The E249 prerequisite mined 2,035 exact production compiler-tree constraint
shadows from 65 document groups and committed the identity-bound train/held-out
corpus. These events certify grammar legality, not semantic preference. E252-E254
remain fail-closed because no counterfactual set-valued evidence exists. Full
telemetry and diagnostics:
[iter-e249-exact-event-mining-20260716.md](iter-e249-exact-event-mining-20260716.md).

E249 then moved all 319 held-out constraint shadows strongly in the requested
direction (chosen win `0→0.7649`, margin win `0→0.6489`) while structural
similarity and reward regressed on every suite. Syntax remained 1.0, eight ship
thresholds failed, and AgentV passed 0/5. This falsifies constraint shadows as
semantic preference labels: keep their legality guarantee in deterministic
decoding and require counterfactual semantic evidence before another local
preference train. Full evidence:
[iter-e249-local-ce-margin-20260716.md](iter-e249-local-ce-margin-20260716.md).

The original E252 zero-event diagnostic was invalidated: an isolated worktree
without bridge dependencies silently lost the official schema and AST/judge
parity. The generalized repair commits a parity-checked schema snapshot, uses
the Lark AST fallback, preserves partial-prefix newlines, and derives statement
separation and candidate admission from grammar/schema semantics. A corrected
32-record probe accepted every production trace and produced six judge-qualified
events across three groups, including three set-valued events. All groups map to
train under the stable split, so no corpus or checkpoint was created and E252
remains fail-closed pending held-out recurrence. Full evidence:
[iter-e252-counterfactual-prerequisite-20260716.md](iter-e252-counterfactual-prerequisite-20260716.md).

## V11 B4 AR→diffusion adaptation baseline (fixture-run 2026-07-16)

Track B4 (DiffuGPT/DiffuLLaMA, [arXiv:2410.17891](https://arxiv.org/abs/2410.17891),
**Adapted**: only the drop-the-causal-mask move is reused — no attention-mask
annealing, shift operation, or their training recipe): the pretrained SmolLM2-135M
causal LM becomes the *denoiser* tower (`denoiser_backend="hf"`,
`models/hf_denoiser.py`) with full bidirectional visibility via an explicit 4D
attention mask, fresh OpenUI-vocabulary embeddings (weight-tied lm_head), and the
context tower's hiddens prepended as projected prefix states. Matched pair
differing only in the denoiser backbone; parallel MaskGIT decode on both rows.

| ID | Isolated lever | Trainable params | Status |
| --- | --- | ---: | --- |
| E255 | From-scratch DenoiserTower control | 1.1M | fixture-run |
| E256 | SmolLM2-135M AR→masked-denoiser adaptation | 135M | fixture-run |
| E257 | C1 De Bruijn relative binder references (`bind_encoding=relative`) | 1.1M | fixture-run |

### V11 measured results (CPU, fixture-grade, 2026-07-16)

Recipe: `--scratch-control --steps 200 --lr 3e-4` (matrix default), batch 4, seed
0, scratch context tower, fixture v1 corpus (108 records), no DESIGN.md context;
suites smoke 3 / held_out 5 / adversarial 4 / ood 4 / rico_held 0 (fixture corpus
has no RICO records; ship bar remains n=1500). AgentV published per row. JSON:
[quality-matrix-results-iter-v10-b4-20260716.json](quality-matrix-results-iter-v10-b4-20260716.json);
narrative: [iter-e255-e256-b4-ar-adaptation-20260716.md](iter-e255-e256-b4-ar-adaptation-20260716.md).

Both rows fail the honest gates (syntax/meaningful parse 0.0 — placeholder-policy
rejections, as across the V9/V11 fixture runs). At this budget the adaptation is
**behind** the matched scratch control on every secondary signal: train loss 8.51
vs 3.75, structural similarity 0.09–0.16 vs 0.28–0.37, component_type_recall
0–0.19 vs 0.22–0.75, decode latency ~2× (30–37s vs ~15s per record). An
`--lr 3e-5` variant of E256 (separate run root,
[quality-matrix-results-iter-v10-b4-lr3e5-20260716.json](quality-matrix-results-iter-v10-b4-lr3e5-20260716.json))
checks the LR-destroys-pretrained-weights confound; see the iter doc for its
numbers. **Wiring evidence only — this fixture budget can neither confirm nor
kill the DiffuLLaMA hypothesis**: 200 CPU steps on 108 records is far below any
adaptation budget in the paper, so the B4 verdict requires a GPU-scale run with
per-arm LR selection. No gate weakened; nothing promoted.

**E257 (C1)** — identical recipe to E255, differing only in
`bind_encoding=relative` (nameless `<BINDDEF>` definitions + signed
statement-delta `<BINDREL_±k>` references; scope legality verifier-enforced).
Fixture result: syntax parse **0.667/0.6/0.25/0.5** (smoke/held_out/
adversarial/ood) vs **0.0** on the matched absolute control, train loss 3.27
vs 3.75, decode p50 1–8s vs ~15s; meaningful parse 0.0 on both (failures shift
to `empty_root_stack` — the valid-but-empty wall is Track A's target, not
C1's). JSON:
[quality-matrix-results-iter-v10-c1-20260716.json](quality-matrix-results-iter-v10-c1-20260716.json);
narrative:
[iter-e257-c1-relative-bind-20260716.md](iter-e257-c1-relative-bind-20260716.md).
Same honesty envelope as the B4 pair.

## V12 A2 ASAp distribution-aware constrained decode (fixture-run 2026-07-17)

Track A2 (Grammar-Aligned Decoding / ASAp,
[2405.21047](https://arxiv.org/abs/2405.21047), **Adapted**: only the adaptive
removal of observed constraint-violating mass is reused, transplanted from
ASAp's prefix trie onto the MaskGIT canvas position — no
sampling-until-acceptance loop, no convergence guarantee): `asap_decode`
(`models/parallel_decode.py::AsapLedger`) removes admit-reject and grammar
stream hard-error mass from the next proposal at that position and gives
unmask ordering the post-removal confidence. Decode-only, eval-only row routed
through the frozen E255 checkpoint via `--parent` — matched pair differing
only in `asap_decode` (enforced by `tests/test_scripts/test_quality_matrix_v14.py`).

| ID | Isolated lever | Baseline | Status |
| --- | --- | --- | --- |
| E277 | `asap_decode=True` (A2 ASAp mass removal) | E255 recorded eval | fixture-run |

### V14 measured results (CPU, fixture-grade, 2026-07-17)

Recipe: eval-only overlay on `qx_e255_b4_scratch_control/best_weighted_nll.pt`,
fixture v1 corpus, suites smoke 3 / held_out 5 / adversarial 4 / ood 4 /
rico_held 0, parallel MaskGIT decode, `--rico-limit 3`. JSON:
[quality-matrix-results-iter-v14-a2-20260717.json](quality-matrix-results-iter-v14-a2-20260717.json);
narrative + honesty envelope:
[iter-e277-a2-asap-decode-20260717.md](iter-e277-a2-asap-decode-20260717.md).
Ledger telemetry proves the lever is live: 204–334 `asap_penalties` across
32–53 distinct positions per suite; two runs produced byte-identical metrics.

## V15 C2 dynamic pseudo-embeddings (fixture-run 2026-07-17)

Track C2 (DyVo [2410.07722](https://arxiv.org/abs/2410.07722), **Adapted**:
dynamic-vocabulary embedding only): `runtime_symbol_features="replace"`
cancels the learned `<SYM_i>`/`<BIND_j>` pool rows with deterministic
byte-compositional vectors through the V8 delta path (weight tying and
batching untouched; same surface → identical vector at every slot, by
construction and by test).

| ID | Isolated lever | Baseline | Status |
| --- | --- | --- | --- |
| E278 | `runtime_symbol_features="replace"` (C2) | E255 recorded eval | fixture-run |

Recipe: scratch-control 200 CPU steps, fixture v1 corpus, matched vs E255 on
everything but the mode. JSON:
[quality-matrix-results-iter-v15-c2-20260717.json](quality-matrix-results-iter-v15-c2-20260717.json);
narrative: [iter-e278-c2-pseudo-embeddings-20260717.md](iter-e278-c2-pseudo-embeddings-20260717.md).
Honest gates fail on both rows; structural similarity dips vs control
(0.19–0.29 vs 0.28–0.37) — an honest fixture-scale negative. The
binding-consistency probe on the trained checkpoint reports same-surface
hidden cosine 0.9998 vs cross-surface 0.9679 (margin +0.032):
[binding-consistency-e278-20260717.json](binding-consistency-e278-20260717.json).
The run also flushed a latent stale-feature leak (fixed at the source in
`training_loss`; complementary to PR #275).

Honest gates still fail on both rows (syntax/meaningful parse 0.0 — the
fixture-scale placeholder-policy wall Track A targets at frontier scale, not
here). Secondary signals, structural similarity E255 → E277:
smoke 0.30 → 0.265, held_out 0.323 → 0.248, adversarial 0.281 → 0.370,
ood 0.372 → 0.278 — decode behavior demonstrably diverges under the ledger,
with mixed noise-level deltas at n≤5 per suite. **Wiring evidence only**: the
fixture budget cannot decide the ASAp hypothesis; the A2 verdict requires the
frontier checkpoints (GPU host, n=1500 RICO bar) where the A1-diagnosed
constraint distortion actually binds.

E256 then ran the repaired path across all 65 E230 document records. It
persisted 16 independently judged counterfactual events and their full probes
as an immutable source corpus: 14 train and two held-out events across eight
groups, with eight set-valued comparisons. This clears the predefined E252
corpus prerequisite but remains narrow root-decision evidence; training and all
quality gates are still pending. Full evidence:
[iter-e256-counterfactual-corpus-20260716.md](iter-e256-counterfactual-corpus-20260716.md).

E252 trained for 30 matched CPU updates on that corpus. Held-out FTPO loss and
margin improved, but good-token mass decreased; the full result is decisively
negative. Syntax stayed 1.0 through the deterministic compiler layer while
placeholder fidelity fell to zero on all suites, structure/reward regressed on
every suite versus E248, 13 thresholds failed, and AgentV passed 0/5. This
falsifies narrow root-only counterfactual supervision. E253/E254 remain blocked
until evidence covers deeper semantic decisions and more held-out groups; a
tether or sampling change is not a substitute for support. Full evidence:
[iter-e252-ftpo-set-20260716.md](iter-e252-ftpo-set-20260716.md).

E258 replaced chronological state truncation with deterministic stratification
over compiler-derived decision kinds and relative trajectory-depth quartiles.
Across 65 records it replayed 260 states and produced 18 qualified events across
six decision kinds, including bound components, child references, populated-root
closure, and symbols. The sampler hypothesis is confirmed, but only eight prompt
groups and one held-out group qualified, so this export is not admitted for
training. E253/E254 remain blocked pending broader group support. Full evidence:
[iter-e258-counterfactual-depth-probe-20260716.md](iter-e258-counterfactual-depth-probe-20260716.md).

E259 doubled the uniform budget to eight states per record: 520 states and 1,528
legal candidates produced 38 qualified events across eight decision kinds. The
extra depth added roles but no prompt groups; support remained eight groups with
one held-out group. This falsifies state-count scaling as the group-coverage fix,
so the export is not admitted and E253/E254 remain blocked. The next data lever
must derive broad exact states from grammar/AST-aligned judged trajectories, not
probe more states from the same poor model completions. Full evidence:
[iter-e259-expanded-counterfactual-probe-20260716.md](iter-e259-expanded-counterfactual-probe-20260716.md).

E260 tested that grammar/AST lever on 10 records. Forty exact gold-derived states
produced 30 qualified events across 11 decision kinds and nine prompt groups,
including six events from two stable held-out groups. Gold remained outside model
context; retained probes pair a gold-selected completion with policy-completed
legal alternatives and independent judge evidence. The bounded hypothesis is
confirmed, but training remains blocked until the identical all-record run is
persisted and audited. Full evidence:
[iter-e260-gold-ast-counterfactual-probe-20260716.md](iter-e260-gold-ast-counterfactual-probe-20260716.md).

E261 completed that all-record run. Across 65 accepted document traces, 260
exact gold-AST states and 736 legal candidates produced 239 independently judged
events across 14 decision kinds and 64 prompt groups. The immutable committed
corpus contains 200 train and 39 held-out events across 53/11 groups, including
108 set-valued comparisons; every retained probe is same-state verified and
pairs a gold-AST selected completion with policy-completed alternatives. The
corpus prerequisite is satisfied and a new semantic preference experiment is
unblocked, but model quality and ship gates remain unmeasured. Full evidence:
[iter-e261-gold-ast-counterfactual-corpus-20260716.md](iter-e261-gold-ast-counterfactual-corpus-20260716.md).

## V12 B1 choice-sequence codec (registered 2026-07-17; matrix row unrun)

Track B1 (SLM-42): a pure grammar-choice output stream — the model predicts
only semantic decisions (which production, which slot filler); all non-lexical
surface syntax is reconstructed by a deterministic detokenizer through the
official lang-core serializer (fail-closed, so parse is a meaningful honest
primary — the detokenizer never invents syntax for an invalid stream). New
`--output-tokenizer choice` beside compositional/lexer; codec in
`dsl/production_codec.py` (`encode_choices`/`decode_choices`), tokenizer in
`models/choice_tokenizer.py` (sidecar kind `choice_codec`). B2 canonical
alignment laws pinned in `tests/test_dsl/test_choice_codec.py`.

| ID | Isolated lever | Status |
| --- | --- | --- |
| E262 | B1 pure grammar-choice output stream (`output_tokenizer=choice`) vs E255 lexer control (same diffusion masking, non-LTR MaskGIT decode) | registered / unrun |

E2 semantic density (36 fixture seeds, measured 2026-07-17): choice stream
carries 842 decisions / 3713.2 bits vs production 1019 / 4391.9 and surface
1535 / 8368.0 — `surface_to_choice_bit_ratio` 2.254 (production: 1.905),
`production_to_choice_bit_ratio` 1.183; structural/punct/name categories
collapse to 0 (arity choices remain, honestly categorized). A fixture-scale
CPU wiring smoke of the identical harness path (choice + matched lexer
control) scored honest 0.0 parse on both arms at 120–2500 steps —
wiring evidence only; the matrix row needs the standard eval corpus and
budget. JSON + narrative:
[iter-b1-choice-sequence-codec-20260717.json](iter-b1-choice-sequence-codec-20260717.json),
[iter-b1-choice-sequence-codec-20260717.md](iter-b1-choice-sequence-codec-20260717.md).
No checkpoint was created or promoted; MODEL_CARD unchanged. v1 caveat: the
surface-DFA token gate is bypassed for choice ids (validation moves entirely
to the fail-closed detokenizer); a choice-native legal-decision gate is the
follow-up.

E277 then repeated E252's matched 30-step set-FTPO recipe using the committed
E261 corpus. Broad support prevented the E252 fidelity collapse: fidelity and
meaningful-program rates exactly matched the E248 parent, while deterministic
syntax remained 1.0 with no fallback or timeout. The objective still failed:
held-out FTPO loss worsened from 2.7660 to 3.0144, bad-token mass more than
doubled, structure regressed on all five suites, ten thresholds failed, and
AgentV passed 0/5. The checkpoint is rejected. The next lever must guard updates
against held-out exact-state and parent-semantic regressions, not add syntax
training or duration. This run originally emitted E262 before concurrent B1
claimed that ID; the measured preference result is canonically E277. Full
evidence:
[iter-e277-broad-gold-ast-ftpo-20260716.md](iter-e277-broad-gold-ast-ftpo-20260716.md).

E278 added a generalized held-out Pareto guard to that same objective. Every
five steps, held-out loss and bad-token mass had to be no worse while
good-token mass and mean margin had to be no worse. None of steps 5–30 was
eligible, so the harness restored step 0. All 374 restored tensors, the config,
and tokenizer sidecars are bit-identical to E228. A current-code E248 parent
control exactly reproduced every E278 suite metric and gate failure, proving
that differences from the historical E248 report are evaluator/decoder drift,
not a training gain. The guard is retained; the parent-equivalent E278 artifact
is rejected and not promoted. Full evidence:
[iter-e278-guarded-gold-ast-ftpo-20260716.md](iter-e278-guarded-gold-ast-ftpo-20260716.md).

E265 enforced that Pareto contract on every optimizer proposal with
optimizer-consistent backtracking. Three of 30 updates were accepted and all
four aggregate held-out metrics improved, proving a safe local direction
exists. The aggregate nevertheless hid severe per-decision-kind regressions
(`grammar_comma` loss `1.3764→3.1417`), and full-eval fidelity/reward fell on
most suites while five gates still failed. The naive implementation also took
50m09s for 142 candidate scales and 5,538 held-out event forwards. Reject the
checkpoint. The next guard must be stratified by grammar/AST decision kind and
validation must be batched/cached without weakening the contract. Full
evidence:
[iter-e265-safe-gold-ast-ftpo-20260717.md](iter-e265-safe-gold-ast-ftpo-20260717.md).

E266 replaced the aggregate-only contract with the same four guards applied to
every grammar/AST `decision_kind`, and batched same-length held-out states with
cached frozen context. All 30 proposals and 150 scales were rejected, proving
the tested global FTPO direction has no safe per-kind Pareto update. The model
was restored bit-identically to E228 and a same-code parent control reproduced
every suite metric. Batching cut the local stage from 3,009.05s to 79.77s
(37.7×) despite checking more scales. Retain the guard and batching; reject the
artifact. The next lever is decision-kind block-coordinate proposals, not more
global FTPO duration or a literal special case. Full evidence:
[iter-e266-stratified-safe-ftpo-20260717.md](iter-e266-stratified-safe-ftpo-20260717.md).

E267 averaged train losses within each grammar/AST `decision_kind` before
proposing an update, testing whether E266's single-event gradients were simply
too noisy. All 30 category blocks and 150 scales were rejected; the restored
model is bit-identical to E228 and full evaluation exactly matches the current
parent control. The batched stage remained practical at 90.27s. Category
averaging therefore does not produce a safe FTPO direction. The next lever
must construct a conflict-projected or minimum-norm combination of per-kind
gradients, not vary duration or scalar learning rate. Full evidence:
[iter-e267-block-stratified-ftpo-20260717.md](iter-e267-block-stratified-ftpo-20260717.md).

E268 constructed all 14 grammar/AST decision-kind gradients per step and
deterministically applied pairwise PCGrad before the unchanged stratified
guard. It projected 2,220 of 5,460 ordered task pairs, yet every one of 30
proposals and 150 scales regressed at least one per-kind metric. The restored
model and full evaluation exactly match the parent; five gates fail and AgentV
is 2/5. The local stage took 2,338.56s, 25.9x E267, so this implementation is
also operationally rejected. Pairwise projection does not certify a common
descent direction. The next generalized lever is a deterministic minimum-norm
convex combination with an explicit common-descent certificate, benchmarked
for one step before a full run. Full evidence:
[iter-e268-projected-stratified-ftpo-20260717.md](iter-e268-projected-stratified-ftpo-20260717.md).

E269 replaced PCGrad with the minimum-norm convex combination from MGDA and
used a one-step preflight before authorizing matched compute. After repairing
inactive zero-gradient handling and fail-closed optimizer bypass, the final
solver found a strict common-descent direction for 13 active train objectives.
All five scales still regressed held-out metrics in `component_bound`,
`grammar_comma`, `lit`, and `sym`; the parent was restored and full evaluation
retained five failures with AgentV 2/5. The 219.11s one-step cost projects to
about 110 minutes for 30 steps, so the full run was correctly canceled. The
next lever is train/held-out gradient-alignment and provenance diagnosis, not
another optimizer or scalar tuning pass. Full evidence:
[iter-e269-mgda-stratified-ftpo-20260717.md](iter-e269-mgda-stratified-ftpo-20260717.md).

E270 profiled frozen-parent train and held-out FTPO gradients without an
optimizer. Same-kind split gradients are nonnegative for every shared
decision kind, but the full matrix exposes severe cross-kind conflicts (for
example held-out `grammar_comma` vs train
`grammar_rsqb_bound_populated`, cosine `-0.9941`). MGDA still produces a raw
combined direction with positive dot product against every active held-out
FTPO-loss gradient; `grammar_comma` is weakest at cosine `0.0032`. Therefore
E269's rejected finite steps are an optimizer-geometry mismatch: AdamW's
preconditioned/sign-like first update is not the raw gradient direction the
MGDA certificate covers. The next diagnostic must certify the actual
optimizer-transformed step before any new training. Full evidence:
[iter-e270-preference-gradient-alignment-20260717.md](iter-e270-preference-gradient-alignment-20260717.md).

E271 analytically profiled the exact fresh Adam/AdamW first-step directions
without mutating the model. Both transforms reverse held-out
`grammar_comma` (cosine about `-0.00913`) and train-only `grammar_lsqb`
(`-0.00345`), while their values are nearly identical; decoupled weight decay
is not the cause. Adam's adaptive sign-like preconditioning breaks the raw
MGDA common-descent certificate. The next bounded training lever is a one-step
MGDA plus SGD preflight under the unchanged stratified guard, not another
gradient mixer or AdamW scalar tune. Full evidence:
[iter-e271-preference-optimizer-geometry-20260717.md](iter-e271-preference-optimizer-geometry-20260717.md).

E272 applied the certified MGDA raw gradient with collinear SGD under the
unchanged strict guard. Aggregate held-out FTPO loss improved at every scale,
but all five scales regressed nine guarded probability/margin metrics across
six decision kinds. The parent was restored; five ship gates fail and AgentV
is 2/5. This rules out optimizer geometry as the final blocker: the solver's
objective is incomplete because it certifies only loss while the contract also
guards bad mass, good mass, and mean margin per kind. The next diagnostic must
cover the full metric-gradient constraint set before any training. Full
evidence:
[iter-e272-mgda-sgd-preflight-20260717.md](iter-e272-mgda-sgd-preflight-20260717.md).

E273 differentiated all four guarded metrics for every train decision kind:
56 objectives, 55 active. Their minimum-norm vector is effectively zero
(`norm_sq=3.90e-8`) and still lacks common descent; twelve held-out objectives
oppose it. Probability-mass objectives dominate the conflict. No
metric-complete optimizer run is justified. Before changing data or model
capacity, verify whether good/bad mass is evaluated in the wrong probability
space: it currently uses full-vocabulary softmax although constrained decoding
chooses only among `legal_token_ids`. Full evidence:
[iter-e273-metric-complete-feasibility-20260717.md](iter-e273-metric-complete-feasibility-20260717.md).

E274 repeated the frozen-parent profile with good/bad probability conditioned
only on each event's grammar-derived legal candidates. The train-side result
flips from no feasible direction to strict common descent (`norm_sq=3.81e-4`,
minimum active-task dot `3.36e-4`), proving full-vocabulary mass created a false
Pareto conflict for constrained decisions. Training remains blocked: eleven
held-out objectives oppose the corrected direction, and raw gradient scale
assigns `0.9964` of the minimum-norm mixture to `lit:good_probability_mass`.
The next diagnostic must normalize objective gradients before combining them;
do not change duration or add token-specific cases. Full evidence:
[iter-e274-legal-conditioned-metric-feasibility-20260717.md](iter-e274-legal-conditioned-metric-feasibility-20260717.md).

E275 unit-normalized every nonzero legal-conditioned metric gradient before
minimum-norm combination, then checked the direction against the original
unscaled objectives. All 55 active train objectives align positively and the
single-metric weight collapse disappears. Held-out regressions fall from eleven
to three: component-bound good/bad mass and literal loss. The direction remains
unsafe, so no training ran. The next diagnostic must split kind-level averages
by grammar/AST decision signatures derived from legal/good/bad sets, not add
literal or component special cases. Full evidence:
[iter-e275-normalized-metric-geometry-20260717.md](iter-e275-normalized-metric-geometry-20260717.md).

E276 kept E275's train direction fixed and evaluated 21 held-out signatures
derived from decision kind plus legal/good/bad token sets. Nine signatures have
no train counterpart. Seventeen objective regressions concentrate in seven
signatures: four are absent from train and the other three have only one to
three train examples. Coarse kind-level averaging was hiding sparse semantic
support. No training ran. The next lever is judged, leakage-safe synthesis with
minimum support per grammar-derived signature, followed by this same profile;
do not add token/component special cases. Full evidence:
[iter-e276-decision-signature-alignment-20260717.md](iter-e276-decision-signature-alignment-20260717.md).

E277 added signature-diverse gold-AST state sampling, sharded trace ingestion,
manifested support coverage, and fail-closed admission before corpus writes.
The completed 65-record synthesis produced 362 independently backed events,
but only 14/23 held-out support signatures have train coverage. The strict
builder rejected the corpus before publication; more event volume alone did
not repair support. Seven missing states exist in train gold ASTs and two
require new leakage-safe judged records. Target grammar-state metadata next;
do not train this candidate or hard-code component names in the compiler. Full
evidence:
[iter-e277-signature-coverage-synthesis-20260717.md](iter-e277-signature-coverage-synthesis-20260717.md).

E283 replaced bounded random coverage repair with exact compiler-state targets
derived from decision kind, legal token set, and selected token. Six judged
E230 train-group records repaired seven signatures. Two fresh generation
records synthesized from grammar/component semantics independently passed the
pairing judge, meaningful-program verifier, and exact compiler-state check;
held-out prompts/programs were not copied. The combined immutable corpus has
372 judge-backed events (311 train / 61 held-out), all 23 held-out support
signatures are covered, and the strict admission gate passes. Both source
records and preference events are committed and visible to the training-data
API. No model training ran. Full evidence:
[iter-e283-signature-support-repair-20260717.md](iter-e283-signature-support-repair-20260717.md).

E284 reran E276's frozen-parent legal-conditioned, unit-normalized profile on
the admitted E283 corpus. The kind-level train direction still has common
descent across all 63 active train objectives, but it opposes 35 held-out
objectives across 13 exact decision signatures. Only 20/26 held-out objective
signatures have an exact train counterpart; six are absent and one additional
signature has a train-count deficit. Stable grammar-state support is therefore
necessary but insufficient when sampled judged bad-token sets change the FTPO
objective. No training ran. Profile at exact decision-signature train strata
next; do not add token/component cases or increase duration. Full evidence:
[iter-e284-signature-support-profile-20260717.md](iter-e284-signature-support-profile-20260717.md).

E285 attempted the next read-only exact-signature profile with the same frozen
checkpoint and E283 corpus. The legacy direct invocation had no cumulative
experiment deadline and remained incomplete beyond 25 minutes, so it was
operator-stopped without an output artifact. It is **invalid evidence** and no
training or checkpoint mutation occurred. The autoresearch harness now enforces
one configurable cumulative `max_wall_minutes` budget, defaulting to and capped
at three minutes across all compiled stages. Future comparisons must also keep
the same declared training budget; runtime expiry is a stopped run, never a
quality result. Full record:
[iter-e285-exact-signature-profile-aborted-20260717.md](iter-e285-exact-signature-profile-aborted-20260717.md).

E286 tested chunked batched vector-Jacobian products as a generalized
acceleration for the same full-corpus profile. A batch-16 E284 reproduction
remained incomplete at 283.62 seconds and was killed by the five-minute process
envelope without producing a report. The implementation was removed: a faster
unit test is not enough when the real harness misses its operational gate. This
is **invalid evidence**, no training ran, and E285 remains unresolved. The loop
pivots to the already registered matched B3 capacity arms, whose equal
width/depth/token/step recipes fit the bounded experiment policy. Full record:
[iter-e286-batched-signature-profile-rejected-20260717.md](iter-e286-batched-signature-profile-rejected-20260717.md).

## V16 C3 corpus-mined macro tokens (fixture-run 2026-07-17)

Track C3 (Stitch [arXiv:2211.16605](https://arxiv.org/abs/2211.16605) /
LILO [arXiv:2310.19791](https://arxiv.org/abs/2310.19791), **Adapted**: only
the greedy-MDL compression objective is reused — no lambda-calculus
anti-unification, no learned library): recurring fixed-vocabulary token spans
are mined offline from the canonicalized training corpus
(`data/macro_induction.py`, `net_gain = freq*(len-1) - len`) and bound to
reserved `<MACRO_i>` ids (tokenizer v3, 64 rows). Expansion is deterministic
and lossless at decode; the table persists in the tokenizer sidecar so train
and decode cannot disagree. Macros never contain `<SYM_i>`/`<BIND_j>`/
`<STATE_k>` or `NL`, sidestepping the alpha-equivalence hashing pitfall
([arXiv:2401.02948](https://arxiv.org/abs/2401.02948)). New
`macro_substitution` diffusion policy masks whole macro blocks.

| ID | Isolated lever | Status |
| --- | --- | --- |
| E280 | C3 `macro_tokens=true` on the lexer/diffusion base | fixture-run |

### V16 measured results (CPU, fixture-grade, 2026-07-17)

Recipe: `--steps 80 --scratch-control --no-design-md-context --rico-limit 3`,
batch 4, seed 0, lr 3e-4, fixture v1 corpus (108 records). JSON:
[quality-matrix-results-iter-v16-c3-20260717.json](quality-matrix-results-iter-v16-c3-20260717.json);
narrative: [iter-e280-c3-macro-tokens-20260717.md](iter-e280-c3-macro-tokens-20260717.md).

Induction: 16 macros (cap), corpus 4,964 → 3,261 tokens incl. table (−34.3%),
description length −35.5%; matched-recipe training throughput
`seen_target_tokens` 15,417 → 10,118 (−34.4%). The 16-entry table round-trips
through the checkpoint sidecar and evals score expanded output. All honest
gates fail (syntax/meaningful parse 0.0, struct sim 0.05–0.17, train loss
5.61 @80) — consistent with every 80-step fixture row. **Wiring evidence
only**: no matched no-macro control row in this run; whether sequence
compression buys quality is the open frontier-scale matched pair. No gate
weakened; nothing promoted.

## V17 C4 names-disappear matched pair (fixture-run 2026-07-17)

Track C4 ("When Names Disappear" [arXiv:2510.03178](https://arxiv.org/abs/2510.03178)):
does anonymizing binder/state identifiers to `<BIND_j>`/`<STATE_k>` — the
assumption C1–C3 build on — hurt this DSL the way it hurts general code
models? One lever (`symbol_anonymization`); placeholders keep `<SYM_i>` in
both arms. Both arms decode unconstrained (`grammar_constrained=False`,
per-experiment knob) because the NAME gate admits only `<BIND_j>` ids and
would confound the comparison; surface mode + constrained decode / macros /
relative binding fail closed.

| ID | Isolated lever | Status |
| --- | --- | --- |
| E281 | Anonymized-symbol control (unconstrained decode) | fixture-run |
| E282 | Surface binder/state identifiers via byte channel | fixture-run |

### V17 measured results (CPU, fixture-grade, 2026-07-17)

Recipe: `--steps 80 --scratch-control --no-design-md-context --rico-limit 3`,
batch 4, seed 0, lr 3e-4, fixture v1 corpus. JSON:
[quality-matrix-results-iter-v17-c4-20260717.json](quality-matrix-results-iter-v17-c4-20260717.json);
narrative: [iter-e281-e282-c4-names-disappear-20260717.md](iter-e281-e282-c4-names-disappear-20260717.md).

Syntax/meaningful parse 0.0 on both arms (fixture wall); structural
similarity favors the **surface** arm on 5/5 suites (0.23/0.17/0.16/0.18/0.11
vs 0.12/0.09/0.11/0.09/0.03) despite 1.72× longer targets at the same step
budget. **Verdict: open** — the primary metric never leaves zero, so the
threat is neither confirmed nor refuted; the secondary signal is a small
adverse data point for the anonymization defense, to be settled by a
frontier-scale replicated pair. No gate weakened; nothing promoted.

## E292 complete ChoiceTokenizer loss suite (CPU scratch, 2026-07-17)

E292 fixes a measurement-only omission: generic ChoiceTokenizer kinds
(`sym`/`bind`/`state`) now contribute binding masks, and generic structural
kinds contribute structural masks. The matched d64/h2 choice arm ran 107 steps
and 5,022 target tokens. Its frozen suite is complete with weighted NLL 7.2265
(binding 8.0201 over 112 masks; structural 5.6419 over 210 masks). The
checkpoint SHA is byte-identical to E288-E291, proving the accounting fix did
not alter model behavior.

Frozen honest ship evaluation kept grammar constraints, prompt-derived
slot-contract constraints, no DESIGN.md context, and no unconstrained fallback.
Parse is 1.0 on all five small suites, but meaningful rate is 0.0 everywhere;
component recall is 0.04 on held_out and 0.0 elsewhere. AgentV is 0/5 and 15
gates fail. **Verdict:** measurement fixed, model rejected. Binding and
component selection—not syntax/runtime—are the next bounded quality lever.
See [the narrative](iter-e292-choice-loss-suite-completeness-20260717.md) and
[machine-readable results](choice-loss-suite-results-iter-e292-20260717.json).

## E293 choice-native component plan (CPU scratch, 2026-07-17)

The choice arm now supports grammar-role component-plan supervision by replaying
its native pushdown state rather than the surface compiler. A provenance audit
also fixed a summary bug: E292's unset outer context flag was reported as
no-DESIGN even though the factory and checkpoint enabled DESIGN context.

The actual E292-matched DESIGN-context arm reaches adversarial meaningful 0.5
and AgentV 1/5 with plan decode bias off, versus E292's 0.0 / 0/5; bias 1 erases
that gain. In the policy-correct no-DESIGN follow-up (107 steps / 5,022 tokens),
plan loss falls 5.6761→3.2616 and root/bound metrics reach 0.5, but meaningful
rate remains zero.

The frozen honest same-checkpoint ablation confirms the path is active: decode
bias 1 applies 752 times, changes 38 choices, and reduces gate failures 17→13
versus bias off. It improves most fidelity/structure cells, but meaningful rate
remains 0.0 everywhere and AgentV remains 0/5. An earlier DESIGN-context
calibration produced one meaningful adversarial row, but is not a matched
comparison. **Verdict:** harness/provenance repaired; a DESIGN-context training
signal does not transfer to the no-DESIGN policy; no promotion. See
[the narrative](iter-e293-choice-component-plan-20260717.md) and
[machine-readable results](choice-component-plan-results-iter-e293-20260717.json).

## E294 no-DESIGN no-plan control (CPU scratch, 2026-07-17)

E294 supplies the missing control for E293 `r3`: identical 107-step /
5,022-token choice recipe with no DESIGN context and both plan weights zero.
Weighted NLL is 7.4977 versus E293's 7.5550. Its frozen honest board is exactly
identical to E293 with decode bias off, despite 69/73 shared non-head tensors
differing: meaningful 0.0 everywhere, AgentV 0/5, 17 failures.

**Verdict:** plan training alone does not change discrete outputs at this
resolution; enabling its learned legal-candidate bias cuts failures 17→13 but
still produces no meaningful programs. Secondary-ranking evidence only; no
promotion. See [the narrative](iter-e294-no-design-plan-control-20260717.md)
and [machine-readable results](choice-plan-control-results-iter-e294-20260717.json).

## E295 deterministic DESIGN-context dropout (CPU scratch, 2026-07-17)

E295 adds cache-safe record-level `--design-md-dropout` and runs a matched 50%
arm: exactly 240/480 DESIGN contexts are omitted. Complete weighted NLL 7.3785
interpolates between E292 all-DESIGN (7.2265) and E294 no-DESIGN (7.4977).
Frozen prompt-only evaluation produces one meaningful adversarial program
(0.25), AgentV 1/5, and 14 failures versus 0.0 / 0/5 / 15–17 in the controls.
The other four suite scoreboards exactly match E294.

**Verdict:** retain and replicate the generalized lever; the isolated
adversarial success is not broad transfer and the checkpoint is not promotable.
See [the narrative](iter-e295-design-context-dropout-20260717.md) and
[machine-readable results](choice-design-dropout-results-iter-e295-20260717.json).

## E496–E497 current-main provenance audits

E496 syncs and verifies the durable E396 checkpoint SHA, then attempts to load
it on clean current-main-derived revision `bccf2355`. Loading fails before
evaluation because `slot_component_head.{weight,bias}` exists in the checkpoint
but not current code. This falsifies current-main reproducibility of E490.
E490's 5/5 result remains branch-only diagnostic evidence from an unreconciled
decoder stack; it is not a deployable-code champion claim.

E497 validates the repaired provenance envelope using the loadable committed
playground fixture. Exact code SHA and clean-worktree state are persisted.
Complete smoke n=3 finishes in 113.9 seconds: parse/meaningful/fidelity 0.0,
structure 0.2203, type recall 0.1667, reward 0.0, one timeout, and AgentV 0/5.
This is fixture-grade negative evidence, not a ship regression.

Full evidence: [E496 compatibility audit](iter-e496-current-main-e396-honest-smoke-20260718.md),
[E496 JSON](iter-e496-current-main-e396-honest-smoke-20260718.json),
[E497 provenance smoke](iter-e497-current-main-playground-provenance-smoke-20260718.md),
and [E497 JSON](iter-e497-current-main-playground-provenance-smoke-20260718.json).

## E499 bounded strict-corpus SFT

E499 compares `dq_strict_fixture_r4_20260718` with the diverse-root
`remediated_roots` control under an identical frozen SmolLM2 choice-codec
recipe: CPU, seed 0, 230,210 trainable parameters, 1,000 target tokens, and a
three-minute internal wall limit under a 170-second process cap.

The matched strict-r4 arm regresses smoke structure `0.1542→0.0375` and
component recall `0.25→0.0`; meaningful rate, fidelity, and reward remain zero.
An integrated strict build restores broad structural coverage but exposes 76
fragment targets that the document-only choice codec cannot encode. Its
choice-compatible r6 follow-up passes a 67/67 codec preflight and is faster,
but reproduces the same quality regression and carries red synthesis-feedback
warnings. All three final evaluations emit AgentEvals plus AgentV and fail.

**Verdict:** keep the strict fixture repairs and compatibility diagnosis, but
reject both candidate corpora as replacements at this budget. No checkpoint
was synced or promoted. Full evidence:
[narrative](iter-e499-strict-corpus-bounded-sft-20260718.md) and
[JSON](iter-e499-strict-corpus-bounded-sft-20260718.json).

## E500 documentized-expression corpus

E500 adds a general, provenance-preserving projection from language-contract
expression tasks to complete documents and an explicit target-kind selector.
The clean projected corpus has 260 choice-compatible rows, 87 independent root
parents, 72 program families, and 241 structural families, with no synthesis
warnings or feedback recommendations. The canonical snapshot is committed as
`e500_documentized_expression_candidate_r2_20260718`.

Matched frozen-SmolLM2 choice runs at 1k and 5k target tokens do not show a
model-quality gain. All four arms have syntax 1.0 but meaningful rate,
fidelity, component recall, and reward 0.0; structural similarity is 0.0375
and AgentV is 0/1. The candidate's lower 1k loss (`27.6250` versus `30.3844`)
reverses at 5k (`12.6778` versus `10.5529`).

**Verdict:** retain the generalized data projection and clean committed
snapshot, but reject all four bounded checkpoints for promotion or bucket
sync. Every process was externally capped at 170 seconds and every train
summary records `max_wall_minutes=3.0`. Full evidence:
[narrative](iter-e500-documentized-expression-corpus-20260718.md) and
[JSON](iter-e500-documentized-expression-corpus-20260718.json).

## E501 E396 warm-start on E500

E501 adds explicit weight-only initialization for training on a new corpus
without weakening the bit-exact resume data guard. The frozen E396 parent is
compared with three CPU/frozen-HF continuation arms on the committed E500
corpus under an identical honest smoke `n=3` recipe.

The published task-balanced mixture samples generation at 33.9% and regresses
structure `0.2117→0.1458`. Uniform sampling restores 93.4% generation exposure
at 5k tokens and produces component recall `0.1667`, but structure collapses
to `0.0889`; meaningful rate, fidelity, and reward stay zero. The uniform 1k
arm preserves the parent and reaches structure `0.2317`, but moves no semantic
gate. AgentV is 0/1 for every arm.

**Verdict:** keep `--initialize-from`, reject all E501 checkpoints, and treat
the 1k boundary as evidence that longer continuation needs explicit retention
or lower-update safeguards. Every train records the three-minute wall cap and
all checkpoints remain local. Full evidence:
[narrative](iter-e501-e396-e500-warm-start-20260719.md) and
[JSON](iter-e501-e396-e500-warm-start-20260719.json).

## E502 checkpoint-prior retention

E502 first sweeps the E501 1k continuation learning rate from `3e-4` to
`1e-4` and `3e-5`. Both lower-LR arms collapse to structure `0.1133–0.1167`,
showing that optimizer magnitude does not explain the behavioral drift. The
checkpoint loader was retaining tensors/tokenizers while silently using
slot-component lexeme/span priors rebuilt from E500.

Restoring those learned serving priors raises matched 1k structure to `0.3169`
and recall to `0.0833`, versus E501's `0.2317`/`0.0`. A 5k stress arm still
collapses to structure `0.0927` with recall `0.1667`. Meaningful rate,
fidelity, reward, and AgentV remain zero for every E502 arm.

**Verdict:** keep prior-preserving initialization and complete slot-head recipe
telemetry; reject all E502 checkpoints. Prior transfer fixes experiment
attribution and the short continuation, but longer training needs an explicit
weight-retention or replay objective. Every process was capped at 170 seconds
and every train records `max_wall_minutes=3.0`. Full evidence:
[narrative](iter-e502-initialization-prior-retention-20260719.md) and
[JSON](iter-e502-initialization-prior-retention-20260719.json).

## E503 initialized-weight retention

E503 adds an explicit per-step contraction toward the E396 initialization and
records final RMS weight drift. Four matched 5k-token arms use the same CPU,
frozen-HF, E500 uniform-sampling recipe and honest smoke `n=3` scoreboard.

Retention reduces RMS drift monotonically from `0.003123` at 0% to `0.000811`
at 5%. Structure rises from `0.0927` to `0.2029`, but component recall falls
from `0.1667` to zero. The 3% midpoint reaches structure `0.1667` and recall
`0.0833`. Meaningful rate, fidelity, reward, and AgentV remain zero throughout.

**Verdict:** keep the measurable retention control, but reject all E503
checkpoints. Strong anchoring exchanges duplicate-subtree spam for trivial
empty layouts rather than improving semantics. Parent replay is the next
matched lever. Every process was capped at 170 seconds and every train records
`max_wall_minutes=3.0`. Full evidence:
[narrative](iter-e503-initialized-weight-retention-20260719.md) and
[JSON](iter-e503-initialized-weight-retention-20260719.json).

## E504 provenance-preserving parent replay

E504 adds deterministic parent-corpus replay to the canonical training loop,
with namespaced replay IDs, full-state fingerprints for both corpora, and
requested-versus-effective exposure telemetry. Four matched replay fractions
and one adaptive replay-plus-retention follow-up use E396 initialization, the
E500 primary corpus, and honest smoke `n=3`.

Fifty-percent replay reduces RMS drift from `0.003123` to `0.002796` and raises
structure from `0.0927` to `0.2469`, but component recall falls from `0.1667`
to `0.0833`; meaningful rate, fidelity, reward, and AgentV remain zero. Adding
1% retention to that arm cuts drift to `0.001775` but collapses structure to
`0.0634` and recall to zero.

**Verdict:** keep exact replay provenance and exposure telemetry; reject all
five E504 checkpoints. High replay restores hierarchy but not semantics, and
retention compounds the failure. Measure primary-versus-replay objective
conflict before changing synthesis or adding another regularizer. Every process
was capped at 170 seconds and every train records `max_wall_minutes=3.0`. Full
evidence: [narrative](iter-e504-parent-corpus-replay-20260719.md) and
[JSON](iter-e504-parent-corpus-replay-20260719.json).

## E505 source-stratified replay loss attribution

E505 adds bounded primary-versus-replay masked-token loss summaries without
changing the training objective. A matched replication of E504's 50% replay arm
records primary proxy `3.8422→3.3724` and replay proxy `3.4087→2.9217` from the
first to last 20 examples. Both improve; the primary-minus-replay gap widens
3.95% from `0.4335` to `0.4506`.

The matched decode exactly reproduces E504 structure `0.2469`, recall `0.0833`,
and zero meaningful/fidelity/reward. A same-checkpoint decode ablation enabling
slot-contract constraint raises fidelity to `0.1667` and reward to `0.2623`,
but structure falls to `0.2039` and meaningful rate/AgentV remain zero.

**Verdict:** keep source-loss telemetry; reject the E505 checkpoint. Simple
primary-loss divergence is falsified, while scalar loss cannot establish
gradient conflict. Validate constrained slot-contract decode on a larger capped
diagnostic or measure gradient alignment next. Every process was capped at 170
seconds and the train records `max_wall_minutes=3.0`. Full evidence:
[narrative](iter-e505-replay-loss-attribution-20260719.md) and
[JSON](iter-e505-replay-loss-attribution-20260719.json).

## E506 larger constrained slot-contract decode

E506 evaluates the rejected E505 checkpoint on all 13 held-out, OOD, and
adversarial records with constrained slot-contract decode off versus on.
Enabling the constraint raises aggregate meaningful rate `0→0.1538`, fidelity
`0→0.2538`, structure `0.1271→0.1669`, recall `0.1410→0.2654`, reward
`0→0.5454`, and AST node F1 `0.1524→0.2385`. AST edge F1 falls
`0.0192→0`.

**Verdict:** retain constrained slot-contract decode as the leading inference
policy, but do not promote the checkpoint. AgentV remains 0/3 in both arms and
the 96-token diagnostic canvas is below every suite's gold p95. Both processes
were capped at 170 seconds. Full evidence:
[narrative](iter-e506-slot-contract-decode-20260719.md) and
[JSON](iter-e506-slot-contract-decode-20260719.json).

## E507 length-safe OOD contract decode

E507 repeats E506's OOD comparison with a 160-token canvas, above the suite's
gold p95 of 143. Both arms exactly reproduce their 96-token quality metrics.
Contract-on retains meaningful `0.25`, fidelity `0.2583`, structure `0.2281`,
recall `0.3333`, reward `0.692`, and AST node F1 `0.3389`; contract-off remains
zero on semantic metrics.

**Verdict:** the constrained decode gain is not a canvas-truncation artifact.
Keep it as the leading inference policy, but do not promote: AgentV remains
0/1 and generation is still diagnostic. Both processes were capped at 170
seconds. Full evidence:
[narrative](iter-e507-length-safe-ood-contract-decode-20260719.md) and
[JSON](iter-e507-length-safe-ood-contract-decode-20260719.json).

## E508 default-generation OOD replication

E508 raises the length-safe constrained OOD policy from four generation steps
and one attempt to the checkpoint defaults of eight steps and four attempts.
Every quality metric exactly reproduces E507: meaningful `0.25`, fidelity
`0.2583`, structure `0.2281`, recall `0.3333`, reward `0.692`, and AST node F1
`0.3389`.

**Verdict:** grammar-LTR primary decode succeeds on its first path, so these
denoising/retry controls are not the remaining blocker. Keep the constrained
policy, stop tuning these controls, and focus next on semantic component
correctness. AgentV remains 0/1. The process was capped at 170 seconds. Full
evidence: [narrative](iter-e508-default-generation-ood-contract-decode-20260719.md)
and [JSON](iter-e508-default-generation-ood-contract-decode-20260719.json).

## E509 honest slot contract in context

E509 adds the honest request slot contract to model context while retaining
E508's constrained decode, length-safe 160-token canvas, and default generation
settings. Structure rises `0.2281→0.2406` and binding-aware coverage
`0.75→1.0`, but meaningful `0.25`, fidelity `0.2583`, recall `0.3333`, reward
`0.692`, AST node F1 `0.3389`, and AgentV `0/1` are unchanged.

**Verdict:** inventory visibility is not the semantic blocker. Do not promote;
target component selection and placeholder semantic-role mapping next. The
process was capped at 170 seconds. Full evidence:
[narrative](iter-e509-slot-contract-context-20260719.md) and
[JSON](iter-e509-slot-contract-context-20260719.json).

## E510 component-plan decode

E510 activates the E505 checkpoint's trained component-plan head at decode
weight 4 while retaining E509's honest contract, length-safe canvas, and
default generation policy. On all four OOD records, meaningful rises
`0.25→0.50`, fidelity `0.2583→0.6583`, structure `0.2406→0.3446`, recall
`0.3333→0.3958`, reward `0.6920→0.8405`, AST node F1 `0.3389→0.4679`, and AST
edge F1 `0→0.1625`.

**Verdict:** retain weight 4 as the leading diagnostic policy and expand it
across held-out and adversarial suites. Do not promote: strict binding-aware
meaning remains zero and AgentV remains 0/1. The process was capped at 170
seconds. Full evidence: [narrative](iter-e510-component-plan-decode-20260719.md)
and [JSON](iter-e510-component-plan-decode-20260719.json).

## E511 length-safe three-suite component-plan decode

E511 expands component-plan weight 4 to all 13 held-out, OOD, and adversarial
records with a 192-token canvas above every suite's gold p95. Aggregate
meaningful is `0.3846`, fidelity `0.6718`, structure `0.3440`, recall `0.4615`,
reward `0.6272`, AST node F1 `0.4654`, and AST edge F1 `0.1748`. OOD exactly
reproduces E510 quality; held-out and adversarial reach meaningful `0.20` and
`0.50`.

**Verdict:** the component-plan gain generalizes, so retain weight 4 as the
leading diagnostic policy. Do not promote: strict binding-aware meaning remains
zero across all suites and AgentV remains 0/3. Target placeholder semantic-role
supervision and anti-spam behavior next. The process was capped at 170 seconds.
Full evidence: [narrative](iter-e511-three-suite-component-plan-20260719.md)
and [JSON](iter-e511-three-suite-component-plan-20260719.json).

## E512 slot-to-component decode-weight ablation

E512 exposes the existing slot-to-component bias through the canonical eval CLI
and compares weight 8 with E510's weight 4 on the same four OOD records. Spam
prevalence falls `3→1`, but role mismatch stays `4→4`; meaningful regresses
`0.50→0.25`, fidelity `0.6583→0.3417`, structure `0.3446→0.2869`, reward
`0.8405→0.7245`, and AST edge F1 `0.1625→0.10`.

**Verdict:** reject weight 8 and retain weight 4. Improve slot-role supervision
and anti-spam calibration during training rather than increasing decode bias.
Strict binding-aware meaning remains zero and AgentV remains 0/1. The process
was capped at 170 seconds. Full evidence:
[narrative](iter-e512-slot-component-weight-20260719.md) and
[JSON](iter-e512-slot-component-weight-20260719.json).

## E513 durable slot-role supervision continuation

E513 warm-starts the bucket-backed E396 checkpoint on E500 with 50% exact E357
replay, raises slot-component loss `1→4`, adds focal gamma 2, and supplies the
honest slot contract in context. The CPU HF-context run completes 101 steps /
5,000 target tokens in 79.6 seconds under `max_wall_minutes=3`, then uploads and
verifies checkpoint SHA `59253c67…a88a9548` in the OpenUI bucket.

Matched E514 OOD evaluation under E510's weight-4 policy regresses meaningful
`0.50→0.00`, fidelity `0.6583→0.4917`, structure `0.3446→0.2750`, recall
`0.3958→0.2083`, AST node F1 `0.4679→0.3500`, and AST edge F1
`0.1625→0.0625`; strict v2 remains zero and AgentV remains 0/1.

**Verdict:** retain the uploaded checkpoint as diagnostic evidence but reject
promotion and broader evaluation. Full evidence:
[narrative](iter-e513-slot-role-supervision-20260719.md) and
[JSON](iter-e513-slot-role-supervision-20260719.json).

## E515 focal-loss decomposition

E515 is matched to E513 except focal gamma returns `2→0`. It completes 101 CPU
HF-context steps / 5,000 target tokens in 105.8 seconds under
`max_wall_minutes=3`, then uploads and verifies checkpoint SHA
`97f2e426…24721c1b` in the OpenUI bucket.

Matched E516 OOD evaluation recovers meaningful `0.00→0.25`, fidelity
`0.4917→0.6583`, structure `0.2750→0.3213`, recall `0.2083→0.2708`, reward
`0.7695→0.8270`, and AST node F1 `0.3500→0.4292` versus E513. It still trails
E510 on meaningful, structure, recall, reward, AST node F1, and AST edge F1;
strict v2 remains zero and AgentV remains 0/1.

**Verdict:** focal gamma 2 is harmful. Retain focal gamma zero for future
controls, reject slot-component loss 4 for promotion, and target role-labeled
data rather than more loss scaling. Full evidence:
[narrative](iter-e515-focal-loss-decomposition-20260719.md) and
[JSON](iter-e515-focal-loss-decomposition-20260719.json).

## E517 slot-loss context control

E517 is matched to E515 except slot-component loss returns `4→1`; focal gamma
stays zero and training-time honest contract context stays enabled. It completes
101 CPU HF-context steps / 5,000 target tokens in 130.7 seconds under
`max_wall_minutes=3`, then uploads and verifies checkpoint SHA
`2b572a04…e24b60e3`.

Matched E518 OOD evaluation regresses meaningful `0.25→0.00`, fidelity
`0.6583→0.4083`, structure `0.3213→0.2250`, recall `0.2708→0.2083`, reward
`0.8270→0.7445`, and AST node F1 `0.4292→0.2833` versus E515. Strict v2 stays
zero and AgentV stays 0/1.

**Verdict:** reject E517. Slot loss and contract context interact, but neither
context-conditioned arm approaches E510. Stop objective-scale tuning and
target training-time role-label representation. Full evidence:
[narrative](iter-e517-slot-loss-context-control-20260719.md) and
[JSON](iter-e517-slot-loss-context-control-20260719.json).

## E519 honest slot-contract training context

E519 adds canonical `train_model --honest-slot-contract`, preventing
training-time slot context from falling back to gold record placeholders and
recording both authority flags in train summaries (`harness.model_build.train`
`v7`). The matched run completes 101 CPU HF-context steps / 5,000 target tokens
in 103.2 seconds under `max_wall_minutes=3` from clean commit `950007f`, then
uploads and verifies checkpoint SHA `d82155b0…6c91805f`.

E520 exactly matches E518 on every OOD quality metric and decoder counter:
meaningful 0.0, fidelity 0.4083, structure 0.2250, recall 0.2083, reward
0.7445, AST node F1 0.2833, AST edge F1 0.0625, and AgentV 0/1. The authority
change still perturbs 102/106 tensors (max absolute delta `9.67e-05`).

**Verdict:** retain the honest harness fix, reject checkpoint promotion, and
make role labels visible in training prompts rather than restoring privileged
gold inventory. Full evidence:
[narrative](iter-e519-honest-slot-context-20260719.md) and
[JSON](iter-e519-honest-slot-context-20260719.json).

## E521 visible slot-contract data

E521 applies the existing canonical prompt-slot-contract projection to the E500
source recipe. The audit finds full placeholder visibility in only 13/260 E500
rows and 0/209 generation rows, versus 998/998 E357 replay rows. The successful
strict build admits 244 rows; all 244 expose every declared placeholder, mean
quality is 0.9643, and no quality record is rejected.

**Verdict:** publish the immutable E521 snapshot for a matched bounded
continuation. Keep semantic dedup unchanged; its 18 near-duplicate removals
produce one ProgramSpec yield candidate but no warning. E521 is data evidence
only until a checkpoint receives the standard honest suites. Full evidence:
[narrative](iter-e521-visible-slot-contract-data-20260719.md) and
[JSON](iter-e521-visible-slot-contract-data-20260719.json).

## E522 visible-inventory continuation

E522 holds the E519 parent, replay, token budget, objective weights, honest
context authority, and E520 evaluator fixed while replacing E500 with E521.
The clean run completes 99 CPU HF-context steps / 5,059 target tokens in 120.7
seconds and uploads a bucket-verified checkpoint.

E523 raises OOD fidelity `0.4083→0.8667`, recall `0.2083→0.2708`, AST node F1
`0.2833→0.3437`, and AST edge F1 `0.0625→0.1007`. Structure regresses
`0.2250→0.1955`, reward regresses `0.7445→0.2093`, meaningful and strict
meaning remain zero, and AgentV remains 0/1.

**Verdict:** retain visible inventory as positive slot-grounding evidence but
reject E522. The next intervention must restore component hierarchy while
preserving the fidelity gain. Full evidence:
[narrative](iter-e522-visible-slot-continuation-20260719.md) and
[JSON](iter-e522-visible-slot-continuation-20260719.json).

## E524 visible component-contract data

E524 appends exact component type/count inventories to the immutable E521
prompts after all admission gates. The published r4 snapshot preserves all 244
E521 IDs and OpenUI targets, exposes exact component contracts in 244/244 rows,
retains full declared-slot visibility, and has zero quality rejects, warnings,
or synthesis recommendations.

**Verdict:** publish E524 for one matched bounded continuation against E522.
This is conditional-contract data evidence, not an unconditional model or ship
claim. Three membership-changing diagnostics remain local and are not evidence.
Full evidence:
[narrative](iter-e524-visible-component-contract-data-20260719.md) and
[JSON](iter-e524-visible-component-contract-data-20260719.json).

## E525 visible component-contract continuation

E525 holds the E522 parent, replay, token budget, objective weights, authority,
and evaluator fixed while replacing E521 with membership-identical E524. The
clean run completes 99 CPU HF-context steps / 5,059 target tokens in 76.7
seconds and has a bucket-verified checkpoint.

E526 raises component recall `0.2708→0.4167`, but fidelity regresses
`0.8667→0.4667`, structure `0.1955→0.1452`, AST node F1
`0.3437→0.3041`, and AST edge F1 `0.1007→0.0774`. Meaningful and strict
meaning remain zero and AgentV remains 0/1.

**Verdict:** reject E525. Exact component counts teach inventory recall but do
not recover reference hierarchy and trade away slot fidelity. Full evidence:
[narrative](iter-e525-visible-component-continuation-20260719.md) and
[JSON](iter-e525-visible-component-continuation-20260719.json).

## E527 visible component-types data

E527 removes exact counts from E524 while retaining unique component types and
slot inventory. The projection preserves all 244 E521 IDs/targets, passes every
quality check, and emits zero feedback actions.

**Verdict:** publish for one matched continuation testing whether a weaker
conditional contract preserves recall without E525’s fidelity/hierarchy
regression. Data evidence only. Full evidence:
[narrative](iter-e527-visible-component-types-data-20260719.md) and
[JSON](iter-e527-visible-component-types-data-20260719.json).

## E528 visible component-types continuation

E528 holds the E525 parent, replay, token budget, objective weights, authority,
and evaluator fixed while replacing exact component counts with E527's
membership-identical type-only contracts. The clean CPU HF-context run finishes
99 steps / 5,059 target tokens in 146.8 seconds under the three-minute cap and
syncs a verified nine-file checkpoint bundle.

Matched E529 OOD n=4 recovers meaningful rate 0.0→0.25, fidelity
0.4667→0.55, and reward 0.1668→0.5778 versus E525, while component recall falls
0.4167→0.3542, structure 0.1452→0.1136, and AST node F1 0.3041→0.2270.
Strict binding-aware meaning remains 0.0 and AgentV remains 0/1.

**Verdict:** reject E528. The type-only contract is a conditional positive
signal for v1 meaning and reward, but it does not restore hierarchy or pass
strict gates. Target semantic roles and reference-graph construction next;
do not expose gold counts or weaken gates. Full evidence:
[narrative](iter-e528-visible-component-types-continuation-20260719.md) and
[JSON](iter-e528-visible-component-types-continuation-20260719.json).

## E530 visible semantic-role data

E530 adds prompt-visible semantic namespaces and schema-compatible owning-type
candidates derived only from slots and component types already visible in
E521. It exposes no exact counts or gold parent/child graph. An initial
recipe-drift build is retained as invalid evidence because default producers
changed membership. The corrected producer-free projection preserves all 244
IDs, targets, and placeholder lists; all 244 rows gain role contracts, 174 gain
typed candidates, and strict feedback is clean.

**Verdict:** publish only corrected `r2` for one matched bounded continuation.
Data evidence only; no learned-behavior or ship claim. Full evidence:
[narrative](iter-e530-visible-semantic-roles-data-20260719.md) and
[JSON](iter-e530-visible-semantic-roles-data-20260719.json).

## E531 visible semantic-role continuation

E531 holds E528's E396 parent, exact E357 replay, 50% replay, 5k token budget,
objectives, authority, tokenizer, and evaluator fixed while replacing E527
type-only prompts with membership-identical E530 semantic-role prompts. The
clean CPU HF-context train completes 99 steps / 5,059 target tokens in 99.72
seconds under the three-minute cap and persists a verified nine-file bucket
bundle.

Matched E532 OOD n=4 improves structure 0.1136→0.1431 and AST node F1
0.2270→0.2543, but meaningful falls 0.25→0.0, fidelity 0.55→0.4667,
component recall 0.3542→0.2917, reward 0.5778→0.3685, and AST edge F1
0.0801→0.0455. Strict meaning remains 0.0 and AgentV remains 0/1.

**Verdict:** reject E531. Prompt-visible semantic grouping alone does not
construct correct reference graphs. Target an explicit visible-contract
reference-edge lever next without exposing gold topology or weakening gates.
Full evidence:
[narrative](iter-e531-visible-semantic-roles-continuation-20260719.md) and
[JSON](iter-e531-visible-semantic-roles-continuation-20260719.json).

## E533 honest visible-role inference

E533 closes E531's train/inference authority gap without gold leakage. It
normalizes only official component names already present in prompt prose and
the honest visible slot contract, then holds the E531 checkpoint and every
E532 evaluator setting fixed.

Matched OOD n=4 keeps syntax at 1.0 but regresses fidelity 0.4667→0.3833,
structure 0.1431→0.1159, component recall 0.2917→0.2292, AST node F1
0.2543→0.1627, and AST edge F1 0.0455→0.0417. Reward remains 0.3685;
meaningful and strict meaning remain 0.0; AgentV remains 0/1.

**Verdict:** reject E533 and do not retrain this prompt-conditioning lever.
Keep the opt-in harness for honest matched evaluation, then target explicit
grammar/reference construction without hidden gold inputs. Full evidence:
[narrative](iter-e533-visible-role-inference-20260719.md) and
[JSON](iter-e533-visible-role-inference-20260719.json).

## E534 honest visible-role decode bias

E534 tests whether E533's visible semantic-role contract was uninformative or
merely ignored by the model. It holds the E531 checkpoint and matched E533 OOD
n=4 recipe fixed, adding a weight-4 bias only to legal bound-component choices
that are schema-compatible with prompt-mentioned component names and honest
visible slots.

All 18 eligible decisions changed. Meaningful v1 improves 0.0→0.25,
placeholder fidelity 0.3833→1.0, structure 0.1159→0.1959, component recall
0.2292→0.5417, and reward 0.3685→0.7402. AST node/edge F1 remain
0.1627/0.0417, strict meaning remains 0.0, reference graphs remain invalid,
and AgentV remains 0/1.

**Verdict:** retain the opt-in causal inference lever, but do not promote or
claim ship readiness from the diagnostic subset. The contract contains useful
signal; the next lever must construct valid references/topology from visible
authority. Full evidence:
[narrative](iter-e534-visible-role-decode-bias-20260719.md) and
[JSON](iter-e534-visible-role-decode-bias-20260719.json).

## E535 visible generated-reference completeness

E535 holds the E531 checkpoint and complete E534 OOD n=4 recipe fixed, adding
only a weight-4 preference for each unused legal reference to an
already-generated bound element while the root is open. The intervention uses
no gold graph and fails closed outside honest slot-constrained choice decode.

Every aggregate is identical to E534, including meaningful 0.25, structure
0.1959, component recall 0.5417, AST node/edge F1 0.1627/0.0417, strict meaning
0.0, reference-graph exact 0.0, and AgentV 0/1. Telemetry reports zero
applications and zero choice changes. E536 later showed that legal reference
alternatives did exist: all streams used structural mode, so E535's v0.5 `r=`
guard was unreachable.

**Verdict:** reject E535 and do not train it. A future topology lever must act
at completion-path or declaration/reference-plan scope and prove non-zero
structural-mode reachability before consuming a training run. Full evidence:
[narrative](iter-e535-visible-reference-completeness-20260719.md) and
[JSON](iter-e535-visible-reference-completeness-20260719.json).

## E536 choice decision evidence

E536 persists bounded `choice_decision_trace/v1` evidence—the actual generated
choice tokens and legal reference decisions—without gold topology. It holds the
E535 recipe fixed and reproduces every quality aggregate exactly.

The four streams contain 212 choice tokens and 84 decision rows with legal
references; 10 references are selected. All streams use structural mode. This
corrects E535: its zero applications came from a v0.5 `r=` marker guard, not
from absent alternatives.

**Verdict:** accept E536 as a harness improvement with no model-quality claim.
Use the evidence to test structural-mode terminal-root planning next, and
require non-zero reach before training. Full evidence:
[narrative](iter-e536-choice-decision-evidence-20260719.md) and
[JSON](iter-e536-choice-decision-evidence-20260719.json).

## E538 semantic-role plus component-plan composition

E538 adds component-plan decode weight 4 to E536's exact visible semantic-role
policy on the E531 checkpoint. The lever is causally active (95 applications,
four changed component choices), but all four predictions collapse to a single
inline `Stack` root. Meaningful falls `0.25→0`, fidelity `1.0→0.85`, structure
`0.1959→0.1079`, recall `0.5417→0.2708`, and reward `0.7403→0`.

**Verdict:** reject E538 and do not train it. The older plan-head gain does not
transfer to E531 under direct visible-role decoding. Target explicit
declaration/reference structure rather than another component-type bias. Full
evidence: [narrative](iter-e538-role-plan-composition-20260719.md) and
[JSON](iter-e538-role-plan-composition-20260719.json).

## E539 structural reference aggregation

E539 makes the default-off generated-reference hook reachable inside structural
lists and compares weight 4 with a same-commit v10 weight-zero control. Ten
applications change seven choices. Fidelity improves `0.3833→0.4667` and
validity `0.53→0.58`, while meaningful, recall, reward, strict meaning, and
AgentV remain unchanged; structure slips `0.1159→0.1119`.

**Verdict:** retain the fail-closed hook for diagnosis, but reject weight 4 for
training or promotion. Reference aggregation must be learned or explicitly
phased rather than applied to every list. Full evidence:
[narrative](iter-e539-structural-reference-aggregation-20260719.md),
[control JSON](iter-e539-structural-reference-control-20260719.json), and
[intervention JSON](iter-e539-structural-reference-aggregation-20260719.json).

## E540 reference phase telemetry

E540 adds bounded root-vs-nested generated-frame evidence and reference-bias
counterfactual choices, then replays the E539 intervention unchanged. All
quality metrics reproduce exactly. Nine of ten applications and six of seven
changed choices occur in the structural root list; one changed choice occurs in
a nested `Modal` list.

**Verdict:** accept the observability improvement with no quality or readiness
claim. Test root-list-only bias next; do not train or promote weight 4. Full
evidence:
[narrative](iter-e540-reference-phase-telemetry-20260719.md) and
[JSON](iter-e540-reference-phase-telemetry-20260719.json).

## E541 root-only reference completeness

E541 removes the single nested-list intervention identified by E540. Nine
root-list applications still change six choices, but every quality metric
exactly matches the E539 weight-zero control. E539's single nested Modal choice
caused both its placeholder gain and structural regression.

**Verdict:** keep the safer root-only guard default-off, reject weight 4 for
training or promotion, and stop hand-written completeness-bias iteration.
Learn topology/aggregation targets instead. Full evidence:
[narrative](iter-e541-root-reference-only-20260719.md) and
[JSON](iter-e541-root-reference-only-20260719.json).

## E542 learned root-reference arity

E542 trains an isolated dependency-order-aware terminal-root reference-count
head on a 24-step E531 continuation. The local scratch train completes in
52.93 seconds under the three-minute cap; the target covers 188/244 E530
records and auxiliary loss moves from 3.9565 to 3.3124.

The four-record OOD control reaches meaningful-v1 0.50, fidelity 0.5917,
structure 0.3019, recall 0.4167, and reward 0.7950, while strict meaning and
AgentV remain zero. An initial weight-1 replay exposes and fixes impossible
tokenizer-tail mass. After bounding arity by available generated sections, the
head still changes 7/11 applied choices but every metric exactly matches the
weight-zero control.

**Verdict:** retain the learned target, isolated head, semantic bound, and
telemetry; keep decoding default-off, reject weight 1, and do not promote the
scratch checkpoint. Full evidence:
[narrative](iter-e542-learned-root-reference-arity-20260719.md) and
[JSON](iter-e542-learned-root-reference-arity-20260719.json).

## E543 bounded root-reference arity training

E543 applies E542's semantic section bound to the auxiliary training loss. Its
24-step matched continuation completes in 37.17 seconds under the three-minute
cap. All 106 non-root-head tensors are bit-identical to E542. First-half mean
auxiliary loss improves from 3.7329 to 0.8845 and accuracy from 0.0417 to
0.7500; second-half loss improves from 3.5496 to 0.9414 and accuracy from
0.3333 to 0.5833.

The four-record OOD weight-1 replay remains exactly quality-neutral. It makes
the same 7 changes across 11 applications as E542's bounded replay, and every
metric equals E542 control: meaningful-v1 0.50, fidelity 0.5917, structure
0.3019, recall 0.4167, and reward 0.7950. Strict meaning and AgentV remain
zero.

**Verdict:** retain the bounded loss as a calibration fix, keep decoding
default-off, reject the scratch checkpoint, and move to reference-identity
supervision. Full evidence:
[narrative](iter-e543-bounded-root-reference-arity-20260719.md) and
[JSON](iter-e543-bounded-root-reference-arity-20260719.json).

## E544 bounded root-reference identity

E544 supervises the exact generated-section identities referenced by the
terminal root. The bounded target covers 188/244 records, including 42
nontrivial strict subsets. Its 24-step continuation completes in 40.96 seconds
under the three-minute cap. Mean positive recall rises from 0.3056 in steps
1–12 to 0.5729 in steps 13–24, though second-half exact-set accuracy is only
0.0417.

Telemetry rejected additive identity bias because it could alter arity. The
accepted rank-only operator preserves the best existing reference score and
only permutes legal reference identities. In a same-checkpoint, same-commit
OOD `n=4` comparison, weight 1 raises meaningful-v1 0.00→0.25, structure
0.1250→0.1688, recall 0.1458→0.2708, and AST node F1 0.1833→0.2833. All 11
changed identity decisions are reference-to-reference. Strict meaning, AST edge
F1, and AgentV remain zero.

**Verdict:** retain bounded identity training and rank-only decoding
default-off; reject the scratch checkpoint for promotion. Next test
coverage-conditioned calibration, not stronger decode bias. Full evidence:
[narrative](iter-e544-root-reference-identity-20260719.md) and
[JSON](iter-e544-root-reference-identity-20260719.json).

## E545 root-reference negative weighting

E545 compares matched 24-step E544 continuations with root-reference identity
negative-class weights 1 and 4. The treatment improves only sparse late-window
negative accuracy (0.3333→0.3958); exact-set accuracy and positive recall are
unchanged. The two checkpoints produce byte-identical programs on the OOD
`n=4` replay and every metric is identical: syntax 1.0, meaningful-v1 0.0,
fidelity 0.4250, structure 0.1494, recall 0.2083, reward 0.5078, AST node F1
0.2574, strict-v2 0.0, AST edge F1 0.0, and AgentV 0/1. Both additional
continuations regress from E544.

**Decision:** reject weight 4 and both scratch checkpoints for promotion.
Retain the generalized weighted loss, but next increase sampling exposure to
the 42 strict-subset identity records rather than increasing class weight.
Full evidence: [narrative](iter-e545-root-reference-negative-weight-20260719.md)
and [JSON](iter-e545-root-reference-negative-weight-20260719.json).

## E546 strict-subset root-reference sampling

E546 raises exposure to the 42 records whose terminal root references a
nonempty strict subset of generated sections. Multiplier 5 increases observed
negative-target rows from 7 to 22. Against a matched multiplier-1 control, OOD
`n=4` fidelity rises 0.4250→0.6083, structure 0.1494→0.2038, reward
0.5078→0.8120, AST node F1 0.2574→0.2976, and AST edge F1 0→0.0417.
Component recall regresses 0.2083→0.0625; meaningful-v1 and strict-v2 remain
0.0 and AgentV remains 0/1. Identity decoding applies zero times in both arms,
so the difference is attributable to the training distribution.

**Decision:** retain the sampler capability, reject multiplier 5 and both
scratch checkpoints, and test a moderate multiplier. Full evidence:
[narrative](iter-e546-root-reference-coverage-sampling-20260719.md) and
[JSON](iter-e546-root-reference-coverage-sampling-20260719.json).

## E547 moderate strict-subset exposure

Multiplier 2 sees 15 negative-target rows, between control 7 and multiplier 5
22. OOD `n=4` structure reaches 0.2248 and AST node F1 0.3270, the best values
in the 1/2/5 ladder, while component recall matches control at 0.2083.
Fidelity regresses to 0.2583. Root arity and identity each apply six times and
change two choices. Meaningful-v1, strict-v2, AST edge F1, and AgentV remain
zero.

**Decision:** prefer multiplier 2 for subsequent bounded diagnostics but reject
the checkpoint. Next address semantic-role fidelity rather than increasing
exposure. Full evidence:
[narrative](iter-e547-root-reference-coverage2-20260719.md) and
[JSON](iter-e547-root-reference-coverage2-20260719.json).

## E548 semantic-role decode weight 8

E548 holds the E547 checkpoint and OOD `n=4` recipe fixed, changing only
visible semantic-role decode weight from 4 to 8. Predictions and every
headline metric are identical: fidelity 0.2583, structure 0.2248, component
recall 0.2083, AST node F1 0.3270, meaningful-v1 0.0, strict-v2 0.0, and
AgentV 0/1. Both weights apply and change all 28 eligible semantic-role
choices.

**Decision:** reject scalar weight escalation. Weight 4 already determines the
eligible choices; next address learned semantic-role candidate ordering or
supervision. Full evidence:
[narrative](iter-e548-semantic-role-weight8-20260719.md) and
[JSON](iter-e548-semantic-role-weight8-20260719.json).

## E549 learned slot-component ordering off

E549 holds the E547 checkpoint and OOD `n=4` recipe fixed, changing only
learned slot-component decode weight from 4 to 0. Structure improves
0.2248→0.2713, AST node F1 0.3270→0.3833, and AST edge F1 0→0.0625.
Fidelity falls 0.2583→0.2083, component recall collapses 0.2083→0, and reward
collapses 0.5403→0. Meaningful-v1, strict-v2, and AgentV remain zero.

**Decision:** reject disabling learned ordering. It preserves semantic density
but suppresses topology at full weight; test a midpoint learned weight next.
Full evidence:
[narrative](iter-e549-slot-component-ordering0-20260719.md) and
[JSON](iter-e549-slot-component-ordering0-20260719.json).

## E550 learned slot-component midpoint

Learned weight 2 exactly matches weight 4 on predictions, metrics, and all 28
interventions. Weight 0 collapses semantic density; tested positive weights 2
and 4 yield the same ordering.

**Decision:** close scalar tuning and address supervision, calibration, or
candidate ordering directly. Full evidence:
[narrative](iter-e550-slot-component-ordering2-20260719.md) and
[JSON](iter-e550-slot-component-ordering2-20260719.json).

## E551 slot lexeme prior off

Removing the corpus-derived prior improves OOD `n=4` fidelity
0.2583→0.3000 and reward 0.5403→0.5453, but structure falls 0.2248→0.1594,
recall 0.2083→0.1250, and AST node F1 0.3270→0.2389. Meaning and AgentV stay
zero.

**Decision:** reject removal; calibrate or regularize the prior next. Full
evidence: [narrative](iter-e551-slot-lexeme-prior0-20260719.md) and
[JSON](iter-e551-slot-lexeme-prior0-20260719.json).

## E552 half-strength slot lexeme prior

Prior weight 0.5 yields OOD `n=4` fidelity 0.1333, validity 0.2800, structure
0.2181, recall 0.1250, reward 0.3435, and AST node F1 0.3389. It regresses
weight 1 on fidelity, recall, reward, and structure; meaning and AgentV stay
zero.

**Decision:** reject the midpoint and close scalar prior tuning. Full evidence:
[narrative](iter-e552-slot-lexeme-prior05-20260719.md) and
[JSON](iter-e552-slot-lexeme-prior05-20260719.json).

## E553 corpus-local proportional slot priors

Warm starts now rebuild deterministic slot priors from the active corpus, and
zero-cooccurrence token/component pairs receive negative rather than spurious
positive associations. The valid matched R3 run reaches OOD `n=4` fidelity
0.3000 and reward 0.5453, but structure falls to 0.1244, recall to 0.0625, and
AST node F1 to 0.1556. Meaning and AgentV remain zero.

**Decision:** keep the correctness fixes, reject the checkpoint, and move from
prior calibration to corpus/supervision coverage. R1 and R2 are explicitly
excluded as confounded. Full evidence:
[narrative](iter-e553-slot-prior-proportional-smoothing-20260720.md) and
[JSON](iter-e553-slot-prior-proportional-smoothing-20260720.json).

## E554 next-slot context

Adding the next visible slot to slot-owner encoding raises OOD `n=4` structure
0.1244→0.1594, recall 0.0625→0.1250, and AST node F1 0.1556→0.2389 versus
E553, but fidelity falls 0.3000→0.2583 and reward 0.5453→0.5328. Meaning and
AgentV remain zero.

**Decision:** reject the mixed checkpoint. Full evidence:
[narrative](iter-e554-slot-next-context-20260720.md) and
[JSON](iter-e554-slot-next-context-20260720.json).

## E555 slot-pair interaction

Multiplicative slot-pair context keeps E553 fidelity 0.3000 and reward 0.5453
while raising structure to 0.1594, recall to 0.1250, and AST node F1 to 0.2389.
It also matches E554 topology while recovering its fidelity/reward loss.
Meaning and AgentV remain zero.

**Decision:** retain the Pareto lever, reject checkpoint promotion. Full
evidence: [narrative](iter-e555-slot-pair-interaction-20260720.md) and
[JSON](iter-e555-slot-pair-interaction-20260720.json).

## E556 combined slot context

Combining next-slot text and pair interaction holds structure at 0.1594 and
recall at 0.1250 but drops fidelity to 0.2167 and reward to 0.5203. Meaning and
AgentV remain zero. **Decision:** reject the combination, retain E555 alone,
and close the factorial. Evidence:
[narrative](iter-e556-slot-context-combined-20260720.md) and
[JSON](iter-e556-slot-context-combined-20260720.json).

## E557 full slot-owner class balancing

Changing E555 class-balance power 0.5→1.0 produces identical predictions and
metrics: fidelity 0.3000, structure 0.1594, recall 0.1250, reward 0.5453,
meaning 0, AgentV 0/1. **Decision:** reject and change data/sampling coverage.
Evidence: [narrative](iter-e557-slot-balance1-20260720.md) and
[JSON](iter-e557-slot-balance1-20260720.json).

## E558 rare slot-owner record coverage

The canonical sampler now expands records containing owner labels observed at
most 10 times. Fourfold exposure selected 75/244 records and expanded the pool
to 469. OOD `n=4` fidelity improves 0.3000→0.4250, but structure regresses
0.1594→0.0921, reward 0.5453→0.4075, and AST-node F1 0.2389→0.1393;
binding-aware meaning remains 0 and AgentV remains 0/1. **Decision:** retain
the sampler, reject the checkpoint, and test 2× exposure. Evidence:
[narrative](iter-e558-owner-coverage-20260720.md) and
[JSON](iter-e558-owner-coverage-20260720.json).

## E559 twofold rare slot-owner coverage

Reducing the rare-owner multiplier 4×→2× yields OOD `n=4` fidelity 0.4417,
component recall 0.2708, structure 0.1085, AST-node F1 0.2048, and AST-edge F1
0.0648. Fidelity and recall beat E555, but meaning-v1/v2 remain 0, reward falls
to 0.1643, and AgentV remains 0/1. **Decision:** reject the checkpoint and
narrow eligibility to owner labels observed at most four times. Evidence:
[narrative](iter-e559-owner-coverage2-20260720.md) and
[JSON](iter-e559-owner-coverage2-20260720.json).

## E560 narrow rare slot-owner coverage

Narrowing 2× eligibility from ≤10 to ≤4 owner labels selects 9/244 records.
OOD `n=4` structure improves to 0.2181, component recall to 0.2083, and
AST-node F1 to 0.3389. Fidelity falls to 0.2583 and reward is 0.5403;
meaning-v1/v2 and AST-edge F1 remain 0, with AgentV 0/1. **Decision:** retain
as a topology Pareto lever without promotion and test threshold 7. Evidence:
[narrative](iter-e560-owner-threshold4-20260720.md) and
[JSON](iter-e560-owner-threshold4-20260720.json).

## E561 midpoint rare slot-owner coverage

At 2× exposure, threshold 7 selects 42/244 records and dominates E555 on every
non-semantic OOD `n=4` headline: fidelity 0.5750, structure 0.2419, recall
0.1458, reward 0.5753, AST-node F1 0.3125, and AST-edge F1 0.0385. Meaning
v1/v2 remain 0 and AgentV remains 0/1. **Decision:** retain threshold 7, close
the sampling ladder, and use the checkpoint only for semantic decode research.
Evidence: [narrative](iter-e561-owner-threshold7-20260720.md) and
[JSON](iter-e561-owner-threshold7-20260720.json).

## E562 component-plan decode weight 1

Enabling E561's trained component-plan head at decode weight 1 changes five of
136 applications. OOD `n=4` fidelity improves to 0.7417, structure to 0.2732,
and AST-node F1 to 0.3236, but meaning-v1/v2 remain 0, reward falls to 0.3985,
and AgentV remains 0/1. **Decision:** reject as a semantic fix and test weight
0.5. No checkpoint was created. Evidence:
[narrative](iter-e562-component-plan-decode1-20260720.md) and
[JSON](iter-e562-component-plan-decode1-20260720.json).

## E563 component-plan decode weight 0.5

The midpoint weight changes seven of 130 component-plan applications, but OOD
`n=4` fidelity falls to 0.4083, structure to 0.2019, reward to 0.5178, and
AST-node F1 to 0.2500 versus E561. Meaning-v1/v2 remain 0 and AgentV remains
0/1. **Decision:** reject as a semantic fix and close the component-plan
decode-weight ladder. No checkpoint was created. Evidence:
[narrative](iter-e563-component-plan-decode05-20260720.md) and
[JSON](iter-e563-component-plan-decode05-20260720.json).

## E564 semantic-role decode weight 2

Halving E561's semantic-role decode weight from 4 to 2 changes no matched OOD
`n=4` quality aggregate: fidelity remains 0.5750, structure 0.2419, recall
0.1458, reward 0.5753, AST-node F1 0.3125, and AST-edge F1 0.0385.
Meaning-v1/v2 remain 0 and AgentV remains 0/1. **Decision:** retain as
no-effect negative evidence and test weight 0 as the decisive on/off ablation.
No checkpoint was created. Evidence:
[narrative](iter-e564-semantic-role-decode2-20260720.md) and
[JSON](iter-e564-semantic-role-decode2-20260720.json).

## E565 semantic-role decode weight 0

The decisive on/off ablation retains visible role context but disables its
decode bias. Every matched OOD `n=4` aggregate and failure-reason prevalence
remains identical to E561 and E564; meaning-v1/v2 remain 0 and AgentV remains
0/1. **Decision:** close the semantic-role decode-weight ladder as inactive
for E561 and move to a different semantic mechanism. No checkpoint was
created. Evidence:
[narrative](iter-e565-semantic-role-decode0-20260720.md) and
[JSON](iter-e565-semantic-role-decode0-20260720.json).

## E566 slot-component decode weight 2

Halving E561's learned slot-component decode weight from 4 to 2 retains 16
applications and 14 choice changes, but every matched OOD `n=4` quality
aggregate is identical. Meaning-v1/v2 remain 0 and AgentV remains 0/1.
**Decision:** treat weights 2–4 as one saturated selection regime and test
weight 0 as the decisive head on/off ablation. No checkpoint was created.
Evidence: [narrative](iter-e566-slot-component-decode2-20260720.md) and
[JSON](iter-e566-slot-component-decode2-20260720.json).

## E567 slot-component decode weight 0

Disabling E561's learned slot-component head drops matched OOD `n=4` fidelity
to 0.5333, structure to 0.2194, recall to 0.0833, reward to 0.4110, and
AST-node F1 to 0.2292. Meaning-v1/v2 remain 0 and AgentV remains 0/1.
**Decision:** the head helps non-semantic quality but is not the missing
semantic mechanism; retain weight 4 and close this ladder. No checkpoint was
created. Evidence:
[narrative](iter-e567-slot-component-decode0-20260720.md) and
[JSON](iter-e567-slot-component-decode0-20260720.json).

## E568 design-context E561 continuation

A 48-step E561 warm start with threshold-7 twofold owner sampling completes
in 116.24s and writes local SHA `8dcc0804…0283a12b`. The successful recipe
retains design-metadata context, unlike E561, so this is not a duration-only
comparison. OOD `n=4` reward improves to 0.6920, but fidelity falls to
0.2583, structure to 0.1375, AST-node F1 to 0.1833, and AST-edge F1 to 0.
Meaning-v1/v2 remain 0 and AgentV remains 0/1. **Decision:** reject for
promotion, preserve as a local reward Pareto/recipe-drift diagnostic, and
return to no-design-metadata context. Evidence:
[narrative](iter-e568-design-context-continuation-20260720.md) and
[JSON](iter-e568-design-context-continuation-20260720.json).

## E569 matched E561 continuation

The corrected 48-step no-design-context continuation completes in 75.20s and
writes local SHA `8254fcf7…c6535f73`. OOD `n=4` meaningful-v1 rises to 0.25,
recall to 0.3333, reward to 0.6920, and AST-node F1 to 0.3389. Fidelity
regresses to 0.2583, structure to 0.2031, edge F1 to 0, binding-aware
meaning-v2 remains 0, and AgentV remains 0/1. **Decision:** retain as a local
semantic-coverage Pareto for targeted strict-meaning research; do not promote
or sync. Evidence:
[narrative](iter-e569-matched-continuation-20260720.md) and
[JSON](iter-e569-matched-continuation-20260720.json).

## E570 E569 component-plan decode weight 1

Enabling E569's trained component-plan head changes three of 131 choices.
OOD `n=4` fidelity improves to 0.4917, structure to 0.3350, reward to 0.7695,
AST-node F1 to 0.3821, and edge F1 to 0.0455. Recall falls to 0.2083 and
meaningful-v1 to 0; strict meaning-v2 remains 0 and AgentV remains 0/1.
**Decision:** retain as a topology/reward decode Pareto without promotion and
test weight 0.5. No checkpoint was created. Evidence:
[narrative](iter-e570-e569-component-plan1-20260720.md) and
[JSON](iter-e570-e569-component-plan1-20260720.json).

## E571 E569 component-plan decode weight 0.5

At weight 0.5 the trained head changes one of 107 choices and exactly
reproduces E569's OOD `n=4` aggregates: fidelity 0.2583, structure 0.2031,
recall 0.3333, reward 0.6920, AST-node F1 0.3389, edge F1 0, and
meaningful-v1 0.25. Strict meaning-v2 remains 0 and AgentV remains 0/1.
An earlier completed run with E569's non-LTR policy produced the same
aggregates and is retained as a disclosed control. **Decision:** retain 0.5
only as the E569-equivalent coverage setting; the 0.5-to-1.0 response is a
sharp threshold, so close the scalar ladder and return to training or a
targeted strict-semantic mechanism. No checkpoint was created. Evidence:
[narrative](iter-e571-e569-component-plan05-20260720.md) and
[JSON](iter-e571-e569-component-plan05-20260720.json).

## E572 E569 fidelity-loss weight 2

A matched 48-step E569 warm start with fidelity loss 0.5→2.0 completes in
84.26s. OOD `n=4` fidelity improves to 0.6500, validity to 0.7900, and reward
to 0.8170, but structure falls to 0.1438, recall to 0.1458, AST-node F1 to
0.1833, and meaningful-v1 to 0. Strict meaning-v2 remains 0 and AgentV
remains 0/1. **Decision:** reject as a semantic candidate; retain the local
no-sync checkpoint only as a non-semantic fidelity/reward Pareto. Evidence:
[narrative](iter-e572-e569-fidelity2-20260720.md) and
[JSON](iter-e572-e569-fidelity2-20260720.json).

## E573 E569 fidelity-loss weight 1

A matched 48-step midpoint completes in 109.72s. OOD `n=4` retains
meaningful-v1 0.25 while improving fidelity to 0.4750, validity to 0.6850,
and reward to 0.7570. Structure falls to 0.1813, recall to 0.2708, and
AST-node F1 to 0.2833; strict meaning-v2 remains 0 and AgentV remains 0/1.
**Decision:** retain locally as a fidelity/coverage Pareto, but close the
fidelity scalar ladder because strict semantics do not improve. Evidence:
[narrative](iter-e573-e569-fidelity1-20260720.md) and
[JSON](iter-e573-e569-fidelity1-20260720.json).

## E574 E573 slot-component loss weight 2

Doubling slot-component loss on the matched 48-step E573 recipe completes in
76.23s. Despite 31 slot-head applications and choice changes, every OOD `n=4`
aggregate exactly matches E573, including meaningful-v1 0.25, fidelity 0.4750,
recall 0.2708, reward 0.7570, strict meaning-v2 0, and AgentV 0/1.
**Decision:** reject as a quality improvement and close slot-component loss
scalar tuning. The checkpoint remains local/no-sync negative evidence.
Evidence: [narrative](iter-e574-e569-slotloss2-20260720.md) and
[JSON](iter-e574-e569-slotloss2-20260720.json).

## E575 prompt-derived SemanticPlanV1 soft scorer

E575 connects visible prompt component mentions to a predicted partial
`SemanticPlanV1` and uses the SLM-146 compiler bridge to soft-score legal root
and bound component choices. Candidate legality is unchanged. On a clean,
matched E569 OOD `n=4` 0/1/2 ladder, weight 1 changes 3/52 scored choices and
improves meaningful-v1 0→0.25, fidelity 0.3417→0.4250, validity
0.6050→0.6550, structure 0.1250→0.1688, component recall 0.1458→0.2708,
reward 0.7095→0.7345, and AST-node F1 0.1833→0.2833. Weight 2 changes five
choices but regresses fidelity, validity, and reward below the control.

**Decision:** retain the generalized scorer and weight 1 as a default-off local
decode Pareto; close the scalar ladder. Strict meaning-v2 and AgentV remain
zero, so do not promote or sync. Target predicted topology/binding factors
next. Evidence:
[narrative](iter-e575-prompt-semantic-plan-soft-20260720.md) and
[JSON](iter-e575-prompt-semantic-plan-soft-20260720.json).

## SLM-147 / SPV1-04 — X22 leakage-safe prototype retrieval

First wiring campaign for retrieve-and-edit initialization of the Kapur tree-edit
diffusion baseline (X22). A train-only local index of hard-valid canonical AST
prototypes is built from the SLM-144 fixture plan corpus; retrieval strategies
are compared by seed validity and seed-to-gold token distance under matched
CPU-only (no model) conditions.

Fixture results (`n_train=51`, `n_val=13`, seed 0):

| Arm | Mean seed-to-gold ratio | Mean component coverage | Adapted prototypes |
| --- | --- | --- | --- |
| A minimal seed | 0.621 | 0.453 | 0/13 |
| B random prototype | 0.717 | 0.594 | 8/13 |
| C prompt similarity | 0.656 | 0.618 | 8/13 |
| D AST sketch | 0.744 | 0.565 | 7/13 |
| E SemanticPlan factor | 0.851 | 0.387 | 13/13 |
| F hybrid | 0.851 | 0.387 | 13/13 |
| G oracle nearest (diagnostic) | 0.860 | 0.367 | 13/13 |
| H retrieval-as-context control | 0.621 | 0.453 | 0/13 |

All arms passed leakage checks (no shared split group, prompt, or structure
between query and retrieved prototype). No GPU X22 checkpoint was run; these
numbers are wiring-only distance diagnostics, not ship-gate claims.

**Decision:** the harness and local index are wired. The semantic-plan and
hybrid arms reach the highest seed-to-gold ratio on this small fixture, but the
corpus is too homogeneous to discriminate retrieval strategies. A frontier run
with a trained X22 checkpoint, a larger labeled semantic corpus, and AgentV
evaluation is required before any quality claim.

Evidence:
[narrative](iter-slm147-x22-retrieval-20260720.md) and
[JSON](iter-slm147-x22-retrieval-20260720.json).

## SLM-148 / SPV1-05 — plan-conditioned X22 × conflict-slice campaign

First wiring campaign for the staged seed × recovery factorial combining
SLM-144/145 plan predictors, SLM-146 plan-to-seed compiler, SLM-147
leakage-safe retrieval, and SLM-113 conflict-slice repair. The fixture runs on
the same CPU-only corpus as SLM-147 (`n_train=51`, `n_val=13`, seed 0) and
compares seed validity, seed-to-gold distance, component coverage, and
synthetic conflict-recovery metrics across all preregistered arms.

Screening results (mean over seeds 0–2):

| Arm | Mean seed-to-gold ratio | Mean component coverage |
| --- | --- | --- |
| S0 minimal seed | 0.621 | 0.453 |
| S1 frequency prior | 0.322 | 0.522 |
| S2 learned archetype + role set | 0.621 | 0.453 |
| S3 learned full plan | 0.411 | 0.528 |
| S4 gold binding factor (diagnostic) | 0.621 | 0.453 |
| S5 gold plan oracle (diagnostic) | 0.411 | 1.000 |
| S6 retrieved prototype | 0.851 | 0.387 |
| S7 plan-reranked retrieval | 0.851 | 0.387 |

All promotable seed arms survived screening (every seed was hard-valid). The
gold/oracle seed arms are explicitly non-promotable. Recovery arms were crossed
with the survivors; conflict-slice and full-remask policies run deterministically
on synthetic analyzer slices, with oracle-recovery arms marked diagnostic.
No GPU X22 checkpoint or AgentV evaluation was run; these numbers are
wiring-only distance/recovery diagnostics, not ship-gate claims.

**Decision:** the staged factorial harness is wired. Retrieved/plan-reranked
prototypes reach the highest seed-to-gold ratio on this fixture, but the corpus
is too small to separate plan prediction from retrieval. A frontier run with a
trained X22 checkpoint, SLM-111 beam/depth points, a real conflict analyzer, and
AgentV evaluation is required before any quality claim.

Evidence:
[narrative](iter-slm148-x22-conflict-campaign-20260720.md) and
[JSON](iter-slm148-x22-conflict-campaign-20260720.json).

## E576 prompt-plan binding soft scorer

E576 soft-ranks legal unused terminal-root references whose already-generated
component family matches the prompt-derived predicted `SemanticPlanV1`. On a
clean, matched E569 OOD `n=4` 0/1/2 ladder, both treatment weights find five
compatible binding decisions but cause zero immediate choice changes and zero
deltas in every quality metric. Meaning-v1 remains 0.25, strict meaning-v2 and
AST-edge F1 remain zero, reward remains 0.7345, and AgentV remains 0/1.

**Decision:** reject weights 1 and 2 as quality interventions and keep the
legality-preserving factor default-off for diagnostics only. Do not promote or
sync. Target plan-aware root construction or explicit topology cardinality,
not a stronger binding scalar. Evidence:
[narrative](iter-e576-prompt-plan-binding-soft-20260720.md) and
[JSON](iter-e576-prompt-plan-binding-soft-20260720.json).

## E577 plan-binding score composition

E577 applies predicted plan-binding evidence after learned root-reference
identity ranking so identity cannot overwrite the plan factor. On a clean,
matched E569 OOD `n=4` pair, binding weight 1 changes 2/4 applicable latent
reference decisions versus 0 in E576, validating the composition diagnosis.
All four final programs remain identical to control, however, and every
quality aggregate is unchanged: meaning-v1 0.25, strict meaning-v2 0,
AST-edge F1 0, reward 0.7345, and AgentV 0/1.

**Decision:** retain the corrected factor order, keep binding scoring
default-off, and do not promote or sync. The remaining bottleneck is root
construction/topology cardinality, not reference-ranker ordering. Evidence:
[narrative](iter-e577-plan-binding-order-20260720.md) and
[JSON](iter-e577-plan-binding-order-20260720.json).

## E578 plan-aware Stack root construction

E578 waits for predicted component-family coverage, then soft-scores legal
`Stack` construction, plan-compatible nested child references, and legal
termination. On a clean matched E569 OOD `n=4` 0/1/2 ladder, the root factor
activates 21 times and changes one latent choice at weight 1; weight 2 changes
two. All final programs and quality metrics remain identical to control:
meaning-v1 0.25, strict meaning-v2 0, AST-edge F1 0, reward 0.7345, and AgentV
0/1.

**Decision:** reject weights 1 and 2 as quality interventions, keep the root
factor default-off, and do not promote or sync. The next topology experiment
must test a compiler-validated planned-root seed/state rather than another
score increase. Evidence:
[narrative](iter-e578-plan-root-container-20260720.md) and
[JSON](iter-e578-plan-root-container-20260720.json).

## Verifier-guided repair (mixed status)

Verifier-guided repair status from
[verifier-guided-repair.md](verifier-guided-repair.md). **E62 is wired**;
E60–E61 and E63–E65 remain proposed.
The inner-loop prerequisites (deterministic denoising-NLL suites, token
budgets, full-state resume, source-family manifests, decode trajectory
store) plus the P1–P3 staged plan (mixture search, scaling ladders,
self-distillation, trajectory RL) are in
[promotion-pipeline.md](promotion-pipeline.md).
**E50–E55 are taken by shipped V6** (CoRe / T2M / slot trust / champion) and
**E70–E75 by V7 speculative denoising**; do not reuse those IDs.

| ID | Approach | Primary lever | Status |
| --- | --- | --- | --- |
| E60 | Differential validation | Dual lang-core + Lark parse; quarantine disagreement | proposed |
| E61 | Failure-cone remask | Remask first hard error + structural dependents | proposed |
| E62 | Minimal hard negatives | `data/corrupt` verified invalid→clean repair taxonomy; wiring only, no quality result yet | wired |
| E63 | Gate calibration | ECE / selective accuracy / abstention on `FastPathGate` | proposed |
| E64 | Trajectory-aligned RL | MDPO/d1-style on intermediate MaskGIT states | proposed |
| E65 | Schema generalization | Held-out schemas / rename / `toy-layout` transfer | proposed |

## LDI (local decision interventions) index

LDI0-01 recommits the local-decision architecture and its 34-work source inventory
([`local-decision-interventions.md`](local-decision-interventions.md),
[`local-decision-sources.json`](../../src/slm_training/resources/autoresearch/local-decision-sources.json)).
The runnable local-decision experiments already live in the **V10** section above
(E248 control and E249 measured; E250–E254 unrun / fail-closed). LDI reuses that
campaign, the existing preference harness, and the append-only decode traces, and
introduces no second trainer or orchestration stack.

No LDI matrix rows are claimed here and no E-IDs are reserved: future LDI
experiments draw globally unique E-IDs from the existing allocation process and are
registered as ordinary E rows when they run.

| Namespace | Scope | Status |
| --- | --- | --- |
| LDI0 | Evidence contract and bounded diagnostics — source inventory, separation-of-concerns invariants, E249–E284 falsification chain | inventory registered; no matrix rows claimed |

## Verified-solver matrix (VSS4-02, R0–R6)

The `verified-solver` matrix set adds matched rows and fail-closed hard gates for
verified scope solving, driven by the same runner
(`scripts/run_quality_matrix.py --matrix-set verified-solver`). It measures
correctness authority separately from search efficiency and output quality: a row
can be faster or more accurate on semantic metrics and still fail if it produces
one false certified prune, removes one unknown candidate, or returns an
unverified solved output. Full design, metric groups, and the row table live in
[verified-scope-solver-benchmark.md](verified-scope-solver-benchmark.md); results
are mirrored to
[verified-scope-solver-matrix-results.json](verified-scope-solver-matrix-results.json).

| Row | Method | Control | Single variable | Fixture |
| --- | --- | --- | --- | --- |
| R0 | Current matched control (solver off) | — | baseline | ran |
| R1 | Exact deterministic solver | R0 | exact_closure=on | ran (closed benchmark) |
| R2 | Exact solver + model ranking | R1 | ranker=model | blocked (checkpoint) |
| R3 | Capsule-aware topology solver | R2 | capsule_topology=on | blocked (family C) |
| R4 | Capsule solver + cost-to-go energy | R3 | ranker=energy | blocked (checkpoint) |
| R5 | Deterministic late realization | R3 | late_realization=deterministic | blocked (family E) |
| R6 | AR late realization | R5 | realizer=ar | blocked (checkpoint) |

Hard gates (evaluated before quality gains, fail-closed):
`false_unsupported_count`, `unknown_preservation_violations`,
`certificate_replay_failures`, `solved_without_final_verifier`,
`certified_unsat_with_incomplete_proof`, `candidate_set_parity_failures`,
`surface.semantic_ir_mutation_violations`,
`structured_or_observable_slots_routed_to_ar` — each must equal 0. Every existing
ship gate in this document is retained unchanged; the verified-solver rows never
weaken grammar/schema/dataflow/behavior/adversarial/OOD requirements. Fixture
wiring landed 2026-07-18 (R0/R1 ran on CPU, hard gates PASS); every frontier row
is fully specified but **not run until VSS4-03**. No model or ship claim.

## V18 shared recursive denoiser (SLM-138)

Replace the stacked ``DenoiserTower`` with a shared-recursive transition that
recurses a small set of ``TransformerBlock(cross_attn=True)`` instances.  The
tower keeps the same public contract as ``DenoiserTower`` so the rest of the
codebase (masking, decode, checkpoints) needs no changes.  V18 is a
**wiring-only** slice: it validates the module, factory routing, deep-supervision
plumbing, and checkpoint migration on tiny synthetic records.  Matched-block
evaluation arms and GPU training are deferred.

| ID | Isolated lever | Purpose | Run id |
| --- | --- | --- | --- |
| E300 | Stacked control (byte-identical baseline) | Preserve existing ship recipe | `qx_e300_stacked_control` |
| E301 | Shared recursive R=2, L=2 transition | Test that recurrence trains and round-trips | `qx_e301_recursive_r2_l2` |
| E302 | Shared recursive R=4, L=1 transition | Stress very small transition, many recurrences | `qx_e302_recursive_r4_l1` |
| E303 | Shared recursive + deep supervision | Per-recursion CE weighted depth loss | `qx_e303_recursive_deep_sup` |
| E304 | Warm-start stacked → recursive | ``migrate_to_shared_recursive_denoiser`` smoke | `qx_e304_recursive_migrate` |

```bash
# Fixture wiring (CPU, no GPU training)
python -m scripts.run_slm138_recursive_denoiser_fixture --mode fixture

# Planned matrix dispatch (requires GPU + durable checkpoints for ship claims)
python -m scripts.run_quality_matrix --matrix v18 --only E300,E301,E303 \
  --steps 400 --device cpu --context-backend scratch
```

Primary metric: same honest `--ship-gates` as V4+.  Fixture output:
`outputs/runs/slm138-recursive-denoiser-20260720/` with mirrored design artifacts
`docs/design/iter-slm138-recursive-denoiser-20260720.json` and `.md`.

## V19 stochastic recursive width (SLM-139) — closed

SLM-139 gates on a positive shared-recursive verdict from SLM-138.  SLM-138
landed as a wiring-only fixture with no GPU matched-block evaluation, so the
activation gate returned `no_supported_probabilistic_regime`.  No stochastic
production code was added.  The closeout report is at
`outputs/runs/slm139-stochastic-recursive-width-20260720/` with mirrored design
artifacts `docs/design/iter-slm139-stochastic-recursive-width-20260720.json`
and `.md`.

| ID | Isolated lever | Purpose | Status |
| --- | --- | --- | --- |
| E305 | High-level learned stochastic latent | GRAM-style width vs depth | blocked by SLM-138 gate |
| E306 | Low-level trained stochastic state | Noise-locus ablation | blocked by SLM-138 gate |

## V20 plan-predictor factor heads (SLM-145) — closed

SLM-145 proposed adding learned topology, cardinality, and live-symbol pointer
heads to the SPV1 plan predictor.  Its authorization gate required SPV0-02
(SLM-142) to show, through factor-wise oracle substitution on real or fixture
completions, that each factor carries a material downstream semantic ceiling.
SLM-142 landed extraction/canonicalization/oracle/seed wiring but did not run
the factor-wise experiments, so the gate returned
`blocked_pending_spv0_02_ceiling_evidence`.  No head was implemented.

| ID | Isolated lever | Purpose | Status |
| --- | --- | --- | --- |
| E307 | Learned topology head | Predict plan topology factor | blocked by SLM-142 gate |
| E308 | Learned cardinality head | Predict role-slot cardinality | blocked by SLM-142 gate |
| E309 | Learned live-symbol pointer head | Predict binding pointers | blocked by SLM-142 gate |

The closeout report is at
`outputs/runs/slm145-plan-predictor-factors-20260720/` with mirrored design
artifacts `docs/design/iter-slm145-plan-predictor-factors-20260720.json` and
`.md`.

## EFS4-04 causal diagnosis and architecture disposition (SLM-140)

This is the campaign-level synthesis issue.  It does not introduce a new trainable
lever; it consumes every committed EFS result manifest under `docs/design/` and
emits a preregistered causal diagnosis plus explicit architecture dispositions.

| Hypothesis | Issue | Status | Result refs |
| --- | --- | --- | --- |
| efs0-01-checkpoint-provenance | SLM-103 | MISSING | no committed manifest |
| efs0-02-decode-invariance | SLM-104 | NOT_RUN_BY_GATE | `iter-efs-decode-invariance-20260718.json` |
| efs0-03-meaningful-v2 | SLM-105 | NOT_RUN_BY_GATE | `iter-efs0-03-meaningful-v2-frontier-audit-20260717.json` |
| efs0-04-judge-independence | SLM-106 | MISSING | no committed manifest |
| efs0-05-rejected-lever-readjudication | SLM-107 | NOT_RUN_BY_GATE | `iter-efs0-05-rejected-lever-readjudication-20260719.json` |
| efs1-01-external-ceiling | SLM-108 | MISSING | no committed manifest |
| efs1-02-exposure-ladder | SLM-109 | MISSING | no committed manifest |
| efs1-03-empty-length-bias | SLM-110 | NOT_RUN_BY_GATE | `iter-efs1-03-empty-length-bias-20260719.json` |
| efs2-01-x22-scaling | SLM-111 | NOT_RUN_BY_GATE | `iter-efs2-01-tree-edit-scaling-20260719.json` |
| efs2-02-trigger-telemetry | SLM-112 | NOT_RUN_BY_GATE | `iter-efs2-02-trigger-telemetry-20260719.json` |
| efs2-03-conflict-slice-repair | SLM-113 | NOT_RUN_BY_GATE | `iter-efs2-03-conflict-slice-repair-20260719.json` |
| efs2-04-verifier-cascade | SLM-115 | NOT_RUN_BY_GATE | `iter-efs2-04-verifier-cascade-20260719.json` |
| efs3-01-solver-state-supervision | SLM-118 | NOT_RUN_BY_GATE | `iter-efs3-01-solver-state-supervision-20260719.json` |
| efs3-02-corruption-curriculum | SLM-120 | MISSING | no committed manifest |
| efs3-03-b3-capacity-v2 | SLM-124 | NOT_RUN_BY_GATE | `iter-efs-b3-capacity-v2-20260719.json` |
| efs3-04-candidate-selector | SLM-127 | MISSING | no committed manifest |
| efs3-05-canonical-ast-dedup | SLM-130 | MISSING | no committed manifest |
| efs3-06-ast-sketch-retrieval | SLM-133 | NOT_RUN_BY_GATE | `iter-efs3-06-ast-sketch-retrieval-factorial-20260719.json` |
| efs4-01-trailed-assumptions | SLM-135 | NOT_RUN_BY_GATE | `iter-slm135-trailed-assumptions-20260720.json` |
| efs4-02-shared-recursive-denoiser | SLM-138 | MISSING | no committed manifest |
| efs4-03-stochastic-recursive-state | SLM-139 | MISSING | no committed manifest |

Primary causal diagnosis: **insufficient_valid_evidence**.  Core measurement
branches (checkpoint provenance, decoder invariance, judge independence) are not
all resolved, so no training/data/search/architecture conclusion can be asserted.
No architecture item is promoted; safety-infrastructure items
(compiler-owned exact closure, reversible trailing, verifier cascade) are
recorded as `ADOPT_AS_SAFETY_ONLY` without a quality claim.

```bash
# Regenerate the synthesis (no training, no model download)
python -m scripts.synthesize_efs_campaign \
  --manifest docs/design/evidence-first-semantic-slm-campaign-v1.json \
  --docs-design docs/design \
  --out-json docs/design/iter-efs4-04-causal-synthesis-20260720.json \
  --out-md docs/design/iter-efs4-04-causal-synthesis-20260720.md \
  --graph-output docs/design/iter-efs4-04-causal-synthesis-graph
```

Durable artifacts:
- `docs/design/evidence-first-semantic-slm-campaign-v1.json` (preregistered manifest)
- `docs/design/iter-efs4-04-causal-synthesis-20260720.json`
- `docs/design/iter-efs4-04-causal-synthesis-20260720.md`
- `docs/design/iter-efs4-04-causal-synthesis-graph.{mmd,dot}`

## E579 verifier-gated planned root closure

E579 soft-follows a complete, compiler-decoded and verifier-valid
`Stack([&plan-compatible...], "column")` closure after honest predicted-plan
component coverage. On a clean matched E569 OOD `n=4` weight ladder, weights
1 and 2 remain quality-null, while weight 4 improves structural similarity
from 0.1688 to 0.3013, component recall from 0.2708 to 0.3958, reward from
0.7345 to 0.7480, AST-node F1 from 0.2833 to 0.3976, and AST-edge F1 from
0 to 0.20. Syntax remains 1.0. Strict meaning-v2 remains 0 and AgentV fails
0/1, so the result is structural-only, default-off, and non-promotable. No
checkpoint was created or synced. Evidence:
[narrative](iter-e579-verified-plan-root-20260720.md) and
[JSON](iter-e579-verified-plan-root-20260720.json).

## E580 honest prompt-plan cardinality

E580 preserves repeated authored component mentions and requires their family
counts before the verifier-gated root closure activates. The OOD auth plan is
now correctly `Button, Input, Input`. On a clean matched E569 OOD `n=4`
weight-0/4 pair, weight 4 applies three times but changes no choices; every
quality metric matches control, strict meaning-v2 remains 0, and AgentV fails
0/1. This removes E579's AST-edge gain because that gain closed auth topology
after only one of two requested Inputs. Retain the cardinality correction for
honesty, keep the scorer default-off, and next test count-aware component
generation. No checkpoint was created or synced. Evidence:
[narrative](iter-e580-plan-cardinality-20260720.md) and
[JSON](iter-e580-plan-cardinality-20260720.json).

## E581 count-aware predicted components

E581 scores only still-missing predicted component-family instances before the
honest cardinality-gated root closure. On a clean matched E569 OOD `n=4`
component-weight 0/1/2/4 ladder, weight 4 improves meaning-v1 0→0.25,
fidelity 0.3417→0.4250, validity 0.6050→0.6550, structure 0.1250→0.3231,
recall 0.1458→0.4583, reward 0.7095→0.7480, AST-node F1 0.1833→0.4532,
and AST-edge F1 0→0.20. Strict meaning-v2 remains 0 and AgentV fails 0/1.
Auth's missing serialized `v1` shows that repeated latent Inputs still collapse
without distinct slot assignments. Retain default-off for structural
diagnosis; do not promote or sync. Evidence:
[narrative](iter-e581-count-aware-components-20260720.md) and
[JSON](iter-e581-count-aware-components-20260720.json).

## E582 distinct repeated-slot instances

E582 softly closes a repeated predicted component after its first visible slot
so later authored roles can receive distinct instances. On a clean matched E569
OOD `n=4` weight-0/4 pair, weight 4 improves meaning-v1 0→0.25, fidelity
0.3417→0.4250, validity 0.6050→0.6550, structure 0.1250→0.3119, recall
0.1458→0.4583, reward 0.7095→0.7510, AST-node F1 0.1833→0.4264, and AST-edge
F1 0→0.1667. Auth reaches AST-node F1 0.75, AST-edge F1 0.6667, and an exact
reference graph, but its email slot is assigned to TextContent rather than the
still-required second Input. Strict meaning-v2 remains 0 and AgentV fails 0/1.
Retain default-off for partial structural diagnosis; do not promote or sync.
Evidence: [narrative](iter-e582-distinct-slot-instances-20260720.md) and
[JSON](iter-e582-distinct-slot-instances-20260720.json).

## E583 prompt-local slot-family scoring

E583 derives honest slot-to-family candidates from local authored phrases in
addition to exact schema property names. On a clean matched E569 OOD `n=4`
role-weight 0/4 pair with E582's plan and root settings fixed, weight 4 is
quality-null on meaning, structure, recall, and AST metrics, while fidelity
regresses 0.5083→0.4250, validity 0.7050→0.6550, and reward 0.7760→0.7510.
Auth is byte-identical and still assigns email to TextContent; the only changed
program is modal, where the treatment loses the body placeholder. Strict
meaning-v2 remains 0 and AgentV fails 0/1. Record the honest association, reject
role weight 4 for this recipe, and next isolate learned-versus-visible role
score composition. No checkpoint was created or synced. Evidence:
[narrative](iter-e583-prompt-local-slot-family-20260720.md) and
[JSON](iter-e583-prompt-local-slot-family-20260720.json).

## E584 visible-role-gated slot head

E584 removes the auxiliary learned slot-head bonus from visible-role-mismatched
families without changing base scores or legal candidates. Auth reaches perfect
AST node/edge/tree similarity and exact reference topology, but modal collapses
to an empty Stack. On clean matched E569 OOD `n=4`, reward regresses
0.7760→0.5743, fidelity 0.5083→0.3417, validity 0.7050→0.5050, and recall
0.4583→0.3750. Strict meaning-v2 remains 0 and AgentV fails 0/1. Reject
unconditional gating, keep default-off, and next test confidence-aware
arbitration. No checkpoint was created or synced. Evidence:
[narrative](iter-e584-role-gated-slot-head-20260720.md) and
[JSON](iter-e584-role-gated-slot-head-20260720.json).

## E585 remaining-role coverage abstention

E585 requires every remaining visible slot to have a role-family candidate
before auxiliary learned-head gating activates. On a clean E569 OOD `n=4`
role-weight 0/4 pair, the treatment exactly reproduces E584: perfect auth
topology, modal `Stack([])`, reward 0.5743 versus control 0.7760, and strict
meaning-v2 0 with AgentV 0/1. The check activates after an uncovered role can
already be consumed. Reject it, keep default-off, and next bind confidence to
the original visible contract. No checkpoint was created or synced. Evidence:
[narrative](iter-e585-remaining-role-coverage-20260720.md) and
[JSON](iter-e585-remaining-role-coverage-20260720.json).

## E586 original-contract role coverage

E586 fixes confidence to complete role-family coverage of the original visible
contract. On a clean E569 OOD `n=4` role-weight 0/4 pair, auth retains perfect
AST topology and the incomplete modal no longer collapses to `Stack([])`.
Structure improves 0.3119→0.3819, AST-node F1 0.4264→0.4889, and AST-edge F1
0.1667→0.2500 with recall unchanged. Fidelity regresses 0.5083→0.4250,
validity 0.7050→0.6550, and reward 0.7760→0.7510; strict meaning-v2 remains 0
and AgentV fails 0/1. Retain the stable confidence boundary, keep the role
weight default-off, and next prevent visible placeholders from filling
enum-like schema properties. No checkpoint was created or synced. Evidence:
[narrative](iter-e586-original-role-coverage-20260720.md) and
[JSON](iter-e586-original-role-coverage-20260720.json).

## E587 schema-value role bias

E587 penalizes visible slot pointers only in enum-valued active schema
arguments, without changing legal candidates. On a clean E569 OOD `n=4`
weight 0/1/4 ladder layered on E586, weight 1 improves fidelity
0.4250→0.4667, validity 0.6550→0.6800, and reward 0.7510→0.7635, while
structure falls 0.3819→0.3469, recall 0.4583→0.3958, and AST-node F1
0.4889→0.4056. It corrects Stack direction but leaves nested Input/Button enum
spam. Weight 4 collapses auth topology and reward to 0.6920. Strict meaning-v2
remains 0 and AgentV fails 0/1 throughout. Keep the lever default-off, do not
promote or sync, and next separate required content consumption from optional
schema values. Evidence:
[narrative](iter-e587-schema-value-role-bias-20260720.md) and
[JSON](iter-e587-schema-value-role-bias-20260720.json).

## E588 plan-root closure strength

E588 raises only the existing plan-root closure score on E587's aggressive
schema-value recipe. On a clean E569 OOD `n=4` root-weight 4/8/12 ladder,
weight 8 improves fidelity 0.2583→0.4250, validity 0.5550→0.6550, structure
0.2156→0.4069, recall 0.3333→0.4583, reward 0.6920→0.7585, AST-node F1
0.3389→0.4889, and AST-edge F1 0→0.25. Auth recovers perfect topology and
closes before enum properties. Weight 12 is byte- and metric-identical,
confirming a plateau. Use 8 as the next diagnostic baseline, but do not
promote or sync: strict meaning-v2 remains 0 and AgentV fails 0/1. Evidence:
[narrative](iter-e588-root-closure-strength-20260720.md) and
[JSON](iter-e588-root-closure-strength-20260720.json).

## E589 optional opaque-argument slot penalty

E589 adds a default-off, legality-preserving penalty for visible slot pointers
in optional unconstrained (`{}`) component arguments. On E588's OOD `n=4`
baseline, weights 4 and 8 are identical: fidelity, validity, recall, reward,
meaning-v1, and strict v2 are flat, while structure regresses
0.4069→0.3319, AST-node F1 0.4889→0.4611, and AST-edge F1 0.25→0.2143.
The Button action diverts into a legal nested list rather than closing. Keep
the lever default-off; do not promote or sync. Next directly score legal
closure instead of suppressing one expression class. Evidence:
[narrative](iter-e589-opaque-slot-penalty-20260720.md) and
[JSON](iter-e589-opaque-slot-penalty-20260720.json).

## E590 optional opaque-argument close score

E590 directly scores legal closure at optional unconstrained (`{}`) arguments.
On E588's OOD `n=4` baseline, weights 0/2/4/8 leave every aggregate metric
unchanged. However, 4 and 8 are byte-identical and remove the erroneous second
Button action argument, while 0 and 2 retain it; this avoids E589's nested-list
diversion. Treat 4 as a behavioral scratch threshold only. Keep the lever
default-off and do not promote or sync: strict meaning-v2 is 0 and AgentV is
0/1. Evidence: [narrative](iter-e590-opaque-close-score-20260720.md) and
[JSON](iter-e590-opaque-close-score-20260720.json).

## E591 content-property owner slot score

E591's visible property-owner slot score improves OOD `n=4` fidelity
0.4250→0.5917, validity 0.6550→0.7550, reward 0.7585→0.8085,
AST-node F1 0.4889→0.5198, and AST-edge F1 0.25→0.325 at weights 2 and
4, with structure nearly flat. Use 2 as the next scratch baseline; do not
promote or sync because strict v2 is 0 and AgentV is 0/1. Evidence:
[narrative](iter-e591-property-role-slot-20260720.md) and
[JSON](iter-e591-property-role-slot-20260720.json).

## E592 typed component-array item legality

E592 preserves item schemas in the canonical choice array frame, making raw
placeholder items illegal for component arrays. Against the matched E591 OOD
`n=4` control, meaningful-v1 doubles 0.25→0.50, structure improves
0.4044→0.4169, recall 0.4583→0.5417, reward 0.8085→0.8115, and
AST-edge F1 0.3250→0.3429, with fidelity and validity unchanged. Keep the
invariant as the next scratch baseline; do not promote or sync because strict
v2 remains 0 and AgentV is 0/1. Evidence:
[narrative](iter-e592-array-item-schema-20260720.md) and
[JSON](iter-e592-array-item-schema-20260720.json).

## E593 optional enum-argument close score

E593 directly scores legal closure at optional enum-valued arguments. On the
matched E592 OOD `n=4` baseline, weight 2 removes the Modal size leak but
lowers fidelity 0.5917→0.5417, validity 0.7550→0.6250, and reward
0.8115→0.6447 after dashboard collapses to an empty Card. Weight 4 removes all
reported enum-role mismatches but still regresses structure, recall, reward,
and AST-node F1. Keep the generalized lever default-off and retain E592 as the
baseline; strict v2 remains 0 and AgentV is 0/1. No checkpoint was created or
synced. Evidence:
[narrative](iter-e593-enum-close-score-20260720.md) and
[JSON](iter-e593-enum-close-score-20260720.json).

## E594 inline semantic-plan family score

E594 scores still-missing prompt families only at inline component positions.
On the matched E592 OOD `n=4` baseline, weights 2 and 4 leave every aggregate
metric unchanged. Weight 2 is prediction-identical; weight 4 does not fix the
Modal child family and introduces raw auth placeholders at the root. Keep the
lever default-off and retain E592 as baseline. Strict v2 remains 0, AgentV is
0/1, and no checkpoint was created or synced. Evidence:
[narrative](iter-e594-inline-plan-family-20260720.md) and
[JSON](iter-e594-inline-plan-family-20260720.json).

## E595 action-semantic plan inference

E595 adds Button to predicted partial plans from authored action semantics.
Against E592 on OOD `n=4`, structure improves 0.4169→0.4694, recall
0.5417→0.6250, AST-node F1 0.5198→0.5532, and AST-edge F1
0.3429→0.3875, while reward is flat. The Button is semantically misbound to
Modal body and overgeneralizes to dashboard action prose. Keep this as mixed
diagnostic evidence only; strict v2 remains 0, AgentV is 0/1, and no
checkpoint was created or synced. Evidence:
[narrative](iter-e595-action-plan-inference-20260720.md) and
[JSON](iter-e595-action-plan-inference-20260720.json).

## E596 visible slot-role aliases

E596 maps visible body/action role aliases to compatible schema properties.
The matched E595 OOD `n=4` treatment is prediction- and metric-identical:
structure 0.4694, recall 0.6250, reward 0.8115, strict v2 0, and AgentV 0/1.
The remaining-slot order commits Button to body before property-role scoring.
Do not promote or sync; next repair component-to-slot assignment order.
Evidence: [narrative](iter-e596-slot-role-alias-20260720.md) and
[JSON](iter-e596-slot-role-alias-20260720.json).

## E597 schema-derived semantic-role candidates

E597 first raises the existing role weight from 4 to 8 and 12, then enables a
default-off public-schema candidate set at weights 4 and 8. All four matched
OOD `n=4` arms retain E596's aggregate metrics: structure 0.4694, recall
0.6250, reward 0.8115, strict v2 0, and AgentV 0/1. Schema candidates replace
the erroneous dashboard Button with TextContent but do not repair the modal
Button-to-body binding. Keep the switch default-off and do not promote or
sync; next repair explicit component-to-slot assignment.
Evidence: [narrative](iter-e597-schema-role-candidates-20260720.md) and
[JSON](iter-e597-schema-role-candidates-20260720.json).

## E598 schema owner-slot threshold

E598 tests the existing schema property-owner slot score at weights 4, 6, and
8 on E597's schema-candidate treatment. Only weight 8 repairs the target modal
binding, changing Button from body to confirm and reducing semantic-role
mismatches 3→2. All headline OOD `n=4` metrics remain flat: structure 0.4694,
recall 0.6250, reward 0.8115, strict v2 0, and AgentV 0/1. Treat weight 8 as a
scratch threshold only; do not promote or sync.
Evidence: [narrative](iter-e598-owner-slot-threshold-20260720.md) and
[JSON](iter-e598-owner-slot-threshold-20260720.json).
