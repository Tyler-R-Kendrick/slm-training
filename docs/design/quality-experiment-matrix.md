# Quality experiment matrix — TwoTower OpenUI

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
python -m scripts.diagnose_eval --train-dir outputs/train_data/v1 \
  --test-dir outputs/test_data/v1

# V2 matrix (default)
python -m scripts.run_quality_matrix --matrix v2 --steps 800 --device cpu

# Isolated levers
python -m scripts.run_quality_matrix --only E11,E12,E13 --steps 800

# Champion combo + decode sweep (E17 needs E15 checkpoint)
python -m scripts.run_quality_matrix --only E15,E17 --steps 1200 --gen-steps 16

# V3 focused ship path (length-safe → decode match → template → champion)
python -m scripts.run_quality_matrix --matrix v3 --only E18,E19a,E19b,E20,E29 \
  --steps 400 --device cpu --context-backend scratch
```

## Success criteria (honest gates)

Same policy as `docs/design/adversarial-review.md`. Matrix may evaluate `rico_held` with `--rico-limit` for CPU time; full 1500 is the ship claim.

## Commands

```bash
# Build curriculum corpus (once) — train excludes test fixture structures automatically
python -m scripts.build_train_data --source all --version v1_curriculum \
  --synthesizer quality --curriculum

python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/train_data/v1/manifest.json

# Migrate legacy v1 tokenizer checkpoints (optional; fresh train preferred)
python -m scripts.migrate_checkpoint \
  --checkpoint outputs/runs/legacy/checkpoints/last.pt \
  --train-records outputs/train_data/v1/records.jsonl \
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
  --train-records outputs/train_data/v1/records.jsonl \
  --out-dir outputs/runs/grpo --steps 50 --group-size 4

# Cycle telemetry (train + generate spans)
python -m scripts.bench_telemetry --train-steps 8 --gen-prompts 8
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
| **E9 accel combo** | 0.0* | **0.75** | 0.02 | *smoke poisoned by curriculum-C; **adv fidelity 0.875** |

\*E9 emits `:adv.*` placeholders on smoke prompts — curriculum overfit. Still the first strong fidelity signal.

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

- Default matrix is now `v3` (`--matrix v3`).
- `scripts/diagnose_eval.py` reports `length_budget` and exits 2 when p95 exceeds LTR budget.
- Defaults: `grammar_ltr_max_tokens=192`, stages `(64,128,192,256)`.
- Research: MDLM schedule + remasking tagged **Adapted** in [research-lineage.md](research-lineage.md).

## V3 measured results (CPU, scratch, fixture suites)

See [quality-matrix-results.json](quality-matrix-results.json).

| ID | Smoke parse | Smoke fid | Ship gates | Notes |
| --- | --- | --- | --- | --- |
| E18 | 0.0 | 0.0 | fail | length-safe alone underfit at 80 steps |
| **E20** | **1.0** | **1.0** | **pass** | template fill; held_out parse 0.6 / fid 1.0 |
| **E29** | **0.67** | **0.67** | **pass** | champion stack at 40 steps |

First honest `--ship-gates` clears on the fixture scoreboard. Production claim still needs full `rico_held` (1500) + HF context.

## V4 matrix (critic-guided revision — candidate)

Candidate work only — research background in
[research-correction-critics.md](research-correction-critics.md);
**Adjacent** tags in [research-lineage.md](research-lineage.md). IDs start at
**E30** to avoid colliding with implemented V3 (E18–E29). These levers attack
**semantic remasking beyond confidence** (critique / trust heads / visible
corruption), after V3’s confidence remask (`remask_ratio`, E22) and template-fill
champion. Prefer running on an E29 (or stronger) checkpoint.

| ID | Approach | Primary lever | Expected gate delta | Run id |
| --- | --- | --- | --- | --- |
| E30 | Suffix-rollback LTR | ReMDM-style revisable window \(W\) behind LTR frontier; remask on grammar / entropy triggers; re-denoise (inference-only) | ↑ held_out / adversarial parse under LTR-primary | `qx_e30_suffix_rollback` |
| E31 | BackPlay-lite trust head | Freeze denoiser; train unwired [`FastPathGate`](../../src/slm_training/grammar_fastpath/gate.py) on model’s own token errors; drive remask with gate scores | ↑ fidelity + smarter remask than raw confidence | `qx_e31_trust_gate` |
| E32 | Corruption-aware train | Extend [`_mask_targets`](../../src/slm_training/models/twotower.py): small fraction of visible tokens → wrong ids (uniform / model-sampled); CE to recover gold (RemeDi/GIDD-lite) | ↑ fidelity; enables revise-visible | `qx_e32_visible_corrupt` |
| E33 | Combined remask policy | Budgeted remask \(P_i \propto\) grammar hard-error + gate score + entropy (extends V3 `remask_ratio` / `filter_ids_by_stream`) | ↑ held_out / adv parse; ↓ over-remask | `qx_e33_remask_policy` |
| E34 | Latent falsification MoE | Deferred: shared head + top‑2 OpenUI mechanism experts + parallel latent streams; gated on E30–E33 residual failures | Research; semantic failures DFA misses | `qx_e34_latent_critics` *(not scheduled)* |

Implementation order = table order (risk ascending). **E30** touches decode only
(`twotower.py`, `parallel_decode.py`). **E31** needs frozen-backbone error
mining. **E32** changes the train corruption graph. **E33** composes E30+E31
signals with existing V3 remask. **E34** waits until cheaper remask policies
saturate. Structural remask targets from V5 (`remask_span=statement`, `<SYM>`
ids) make E30–E34 materially easier to ground on semantic units.

```text
# FUTURE — no `--matrix v4` runner yet (do not copy-paste as a command).
# Intended once E30+ is implemented:
#   python -m scripts.run_quality_matrix --matrix v4 --only E30 --device cpu \
#     --seed-checkpoint outputs/runs/qx_e29_champion/checkpoints/last.pt
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
  --steps 80 --device cpu --context-backend scratch --no-design-md-context
```

## V5 measured results (CPU, scratch, 80 steps, fixture suites)

See [quality-matrix-results.json](quality-matrix-results.json) (`matrix_set: v5`).
Tokenizer diagnostic on `fixtures/train_seeds.jsonl`: compositional mean 72.6
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

## X matrix (grammar-native diffusion ablations)

Staged ablations **X0–X8** isolate grammar-first decode levers on top of the
corrected honest baseline. Each experiment runs **3 seeds** (0/1/2) with
**successive halving** on `smoke` → `held_out` → `adversarial` before full
multi-suite eval on survivors.

| ID | Approach | Primary lever | Model | Run id |
| --- | --- | --- | --- | --- |
| X0 | Corrected baseline | twotower + honest DESIGN.md eval | twotower | `gx_x0_baseline` |
| X1 | Data/contract | slot contract in context + inventory decode | twotower | `gx_x1_contract` |
| X2 | Production codec | grammar_diffusion over production+slot heads | grammar_diffusion | `gx_x2_codec` |
| X3 | Block objective | grammar_diffusion block noise schedule | grammar_diffusion | `gx_x3_block_obj` |
| X4 | Confidence schedule | `parallel_unmask=confidence` + calib loss | grammar_diffusion | `gx_x4_confidence` |
| X5 | Extendability decode | ExtendabilityChecker in constrained posterior | grammar_diffusion | `gx_x5_extend` |
| X6 | Grammar curriculum | soft A/B/C mix (anti-leak) | twotower | `gx_x6_curriculum` |
| X7 | Champion combo | X1–X6 stacked + capacity | grammar_diffusion | `gx_x7_champion` |
| X8 | Process optimization | X7 + pref/RL (skip RL when reward variance=0) | grammar_diffusion | `gx_x8_process` |

`grammar_diffusion` is the harness plug-in for
`GrammarDiffusionModel` (production codec + block diffusion). See
`src/slm_training/models/grammar_diffusion.py` and
`src/slm_training/harnesses/model_build/factory.py`.

### Commands

```bash
# Three-seed honest baseline reproduction (X0)
python -m scripts.reproduce_baseline \
  --device cpu --context-backend scratch --steps 80

# Full X matrix with successive halving (default 3 seeds)
python -m scripts.run_grammar_matrix \
  --device cpu --context-backend scratch --steps 80

# Isolated levers
python -m scripts.run_grammar_matrix --only X0,X2,X7 --steps 200

# Disable halving (run every experiment×seed to completion)
python -m scripts.run_grammar_matrix --no-halving --only X0,X1,X2 --steps 80

# Champion + process stage
python -m scripts.run_grammar_matrix --only X7,X8 --steps 400 --gen-steps 16
```

Artifacts: `outputs/runs/grammar_matrix_summary.json`,
`docs/design/grammar-matrix-results.json`,
`outputs/runs/baseline_reproduction_summary.json`.
