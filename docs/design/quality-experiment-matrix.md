# Quality experiment matrix — TwoTower OpenUI

> Agentic workers: implement levers below, then run `python -m scripts.run_quality_matrix`.

**Goal:** Clear honest `--ship-gates` by attacking fidelity=0 / parse≈0 with every approach from the quality brief.

**Architecture:** Each row is an isolatable lever (plus a stacked `combo` run). All runs use scratch context on CPU by default; HF is optional when cached.

**Tech stack:** TwoTower, OpenUI grammar, ship_gates, preference composite reward.
**Research map:** [research-lineage.md](research-lineage.md) (MaskGIT, constrained diffusion, DPO/GRPO surrogates).

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
metrics are achievable (gold-as-prediction = 1.0). Next levers: 2000+ steps (E16),
HF context, and slot contract on all eval suites.
