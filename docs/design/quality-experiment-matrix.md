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

## V9 lattice-guided recursive compiler search (measured E240-E245)

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
| E246 | Full stack width 4 | Quality, validity, abstention, regret, latency | stopped by continuation rule |
| E247 | Full stack width 8 | Width scaling benefit versus verifier/call cost | stopped by continuation rule |

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

## V10 exact-state local preference (proposed, unrun)

The full 25-paper audit, source manifest, objective definition, and honesty boundary
are in [`local-decision-interventions.md`](local-decision-interventions.md). V10
reuses the existing preference harness and append-only decode traces. It does not
introduce an adapter/SAE trainer and does not claim that a local loss produces a
local parameter update.

All rows require one immutable `DecisionEventV1` JSONL, the same parent checkpoint,
split, steps, learning rate, and seed. E252-E254 fail closed unless the training
split contains at least one same-state-verified multi-good or multi-bad event.

| ID | Isolated lever | Required diagnostics | Status |
| --- | --- | --- | --- |
| E248 | Unchanged parent control | Standard five-suite scoreboard | proposed/unrun |
| E249 | Exact-event CE plus margin | Event win/margin and per-kind recurrence | proposed/unrun |
| E250 | Bad-token unlikelihood | Bad probability mass and held-out recurrence | proposed/unrun |
| E251 | Single-pair clipped FTPO | Active weight, chosen/margin win, drift | proposed/unrun |
| E252 | Verifier-backed set FTPO | Set coverage, evidence source, held-out recurrence | proposed/unrun |
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
every checkpoint created. No V10 row was executed by the implementation change.

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
