# Model card — OpenUI TwoTower / grammar-diffusion

Canonical card for checkpoints produced by this repo. Agents **must update
this file whenever a new checkpoint is created or promoted** (full train,
remote train, bootstrap demo, or matrix champion intended for reuse), then
mirror a short summary into [`README.md`](../README.md) → “Model card (summary)”.

Storage: durable full-run weights live in
[`hf://buckets/TKendrick/OpenUI`](https://huggingface.co/buckets/TKendrick/OpenUI)
(`checkpoints/<run_id>/`). Local/git fixture demo:
`src/slm_training/resources/checkpoints/playground_demo/`.

Related: [checkpoint-bucket.md](design/checkpoint-bucket.md),
[adversarial-review.md](design/adversarial-review.md),
[quality-experiment-matrix.md](design/quality-experiment-matrix.md).

---

## Current checkpoint roster

| Role | Run id | Kind | Location | Status |
| --- | --- | --- | --- | --- |
| Playground demo | `playground_demo` | Fixture wiring | `src/slm_training/resources/checkpoints/playground_demo/last.pt` (git) | Demo only — **not** a ship claim |
| Restructure CPU verify | `restructure_cpu_scratch_v0` | Fixture scratch train | `outputs/runs/restructure_cpu_scratch_v0/checkpoints/last.pt` (local) | Train OK; smoke parse **0.0** @ 80 steps — **not** a ship claim ([results](design/restructure-cpu-train-results.json)) |
| Local DirectML verify | `local_directml_adreno_20260714` | Local GPU scratch train | `outputs/runs/local_directml_adreno_20260714/checkpoints/last.pt` (local) | Adreno DirectML train/checkpoint OK @ 5 steps; not evaluated — **not** a ship claim ([results](design/local-directml-train-results.json)) |
| Overnight retrain | `overnight_retrain_200` | CPU scratch train | `/tmp/slm-training-overnight/outputs/runs/overnight_retrain_200/checkpoints/last.pt` (local) | 200 steps; all honest suites parse 0.0 — **not promotable or ship** |
| Overnight retrain extended | `overnight_retrain_1000` | CPU scratch train | `/tmp/slm-training-overnight/outputs/runs/overnight_retrain_1000/checkpoints/last.pt` (local) | 1,000 steps; smoke parse 0.0 at steps 200/400/600/800/1000 — **not promotable or ship** |
| E120 singleton diagnostic | `e120_unsandboxed` | CPU scratch decoder diagnostic | `outputs/runs/iter-e120-unsandboxed-20260715/e120_unsandboxed/checkpoints/last.pt` (local) | 8 steps; guarded singleton/root/arity path verified; `rico_held n=1` parse 0.0 — **not promotable or ship** |
| E121 judged-corpus E53 iteration | `qx_e53_honest_v5_champion` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e121d-e53-judged-20260715/qx_e53_honest_v5_champion/checkpoints/last.pt` (local) | 405 judge-approved records; 8 train + 30 trust-gate steps; bounded smoke parse 0.0 with decode timeout — **not promotable or ship** |
| E123 judged-corpus 32-step iteration | `e123_judged_32step_b` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e123b-judged-20260715/e123_judged_32step_b/checkpoints/last.pt` (local) | 405 judge-approved records; loss 10.97; smoke parse 0.0 with unconstrained fallback and canvas cap — **not promotable or ship** ([results](design/iter-e123-judged-corpus-32step-20260715.md)) |
| E127 schema/slot-contract iteration | `e127_judged_schema_slots` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e127-schema-slots-20260715/e127_judged_schema_slots/checkpoints/last.pt` (local) | 405 judged records; loss 10.71; placeholder validity 0.55 / normalized fidelity 0.25, parse 0.0 — **not promotable or ship** ([results](design/iter-e127-schema-slots-20260715.md)) |
| E128 schema/slot 64-step iteration | `e128_judged_schema_slots_64` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e128-schema-slots-20260715/e128_judged_schema_slots_64/checkpoints/last.pt` (local) | 405 judged records; loss 15.03; higher LTR/fidelity weights regressed placeholder signals; parse 0.0 — **not promotable or ship** ([results](design/iter-e128-schema-slots-64step-20260715.md)) |
| E129 schema/slot 64-step low-weight control | `e129_judged_schema_slots_64_lowweights` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e129-schema-slots-20260715/e129_judged_schema_slots_64_lowweights/checkpoints/last.pt` (local) | 405 judged records; loss 9.89; placeholder/parse 0.0; longer training did not reproduce E127 — **not promotable or ship** ([results](design/iter-e129-schema-slots-64low-20260715.md)) |
| E130 schema/slot seed-1 control | `e130_judged_schema_slots_seed1` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e130-schema-slots-20260715/e130_judged_schema_slots_seed1/checkpoints/last.pt` (local) | 405 judged records; seed 1 loss 15.28; parse and placeholder signals 0.0 — **not promotable or ship** ([results](design/iter-e130-schema-slots-seed1-20260715.md)) |
| E132 generation-focused mixture | `e132_generation_focus` | CPU scratch judged-corpus mixture iteration | `outputs/runs/iter-e132-generation-focus-20260715/e132_generation_focus/checkpoints/last.pt` (local) | 405 judged records; three-prompt smoke parse/placeholder 0.0; task reweighting rejected — **not promotable or ship** ([results](design/iter-e132-generation-focus-20260715.md)) |
| E133 no-fused-LTR path | `e133_no_fuse_ltr` | CPU scratch judged-corpus training-path iteration | `outputs/runs/iter-e133-no-fuse-ltr-20260715/e133_no_fuse_ltr/checkpoints/last.pt` (local) | 405 judged records; three-prompt parse/structure 0.0 with one timeout; fused LTR retained — **not promotable or ship** ([results](design/iter-e133-no-fuse-ltr-20260715.md)) |
| E135 HF context control | `e135_hf_context_control` | CPU HF-context control | `outputs/runs/iter-e135-hf-context-20260715/e135_hf_context_control/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, 8 steps; 3-prompt parse 0.0 but structural similarity 0.2422 / placeholder validity 0.3167; **not promotable or ship** ([results](design/iter-e135-hf-context-control-20260715.md)) |
| E136 HF context 32-step control | `e136_hf_context_32` | CPU HF-context control | `outputs/runs/iter-e136-hf-context-20260715/e136_hf_context_32/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, 32 steps; parse/placeholder 0.0 and structural similarity 0.0825; **not promotable or ship** ([results](design/iter-e136-hf-context-32step-20260715.md)) |
| E137 HF context 16-step midpoint | `e137_hf_context_16` | CPU HF-context control | `outputs/runs/iter-e137-hf-context-20260715/e137_hf_context_16/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, 16 steps; placeholder validity 0.40 / structural similarity 0.2142, parse 0.0; **not promotable or ship** ([results](design/iter-e137-hf-context-16step-20260715.md)) |
| E138 HF context seed-1 8-step control | `e138_hf_context_seed1_8` | CPU HF-context seed variance control | `outputs/runs/iter-e138-hf-seed1-20260715/e138_hf_context_seed1_8/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, seed 1, 8 steps; placeholder validity 0.0 / structural similarity 0.1683, parse 0.0; **not promotable or ship** ([results](design/iter-e138-hf-seed1-8step-20260715.md)) |
| E139 HF context seed-2 8-step control | `e139_hf_context_seed2_8` | CPU HF-context seed variance control | `outputs/runs/iter-e139-hf-seed2-20260715/e139_hf_context_seed2_8/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, seed 2, 8 steps; placeholder validity/structure/parse 0.0 with two timeouts; **not promotable or ship** ([results](design/iter-e139-hf-seed2-8step-20260715.md)) |
| E173 schema-context 32-step control | `e173-schema-context-32step` | CPU HF-context semantic control | `outputs/runs/e173-schema-context-32step/checkpoints/last.pt` (local) | Schema/slot context enabled, loss 11.0876; bounded probe syntax 1.0 but meaningful parse 0.0; **not promotable or ship** ([results](design/iter-e173-schema-context-20260716.md)) |
| E174 unfrozen-context 8-step control | `e174-unfrozen-context-8step` | CPU HF-context semantic control | `outputs/runs/e174-unfrozen-context-8step/checkpoints/last.pt` (local) | Unfrozen context, loss 39.4253; bounded probe syntax 0.0 and parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e174-unfrozen-context-20260716.md)) |
| E175 retrieval 8-step control | `e175-retrieval-8step` | CPU HF-context retrieval control | `outputs/runs/e175-retrieval-8step/checkpoints/last.pt` (local) | Retrieval k=4, loss 27.9708; bounded syntax/parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e175-retrieval-20260716.md)) |
| E176 broad-corpus 8-step control | `e176-broad-corpus-8step` | CPU HF-context corpus control | `outputs/runs/e176-broad-corpus-8step/checkpoints/last.pt` (local) | 1,417-record corpus, loss 34.0464; bounded syntax/parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e176-broad-corpus-20260716.md)) |
| E177 semantic-judge 32-step control | `e177-semantic-judge-32step` | CPU HF-context data-quality control | `outputs/runs/e177-semantic-judge-32step/checkpoints/last.pt` (local) | 496-record published judge-gated corpus, loss 12.2220; E180 bounded decode reaches syntax 1.0 but meaningful parse 0.0 / component recall 0.25; **not promotable or ship** ([results](design/iter-e177-e180-semantic-compiler-20260716.md)) |
| E181 balanced-mixture control | `e181-semantic-balanced-32step` | CPU HF-context mixture control | `outputs/runs/e181-semantic-balanced-32step/checkpoints/last.pt` (local) | Loss 5.5118; bounded syntax 1.0 but meaningful parse 0.0 / component recall 0.25; **not promotable or ship** ([results](design/iter-e181-e194-compiler-alignment-20260716.md)) |
| E184 component-aligned diagnostic | `e184-compiler-aligned-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e184-compiler-aligned-32step/checkpoints/last.pt` (local) | Component-state alignment recovers `Stack` root, but E194 meaningful parse 0.0 / structure 0.3600; **not promotable or ship** ([results](design/iter-e181-e194-compiler-alignment-20260716.md)) |
| E191 all-branch aligned diagnostic | `e191-full-compiler-aligned-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e191-full-compiler-aligned-32step/checkpoints/last.pt` (local) | Random all-branch alignment regresses root selection; E192 meaningful parse 0.0; rejected, **not promotable or ship** ([results](design/iter-e181-e194-compiler-alignment-20260716.md)) |
| E195 stratified-alignment invalid control | `e195-stratified-compiler-aligned-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e195-stratified-compiler-aligned-32step/checkpoints/last.pt` (local) | Mixture was silently unset, so recipe is not comparable; retained as invalid evidence, **not promotable or ship** ([results](design/iter-e195-e199-stratified-alignment-20260716.md)) |
| E196 matched stratified alignment | `e196-stratified-compiler-aligned-matched-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e196-stratified-compiler-aligned-matched-32step/checkpoints/last.pt` (local) | E199 syntax 1.0 with zero compiler fallbacks, but meaningful parse/component recall 0.0; **not promotable or ship** ([results](design/iter-e195-e199-stratified-alignment-20260716.md)) |
| E201 generated-role alignment | `e201-role-stratified-compiler-aligned-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e201-role-stratified-compiler-aligned-32step/checkpoints/last.pt` (local) | E204 component recall 0.25 / placeholder validity 0.70, but recursive children hit the token cap and meaningful parse remains 0.0; **not promotable or ship** ([results](design/iter-e200-e204-layout-role-compiler-20260716.md)) |
| E205 Lark-terminal alignment | `e205-lark-terminal-stratified-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e205-lark-terminal-stratified-32step/checkpoints/last.pt` (local) | E207 syntax 1.0 with zero fallback and structure 0.3125, but empty bound stacks leave meaningful parse/component recall 0.0; **not promotable or ship** ([results](design/iter-e205-e207-lark-terminal-alignment-20260716.md)) |
| E208 occupancy alignment | `e208-list-occupancy-stratified-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e208-list-occupancy-stratified-32step/checkpoints/last.pt` (local) | E209 syntax 1.0 but empty root and meaningful parse 0.0; rejected, **not promotable or ship** ([results](design/iter-e208-e213-contextual-decisions-20260716.md)) |
| E210 scoped occupancy alignment | `e210-list-scope-occupancy-stratified-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e210-list-scope-occupancy-stratified-32step/checkpoints/last.pt` (local) | E211 syntax 1.0 but empty root and meaningful parse 0.0; rejected, **not promotable or ship** ([results](design/iter-e208-e213-contextual-decisions-20260716.md)) |
| E212 contextual decision alignment | `e212-contextual-decision-stratified-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e212-contextual-decision-stratified-32step/checkpoints/last.pt` (local) | E213 recovers populated root and normalized fidelity 0.50, but required schema semantics fail and meaningful parse remains 0.0; **not promotable or ship** ([results](design/iter-e208-e213-contextual-decisions-20260716.md)) |
| E215 overfiltered schema-role control | `e215-schema-role-judged-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e215-schema-role-judged-32step/checkpoints/last.pt` (local) | E214 falsely rejected 27 legal optional-null records; E216 metrics remain diagnostic but the data conclusion is superseded; **not promotable or ship** ([results](design/iter-e214-e216-schema-role-judge-20260716.md)) |
| E219 corrected schema-admission control | `e219-schema-normalized-32step` | CPU HF-context compiler-alignment diagnostic | `outputs/runs/e219-schema-normalized-32step/checkpoints/last.pt` (local) | E220 syntax 1.0 with zero fallback/dead ends, but component recall 0.25 and meaningful parse 0.0; **not promotable or ship** ([results](design/iter-e218-e220-schema-normalization-20260716.md)) |
| E221 task-balanced exposure diagnostic | `e221-canonical-task-balanced` | CPU HF-context compiler-alignment diagnostic | `outputs/autoresearch/e221-task-balanced-exposure-v4/runs/e221-canonical-task-balanced/checkpoints/last.pt` (local) | Effective exposure 29.68/128 falsifies task balancing; strict five-suite eval failed 9 gates and AgentV passed 1/5; **not promotable or ship** ([results](design/iter-e221-task-balanced-exposure-20260716.md)) |
| E222 capacity-aware exposure diagnostic | `e222-capacity-aware-matched` | CPU HF-context sampler diagnostic | `outputs/autoresearch/e222-capacity-aware-exposure/runs/e222-capacity-aware-matched/checkpoints/last.pt` (local) | Effective exposure rose to 83.59/128, but strict smoke parse regressed to 0.0 and 10 gates failed; **not promotable or ship** ([results](design/iter-e222-capacity-aware-exposure-20260716.md)) |
| E223 quota-capacity exposure diagnostic | `e223-quota-capacity-matched` | CPU HF-context sampler diagnostic | `outputs/autoresearch/e223-quota-capacity-exposure/runs/e223-quota-capacity-matched/checkpoints/last.pt` (local) | Task quotas and syntax are deterministic, but every suite has meaningful parse/recall/fidelity 0.0 and 12 gates fail; **not promotable or ship** ([results](design/iter-e223-quota-capacity-exposure-20260716.md)) |
| E224 semantic-exhaustive alignment diagnostic | `e224-semantic-exhaustive-matched` | CPU HF-context AST-role alignment diagnostic | `outputs/autoresearch/e224-semantic-exhaustive-alignment/runs/e224-semantic-exhaustive-matched/checkpoints/last.pt` (local) | E226 honest tree eval reaches syntax 1.0 on all suites and exact contract precision 1.0, but meaningful-program quality fails 5 gates; **not promotable or ship** ([results](design/iter-e226-honest-compiler-policy-20260716.md)) |
| E227 legal-candidate alignment diagnostic | `e227-candidate-set-matched` | CPU HF-context compiler candidate-ranking diagnostic | `outputs/autoresearch/e227-candidate-set-alignment/runs/e227-candidate-set-matched/checkpoints/last.pt` (local) | Syntax 1.0 but empty-layout collapse fails 12 gates and AgentV 0/5; rejected, **not promotable or ship** ([results](design/iter-e227-candidate-set-alignment-20260716.md)) |
| E228 legal-candidate margin diagnostic | `e228-candidate-margin-matched` | CPU HF-context compiler margin diagnostic | `outputs/autoresearch/e228-candidate-margin-alignment/runs/e228-candidate-margin-matched/checkpoints/last.pt` (local) | Syntax/contract precision 1.0 and only 4 failed gates, but AgentV 1/5; best diagnostic, **not promotable or ship** ([results](design/iter-e228-candidate-margin-alignment-20260716.md)) |
| E229 64-step margin continuation | `e229-margin-64step` | CPU HF-context duration diagnostic | `outputs/autoresearch/e229-margin-continuation/runs/e229-margin-64step/checkpoints/last.pt` (local) | Corrected syntax 1.0, but same 4 gates fail and several quality metrics regress vs E228; rejected, **not promotable or ship** ([results](design/iter-e229-margin-continuation-20260716.md)) |
| E230 diverse judged roots | `e230-diverse-roots-32step` | CPU HF-context data-coverage diagnostic | `outputs/autoresearch/e230-diverse-judged-roots/runs/e230-diverse-roots-32step/checkpoints/last.pt` (local) | 126 published judge-passed roots; four gates still fail and adversarial regresses; data repair retained, checkpoint **not promotable or ship** ([results](design/iter-e230-diverse-judged-roots-20260716.md)) |
| E231 component inventory | `e231-component-inventory-32step` | CPU HF-context semantic-inventory diagnostic | `outputs/autoresearch/e231-component-inventory/runs/e231-component-inventory-32step/checkpoints/last.pt` (local) | Inventory recall reaches 0.9167, but bias-off aggregate metrics/component choices are identical; six thresholds fail, checkpoint **not promotable or ship** ([results](design/iter-e231-component-inventory-20260716.md)) |
| E174 unfrozen-context 8-step control | `e174-unfrozen-context-8step` | CPU HF-context semantic control | `outputs/runs/e174-unfrozen-context-8step/checkpoints/last.pt` (local) | Unfrozen context, loss 39.4253; bounded probe syntax 0.0 and parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e174-unfrozen-context-20260716.md)) |
| Matrix honest champion (scratch) | `qx_e53_*` (V6 E53 family) | CPU scratch matrix clear | Primarily `outputs/runs/` (+ docs matrix JSON) | Honest `--ship-gates` on limited `rico_held` n; **not** production HF ship |
| P13 fixture E50 control | `qx_e50_core_remask` | CPU scratch, fixture corpus | `/tmp/slm17-e50-fixture-honest/` (local) | Matched control; held 0.08 / RICO 0.0667 fidelity; parse 0.0, not ship |
| P13 integrated E50 candidate | `qx_e50_core_remask` | CPU scratch, integrated corpus | `/tmp/slm17-e50-new-honest/` (local) | Strict fidelity gain on both smoke suites; parse 0.0, not promotable or ship |
| Frozen X2 baseline | `gx_x2_codec` seeds 0/1/2 | Retired fixed-canvas grammar diffusion | `/tmp/slm-training-fixed-baseline/outputs/topology_baseline/` (local) | 80 steps; all suites parse/fidelity/structure/reward 0.0; comparison only, not ship |
| Topology implementation smoke | `grammar_diffusion_overfit` | CPU scratch fixture topology v2 | pytest temporary checkpoint (local) | 200 steps; smoke n=2 parse/fidelity 0.5, topology composite 0.482; not reusable or ship |
| Topology X9/X14 confirmation | `gx_x9_topology_base`, `gx_x14_buffer` seeds 0/1/2 | CPU scratch topology v2 matrix | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` (local) | 200 steps; all 6 fail multi-suite gates; not promoted or synced |
| ScopeDiff X18 confirmation | `gx_x18_scope_noise_confirm_200` seeds 0/1/2 | CPU scratch topology v2 matrix | `outputs/runs/gx_x18_scope_noise_confirm_200/` (local) | 200 steps; all-suite median parse/fidelity 0.0; not promoted or synced |
| ScopeDiff X21 confirmation | `gx_x21_scoped_topology_confirm_200` seeds 0/1/2 | CPU scratch topology v2 matrix | `outputs/runs/gx_x21_scoped_topology_confirm_200/` (local) | 200 steps; weak structure, parse/fidelity 0.0; not promoted or synced |
| Production HF ship | — | — | `hf://buckets/TKendrick/OpenUI/checkpoints/<run_id>/` | **None registered yet** — fill this row after the first full HF sync |

Update the table in place when a checkpoint is written or superseded. Keep
invalidated / superseded rows in **Checkpoint history** below.

---

## Intended use

- Generate **placeholder OpenUI** layout programs (`openuiLibrary` syntax) from
  natural-language prompts, optionally conditioned on DESIGN.md.
- Train / eval harness research for TwoTower masked diffusion and
  grammar-diffusion codecs with honest multi-suite ship gates.

**Not intended:** production UI without human review; treating fixture-demo or
scratch-matrix clears as production readiness; silent gold-placeholder channels.

---

## Architecture (serving defaults)

| Piece | Default / notes |
| --- | --- |
| Model | TwoTower (context tower + MaskGIT-style denoiser); optional `grammar_diffusion` |
| Context | HF frozen backbone (`HuggingFaceTB/SmolLM2-135M`) for full ship track; scratch for matrix/CI demos |
| Output tokenizer | Compositional `OpenUITokenizer` (default) or V5 lexer (`DSLNativeTokenizer`) |
| Decode | Grammar-constrained LTR / MaskGIT + repair levers (see design docs) |
| Topology experiment | `grammar_diffusion` v2: typed production-tree expansion/contraction with bounded active nodes; no fixed canvas |
| Eval gates | Multi-suite `--ship-gates` (parse, structural, `placeholder_fidelity`, reward) |

---

## How to load

```bash
# Fixture demo (annotate playground)
python -m scripts.serve_playground
# → src/slm_training/resources/checkpoints/playground_demo/last.pt

# Full-run checkpoint from the OpenUI bucket (after sync)
hf buckets sync \
  hf://buckets/TKendrick/OpenUI/checkpoints/<run_id> \
  ./outputs/runs/<run_id>/checkpoints

python -m scripts.evaluate_model \
  --test-dir outputs/data/eval/v1 \
  --run-id <run_id> \
  --ship-gates
```

Sidecars required next to `*.pt`: `.tokenizer.json`, `.meta.json`
(optional `.context.tokenizer.json`).

---

## Training data

| Split | Source | Notes |
| --- | --- | --- |
| Train | `outputs/data/train/v1` (all sources + quality synth) for ship | Fixture upsample = demo only |
| Eval | `outputs/data/eval/v1` suites: smoke, held_out, adversarial, ood, `rico_held` | Ship claims need full `rico_held` (1500) when asserted |

Leakage: structural fingerprints + train/test isolation
([adversarial-review.md](design/adversarial-review.md)).

---

## Evaluation (fill per checkpoint)

| Suite | n | parse | fidelity | struct | reward | Pass? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke (`restructure_cpu_scratch_v0`) | 3 | 0.0 | 0.0 | 0.31 | 0.0 | No — fixture scratch wiring |
| not run (`local_directml_adreno_20260714`) | 0 | — | — | — | — | No — hardware/checkpoint validation only |
| held_out | | | | | | |
| adversarial | | | | | | |
| ood | | | | | | |
| rico_held | | | | | | |
| `rico_held` (`e120_unsandboxed`, diagnostic subset) | 1 | 0.0 | 0.375 | 0.0375 | 0.0 | No — 8-step scratch; 64-token incomplete program |
| `smoke` (`qx_e53_honest_v5_champion`, E121 diagnostic subset) | 1 | 0.0 | 0.0 | 0.0 | 0.0 | No — one 5-second constrained-decode timeout; not a ship evaluation |
| `smoke` (`e123_judged_32step_b`, E123 diagnostic subset) | 1 | 0.0 | 0.1917 | 0.0 | 0.0 | No — unconstrained retry/canvas cap; not a ship evaluation |
| `smoke` (`e127_judged_schema_slots`, E127 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1917 | 0.0 | No — placeholder signals improved but output did not parse; not a ship evaluation |
| `smoke` (`e128_judged_schema_slots_64`, E128 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.0 | No — higher loss weights regressed placeholder signal; not a ship evaluation |
| `smoke` (`e129_judged_schema_slots_64_lowweights`, E129 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.0 | No — lower-weight control did not reproduce E127; not a ship evaluation |
| `smoke` (`e130_judged_schema_slots_seed1`, E130 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.0 | No — seed-1 control did not reproduce E127; not a ship evaluation |
| `smoke` (`e132_generation_focus`, E132 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.1742 | 0.0 | No — task reweighting did not improve quality; not a ship evaluation |
| `smoke` (`e133_no_fuse_ltr`, E133 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.0 | 0.0 | No — no-fused-LTR path worsened feedback; not a ship evaluation |
| `smoke` (`e135_hf_context_control`, E135 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.2422 | 0.0 | No — HF representation improved signals but did not parse; not a ship evaluation |
| `smoke` (`e136_hf_context_32`, E136 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.0825 | 0.0 | No — longer HF run regressed E135; not a ship evaluation |
| `smoke` (`e137_hf_context_16`, E137 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.2142 | 0.0 | No — midpoint improved placeholder signal but did not parse; not a ship evaluation |
| `smoke` (`e138_hf_context_seed1_8`, E138 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.1683 | 0.0 | No — seed-1 control regressed diagnostic signals; not a ship evaluation |
| `smoke` (`e139_hf_context_seed2_8`, E139 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.0 | 0.0 | No — seed-2 control had no quality signal and two timeouts; not a ship evaluation |
| `smoke` (`e177-semantic-judge-32step`, E180 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.607 | No — syntax 1.0, but meaningful component recall 0.25; not a ship evaluation |
| `smoke` (`e181-semantic-balanced-32step`, E181 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.607 | No — mixture-only control did not improve quality; not a ship evaluation |
| `smoke` (`e184-compiler-aligned-32step`, E194 diagnostic subset) | 1 | 0.0 | 0.0 | 0.3600 | 0.0 | No — root/schema constraints improved, but output remained incomplete; not a ship evaluation |
| `smoke` (`e191-full-compiler-aligned-32step`, E192 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.607 | No — all-branch alignment regressed semantic root selection; not a ship evaluation |
| `smoke` (`e196-stratified-compiler-aligned-matched-32step`, E199 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1917 | 0.0 | No — syntax 1.0, but primitive binder declaration makes the layout trivial; not a ship evaluation |
| `smoke` (`e201-role-stratified-compiler-aligned-32step`, E204 diagnostic subset) | 1 | 0.0 | 0.0 | 0.0955 | 0.70 | No — component recall 0.25, but recursive children hit the token cap; not a ship evaluation |
| `smoke` (`e205-lark-terminal-stratified-32step`, E207 diagnostic subset) | 1 | 0.0 | 0.0 | 0.3125 | 0.0 | No — syntax 1.0 without fallback, but bound stacks are empty; not a ship evaluation |
| `smoke` (`e208-list-occupancy-stratified-32step`, E209 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1917 | 0.0 | No — syntax 1.0 but root is empty; not a ship evaluation |
| `smoke` (`e210-list-scope-occupancy-stratified-32step`, E211 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1917 | 0.0 | No — typed scope does not recover root occupancy; not a ship evaluation |
| `smoke` (`e212-contextual-decision-stratified-32step`, E213 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1333 | 0.50 | No — populated root/fidelity recover, but required schema semantics fail; not a ship evaluation |
| `smoke` (`e215-schema-role-judged-32step`, E216 diagnostic subset) | 1 | 0.0 | 0.0 | 0.3458 | 0.25 | No — syntax 1.0, but component recall 0.25 keeps meaningful parse at 0.0; not a ship evaluation |
| `smoke` (`e219-schema-normalized-32step`, E220 diagnostic subset) | 1 | 0.0 | 0.0 | 0.3458 | 0.25 | No — corrected admission preserves syntax, but component recall 0.25 keeps meaningful parse at 0.0; not a ship evaluation |
| `smoke` (`e221-canonical-task-balanced`, strict) | 3 | 0.3333 | 0.0 | 0.2097 | 0.4327 | No — syntax 1.0, but meaningful structure and fidelity fail gates |
| `held_out` (`e221-canonical-task-balanced`, strict) | 5 | 0.0 | 0.0 | 0.1667 | 0.3822 | No — strict gate failure |
| `adversarial` (`e221-canonical-task-balanced`, strict) | 4 | 0.25 | 0.0 | 0.3492 | 0.1593 | No — diagnostic only; checkpoint fails the full gate set |
| `ood` (`e221-canonical-task-balanced`, strict) | 4 | 0.0 | 0.0 | 0.2527 | 0.1593 | No — strict gate failure |
| `rico_held` (`e221-canonical-task-balanced`, strict) | 3 | 0.0 | 0.0 | 0.0901 | 0.0 | No — strict gate failure |
| `smoke` (`e222-capacity-aware-matched`, strict) | 3 | 0.0 | 0.0 | 0.2661 | 0.2123 | No — exposure improved but syntax and semantic quality regressed |
| `held_out` (`e222-capacity-aware-matched`, strict) | 5 | 0.0 | 0.0 | 0.2796 | 0.2548 | No — strict gate failure |
| `adversarial` (`e222-capacity-aware-matched`, strict) | 4 | 0.5 | 0.0 | 0.3845 | 0.4778 | No — suite signal only; full gate set failed |
| `ood` (`e222-capacity-aware-matched`, strict) | 4 | 0.0 | 0.0 | 0.3719 | 0.1593 | No — strict gate failure |
| `rico_held` (`e222-capacity-aware-matched`, strict) | 3 | 0.0 | 0.0 | 0.1501 | 0.0 | No — strict gate failure |
| `smoke` (`e223-quota-capacity-matched`, strict) | 3 | 0.0 | 0.0 | 0.3094 | 0.0 | No — syntax 1.0 but output is semantically trivial |
| `held_out` (`e223-quota-capacity-matched`, strict) | 5 | 0.0 | 0.0 | 0.2514 | 0.0 | No — strict gate failure |
| `adversarial` (`e223-quota-capacity-matched`, strict) | 4 | 0.0 | 0.0 | 0.2905 | 0.0 | No — strict gate failure |
| `ood` (`e223-quota-capacity-matched`, strict) | 4 | 0.0 | 0.0 | 0.2369 | 0.0 | No — strict gate failure |
| `rico_held` (`e223-quota-capacity-matched`, strict) | 3 | 0.0 | 0.0 | 0.0901 | 0.0 | No — strict gate failure |
| `smoke` (`e224-semantic-exhaustive-matched`, E226 honest tree) | 3 | 1.0 | 0.7222 | 0.3628 | 0.3163 | No — meaningful program 0.0; trivial/low-recall layouts |
| `held_out` (`e224-semantic-exhaustive-matched`, E226 honest tree) | 5 | 1.0 | 0.4533 | 0.2309 | 0.2916 | No — meaningful program 0.0 and structure below gate |
| `adversarial` (`e224-semantic-exhaustive-matched`, E226 honest tree) | 4 | 1.0 | 0.7500 | 0.2982 | 0.1873 | No — meaningful program 0.0 |
| `ood` (`e224-semantic-exhaustive-matched`, E226 honest tree) | 4 | 1.0 | 0.4250 | 0.2762 | 0.3520 | No — meaningful program 0.0 |
| `rico_held` (`e224-semantic-exhaustive-matched`, E226 honest tree) | 3 | 1.0 | 0.1667 | 0.2380 | 0.4577 | No — diagnostic n=3; full gate set failed |
| `smoke` (`e227-candidate-set-matched`) | 3 | 1.0 | 0.0000 | 0.3094 | 0.0000 | No — trivial empty layouts |
| `held_out` (`e227-candidate-set-matched`) | 5 | 1.0 | 0.0333 | 0.2739 | 0.1398 | No — meaningful program 0.0; full gate set failed |
| `adversarial` (`e227-candidate-set-matched`) | 4 | 1.0 | 0.0000 | 0.2905 | 0.0000 | No — trivial empty layouts |
| `ood` (`e227-candidate-set-matched`) | 4 | 1.0 | 0.0000 | 0.2369 | 0.0000 | No — trivial empty layouts |
| `rico_held` (`e227-candidate-set-matched`) | 3 | 1.0 | 0.0000 | 0.0901 | 0.0000 | No — diagnostic n=3; full gate set failed |
| `smoke` (`e228-candidate-margin-matched`) | 3 | 1.0 | 0.5278 | 0.4642 | 0.8073 | No — meaningful program 0.3333 below gate |
| `held_out` (`e228-candidate-margin-matched`) | 5 | 1.0 | 0.2800 | 0.3369 | 0.7330 | No — meaningful program 0.0 |
| `adversarial` (`e228-candidate-margin-matched`) | 4 | 1.0 | 0.5417 | 0.4744 | 0.8115 | Suite passes; checkpoint still fails full gate set |
| `ood` (`e228-candidate-margin-matched`) | 4 | 1.0 | 0.2583 | 0.3750 | 0.7265 | No — meaningful program 0.0 |
| `rico_held` (`e228-candidate-margin-matched`) | 3 | 1.0 | 0.1250 | 0.1628 | 0.6865 | No — structure below gate; diagnostic n=3 |
| `smoke` (`e229-margin-64step`) | 3 | 1.0 | 0.5556 | 0.4475 | 0.6073 | No — meaningful program 0.3333 below gate |
| `held_out` (`e229-margin-64step`) | 5 | 1.0 | 0.5600 | 0.3564 | 0.8290 | No — meaningful program 0.0 |
| `adversarial` (`e229-margin-64step`) | 4 | 1.0 | 0.8333 | 0.4387 | 0.9110 | Suite passes; checkpoint still fails full gate set |
| `ood` (`e229-margin-64step`) | 4 | 1.0 | 0.5583 | 0.3481 | 0.8285 | No — meaningful program 0.0 |
| `rico_held` (`e229-margin-64step`) | 3 | 1.0 | 0.2500 | 0.1720 | 0.7360 | No — structure below gate; diagnostic n=3 |
| `smoke` (`e230-diverse-roots-32step`) | 3 | 1.0 | 0.5278 | 0.4642 | 0.8073 | No — meaningful program 0.3333 below gate |
| `held_out` (`e230-diverse-roots-32step`) | 5 | 1.0 | 0.2800 | 0.3369 | 0.7330 | No — meaningful program 0.0 |
| `adversarial` (`e230-diverse-roots-32step`) | 4 | 1.0 | 0.2083 | 0.3477 | 0.3870 | No — semantic quality regressed |
| `ood` (`e230-diverse-roots-32step`) | 4 | 1.0 | 0.2583 | 0.3750 | 0.7265 | No — meaningful program 0.0 |
| `rico_held` (`e230-diverse-roots-32step`) | 3 | 1.0 | 0.1250 | 0.1628 | 0.6865 | No — structure below gate; diagnostic n=3 |
| `smoke` (`e231-component-inventory-32step`) | 3 | 1.0 | 0.1944 | 0.4636 | 0.4910 | No — meaningful program 0.3333 and fidelity below gates |
| `held_out` (`e231-component-inventory-32step`) | 5 | 1.0 | 0.1133 | 0.3302 | 0.4234 | No — meaningful program 0.0 and fidelity below gates |
| `adversarial` (`e231-component-inventory-32step`) | 4 | 1.0 | 0.4583 | 0.4681 | 0.6242 | Suite passes; checkpoint still fails full gate set |
| `ood` (`e231-component-inventory-32step`) | 4 | 1.0 | 0.2083 | 0.3469 | 0.5493 | No — meaningful program 0.0 |
| `rico_held` (`e231-component-inventory-32step`) | 3 | 1.0 | 0.1250 | 0.1628 | 0.6865 | No — structure below gate; diagnostic n=3 |

Recipe for `restructure_cpu_scratch_v0`: device=cpu, steps=80, context=scratch,
fixture train/test `v0`, `--no-sync-checkpoints`, LTR primary, no DESIGN.md in
context. Host: 4c / 15GB RAM, no CUDA, no `HF_TOKEN` (Jobs/bucket skipped).
Evidence: [restructure-cpu-train-results.json](design/restructure-cpu-train-results.json).

Recipe for `local_directml_adreno_20260714`: Qualcomm Adreno X1-85 via
Torch-DirectML (`privateuseone:0`), 5 steps, batch 4, 585-record remediated
corpus, scratch context, 924,386 trainable parameters, no AMP/compile, and
`--no-sync-checkpoints`. Last loss was 61.2962; no eval suite or ship gates ran.
AdamW `aten::lerp.Scalar_out` fell back to CPU. The checkpoint loaded in the CPU
playground, but a real generation did not return within 120 seconds, so it is not
a viable playground candidate. Evidence:
[local-directml-train-results.json](design/local-directml-train-results.json).

Record device, steps, context backend, honesty mode (`honest_slot_contract`),
and whether gates used `--ship-gates`. Link
`docs/design/*-results.json` / run `gates.json` when available.

**Known honest fixture clears (not production):** V6 E50/E53/E55 on CPU scratch
with limited `rico_held` n — see
[quality-experiment-matrix.md](design/quality-experiment-matrix.md).

### P13 matched smoke (SLM-17)

| Checkpoint | Suite | n | Parse | Fidelity | Struct | Reward | Pass? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixture E50 | `held_out` | 5 | 0.0 | 0.08 | 0.0 | 0.0 | No |
| integrated E50 | `held_out` | 5 | 0.0 | 0.12 | 0.0 | 0.0 | Signal only; +0.04 |
| fixture E50 | `rico_held` | 5 | 0.0 | 0.0667 | 0.0 | 0.0 | No |
| integrated E50 | `rico_held` | 5 | 0.0 | 0.10 | 0.0 | 0.0 | Signal only; +0.0333 |

Recipe: E50 on CPU scratch, 80 train steps, batch 4, lr `3e-4`, seed 0,
honest slot contract, four-step best-of-1 decode, no template fill or
DESIGN.md context, and unchanged gates. Checkpoints are local scratch
artifacts with explicit no-sync rationale; this is a bounded matched data
signal, not a full HF-context train or reusable promotion.
Evidence: [data-synthesis.md](design/data-synthesis.md) and
[data-synthesis-results.json](design/data-synthesis-results.json).

### Grammar topology implementation smoke

| Checkpoint | Suite | n | Parse | Fidelity | Struct | Topology composite | Pass? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| retired X2, seeds 0/1/2 | all five | 19 per seed | 0.0 | 0.0 | 0.0 | unavailable | No |
| topology v2 fixture overfit | smoke | 2 | 0.5 | 0.5 | 0.225 | 0.4820 | Wiring only; no ship |
| X9 confirmation median | smoke | 3 | 0.0 | 0.333 | 0.098 | 0.414 | No |
| X9 confirmation median | held_out | 5 | 0.0 | 0.0 | 0.0 | 0.372 | No |
| X9 confirmation median | adversarial | 4 | 0.0 | 0.0 | 0.0 | 0.472 | No |
| X9 confirmation median | ood | 4 | 0.0 | 0.083 | 0.108 | 0.330 | No |
| X9 confirmation median | rico_held | 3 | 0.667 | 0.125 | 0.317 | 0.464 | No — limited slice; other suites fail |
| X14 confirmation median | smoke | 3 | 0.0 | 0.0 | 0.309 | 0.298 | No |
| X14 confirmation median | held_out | 5 | 0.0 | 0.0 | 0.251 | 0.277 | No |
| X14 confirmation median | adversarial | 4 | 0.0 | 0.0 | 0.291 | 0.285 | No |
| X14 confirmation median | ood | 4 | 0.0 | 0.0 | 0.237 | 0.278 | No |
| X14 confirmation median | rico_held | 3 | 0.0 | 0.042 | 0.078 | 0.233 | No |
| X18 confirmation median | smoke | 3 | 0.0 | 0.0 | 0.0 | unavailable | No |
| X18 confirmation median | held_out | 5 | 0.0 | 0.0 | 0.0 | unavailable | No |
| X18 confirmation median | adversarial | 4 | 0.0 | 0.0 | 0.0 | unavailable | No |
| X18 confirmation median | ood | 4 | 0.0 | 0.0 | 0.0 | unavailable | No |
| X18 confirmation median | rico_held | 32 | 0.0 | 0.0 | 0.0 | unavailable | No |
| X21 confirmation median | smoke | 3 | 0.0 | 0.0 | 0.115 | 0.238 | No |
| X21 confirmation median | held_out | 5 | 0.0 | 0.0 | 0.109 | 0.231 | No |
| X21 confirmation median | adversarial | 4 | 0.0 | 0.0 | 0.109 | 0.231 | No |
| X21 confirmation median | ood | 4 | 0.0 | 0.0 | 0.085 | 0.228 | No |
| X21 confirmation median | rico_held | 32 | 0.0 | 0.0 | 0.046 | 0.208 | No |

X2 used CPU scratch, 80 steps, batch 4, seeds 0/1/2, the 1,165-record
curriculum corpus, limited remediated suites, and no checkpoint sync. All three
AgentV bundles ran without execution errors and all ship gates failed. Topology v2
used CPU scratch, 200 steps, batch 2, learning rate `3e-3`, two fixture records,
and honest request slot contracts; AgentV ran 5 checks with zero passes. Neither
checkpoint family was promoted or uploaded. Evidence:
[grammar-fixed-canvas-baseline-results.json](design/grammar-fixed-canvas-baseline-results.json)
and [grammar-topology-smoke-results.json](design/grammar-topology-smoke-results.json).
The X9/X14 confirmation used the same 1,165-record curriculum corpus, CPU scratch
context, 200 steps, batch 4, learning rate `3e-4`, 16 generation phases, and seeds
0/1/2. Six AgentV bundles completed. All checkpoints are local short-budget matrix
artifacts with an explicit no-sync rationale; no reusable champion was designated.
Evidence: [grammar-matrix-results.json](design/grammar-matrix-results.json).
The X16-X21 campaign used 694 immutable ProgramSpec-derived records, including
189 scope-contract rows, and limited RICO n=32. X18 and X21 were confirmed at 200
CPU scratch steps for seeds 0/1/2. Six AgentV bundles completed with 0/5 domain
passes each. The scope-specific heads cannot be scored on these generalization
suites because they contain no scope metadata. Evidence:
[grammar-scope-matrix-results.json](design/grammar-scope-matrix-results.json).

---

## Limitations & honesty

- Smoke parse alone is a canary, not generalization.
- Soft `placeholder_validity` is diagnostic; ship on `placeholder_fidelity`.
- Inventory must come from the user-visible prompt / DESIGN.md under
  `honest_slot_contract=True` (no silent `gold.placeholders`).
- Scratch + short steps ≠ HF + full `rico_held` production claim.

---

## Checkpoint history

| Date (UTC) | Run id | Bucket / path | Metric headline | Notes |
| --- | --- | --- | --- | --- |
| (seed) | `playground_demo` | `src/slm_training/resources/checkpoints/playground_demo/` | wiring demo | Committed fixture; regenerate via `bootstrap_playground` |
| 2026-07-14 | `restructure_cpu_scratch_v0` | `outputs/runs/restructure_cpu_scratch_v0/` (local) | smoke parse 0.0 @ 80 steps; last_loss≈6.97 | Post-restructure CPU budget verify; not ship |
| 2026-07-14 | `restructure_cpu_scratch_v0_cont` | `outputs/runs/restructure_cpu_scratch_v0_cont/` (local) | resume +200 scratch steps; smoke parse still 0.0 | Continues v0; HF Jobs still blocked on missing HF_TOKEN |
| 2026-07-14 | `qx_e0_baseline` (P13 superseded) | `outputs/slm17/matrix-smoke-baseline/` (local) | `rico_held n=3` parse/fidelity 0.0 | Fixture probe; not comparable to E50; scratch/no-sync |
| 2026-07-14 | `qx_e50_core_remask` (P13 superseded) | `outputs/slm17/matrix-smoke-champion/` (local) | `rico_held n=3` parse/fidelity 1.0 | System-recipe probe, not a matched data signal; scratch/no-sync |
| 2026-07-14 | fixture `qx_e50_core_remask` (P13 final) | `/tmp/slm17-e50-fixture-honest/` (local) | held 0.08 / RICO 0.0667 fidelity; parse 0.0 | Equal-recipe fixture control; scratch/no-sync; not ship |
| 2026-07-14 | integrated `qx_e50_core_remask` (P13 final) | `/tmp/slm17-e50-new-honest/` (local) | held 0.12 / RICO 0.10 fidelity; parse 0.0 | Strict two-suite data signal; scratch/no-sync; not promotable or ship |
| 2026-07-14 | `local_directml_adreno_20260714` | `outputs/runs/local_directml_adreno_20260714/` (local) | DirectML train completed @ 5 steps; last_loss≈61.30 | Adreno GPU/checkpoint wiring; one AdamW op used CPU fallback; CPU generation timed out at 120s; no eval/ship claim |
| 2026-07-15 | `overnight_retrain_200` | `/tmp/slm-training-overnight/outputs/runs/overnight_retrain_200/` (local) | 200 CPU scratch steps; last_loss≈6.64; all suites parse 0.0 | Full honest eval with AgentV bundle; no promotion; decode-path investigation continues |
| 2026-07-15 | `overnight_retrain_1000` | `/tmp/slm-training-overnight/outputs/runs/overnight_retrain_1000/` (local) | 1,000 CPU scratch steps; last_loss≈1.12; smoke parse 0.0 at every checkpoint | Extended training did not improve generation quality; no promotion |
| 2026-07-15 | `gx_x2_codec` seeds 0/1/2 | `/tmp/slm-training-fixed-baseline/outputs/topology_baseline/` (local) | all five suites parse/fidelity/structure/reward 0.0 | Frozen format-v1 comparison; AgentV complete; not promoted or synced |
| 2026-07-15 | topology `grammar_diffusion_overfit` | pytest temporary local checkpoint | smoke n=2 parse/fidelity 0.5; topology composite 0.4820 | Implementation smoke only; temporary checkpoint, not promoted or synced |
| 2026-07-15 | `gx_x9_topology_base` seeds 0/1/2 | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` (local) | RICO n=3 median parse 0.667, but held/adversarial/OOD parse 0.0 | 200-step CPU scratch confirmation; all seeds fail multi-suite gates; not promoted or synced |
| 2026-07-15 | `gx_x14_buffer` seeds 0/1/2 | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` (local) | all-suite median parse 0.0 | 200-step CPU scratch confirmation; all seeds fail; not promoted or synced |
| 2026-07-16 | `gx_x18_scope_noise_confirm_200` seeds 0/1/2 | `outputs/runs/gx_x18_scope_noise_confirm_200/` (local) | all-suite median parse/fidelity/structure 0.0 | 200-step CPU scratch confirmation; all seeds fail; no promotion or sync |
| 2026-07-16 | `gx_x21_scoped_topology_confirm_200` seeds 0/1/2 | `outputs/runs/gx_x21_scoped_topology_confirm_200/` (local) | all-suite median parse/fidelity 0.0; weak structure | 200-step CPU scratch confirmation; all seeds fail; no promotion or sync |
| 2026-07-16 | `qx_e53_honest_v5_champion` (E121) | `outputs/runs/iter-e121d-e53-judged-20260715/` (local) | judged corpus 405; smoke n=1 parse/fidelity/structure/reward 0.0; decode timeout | Explicit corpus precedence and evaluator tuple bugs fixed; scratch-only; no promotion |
| 2026-07-16 | `e123_judged_32step_b` (E123) | `outputs/runs/iter-e123b-judged-20260715/` (local) | 32 CPU scratch steps; loss 10.97; smoke parse 0.0, structural similarity 0.1917, 26.75s p50; fallback/canvas cap | Longer training did not improve generation; generation-recipe investigation next; no promotion |
| 2026-07-16 | `e127_judged_schema_slots` (E127) | `outputs/runs/iter-e127-schema-slots-20260715/` (local) | 32 CPU scratch steps; loss 10.71; placeholder validity 0.55 / normalized fidelity 0.25; parse 0.0 | Schema/slot conditioning improves placeholder signal but not syntax; no promotion |
| 2026-07-16 | `e128_judged_schema_slots_64` (E128) | `outputs/runs/iter-e128-schema-slots-20260715/` (local) | 64 CPU scratch steps; loss 15.03; placeholder validity/fidelity 0.0; parse 0.0 | Higher LTR/fidelity weights regressed E127; no promotion |
| 2026-07-16 | `e129_judged_schema_slots_64_lowweights` (E129) | `outputs/runs/iter-e129-schema-slots-20260715/` (local) | 64 CPU scratch steps; loss 9.89; placeholder validity/fidelity 0.0; parse 0.0 | Lower-weight control failed to reproduce E127; data/variance investigation next; no promotion |
| 2026-07-16 | `e130_judged_schema_slots_seed1` (E130) | `outputs/runs/iter-e130-schema-slots-20260715/` (local) | 32 CPU scratch steps, seed 1; loss 15.28; placeholder validity/fidelity 0.0; parse 0.0 | E127 not reproducible; multi-example/task-composition feedback next; no promotion |
| 2026-07-16 | `e132_generation_focus` (E132) | `outputs/runs/iter-e132-generation-focus-20260715/` (local) | 32 CPU scratch steps; three-prompt parse/placeholder 0.0; structural similarity 0.1742 | Task reweighting rejected; architecture/representation or synthesis contract next; no promotion |
| 2026-07-16 | `e133_no_fuse_ltr` (E133) | `outputs/runs/iter-e133-no-fuse-ltr-20260715/` (local) | 32 CPU scratch steps; three-prompt parse/structure 0.0; one 15s timeout | No-fused-LTR path rejected; fused LTR retained; no promotion |
| 2026-07-16 | `e135_hf_context_control` (E135) | `outputs/runs/iter-e135-hf-context-20260715/` (local) | 8 CPU steps with frozen SmolLM2-135M; three-prompt parse 0.0, structural 0.2422, placeholder validity 0.3167; one timeout | HF context is leading representation hypothesis; longer cached control next; no promotion |
| 2026-07-16 | `e136_hf_context_32` (E136) | `outputs/runs/iter-e136-hf-context-20260715/` (local) | 32 CPU steps with frozen SmolLM2-135M; parse/placeholder 0.0, structural 0.0825 | Longer HF training regressed; checkpoint selection/supervision alignment next; no promotion |
| 2026-07-16 | `e137_hf_context_16` (E137) | `outputs/runs/iter-e137-hf-context-20260715/` (local) | 16 CPU steps with frozen SmolLM2-135M; parse 0.0, placeholder validity 0.40, structural 0.2142 | Non-monotonic HF trajectory; explicit early checkpoint selection next; no promotion |
| 2026-07-16 | `e138_hf_context_seed1_8` (E138) | `outputs/runs/iter-e138-hf-seed1-20260715/` (local) | 8 CPU steps with frozen SmolLM2-135M, seed 1; parse 0.0, placeholder validity 0.0, structural 0.1683 | Seed variance is material; multi-seed selection before corpus/loss changes; no promotion |
| 2026-07-16 | `e139_hf_context_seed2_8` (E139) | `outputs/runs/iter-e139-hf-seed2-20260715/` (local) | 8 CPU steps with frozen SmolLM2-135M, seed 2; parse/placeholder/structure 0.0, two timeouts | Seed-0 signal remains unexplained; decoder/checkpoint diagnosis next; no promotion |
| 2026-07-16 | `e173-schema-context-32step` (E173) | `outputs/runs/e173-schema-context-32step/` (local) | 32 CPU steps with frozen HF context plus schema/slot context; loss 11.0876; bounded syntax 1.0, parse 0.0 | Context-only control did not recover semantic hierarchy; no promotion |
| 2026-07-16 | `e174-unfrozen-context-8step` (E174) | `outputs/runs/e174-unfrozen-context-8step/` (local) | 8 CPU steps with unfrozen HF context; loss 39.4253; bounded syntax 0.0, parse 0.0 | Rejected control; retain frozen context; no promotion |
| 2026-07-16 | `e175-retrieval-8step` (E175) | `outputs/runs/e175-retrieval-8step/` (local) | 8 CPU steps with frozen HF context, schema context, retrieval k=4; loss 27.9708; bounded syntax/parse 0.0 | Retrieval rejected; improve semantic coverage; no promotion |
| 2026-07-16 | `e176-broad-corpus-8step` (E176) | `outputs/runs/e176-broad-corpus-8step/` (local) | 8 CPU steps on 1,417-record prompt-contract corpus; loss 34.0464; bounded syntax/parse 0.0 | Broad corpus rejected; targeted judge-gated augmentation next; no promotion |
| 2026-07-16 | `e177-semantic-judge-32step` (E177–E180) | `outputs/runs/e177-semantic-judge-32step/` (local) | 32 CPU steps on 496 published judge-gated records; loss 12.2220; bounded syntax 1.0, meaningful parse 0.0, structure 0.1542 | Deterministic compiler structure fixed; semantic role supervision next; no promotion |
| 2026-07-16 | `e181-semantic-balanced-32step` (E181–E183) | `outputs/runs/e181-semantic-balanced-32step/` (local) | 32 CPU steps; loss 5.5118; bounded syntax 1.0, meaningful parse 0.0, structure 0.1542 | Mixture-only change rejected; root telemetry isolates learned semantic preference; no promotion |
| 2026-07-16 | `e184-compiler-aligned-32step` (E184–E190, E193–E194) | `outputs/runs/e184-compiler-aligned-32step/` (local) | 32 CPU steps; loss 10.0153; E194 meaningful parse 0.0, structure 0.3600 | Component alignment recovered root; generalized grammar/schema fixes retained; no promotion |
| 2026-07-16 | `e191-full-compiler-aligned-32step` (E191–E192) | `outputs/runs/e191-full-compiler-aligned-32step/` (local) | 32 CPU steps; loss 14.8498; E192 syntax 1.0, meaningful parse 0.0, structure 0.1542 | Random all-branch alignment regressed root selection; rejected; no promotion |
| 2026-07-16 | `e195-stratified-compiler-aligned-32step` (E195) | `outputs/runs/e195-stratified-compiler-aligned-32step/` (local) | 32 CPU steps; loss 17.0750; 597 aligned states | Invalid comparison: persisted mixture was not loaded; resolver fixed; no promotion |
| 2026-07-16 | `e196-stratified-compiler-aligned-matched-32step` (E196–E199) | `outputs/runs/e196-stratified-compiler-aligned-matched-32step/` (local) | 32 CPU steps; loss 7.8562; E199 syntax 1.0, meaningful parse 0.0, structure 0.1917 | Stratification fixes root/binder choice and enum completion, but declaration role remains wrong; no promotion |
| 2026-07-16 | `e201-role-stratified-compiler-aligned-32step` (E200–E204) | `outputs/runs/e201-role-stratified-compiler-aligned-32step/` (local) | 32 CPU steps; loss 9.1521; E204 meaningful parse 0.0, structure 0.0955, placeholder validity 0.70 | Generated role constraints work locally, but recursive children remain incomplete; no promotion |
| 2026-07-16 | `e205-lark-terminal-stratified-32step` (E205–E207) | `outputs/runs/e205-lark-terminal-stratified-32step/` (local) | 32 CPU steps; loss 8.1997; E207 syntax 1.0, meaningful parse 0.0, structure 0.3125 | Terminal alignment and enum paths fix syntax/fallback, but empty child collections remain trivial; no promotion |
| 2026-07-16 | `e208-list-occupancy-stratified-32step` (E208–E209) | `outputs/runs/e208-list-occupancy-stratified-32step/` (local) | 32 CPU steps; loss 7.4938; E209 syntax 1.0, meaningful parse 0.0 | Occupancy-only alignment produces an empty root; rejected, no promotion |
| 2026-07-16 | `e210-list-scope-occupancy-stratified-32step` (E210–E211) | `outputs/runs/e210-list-scope-occupancy-stratified-32step/` (local) | 32 CPU steps; loss 7.5847; E211 syntax 1.0, meaningful parse 0.0 | Root/bound occupancy scope remains insufficient; rejected, no promotion |
| 2026-07-16 | `e212-contextual-decision-stratified-32step` (E212–E213) | `outputs/runs/e212-contextual-decision-stratified-32step/` (local) | 32 CPU steps; loss 7.5117; E213 normalized fidelity 0.50, meaningful parse 0.0 | Contextual binder role recovers root occupancy; required schema semantics next; no promotion |
| 2026-07-16 | `e215-schema-role-judged-32step` (E214–E216) | `outputs/runs/e215-schema-role-judged-32step/` (local) | 32 CPU steps on 447 overfiltered records; loss 12.4024; E216 syntax 1.0, meaningful parse 0.0 | Superseded after 27 false-positive optional-null rejects were found; no promotion |
| 2026-07-16 | `e219-schema-normalized-32step` (E218–E220) | `outputs/runs/e219-schema-normalized-32step/` (local) | 32 CPU steps on 480 corrected records; loss 13.2406; E220 syntax 1.0, meaningful parse 0.0 | Restores legal optional omissions and schema-normalizes producers; semantic coverage unchanged; no promotion |
| 2026-07-16 | `e221-canonical-task-balanced` (E221) | `outputs/autoresearch/e221-task-balanced-exposure-v4/runs/e221-canonical-task-balanced/` (local) | 32 CPU steps on canonical E218; loss 14.1748; strict eval syntax 1.0 on all suites but meaningful parse 0.3333/0/0.25/0/0 and 9 failed gates | Task balancing did not improve effective exposure; AgentV 1/5; checkpoint SHA `85f0fb0c…bd7cd`; no sync or promotion |
| 2026-07-16 | `e222-capacity-aware-matched` (E222) | `outputs/autoresearch/e222-capacity-aware-exposure/runs/e222-capacity-aware-matched/` (local) | 32 CPU steps on canonical E218; loss 11.7409; exposure 83.59/128; strict meaningful parse 0/0/0.5/0/0 and 10 failed gates | Sampler mechanism confirmed but semantic non-regression falsified; AgentV 1/5; checkpoint SHA `960e13f1…3f348c5`; no sync or promotion |
| 2026-07-16 | `e223-quota-capacity-matched` (E223) | `outputs/autoresearch/e223-quota-capacity-exposure/runs/e223-quota-capacity-matched/` (local) | 32 CPU steps on canonical E218; loss 11.9060; exposure 81.11/128; syntax 1.0 but meaningful parse/recall/fidelity 0.0 on all suites | Allocation mechanism confirmed but quality recovery falsified; AgentV 0/5; checkpoint SHA `2db1e797…28a5ab87`; no sync or promotion |
| 2026-07-16 | `e224-semantic-exhaustive-matched` (E224–E226) | `outputs/autoresearch/e224-semantic-exhaustive-alignment/runs/e224-semantic-exhaustive-matched/` (local) | 32 CPU steps; loss 15.9786; E226 honest tree eval syntax 1.0 on all suites, fidelity 0.1667–0.75, meaningful program 0/0/0/0/0.3333 | E225 superseded after request telemetry dropped schema/contract; AgentV 1/5; checkpoint SHA `c9f38df1…22bb8ef`; no sync or promotion |
| 2026-07-16 | `e227-candidate-set-matched` (E227) | `outputs/autoresearch/e227-candidate-set-alignment/runs/e227-candidate-set-matched/` (local) | 32 CPU steps; loss 12.3030; syntax 1.0 on all suites, meaningful program 0.0 throughout | Legal-candidate loss fell to 2.4120 but empty-layout collapse failed 12 gates; AgentV 0/5; checkpoint SHA `b99bdf78…269577`; no sync or promotion |
| 2026-07-16 | `e228-candidate-margin-matched` (E228) | `outputs/autoresearch/e228-candidate-margin-alignment/runs/e228-candidate-margin-matched/` (local) | 32 CPU steps; loss 14.6153; syntax/contract 1.0, meaningful program 0.3333/0/0.5/0/0.6667 | Margin restores populated topology and reduces failures to 4; AgentV 1/5; checkpoint SHA `7a9be4a6…f5b093a`; no sync or promotion |
| 2026-07-16 | `e229-margin-64step` (E229) | `outputs/autoresearch/e229-margin-continuation/runs/e229-margin-64step/` (local) | Resumed to 64 CPU steps; loss 9.4505; corrected syntax 1.0, meaningful program 0.3333/0/0.5/0/0.6667 | Same 4 gates fail and quality regresses vs E228; AgentV 1/5; checkpoint SHA `23f31fa9…97cf0f4`; no sync or promotion |
| 2026-07-16 | `e230-diverse-roots-32step` (E230) | `outputs/autoresearch/e230-diverse-judged-roots/runs/e230-diverse-roots-32step/` (local) | 32 CPU steps on 126 published judged roots; loss 19.1868; syntax 1.0 and meaningful program 0.3333/0/0.25/0/0.6667 | Same 4 gates fail; adversarial regresses; AgentV 1/5; checkpoint SHA `009b1ab3…03198`; no sync or promotion |
| 2026-07-16 | `e231-component-inventory-32step` (E231) | `outputs/autoresearch/e231-component-inventory/runs/e231-component-inventory-32step/` (local) | 32 CPU steps; loss 19.9879; inventory recall 0.9167; syntax 1.0 and meaningful program 0.3333/0/0.5/0/0.6667 | Bias-off aggregate/component choices identical; 6 thresholds fail; AgentV 1/5; checkpoint SHA `136aa004…d475de`; no sync or promotion |

Append a row for every new or replaced checkpoint. Do not delete history.

---

## Agent checklist (after each checkpoint)

1. Sync durable weights (HF bucket for full runs) —
   [checkpoint-bucket.md](design/checkpoint-bucket.md).
2. Update **Current checkpoint roster** + **Evaluation** + **Checkpoint history**
   in this file.
3. Refresh the **Model card (summary)** section in [`README.md`](../README.md)
   (keep it short; link here for detail).
4. Point measured-results / matrix docs at the new run id when relevant.
5. Commit docs with the checkpoint-producing change.
