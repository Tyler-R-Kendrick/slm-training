# Model card — OpenUI TwoTower / grammar-diffusion

Canonical card for checkpoints produced by this repo. Agents **must update
this file whenever a new checkpoint is created or promoted** (full train,
remote train, bootstrap demo, or matrix champion intended for reuse), then
mirror a short summary into [`README.md`](../README.md) → “Model card (summary)”.

Storage: durable full-run weights live in
[`hf://buckets/TKendrick/OpenUI`](https://huggingface.co/buckets/TKendrick/OpenUI)
(`checkpoints/<run_id>/`). Local/git fixture demo:
`src/slm_training/resources/checkpoints/playground_demo/`.

Provenance is **fail-closed**: a row citing a `frontier` / `ship_candidate`
checkpoint must carry a verified `CheckpointReferenceV1` that resolves from a
fresh clone (`python -m scripts.verify_checkpoint_references --check`).
Gitignored `outputs/` rows below are honest **local / diagnostic** evidence, not
frontier claims; see the migration record
[checkpoint-reference-backfill-20260717.md](design/checkpoint-reference-backfill-20260717.md).

Related: [checkpoint-bucket.md](design/checkpoint-bucket.md),
[checkpoint-provenance.md](design/checkpoint-provenance.md),
[adversarial-review.md](design/adversarial-review.md),
[quality-experiment-matrix.md](design/quality-experiment-matrix.md).

---

## Current checkpoint roster

| Role | Run id | Kind | Location | Status |
| --- | --- | --- | --- | --- |
| Playground demo | `playground_demo` | Fixture wiring | `src/slm_training/resources/checkpoints/playground_demo/last.pt` (git) | E497 clean-revision honest smoke: parse/meaningful/fidelity 0.0, structure 0.2203, AgentV 0/5, one timeout. Demo only — **not** a quality or ship claim |
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
| E232 role component plan | `e232-role-component-plan-32step` | CPU HF-context grammar-role planning diagnostic | `outputs/autoresearch/e232-role-component-plan/runs/e232-role-component-plan-32step/checkpoints/last.pt` (local) | Root/count targets learn and causally improve adversarial quality, but four frontier thresholds fail and stronger calibration is flat; **not promotable or ship** ([results](design/iter-e232-role-component-plan-20260716.md)) |
| E233 resolved-AST component edges | `e233-component-edges-32step` | CPU HF-context AST-edge planning diagnostic | `outputs/autoresearch/e233-component-edges/runs/e233-component-edges-32step/checkpoints/last.pt` (local) | Edge target learns, but edge on/off suite aggregates are identical and four thresholds fail; **not promotable or ship** ([results](design/iter-e233-component-edges-20260716.md)) |
| E234 edge decision alignment | `e234-edge-decision-alignment-32step` | CPU HF-context legal-decision alignment diagnostic | `outputs/autoresearch/e234-edge-decision-alignment/runs/e234-edge-decision-alignment-32step/checkpoints/last.pt` (local) | Decision accuracy learns and changes five choices, but edge on/off suite aggregates are identical and four thresholds fail; **not promotable or ship** ([results](design/iter-e234-edge-decision-alignment-20260716.md)) |
| E235 binder-instance plan | `e235-binder-instance-plan-32step` | CPU HF-context grammar-binder planning diagnostic | `outputs/autoresearch/e235-binder-instance-plan/runs/e235-binder-instance-plan-32step/checkpoints/last.pt` (local) | Binder accuracy learns with full bound-row coverage and changes four choices, but on/off suite aggregates are identical and nine thresholds fail; **not promotable or ship** ([results](design/iter-e235-binder-instance-plan-20260716.md)) |
| E236 binder topology | `e236-binder-topology-32step` | CPU HF-context binder-reference diagnostic | `outputs/autoresearch/e236-binder-topology/runs/e236-binder-topology-32step/checkpoints/last.pt` (local) | Topology objective fails to learn, changes zero of 38 applied choices, and semantic metrics collapse; twelve thresholds fail; **not promotable or ship** ([results](design/iter-e236-binder-topology-20260716.md)) |
| E237 detached topology | `e237-detached-topology-32step` | CPU HF-context gradient-routing diagnostic | `outputs/autoresearch/e237-detached-topology/runs/e237-detached-topology-32step/checkpoints/last.pt` (local) | Detaching already-frozen context is a no-op and exactly reproduces E236; twelve thresholds fail; **not promotable or ship** ([results](design/iter-e237-detached-topology-20260716.md)) |
| E238 binder arity (invalidated) | `e238-binder-arity-32step` | CPU HF-context arity diagnostic | `outputs/autoresearch/e238-binder-arity/runs/e238-binder-arity-32step/checkpoints/last.pt` (local) | Optional-head RNG shifted matched stochastic draws; ten thresholds fail and causal training comparison is invalid; **not promotable or ship** ([results](design/iter-e238-binder-arity-confounded-20260716.md)) |
| E239 isolated binder arity | `e239d-binder-arity-fully-isolated-32step` | CPU HF-context isolated arity diagnostic | `outputs/autoresearch/e239-binder-arity-corrected/runs/e239d-binder-arity-fully-isolated-32step/checkpoints/last.pt` (local) | 104/104 shared tensors are bit-exact against control; 29 changed choices improve smoke syntax only, meaningful rate stays 0 and eleven thresholds fail; **not promotable or ship** ([results](design/iter-e239-binder-arity-isolated-20260716.md)) |
| E249 exact-event CE plus margin | `qx_e249_local_ce_margin` | CPU HF-context exact-state preference diagnostic | `outputs/autoresearch/e249-local-ce-margin/runs/qx_e249_local_ce_margin/checkpoints/last.pt` (local) | Held-out lexical chosen win rises 0→0.7649, but structure/reward regress on every suite and AgentV is 0/5; rejected, **not promotable or ship** ([results](design/iter-e249-local-ce-margin-20260716.md)) |
| E252 verifier-backed set FTPO | `qx_e252_local_ftpo_set` | CPU HF-context judged counterfactual preference diagnostic | `outputs/autoresearch/e252-ftpo-set/runs/qx_e252_local_ftpo_set/checkpoints/last.pt` (local) | Syntax stays 1.0, but fidelity collapses to 0, structure/reward regress on every suite, 13 thresholds fail, and AgentV is 0/5; rejected, **not promotable or ship** ([results](design/iter-e252-ftpo-set-20260716.md)) |
| E277 broad gold-AST set FTPO | `qx_e262_broad_gold_ast_ftpo_set` | CPU HF-context judged exact-state preference diagnostic | `outputs/autoresearch/e262-broad-gold-ast-ftpo/runs/qx_e262_broad_gold_ast_ftpo_set/checkpoints/last.pt` (local) | Executed as E262 before concurrent ID reconciliation; syntax/fidelity match E248, but held-out FTPO loss worsens, structure regresses on every suite, 10 thresholds fail, and AgentV is 0/5; rejected, **not promotable or ship** ([results](design/iter-e277-broad-gold-ast-ftpo-20260716.md)) |
| E278 guarded gold-AST set FTPO | `qx_e278_guarded_gold_ast_ftpo_set` | CPU HF-context guarded exact-state preference diagnostic | `outputs/autoresearch/e278-guarded-gold-ast-ftpo/runs/qx_e278_guarded_gold_ast_ftpo_set/checkpoints/last.pt` (local) | No trained step passed the four-metric held-out Pareto guard; step 0 was restored and all 374 tensors are bit-identical to E228. Current-code parent control exactly reproduces the five failing gates; no model gain, **not promotable or ship** ([results](design/iter-e278-guarded-gold-ast-ftpo-20260716.md)) |
| E265 safe gold-AST set FTPO | `qx_e265_safe_gold_ast_ftpo_set` | CPU HF-context backtracked exact-state preference diagnostic | `outputs/autoresearch/e265-safe-gold-ast-ftpo/runs/qx_e265_safe_gold_ast_ftpo_set/checkpoints/last.pt` (local) | 3/30 proposals improve the aggregate held-out Pareto guard, but decision-kind regressions are masked, fidelity/reward fall on most suites, five gates fail, and AgentV is 2/5; rejected, **not promotable or ship** ([results](design/iter-e265-safe-gold-ast-ftpo-20260717.md)) |
| E266 stratified safe set FTPO | `qx_e266_stratified_safe_gold_ast_ftpo_set` | CPU HF-context decision-kind-stratified preference diagnostic | `outputs/autoresearch/e266-stratified-safe-gold-ast-ftpo/runs/qx_e266_stratified_safe_gold_ast_ftpo_set/checkpoints/last.pt` (local) | All 30 proposals fail at least one grammar/AST decision-kind guard; parent is restored bit-identically, current control reproduces all metrics, five gates fail, and AgentV is 2/5; **not promotable or ship** ([results](design/iter-e266-stratified-safe-ftpo-20260717.md)) |
| E267 block-coordinate safe set FTPO | `qx_e267_block_stratified_safe_gold_ast_ftpo_set` | CPU HF-context decision-kind block preference diagnostic | `outputs/autoresearch/e267-block-stratified-safe-ftpo/runs/qx_e267_block_stratified_safe_gold_ast_ftpo_set/checkpoints/last.pt` (local) | All 30 category-block proposals fail the stratified guard; parent is restored bit-identically, full evaluation matches E266/current control, five gates fail, and AgentV is 2/5; **not promotable or ship** ([results](design/iter-e267-block-stratified-ftpo-20260717.md)) |
| E268 projected safe set FTPO | `qx_e268_projected_stratified_safe_gold_ast_ftpo_set` | CPU HF-context conflict-projected preference diagnostic | `outputs/autoresearch/e268-projected-stratified-safe-ftpo/runs/qx_e268_projected_stratified_safe_gold_ast_ftpo_set/checkpoints/last.pt` (local) | PCGrad projects 2,220 conflicting ordered task pairs, but all 30 proposals fail the stratified guard; parent is restored bit-identically, five gates fail, and AgentV is 2/5; **not promotable or ship** ([results](design/iter-e268-projected-stratified-ftpo-20260717.md)) |
| E269 MGDA safe set FTPO | `qx_e269_mgda_stratified_safe_gold_ast_ftpo_set` | CPU HF-context minimum-norm preference preflight | `outputs/autoresearch/e269-mgda-one-step-final/runs/qx_e269_mgda_stratified_safe_gold_ast_ftpo_set/checkpoints/last.pt` (local) | MGDA certifies common train-objective descent, but every scale regresses held-out decision kinds; parent is restored, five gates fail, and AgentV is 2/5; **not promotable or ship** ([results](design/iter-e269-mgda-stratified-ftpo-20260717.md)) |
| E272 MGDA plus SGD preflight | `qx_e272_mgda_sgd_stratified_safe_gold_ast_ftpo_set` | CPU HF-context metric-completeness preflight | `outputs/autoresearch/e272-mgda-sgd-one-step/runs/qx_e272_mgda_sgd_stratified_safe_gold_ast_ftpo_set/checkpoints/last.pt` (local) | Collinear SGD improves aggregate held-out FTPO loss, but every scale regresses per-kind probability/margin metrics; parent restored, five gates fail, AgentV 2/5; **not promotable or ship** ([results](design/iter-e272-mgda-sgd-preflight-20260717.md)) |
| E174 unfrozen-context 8-step control | `e174-unfrozen-context-8step` | CPU HF-context semantic control | `outputs/runs/e174-unfrozen-context-8step/checkpoints/last.pt` (local) | Unfrozen context, loss 39.4253; bounded probe syntax 0.0 and parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e174-unfrozen-context-20260716.md)) |
| Matrix honest champion (scratch) | `qx_e53_*` (V6 E53 family) | CPU scratch matrix clear | Primarily `outputs/runs/` (+ docs matrix JSON) | Honest `--ship-gates` on limited `rico_held` n; **not** production HF ship |
| P13 fixture E50 control | `qx_e50_core_remask` | CPU scratch, fixture corpus | `/tmp/slm17-e50-fixture-honest/` (local) | Matched control; held 0.08 / RICO 0.0667 fidelity; parse 0.0, not ship |
| P13 integrated E50 candidate | `qx_e50_core_remask` | CPU scratch, integrated corpus | `/tmp/slm17-e50-new-honest/` (local) | Strict fidelity gain on both smoke suites; parse 0.0, not promotable or ship |
| Frozen X2 baseline | `gx_x2_codec` seeds 0/1/2 | Retired fixed-canvas grammar diffusion | `/tmp/slm-training-fixed-baseline/outputs/topology_baseline/` (local) | 80 steps; all suites parse/fidelity/structure/reward 0.0; comparison only, not ship |
| Topology implementation smoke | `grammar_diffusion_overfit` | CPU scratch fixture topology v2 | pytest temporary checkpoint (local) | 200 steps; smoke n=2 parse/fidelity 0.5, topology composite 0.482; not reusable or ship |
| Topology X9/X14 confirmation | `gx_x9_topology_base`, `gx_x14_buffer` seeds 0/1/2 | CPU scratch topology v2 matrix | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` (local) | 200 steps; all 6 fail multi-suite gates; not promoted or synced |
| ScopeDiff X18 confirmation | `gx_x18_scope_noise_confirm_200` seeds 0/1/2 | CPU scratch topology v2 matrix | `outputs/runs/gx_x18_scope_noise_confirm_200/` (local) | 200 steps; all-suite median parse/fidelity 0.0; not promoted or synced |
| ScopeDiff X21 confirmation | `gx_x21_scoped_topology_confirm_200` seeds 0/1/2 | CPU scratch topology v2 matrix | `outputs/runs/gx_x21_scoped_topology_confirm_200/` (local) | 200 steps; weak structure, parse/fidelity 0.0; not promoted or synced |
| EFS0-04 X22 reproduction | `gx_x22_kapur_tree_edit_s0` | CPU scratch tree-edit diffusion | `outputs/runs/gx_x22_kapur_tree_edit_s0/checkpoints/last.pt` (local) | 80-step seed-0 audit-material replay; SHA `a9cfb450…02ff6`; syntax 1.0 but meaningful parse 0.333/0.2/0/0/0.667; ship gates fail, no sync or promotion ([results](design/iter-efs0-04-x22-reproduction-20260717.md)) |
| B3 five-minute lexer control | `capacity_lexer_v1__d64_h2_c1_dn2_t5000_x1__s0` | CPU scratch matched-capacity control | `outputs/ladders/b3-matched-5m-e287-r2/runs/capacity_lexer_v1__d64_h2_c1_dn2_t5000_x1__s0/checkpoints/last.pt` (local) | 53 steps / 5,004 target tokens; all-suite parse/meaningful/fidelity 0.0; AgentV 0/5 — **not promotable or ship** ([results](design/iter-b3-capacity-ladder-20260717.md)) |
| B3 five-minute choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` | CPU scratch matched-capacity choice codec | `outputs/ladders/b3-matched-5m-e287-r2/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/checkpoints/last.pt` (local) | E288 frozen eval restores deterministic parse 1.0 on all suites, but meaningful/fidelity remain 0.0 and AgentV 0/5 — **not promotable or ship** ([results](design/iter-e288-choice-native-gate-20260717.md)) |
| E289 cached choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` | CPU scratch matched-capacity choice codec | `outputs/ladders/e289-choice-state-cache/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/checkpoints/last.pt` (local) | Same SHA as E288; exact symbolic-state cache preserves all-suite parse 1.0 and improves p50 2.65×–5.86×, but meaningful/fidelity and AgentV remain zero — **not promotable or ship** ([results](design/iter-e289-choice-state-cache-20260717.md)) |
| E290 direct-candidate choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` | CPU scratch matched-capacity choice codec | `outputs/ladders/e290-choice-direct-candidates/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/checkpoints/last.pt` (local) | Same SHA as E288/E289; exact grammar-derived candidates preserve parse 1.0 and improve p95 1.14×–1.19× but regress p50; semantic metrics and AgentV remain zero — **not promotable or ship** ([results](design/iter-e290-choice-direct-candidates-20260717.md)) |
| E291 completion-cached choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` | CPU scratch matched-capacity choice codec | `outputs/ladders/e291-choice-completion-cache/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/checkpoints/last.pt` (local) | Same SHA as E288–E290; exact completion caching improves p50 1.29×–1.99× and p95 1.51×–1.93× vs E290, but semantic metrics and AgentV remain zero — **not model-promotable or ship** ([results](design/iter-e291-choice-completion-cache-20260717.md)) |
| E292 complete-loss choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` | CPU scratch matched-capacity choice codec | `outputs/ladders/e292-choice-loss-suite-complete-r2/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/checkpoints/last.pt` (local) | Same SHA as E288–E291; fixed metric classification makes all five loss categories complete (weighted NLL 7.2265, binding NLL 8.0201); honest ship board has parse 1.0 but meaningful 0.0 and AgentV 0/5 — **not promotable or ship** ([results](design/iter-e292-choice-loss-suite-completeness-20260717.md)) |
| E293 choice-native component plan | `e293-choice-component-plan-r3` | CPU scratch matched-capacity semantic diagnostic | `outputs/runs/e293-choice-component-plan-r3/checkpoints/last.pt` (local) | Plan loss improves root accuracy/bound recall to 0.5 and legal decode bias reduces gate failures 17→13, but matched no-DESIGN meaningful rate stays 0.0 and AgentV 0/5 — **not promotable or ship** ([results](design/iter-e293-choice-component-plan-20260717.md)) |
| E294 no-DESIGN choice control | `e294-choice-no-design-control-r1` | CPU scratch matched-capacity no-plan control | `outputs/runs/e294-choice-no-design-control-r1/checkpoints/last.pt` (local) | Complete weighted NLL 7.4977; honest board exactly matches E293 decode-off (meaningful 0.0, AgentV 0/5, 17 failures), isolating E293's gain to its decode head — **not promotable or ship** ([results](design/iter-e294-no-design-plan-control-20260717.md)) |
| E295 DESIGN-dropout choice arm | `e295-choice-design-dropout-r1` | CPU scratch matched-capacity 50% DESIGN dropout | `outputs/runs/e295-choice-design-dropout-r1/checkpoints/last.pt` (local) | Complete weighted NLL 7.3785; prompt-only adversarial meaningful 0.25, AgentV 1/5, but four suites remain at 0.0 and 14 gates fail — **not promotable or ship** ([results](design/iter-e295-design-context-dropout-20260717.md)) |
| E396 durable diagnostic checkpoint | `e396-balanced-type-head-continuation-r1` | CPU frozen SmolLM2 full-state continuation | `hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1/` | Exact SHA `feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`; bucket artifacts verified. E498 restores current-main loading and records 20 learned head applications, improving smoke structure 0.17197→0.27057, but meaningful/recall/reward remain zero and AgentV fails. **Durable and load-compatible diagnostic; not champion, promotable, or ship** ([train](design/iter-e396-e399-balanced-type-supervision-20260718.md), [branch-only gates](design/iter-e490-e396-json-number-typed-any-full-ship-gates-20260718.md), [current-main diagnostic](design/iter-e498-current-main-slot-component-restore-20260718.md)) |
| E499 diverse-root control | `e499-remediated-roots-hf-choice-control-r4` | CPU frozen SmolLM2 bounded corpus control | `outputs/runs/e499-remediated-roots-hf-choice-control-r4/checkpoints/last.pt` (local) | 10 steps / 1,023 target tokens; smoke n=1 syntax 1.0, structure 0.1542, component recall 0.25, meaningful/fidelity/reward 0.0, AgentV 0/1. SHA `bb4bec5f…f359fb6`; **diagnostic, not promotable or ship** ([results](design/iter-e499-strict-corpus-bounded-sft-20260718.md)) |
| E499 strict-r4 candidate | `e499-strict-r4-hf-choice-candidate-r4` | CPU frozen SmolLM2 bounded strict-corpus candidate | `outputs/runs/e499-strict-r4-hf-choice-candidate-r4/checkpoints/last.pt` (local) | Matched 9 steps / 1,034 target tokens; smoke structure regresses to 0.0375 and recall to 0.0, AgentV 0/1. SHA `81b2cb66…bcfbaf1`; rejected, **not promotable or ship** ([results](design/iter-e499-strict-corpus-bounded-sft-20260718.md)) |
| E499 choice-compatible strict candidate | `e499-choice-compatible-strict-hf-choice-candidate-r6` | CPU frozen SmolLM2 bounded document-only strict candidate | `outputs/runs/e499-choice-compatible-strict-hf-choice-candidate-r6/checkpoints/last.pt` (local) | 9 steps / 1,091 target tokens; 67/67 codec-compatible rows and 5.88s smoke p50, but structure 0.0375, recall/meaningful/fidelity/reward 0.0, AgentV 0/1. SHA `7230ace9…2e2fab`; rejected, **not promotable or ship** ([results](design/iter-e499-strict-corpus-bounded-sft-20260718.md)) |
| E500 1k document control | `e500-document-control-hf-choice-r1` | CPU frozen SmolLM2 bounded document control | `outputs/runs/e500-document-control-hf-choice-r1/checkpoints/last.pt` (local) | 9 steps / 1,028 target tokens; loss 30.3844, smoke syntax 1.0 and structure 0.0375 with semantic metrics zero, AgentV 0/1. SHA `a40f39a5…772d6834`; **diagnostic, not promotable or ship** ([results](design/iter-e500-documentized-expression-corpus-20260718.md)) |
| E500 1k projected candidate | `e500-documentized-expression-hf-choice-r2` | CPU frozen SmolLM2 bounded projected corpus | `outputs/runs/e500-documentized-expression-hf-choice-r2/checkpoints/last.pt` (local) | 11 steps / 1,039 target tokens; loss 27.6250 but smoke exactly matches the control's red semantic metrics, AgentV 0/1. SHA `f54cea08…773d3f0`; rejected, **not promotable or ship** ([results](design/iter-e500-documentized-expression-corpus-20260718.md)) |
| E500 5k document control | `e500-document-control-hf-choice-r3-5k` | CPU frozen SmolLM2 bounded document control | `outputs/runs/e500-document-control-hf-choice-r3-5k/checkpoints/last.pt` (local) | 43 steps / 5,040 target tokens; loss 10.5529, smoke syntax 1.0 and structure 0.0375 with semantic metrics zero, AgentV 0/1. SHA `9f752ae0…0b2b53`; **diagnostic, not promotable or ship** ([results](design/iter-e500-documentized-expression-corpus-20260718.md)) |
| E500 5k projected candidate | `e500-documentized-expression-hf-choice-r4-5k` | CPU frozen SmolLM2 bounded projected corpus | `outputs/runs/e500-documentized-expression-hf-choice-r4-5k/checkpoints/last.pt` (local) | 50 steps / 5,062 target tokens; loss regresses to 12.6778 and smoke matches the control's red semantic metrics, AgentV 0/1. SHA `a0ed6a58…dda5623`; rejected, **not promotable or ship** ([results](design/iter-e500-documentized-expression-corpus-20260718.md)) |
| E501 task-balanced 5k warm-start | `e501-e396-e500-init-r1` | CPU frozen SmolLM2 E396→E500 diagnostic | `outputs/runs/e501-e396-e500-init-r1/checkpoints/last.pt` (local) | 96 steps / 5,060 target tokens; structure regresses 0.2117→0.1458 and semantic metrics remain zero, AgentV 0/1. SHA `f86b83d3…cc9cf15`; rejected, **not promotable or ship** ([results](design/iter-e501-e396-e500-warm-start-20260719.md)) |
| E501 uniform 5k warm-start | `e501-e396-e500-uniform-init-r2` | CPU frozen SmolLM2 generation-heavy E396→E500 diagnostic | `outputs/runs/e501-e396-e500-uniform-init-r2/checkpoints/last.pt` (local) | 99 steps / 5,019 target tokens; recall reaches 0.1667 but structure collapses to 0.0889 and meaningful/fidelity/reward remain zero, AgentV 0/1. SHA `14605459…736e4e7`; rejected, **not promotable or ship** ([results](design/iter-e501-e396-e500-warm-start-20260719.md)) |
| E501 uniform 1k warm-start | `e501-e396-e500-uniform-init-r3-1k` | CPU frozen SmolLM2 short E396→E500 diagnostic | `outputs/runs/e501-e396-e500-uniform-init-r3-1k/checkpoints/last.pt` (local) | 22 steps / 1,039 target tokens; structure improves slightly 0.2117→0.2317 but all semantic metrics remain zero, AgentV 0/1. SHA `d84d34c0…b5be2ffd`; diagnostic only, **not promotable or ship** ([results](design/iter-e501-e396-e500-warm-start-20260719.md)) |
| E502 1e-4 warm-start | `e502-e396-e500-uniform-lr1e4-r1` | CPU frozen SmolLM2 lower-LR diagnostic | `outputs/runs/e502-e396-e500-uniform-lr1e4-r1/checkpoints/last.pt` (local) | 22 steps / 1,039 tokens; structure 0.1133, recall 0.1667, semantic metrics zero, AgentV 0/1. SHA `fcd51266…f047255e`; rejected, **not promotable or ship** ([results](design/iter-e502-initialization-prior-retention-20260719.md)) |
| E502 3e-5 warm-start | `e502-e396-e500-uniform-lr3e5-r2` | CPU frozen SmolLM2 lower-LR diagnostic | `outputs/runs/e502-e396-e500-uniform-lr3e5-r2/checkpoints/last.pt` (local) | 22 steps / 1,039 tokens; structure 0.1167, recall 0.0833, semantic metrics zero, AgentV 0/1. SHA `528c86a6…a62677c`; rejected, **not promotable or ship** ([results](design/iter-e502-initialization-prior-retention-20260719.md)) |
| E502 retained-prior 1k | `e502-e396-e500-prior-retained-lr3e4-r3` | CPU frozen SmolLM2 prior-retention diagnostic | `outputs/runs/e502-e396-e500-prior-retained-lr3e4-r3/checkpoints/last.pt` (local) | 22 steps / 1,039 tokens; structure 0.3169 and recall 0.0833, but semantic metrics zero and AgentV 0/1. SHA `e1e833cb…0746cb6a`; diagnostic only, **not promotable or ship** ([results](design/iter-e502-initialization-prior-retention-20260719.md)) |
| E502 retained-prior 5k | `e502-e396-e500-prior-retained-lr3e4-r4-5k` | CPU frozen SmolLM2 prior-retention stress diagnostic | `outputs/runs/e502-e396-e500-prior-retained-lr3e4-r4-5k/checkpoints/last.pt` (local) | 99 steps / 5,019 tokens; structure collapses to 0.0927 with recall 0.1667 and semantic metrics zero, AgentV 0/1. SHA `6f937374…4a46a726`; rejected, **not promotable or ship** ([results](design/iter-e502-initialization-prior-retention-20260719.md)) |
| E503 0% retention control | `e503-e396-e500-retention0-r1-5k` | CPU frozen SmolLM2 initialized-weight control | `outputs/runs/e503-e396-e500-retention0-r1-5k/checkpoints/last.pt` (local) | 99 steps / 5,019 tokens; RMS drift 0.003123, structure 0.0927, recall 0.1667, semantic metrics zero, AgentV 0/1. SHA `af6e9b1c…8a0af431`; rejected, **not promotable or ship** ([results](design/iter-e503-initialized-weight-retention-20260719.md)) |
| E503 1% retention | `e503-e396-e500-retention001-r2-5k` | CPU frozen SmolLM2 initialized-weight diagnostic | `outputs/runs/e503-e396-e500-retention001-r2-5k/checkpoints/last.pt` (local) | RMS drift 0.002071, structure 0.0900, recall 0.1667, semantic metrics zero, AgentV 0/1. SHA `7c5f016f…1be75711`; rejected, **not promotable or ship** ([results](design/iter-e503-initialized-weight-retention-20260719.md)) |
| E503 5% retention | `e503-e396-e500-retention005-r3-5k` | CPU frozen SmolLM2 initialized-weight diagnostic | `outputs/runs/e503-e396-e500-retention005-r3-5k/checkpoints/last.pt` (local) | RMS drift 0.000811 and structure 0.2029, but recall and semantic metrics are zero, AgentV 0/1. SHA `4093f1aa…af8d2031`; rejected, **not promotable or ship** ([results](design/iter-e503-initialized-weight-retention-20260719.md)) |
| E503 3% retention | `e503-e396-e500-retention003-r4-5k` | CPU frozen SmolLM2 initialized-weight midpoint | `outputs/runs/e503-e396-e500-retention003-r4-5k/checkpoints/last.pt` (local) | RMS drift 0.001163, structure 0.1667, recall 0.0833, semantic metrics zero, AgentV 0/1. SHA `2dbb52db…5751455b`; rejected, **not promotable or ship** ([results](design/iter-e503-initialized-weight-retention-20260719.md)) |
| E504 0% replay control | `e504-e396-e500-replay000-r1-5k` | CPU frozen SmolLM2 replay control | `outputs/runs/e504-e396-e500-replay000-r1-5k/checkpoints/last.pt` (local) | RMS drift 0.003123, structure 0.0927, recall 0.1667, semantic metrics zero, AgentV 0/1. SHA `35cd38e0…56334c87`; rejected, **not promotable or ship** ([results](design/iter-e504-parent-corpus-replay-20260719.md)) |
| E504 12.5% parent replay | `e504-e396-e500-replay0125-r2-5k` | CPU frozen SmolLM2 E357 replay diagnostic | `outputs/runs/e504-e396-e500-replay0125-r2-5k/checkpoints/last.pt` (local) | Structure 0.1558, recall zero, semantic metrics zero, AgentV 0/1. SHA `da63b403…1b725d3`; rejected, **not promotable or ship** ([results](design/iter-e504-parent-corpus-replay-20260719.md)) |
| E504 25% parent replay | `e504-e396-e500-replay025-r3-5k` | CPU frozen SmolLM2 E357 replay diagnostic | `outputs/runs/e504-e396-e500-replay025-r3-5k/checkpoints/last.pt` (local) | Structure 0.0964, recall 0.0833, semantic metrics zero, AgentV 0/1. SHA `91ab3f73…c3d85b4`; rejected, **not promotable or ship** ([results](design/iter-e504-parent-corpus-replay-20260719.md)) |
| E504 50% parent replay | `e504-e396-e500-replay050-r4-5k` | CPU frozen SmolLM2 E357 replay diagnostic | `outputs/runs/e504-e396-e500-replay050-r4-5k/checkpoints/last.pt` (local) | RMS drift 0.002796 and structure 0.2469, but recall 0.0833 and semantic metrics zero, AgentV 0/1. SHA `7d7e056e…c90294f9`; rejected, **not promotable or ship** ([results](design/iter-e504-parent-corpus-replay-20260719.md)) |
| E504 50% replay + 1% retention | `e504-e396-e500-replay050-retention001-r5-5k` | CPU frozen SmolLM2 interaction diagnostic | `outputs/runs/e504-e396-e500-replay050-retention001-r5-5k/checkpoints/last.pt` (local) | RMS drift 0.001775, but structure collapses to 0.0634 and semantic metrics remain zero, AgentV 0/1. SHA `1fc2fc23…a36036c`; rejected, **not promotable or ship** ([results](design/iter-e504-parent-corpus-replay-20260719.md)) |
| E505 50% replay loss attribution | `e505-e396-e500-replay050-loss-attribution-r1-5k` | CPU frozen SmolLM2 source-loss diagnostic | `outputs/runs/e505-e396-e500-replay050-loss-attribution-r1-5k/checkpoints/last.pt` (local) | Primary/replay loss proxies both decline; matched structure 0.2469 and recall 0.0833, but meaningful/fidelity/reward zero, AgentV 0/1. SHA `8fd11acd…525967e8`; rejected, **not promotable or ship** ([results](design/iter-e505-replay-loss-attribution-20260719.md)) |
| E513 durable slot-role continuation | `e513-e396-e500-replay050-slotrole4-focal2-r3-5k` | CPU frozen SmolLM2 slot-role diagnostic | `hf://buckets/TKendrick/OpenUI/checkpoints/e513-e396-e500-replay050-slotrole4-focal2-r3-5k/` | 101 steps / 5,000 target tokens in 79.6s under the three-minute cap; bucket verified, SHA `59253c67…a88a9548`. E514 OOD meaningful 0.0, fidelity 0.4917, structure 0.2750, AgentV 0/1; rejected, **durable diagnostic only, not promotable or ship** ([results](design/iter-e513-slot-role-supervision-20260719.md)) |
| E515 focal-zero slot-role control | `e515-e396-e500-replay050-slotrole4-focal0-r1-5k` | CPU frozen SmolLM2 focal-loss diagnostic | `hf://buckets/TKendrick/OpenUI/checkpoints/e515-e396-e500-replay050-slotrole4-focal0-r1-5k/` | 101 steps / 5,000 target tokens in 105.8s under the three-minute cap; bucket verified, SHA `97f2e426…24721c1b`. E516 OOD meaningful 0.25, fidelity 0.6583, structure 0.3213, AgentV 0/1; focal 2 rejected and this control **not promotable or ship** ([results](design/iter-e515-focal-loss-decomposition-20260719.md)) |
| E517 slot-loss-1 context control | `e517-e396-e500-replay050-slotrole1-context-r1-5k` | CPU frozen SmolLM2 context interaction diagnostic | `hf://buckets/TKendrick/OpenUI/checkpoints/e517-e396-e500-replay050-slotrole1-context-r1-5k/` | 101 steps / 5,000 target tokens in 130.7s under the three-minute cap; bucket verified, SHA `2b572a04…e24b60e3`. E518 OOD meaningful 0.0, fidelity 0.4083, structure 0.2250, AgentV 0/1; rejected, **durable diagnostic only, not promotable or ship** ([results](design/iter-e517-slot-loss-context-control-20260719.md)) |
| E519 honest slot-context control | `e519-e396-e500-replay050-slotrole1-honest-context-r1-5k` | CPU frozen SmolLM2 authority diagnostic | `hf://buckets/TKendrick/OpenUI/checkpoints/e519-e396-e500-replay050-slotrole1-honest-context-r1-5k/` | 101 steps / 5,000 target tokens in 103.2s from clean harness v7; bucket verified, SHA `d82155b0…6c91805f`. E520 exactly matches E518 quality (meaningful 0.0, fidelity 0.4083, structure 0.2250, AgentV 0/1); honest path retained, checkpoint **not promotable or ship** ([results](design/iter-e519-honest-slot-context-20260719.md)) |
| E522 visible-inventory continuation | `e522-e396-e521-replay050-slotrole1-honest-context-r2-5k` | CPU frozen SmolLM2 data-authority diagnostic | `hf://buckets/TKendrick/OpenUI/checkpoints/e522-e396-e521-replay050-slotrole1-honest-context-r2-5k/` | 99 steps / 5,059 target tokens in 120.7s; bucket verified, SHA `97cb10f4…bf420ce`. E523 fidelity rises to 0.8667 and recall to 0.2708, but meaningful remains 0.0, structure falls to 0.1955, and AgentV is 0/1; **not promotable or ship** ([results](design/iter-e522-visible-slot-continuation-20260719.md)) |
| E525 visible-component continuation | `e525-e396-e524-replay050-slotrole1-honest-context-r2-5k` | CPU frozen SmolLM2 conditional-contract diagnostic | `hf://buckets/TKendrick/OpenUI/checkpoints/e525-e396-e524-replay050-slotrole1-honest-context-r2-5k/` | 99 steps / 5,059 target tokens in 76.7s; bucket verified, SHA `dbd11811…e55e4b9`. E526 recall rises to 0.4167, but fidelity falls to 0.4667, structure to 0.1452, meaningful remains 0.0, and AgentV is 0/1; **not promotable or ship** ([results](design/iter-e525-visible-component-continuation-20260719.md)) |
| E528 visible-component-types continuation | `e528-e396-e527-replay050-slotrole1-honest-context-r1-5k` | CPU frozen SmolLM2 type-contract diagnostic | `hf://buckets/TKendrick/OpenUI/checkpoints/e528-e396-e527-replay050-slotrole1-honest-context-r1-5k/` | 99 steps / 5,059 target tokens in 146.8s; bucket verified, SHA `6a2180d7…306976d5`. E529 meaningful reaches 0.25 and reward 0.5778, but structure falls to 0.1136, strict meaning is 0.0, and AgentV is 0/1; **not promotable or ship** ([results](design/iter-e528-visible-component-types-continuation-20260719.md)) |
| E531 visible-semantic-role continuation | `e531-e396-e530-replay050-slotrole1-honest-context-r1-5k` | CPU frozen SmolLM2 semantic-role diagnostic | `hf://buckets/TKendrick/OpenUI/checkpoints/e531-e396-e530-replay050-slotrole1-honest-context-r1-5k/` | 99 steps / 5,059 target tokens in 99.72s; bucket verified, SHA `6b8c1abc…74a6154`. E532 structure reaches 0.1431, but meaningful is 0.0, fidelity 0.4667, reward 0.3685, strict meaning 0.0, and AgentV 0/1; **not promotable or ship** ([results](design/iter-e531-visible-semantic-roles-continuation-20260719.md)) |
| E542 learned root-arity continuation | `e542-e531-root-reference-arity1-r1-24s` | CPU frozen SmolLM2 learned-topology scratch diagnostic | `outputs/runs/e542-e531-root-reference-arity1-r1-24s/checkpoints/last.pt` (local) | 24 steps / 1,270 target tokens in 52.93s; SHA `2d5cd4b3…6854c5d8`. OOD `n=4` control reaches meaningful 0.50, fidelity 0.5917, structure 0.3019, reward 0.7950, but learned weight 1 is quality-neutral, strict meaning 0.0, and AgentV 0/1; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e542-learned-root-reference-arity-20260719.md)) |
| E543 bounded root-arity continuation | `e543-e531-root-reference-bounded-r1-24s` | CPU frozen SmolLM2 bounded-topology scratch diagnostic | `outputs/runs/e543-e531-root-reference-bounded-r1-24s/checkpoints/last.pt` (local) | 24 steps / 1,270 target tokens in 37.17s; SHA `c6be3791…51d7f90`. Bounded loss sharply improves head calibration, but OOD `n=4` decisions and quality exactly match E542, strict meaning is 0.0, and AgentV is 0/1; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e543-bounded-root-reference-arity-20260719.md)) |
| E544 root-reference identity continuation | `e544-e543-root-identity1-r2-24s` | CPU frozen SmolLM2 bounded-identity scratch diagnostic | `outputs/runs/e544-e543-root-identity1-r2-24s/checkpoints/last.pt` (local) | 24 steps / 1,270 target tokens in 40.96s; SHA `3b6e3c00…474f20c`. Same-checkpoint rank-only identity decoding improves OOD `n=4` meaningful 0.00→0.25, structure 0.1250→0.1688, recall 0.1458→0.2708, and AST node F1 0.1833→0.2833, but strict meaning is 0.0 and AgentV 0/1; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e544-root-reference-identity-20260719.md)) |
| E545 identity negative-weight-1 control | `e545-e544-root-identity-neg1-control-r1-24s` | CPU frozen SmolLM2 matched class-weight scratch diagnostic | `outputs/runs/e545-e544-root-identity-neg1-control-r1-24s/checkpoints/last.pt` (local) | 24 steps / 1,270 target tokens in 30.64s; SHA `9e54d470…76fa1`. OOD `n=4` meaningful 0.0, structure 0.1494, recall 0.2083, strict meaning 0.0, AgentV 0/1; regresses from E544, explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e545-root-reference-negative-weight-20260719.md)) |
| E545 identity negative-weight-4 treatment | `e545-e544-root-identity-neg4-r2-24s` | CPU frozen SmolLM2 matched class-weight scratch diagnostic | `outputs/runs/e545-e544-root-identity-neg4-r2-24s/checkpoints/last.pt` (local) | 24 steps / 1,270 target tokens in 28.64s; SHA `14dd4404…61ae`. Sparse late negative accuracy improves 0.3333→0.3958, but predictions and every OOD metric exactly match the control; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e545-root-reference-negative-weight-20260719.md)) |
| E546 strict-subset multiplier-1 control | `e546-e544-strict-subset1-control-r1-24s` | CPU frozen SmolLM2 matched sampling scratch diagnostic | `outputs/runs/e546-e544-strict-subset1-control-r1-24s/checkpoints/last.pt` (local) | 24 steps / 1,270 target tokens in 29.10s; SHA `46aba904…0fc55`. OOD `n=4` meaningful 0.0, fidelity 0.4250, structure 0.1494, recall 0.2083, AgentV 0/1; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e546-root-reference-coverage-sampling-20260719.md)) |
| E546 strict-subset multiplier-5 treatment | `e546-e544-strict-subset5-r2-24s` | CPU frozen SmolLM2 matched sampling scratch diagnostic | `outputs/runs/e546-e544-strict-subset5-r2-24s/checkpoints/last.pt` (local) | 24 steps / 1,318 target tokens in 30.50s; SHA `a1a6bfc9…b4efe2`. OOD fidelity, structure, reward, and AST F1 improve but recall falls to 0.0625; meaning 0.0 and AgentV 0/1; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e546-root-reference-coverage-sampling-20260719.md)) |
| E547 strict-subset multiplier-2 treatment | `e547-e544-strict-subset2-r1-24s` | CPU frozen SmolLM2 moderate sampling scratch diagnostic | `outputs/runs/e547-e544-strict-subset2-r1-24s/checkpoints/last.pt` (local) | 24 steps / 1,304 target tokens in 36.48s; SHA `37002bfd…0fc57`. OOD structure 0.2248 and AST node F1 0.3270 lead the multiplier ladder while recall stays 0.2083, but fidelity falls to 0.2583, meaning 0.0, AgentV 0/1; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e547-root-reference-coverage2-20260719.md)) |
| E551 no-lexeme-prior treatment | `e551-e544-strict-subset2-no-lexeme-r1-24s` | CPU frozen SmolLM2 prior-calibration scratch diagnostic | `outputs/runs/e551-e544-strict-subset2-no-lexeme-r1-24s/checkpoints/last.pt` (local) | 24 steps / 1,304 target tokens in 41.85s; SHA `e7921e66…dac32fc6`. Fidelity improves to 0.3000, but structure falls to 0.1594 and recall to 0.1250; meaning 0.0, AgentV 0/1; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e551-slot-lexeme-prior0-20260719.md)) |
| E552 half-strength lexeme-prior treatment | `e552-e544-strict-subset2-lexeme05-r1-24s` | CPU frozen SmolLM2 prior-calibration scratch diagnostic | `outputs/runs/e552-e544-strict-subset2-lexeme05-r1-24s/checkpoints/last.pt` (local) | 24 steps / 1,304 target tokens in 34.75s; SHA `49a9c111…a151fc04`. OOD fidelity 0.1333, structure 0.2181, recall 0.1250, reward 0.3435; meaning 0.0, AgentV 0/1; explicit no-sync scratch, **not promotable or ship** ([results](design/iter-e552-slot-lexeme-prior05-20260719.md)) |
| CAP5 evidence package | `cap5-03-evidence` | CAP0–CAP4 reproducible evidence package | `docs/design/calculated-arity-adaptive-precision-results.md` | Reproducible exact-calculation fixtures, claim ledger, artifact index, and negative-result registry; **not a checkpoint or ship claim** ([results](design/calculated-arity-adaptive-precision-results.md)) |
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
| Verified-solver decode | `verified_solver_decode` (VSS1-03) **off by default**: opt-in certificate-checked exact-closure pruning of the compiler-tree forest before soft ranking, on the DSL-native path only. **Experimental and unmeasured — no checkpoint uses it and it carries no ship/quality claim** ([config glossary](design/quality-experiment-matrix.md#configuration-glossary--verified-solver-decode-vss1-03)). |
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
| `ood` (`e542-e531-root-reference-arity1-r1-24s`, diagnostic subset) | 4 | 1.0 | 0.5917 | 0.3019 | 0.7950 | No — meaningful-v1 0.50, strict-v2 0.0, AgentV 0/1; learned weight 1 exactly matches control |
| `ood` (`e543-e531-root-reference-bounded-r1-24s`, diagnostic subset) | 4 | 1.0 | 0.5917 | 0.3019 | 0.7950 | No — bounded training improves head calibration but decisions and quality exactly match E542; meaningful-v1 0.50, strict-v2 0.0, AgentV 0/1 |
| `ood` (`e544-e543-root-identity1-r2-24s`, diagnostic subset) | 4 | 1.0 | 0.4333 | 0.1688 | 0.7370 | No — rank-only identity weight 1 raises meaningful-v1 0.00→0.25 and recall 0.1458→0.2708 versus same-checkpoint control, but strict-v2 0.0, AST edge F1 0.0, AgentV 0/1 |
| `ood` (`e545-e544-root-identity-neg1-control-r1-24s`, diagnostic subset) | 4 | 1.0 | 0.4250 | 0.1494 | 0.5078 | No — meaningful-v1 0.0, strict-v2 0.0, AST edge F1 0.0, AgentV 0/1; regresses from E544 |
| `ood` (`e545-e544-root-identity-neg4-r2-24s`, diagnostic subset) | 4 | 1.0 | 0.4250 | 0.1494 | 0.5078 | No — byte-identical to weight-1 control; increased negative weight is quality-neutral, AgentV 0/1 |
| `ood` (`e546-e544-strict-subset1-control-r1-24s`, diagnostic subset) | 4 | 1.0 | 0.4250 | 0.1494 | 0.5078 | No — meaningful-v1/strict-v2 0.0, recall 0.2083, AgentV 0/1 |
| `ood` (`e546-e544-strict-subset5-r2-24s`, diagnostic subset) | 4 | 1.0 | 0.6083 | 0.2038 | 0.8120 | No — AST edge F1 0.0417, but recall regresses to 0.0625, meaningful-v1/strict-v2 0.0, AgentV 0/1 |
| `ood` (`e547-e544-strict-subset2-r1-24s`, diagnostic subset) | 4 | 1.0 | 0.2583 | 0.2248 | 0.5403 | No — recall 0.2083 and AST node F1 0.3270, but meaningful-v1/strict-v2 and AST edge F1 0.0, AgentV 0/1 |
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
| `smoke` (B3 lexer control) | 3 | 0.0 | 0.0 | 0.0125 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `held_out` (B3 lexer control) | 5 | 0.0 | 0.0 | 0.1166 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `adversarial` (B3 lexer control) | 4 | 0.0 | 0.0 | 0.0346 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `ood` (B3 lexer control) | 4 | 0.0 | 0.0 | 0.0833 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `rico_held` (B3 lexer control) | 3 | 0.0 | 0.0 | 0.2528 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `smoke` (B3 choice arm) | 3 | 0.0 | 0.0 | 0.0 | 0.0 | No — empty predictions; AgentV row failed |
| `held_out` (B3 choice arm) | 5 | 0.0 | 0.0 | 0.0 | 0.0 | No — empty predictions; AgentV row failed |
| `adversarial` (B3 choice arm) | 4 | 0.0 | 0.0 | 0.0 | 0.0 | No — empty predictions; AgentV row failed |
| `ood` (B3 choice arm) | 4 | 0.0 | 0.0 | 0.0 | 0.0 | No — empty predictions; AgentV row failed |
| `rico_held` (B3 choice arm) | 3 | 0.0 | 0.0 | 0.0 | 0.0 | No — empty predictions; AgentV row failed |
| `smoke` (E288 choice-native gate) | 3 | 1.0 | 0.0 | 0.3094 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `held_out` (E288 choice-native gate) | 5 | 1.0 | 0.0 | 0.2514 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `adversarial` (E288 choice-native gate) | 4 | 1.0 | 0.0 | 0.2905 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `ood` (E288 choice-native gate) | 4 | 1.0 | 0.0 | 0.2369 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `rico_held` (E288 choice-native gate) | 3 | 1.0 | 0.0 | 0.0901 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `smoke` (E289 cached choice) | 3 | 1.0 | 0.0 | 0.3094 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `held_out` (E289 cached choice) | 5 | 1.0 | 0.0 | 0.2514 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `adversarial` (E289 cached choice) | 4 | 1.0 | 0.0 | 0.2905 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `ood` (E289 cached choice) | 4 | 1.0 | 0.0 | 0.2369 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `rico_held` (E289 cached choice) | 3 | 1.0 | 0.0 | 0.0901 | 0.0 | No — meaningful 0.0; AgentV row failed |
| `smoke` (E290 direct candidates) | 3 | 1.0 | 0.0 | 0.3094 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `held_out` (E290 direct candidates) | 5 | 1.0 | 0.0 | 0.2514 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `adversarial` (E290 direct candidates) | 4 | 1.0 | 0.0 | 0.2905 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `ood` (E290 direct candidates) | 4 | 1.0 | 0.0 | 0.2369 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `rico_held` (E290 direct candidates) | 3 | 1.0 | 0.0 | 0.0901 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `smoke` (E291 completion cache) | 3 | 1.0 | 0.0 | 0.3094 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `held_out` (E291 completion cache) | 5 | 1.0 | 0.0 | 0.2514 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `adversarial` (E291 completion cache) | 4 | 1.0 | 0.0 | 0.2905 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `ood` (E291 completion cache) | 4 | 1.0 | 0.0 | 0.2369 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `rico_held` (E291 completion cache) | 3 | 1.0 | 0.0 | 0.0901 | 0.0 | No — meaningful 0.0; AgentV rows failed |
| `smoke` (E292 complete-loss honest eval) | 3 | 1.0 | 0.7222 | 0.2958 | 0.0 | No — meaningful/component recall 0.0; AgentV row failed |
| `held_out` (E292 complete-loss honest eval) | 5 | 1.0 | 0.4800 | 0.2784 | 0.1414 | No — meaningful 0.0, component recall 0.04; AgentV row failed |
| `adversarial` (E292 complete-loss honest eval) | 4 | 1.0 | 0.4167 | 0.2330 | 0.0 | No — meaningful/component recall 0.0; AgentV row failed |
| `ood` (E292 complete-loss honest eval) | 4 | 1.0 | 0.1667 | 0.2731 | 0.0 | No — meaningful/component recall 0.0; AgentV row failed |
| `rico_held` (E292 complete-loss honest eval) | 3 | 1.0 | 0.0000 | 0.0901 | 0.0 | No — meaningful/component recall 0.0; AgentV row failed |
| `smoke` (E293 matched plan) | 3 | 1.0 | 0.7222 | 0.2681 | 0.0000 | No — meaningful/component recall 0.0; AgentV row failed |
| `held_out` (E293 matched plan) | 5 | 1.0 | 0.5600 | 0.3328 | 0.1474 | No — meaningful 0.0, component recall 0.04; AgentV row failed |
| `adversarial` (E293 matched plan) | 4 | 1.0 | 0.8333 | 0.2843 | 0.0000 | No — meaningful/component recall 0.0; AgentV row failed |
| `ood` (E293 matched plan) | 4 | 1.0 | 0.5167 | 0.3617 | 0.1893 | No — meaningful 0.0, component recall 0.0625; AgentV row failed |
| `rico_held` (E293 matched plan) | 3 | 1.0 | 0.2500 | 0.1381 | 0.0000 | No — meaningful/component recall 0.0; limited n=3 diagnostic |
| `smoke` (E294 no-DESIGN control) | 3 | 1.0 | 0.3333 | 0.3500 | 0.0000 | No — meaningful/component recall 0.0; AgentV row failed |
| `held_out` (E294 no-DESIGN control) | 5 | 1.0 | 0.0000 | 0.2514 | 0.0000 | No — meaningful/component recall/fidelity 0.0; AgentV row failed |
| `adversarial` (E294 no-DESIGN control) | 4 | 1.0 | 0.2500 | 0.2363 | 0.0000 | No — meaningful/component recall 0.0; AgentV row failed |
| `ood` (E294 no-DESIGN control) | 4 | 1.0 | 0.0000 | 0.2369 | 0.0000 | No — meaningful/component recall/fidelity 0.0; AgentV row failed |
| `rico_held` (E294 no-DESIGN control) | 3 | 1.0 | 0.0000 | 0.0901 | 0.0000 | No — meaningful/component recall 0.0; limited n=3 diagnostic |
| `smoke` (E295 DESIGN dropout) | 3 | 1.0 | 0.3333 | 0.3500 | 0.0000 | No — meaningful/component recall 0.0; AgentV row failed |
| `held_out` (E295 DESIGN dropout) | 5 | 1.0 | 0.0000 | 0.2514 | 0.0000 | No — meaningful/component recall/fidelity 0.0; AgentV row failed |
| `adversarial` (E295 DESIGN dropout) | 4 | 1.0 | 0.2500 | 0.2697 | 0.2343 | No — meaningful/component recall only 0.25; checkpoint fails overall |
| `ood` (E295 DESIGN dropout) | 4 | 1.0 | 0.0000 | 0.2369 | 0.0000 | No — meaningful/component recall/fidelity 0.0; AgentV row failed |
| `rico_held` (E295 DESIGN dropout) | 3 | 1.0 | 0.0000 | 0.0901 | 0.0000 | No — meaningful/component recall 0.0; limited n=3 diagnostic |
| `smoke` (E497 current-main playground audit) | 3 | 0.0 | 0.0 | 0.2203 | 0.0 | No — type recall 0.1667, AgentV 0/5, one timeout; fixture wiring only |
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
| `smoke` (`e232-role-component-plan-32step`) | 3 | 1.0 | 0.5278 | 0.4642 | 0.8073 | No — meaningful program 0.3333 |
| `held_out` (`e232-role-component-plan-32step`) | 5 | 1.0 | 0.1800 | 0.3335 | 0.5732 | No — meaningful program 0.0 |
| `adversarial` (`e232-role-component-plan-32step`) | 4 | 1.0 | 0.5417 | 0.4744 | 0.8115 | Suite passes; checkpoint still fails full gate set |
| `ood` (`e232-role-component-plan-32step`) | 4 | 1.0 | 0.2083 | 0.3469 | 0.5493 | No — meaningful program 0.0 |
| `rico_held` (`e232-role-component-plan-32step`) | 3 | 1.0 | 0.1250 | 0.1628 | 0.6865 | No — structure below gate; diagnostic n=3 |
| `smoke` (`e233-component-edges-32step`) | 3 | 1.0 | 0.5278 | 0.4642 | 0.8073 | No — meaningful program 0.3333 |
| `held_out` (`e233-component-edges-32step`) | 5 | 1.0 | 0.2800 | 0.3369 | 0.7330 | No — meaningful program 0.0 |
| `adversarial` (`e233-component-edges-32step`) | 4 | 1.0 | 0.4167 | 0.3895 | 0.6148 | No — meaningful program 0.25 |
| `ood` (`e233-component-edges-32step`) | 4 | 1.0 | 0.2583 | 0.3750 | 0.7265 | No — meaningful program 0.0 |
| `rico_held` (`e233-component-edges-32step`) | 3 | 1.0 | 0.1250 | 0.1628 | 0.6865 | No — structure below gate; diagnostic n=3 |
| `smoke` (`e234-edge-decision-alignment-32step`) | 3 | 1.0 | 0.5278 | 0.4642 | 0.8073 | No — meaningful program 0.3333 |
| `held_out` (`e234-edge-decision-alignment-32step`) | 5 | 1.0 | 0.2800 | 0.3369 | 0.7330 | No — meaningful program 0.0 |
| `adversarial` (`e234-edge-decision-alignment-32step`) | 4 | 1.0 | 0.2917 | 0.3619 | 0.5743 | No — meaningful program 0.25 |
| `ood` (`e234-edge-decision-alignment-32step`) | 4 | 1.0 | 0.2583 | 0.3750 | 0.7265 | No — meaningful program 0.0 |
| `rico_held` (`e234-edge-decision-alignment-32step`) | 3 | 1.0 | 0.1250 | 0.1628 | 0.6865 | No — structure below gate; diagnostic n=3 |
| `smoke` (`e235-binder-instance-plan-32step`) | 3 | 1.0 | 0.1111 | 0.4122 | 0.2497 | No — meaningful program, fidelity, and reward below gates |
| `held_out` (`e235-binder-instance-plan-32step`) | 5 | 1.0 | 0.0333 | 0.2739 | 0.1398 | No — meaningful program, structure, and fidelity below gates |
| `adversarial` (`e235-binder-instance-plan-32step`) | 4 | 1.0 | 0.2083 | 0.3556 | 0.3870 | Suite passes; checkpoint still fails full gate set |
| `ood` (`e235-binder-instance-plan-32step`) | 4 | 1.0 | 0.0000 | 0.2369 | 0.0000 | No — meaningful program and structure below gates |
| `rico_held` (`e235-binder-instance-plan-32step`) | 3 | 1.0 | 0.0833 | 0.1371 | 0.4577 | No — structure below gate; diagnostic n=3 |
| `smoke` (`e236-binder-topology-32step`) | 3 | 1.0 | 0.0000 | 0.3094 | 0.0000 | No — semantic metrics collapse |
| `held_out` (`e236-binder-topology-32step`) | 5 | 1.0 | 0.0000 | 0.2514 | 0.0000 | No — semantic metrics collapse |
| `adversarial` (`e236-binder-topology-32step`) | 4 | 1.0 | 0.0000 | 0.2905 | 0.0000 | No — meaningful program below gate |
| `ood` (`e236-binder-topology-32step`) | 4 | 1.0 | 0.0000 | 0.2369 | 0.0000 | No — semantic metrics collapse |
| `rico_held` (`e236-binder-topology-32step`) | 3 | 1.0 | 0.0000 | 0.0901 | 0.0000 | No — semantic metrics collapse; diagnostic n=3 |
| `smoke` (`e237-detached-topology-32step`) | 3 | 1.0 | 0.0000 | 0.3094 | 0.0000 | No — exact E236 reproduction |
| `held_out` (`e237-detached-topology-32step`) | 5 | 1.0 | 0.0000 | 0.2514 | 0.0000 | No — exact E236 reproduction |
| `adversarial` (`e237-detached-topology-32step`) | 4 | 1.0 | 0.0000 | 0.2905 | 0.0000 | No — exact E236 reproduction |
| `ood` (`e237-detached-topology-32step`) | 4 | 1.0 | 0.0000 | 0.2369 | 0.0000 | No — exact E236 reproduction |
| `rico_held` (`e237-detached-topology-32step`) | 3 | 1.0 | 0.0000 | 0.0901 | 0.0000 | No — exact E236 reproduction; diagnostic n=3 |
| `smoke` (`e238-binder-arity-32step`) | 3 | 0.6667 | 0.5278 | 0.2042 | 0.4770 | Invalidated — optional-head RNG confound |
| `held_out` (`e238-binder-arity-32step`) | 5 | 0.6000 | 0.3933 | 0.1326 | 0.4406 | Invalidated — optional-head RNG confound |
| `adversarial` (`e238-binder-arity-32step`) | 4 | 0.7500 | 0.8333 | 0.2174 | 0.6658 | Invalidated — optional-head RNG confound |
| `ood` (`e238-binder-arity-32step`) | 4 | 0.5000 | 0.3500 | 0.0888 | 0.3585 | Invalidated — optional-head RNG confound |
| `rico_held` (`e238-binder-arity-32step`) | 3 | 0.6667 | 0.1250 | 0.0291 | 0.4297 | Invalidated — optional-head RNG confound; diagnostic n=3 |
| `smoke` (`e239d-binder-arity-fully-isolated-32step`) | 3 | 0.3333 | 0.5833 | 0.2591 | 0.0000 | No — meaningful rate 0; 11 total failures |
| `held_out` (`e239d-binder-arity-fully-isolated-32step`) | 5 | 0.2000 | 0.5667 | 0.1338 | 0.0000 | No — meaningful rate 0 |
| `adversarial` (`e239d-binder-arity-fully-isolated-32step`) | 4 | 0.0000 | 0.8333 | 0.1912 | 0.0000 | No — syntax and meaningful rate 0 |
| `ood` (`e239d-binder-arity-fully-isolated-32step`) | 4 | 0.0000 | 0.5250 | 0.1775 | 0.0000 | No — syntax and meaningful rate 0 |
| `rico_held` (`e239d-binder-arity-fully-isolated-32step`) | 3 | 0.0000 | 0.3750 | 0.0971 | 0.0000 | No — syntax and meaningful rate 0; diagnostic n=3 |
| `smoke` (`qx_e249_local_ce_margin`) | 3 | 1.0000 | 0.5278 | 0.1742 | 0.7653 | No — meaningful rate below gate; structure regressed vs E248 |
| `held_out` (`qx_e249_local_ce_margin`) | 5 | 1.0000 | 0.2800 | 0.1088 | 0.6910 | No — meaningful rate 0; structure regressed vs E248 |
| `adversarial` (`qx_e249_local_ce_margin`) | 4 | 1.0000 | 0.5417 | 0.1927 | 0.7695 | No — structure below gate and regressed vs E248 |
| `ood` (`qx_e249_local_ce_margin`) | 4 | 1.0000 | 0.2167 | 0.1469 | 0.6720 | No — meaningful rate 0; structure regressed vs E248 |
| `rico_held` (`qx_e249_local_ce_margin`) | 3 | 1.0000 | 0.1250 | 0.0727 | 0.6445 | No — structure below gate; diagnostic n=3 |
| `smoke` (`qx_e252_local_ftpo_set`) | 3 | 1.0000 | 0 | 0.1353 | 0.6070 | No — fidelity collapsed; 13 total failures |
| `held_out` (`qx_e252_local_ftpo_set`) | 5 | 1.0000 | 0 | 0.1239 | 0.6070 | No — fidelity collapsed; structure regressed vs E248 |
| `adversarial` (`qx_e252_local_ftpo_set`) | 4 | 1.0000 | 0 | 0.0978 | 0.6070 | No — fidelity collapsed; structure regressed vs E248 |
| `ood` (`qx_e252_local_ftpo_set`) | 4 | 1.0000 | 0 | 0.1906 | 0.6070 | No — fidelity collapsed; structure regressed vs E248 |
| `rico_held` (`qx_e252_local_ftpo_set`) | 3 | 1.0000 | 0 | 0.0368 | 0.6070 | No — meaningful rate 0; diagnostic n=3 |
| `smoke` (`qx_e262_broad_gold_ast_ftpo_set`) | 3 | 1.0000 | 0.5278 | 0.1742 | 0.7653 | No — meaningful rate and structure below gate; 10 total failures |
| `held_out` (`qx_e262_broad_gold_ast_ftpo_set`) | 5 | 1.0000 | 0.2800 | 0.1088 | 0.6910 | No — meaningful rate 0; structure regressed vs E248 |
| `adversarial` (`qx_e262_broad_gold_ast_ftpo_set`) | 4 | 1.0000 | 0.5417 | 0.1927 | 0.7695 | No — structure below gate and regressed vs E248 |
| `ood` (`qx_e262_broad_gold_ast_ftpo_set`) | 4 | 1.0000 | 0.2583 | 0.1469 | 0.6845 | No — meaningful rate 0; structure regressed vs E248 |
| `rico_held` (`qx_e262_broad_gold_ast_ftpo_set`) | 3 | 1.0000 | 0.1250 | 0.0727 | 0.6445 | No — structure below gate; diagnostic n=3 |

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

### E499 bounded strict-corpus diagnostic

All rows use frozen local SmolLM2 context, the choice codec, no DESIGN context,
the same 1,000-target-token budget, and honest constrained smoke `n=1`.

| Checkpoint | n | Syntax | Meaningful | Fidelity | Structure | Component recall | Reward | AgentV | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e499-remediated-roots-hf-choice-control-r4` | 1 | 1.0 | 0.0 | 0.0 | 0.1542 | 0.25 | 0.0 | 0/1 | No |
| `e499-strict-r4-hf-choice-candidate-r4` | 1 | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | 0/1 | No |
| `e499-choice-compatible-strict-hf-choice-candidate-r6` | 1 | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | 0/1 | No |

This is a one-record scratch diagnostic, not a ship evaluation. All checkpoints
were explicitly kept local with `--no-sync-checkpoints`.

### E500 documentized-expression diagnostic

All rows use CPU, frozen local SmolLM2 context, the choice codec, seed 0,
batch 4, learning rate `3e-4`, no DESIGN context, an honest prompt-derived slot
contract, and a three-minute wall limit. The only matched variable is the
96-row document control versus the 260-row documentized-expression corpus.

| Checkpoint | Tokens | n | Syntax | Meaningful | Fidelity | Structure | Component recall | Reward | AgentV | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e500-document-control-hf-choice-r1` | 1,028 | 1 | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | 0/1 | No |
| `e500-documentized-expression-hf-choice-r2` | 1,039 | 1 | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | 0/1 | No |
| `e500-document-control-hf-choice-r3-5k` | 5,040 | 1 | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | 0/1 | No |
| `e500-documentized-expression-hf-choice-r4-5k` | 5,062 | 1 | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | 0/1 | No |

All four runs emitted AgentEvals JSONL and pinned AgentV result bundles without
execution errors. They are bounded diagnostics, not ship evaluations, and were
explicitly kept local with `--no-sync-checkpoints`.

### E501 E396-to-E500 warm-start diagnostic

All rows use the same frozen E396 parent, CPU/frozen local SmolLM2 context,
choice output, and honest constrained smoke `n=3`. Every process is capped at
170 seconds and every train summary records `max_wall_minutes=3.0`.

| Checkpoint | Sampling | Tokens | Syntax | Meaningful | Fidelity | Structure | Component recall | Reward | AgentV | Pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Frozen E396 parent | — | 0 | 1.0 | 0.0 | 0.0 | 0.2117 | 0.0 | 0.0 | 0/1 | No |
| `e501-e396-e500-init-r1` | equal task groups | 5,060 | 1.0 | 0.0 | 0.0 | 0.1458 | 0.0 | 0.0 | 0/1 | No |
| `e501-e396-e500-uniform-init-r2` | uniform records | 5,019 | 1.0 | 0.0 | 0.0 | 0.0889 | 0.1667 | 0.0 | 0/1 | No |
| `e501-e396-e500-uniform-init-r3-1k` | uniform records | 1,039 | 1.0 | 0.0 | 0.0 | 0.2317 | 0.0 | 0.0 | 0/1 | No |

### E502 checkpoint-prior retention diagnostic

All rows use CPU, frozen local SmolLM2-135M, the committed E500 corpus,
honest prompt-derived slots, constrained decode without fallback, smoke `n=3`,
and `max_wall_minutes=3.0`. Version stamp: code `3e4f906`, train v2 for the
lower-LR controls and v3 for retained-prior arms; eval v1, meaningful v2,
scoring v1, ship gates v1.

| Run | LR | Tokens | Priors restored | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AgentV | Promote |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e502-e396-e500-uniform-lr1e4-r1` | 1e-4 | 1,039 | No | 1.0 | 0.0 | 0.0 | 0.1133 | 0.1667 | 0.0 | 0/1 | No |
| `e502-e396-e500-uniform-lr3e5-r2` | 3e-5 | 1,039 | No | 1.0 | 0.0 | 0.0 | 0.1167 | 0.0833 | 0.0 | 0/1 | No |
| `e502-e396-e500-prior-retained-lr3e4-r3` | 3e-4 | 1,039 | Lexeme + span | 1.0 | 0.0 | 0.0 | 0.3169 | 0.0833 | 0.0 | 0/1 | No |
| `e502-e396-e500-prior-retained-lr3e4-r4-5k` | 3e-4 | 5,019 | Lexeme + span | 1.0 | 0.0 | 0.0 | 0.0927 | 0.1667 | 0.0 | 0/1 | No |

All four E502 checkpoints are rejected local diagnostics with explicit
`--no-sync-checkpoints`; the frozen parent remains the only bucket artifact.

### E503 initialized-weight retention diagnostic

All rows use the same E396 parent, E500 corpus, 5,019 target tokens, CPU/frozen
local SmolLM2 context, honest constrained smoke `n=3`, and external 170-second
process cap. Train v4 records the anchored parameter count and final RMS drift;
eval v1, meaningful v2, scoring v1, ship gates v1.

| Run | Retention | RMS drift | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AgentV | Promote |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e503-e396-e500-retention0-r1-5k` | 0.00 | 0.003123 | 1.0 | 0.0 | 0.0 | 0.0927 | 0.1667 | 0.0 | 0/1 | No |
| `e503-e396-e500-retention001-r2-5k` | 0.01 | 0.002071 | 1.0 | 0.0 | 0.0 | 0.0900 | 0.1667 | 0.0 | 0/1 | No |
| `e503-e396-e500-retention003-r4-5k` | 0.03 | 0.001163 | 1.0 | 0.0 | 0.0 | 0.1667 | 0.0833 | 0.0 | 0/1 | No |
| `e503-e396-e500-retention005-r3-5k` | 0.05 | 0.000811 | 1.0 | 0.0 | 0.0 | 0.2029 | 0.0 | 0.0 | 0/1 | No |

All four checkpoints are rejected local diagnostics with explicit
`--no-sync-checkpoints`; no promotion or production claim is made.

### E504 parent-corpus replay diagnostic

All rows use the same E396 parent, E500 primary corpus, approximately 5,000
target tokens, CPU/frozen local SmolLM2 context, honest constrained smoke
`n=3`, and external 170-second process cap. Train v5 records immutable replay
fingerprints and effective exposure; eval v1, meaningful v2, scoring v1, ship
gates v1.

| Run | Replay | Retention | RMS drift | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AgentV | Promote |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e504-e396-e500-replay000-r1-5k` | 0% | 0% | 0.003123 | 1.0 | 0.0 | 0.0 | 0.0927 | 0.1667 | 0.0 | 0/1 | No |
| `e504-e396-e500-replay0125-r2-5k` | 12.5% | 0% | 0.003219 | 1.0 | 0.0 | 0.0 | 0.1558 | 0.0 | 0.0 | 0/1 | No |
| `e504-e396-e500-replay025-r3-5k` | 25% | 0% | 0.003098 | 1.0 | 0.0 | 0.0 | 0.0964 | 0.0833 | 0.0 | 0/1 | No |
| `e504-e396-e500-replay050-r4-5k` | 50% | 0% | 0.002796 | 1.0 | 0.0 | 0.0 | 0.2469 | 0.0833 | 0.0 | 0/1 | No |
| `e504-e396-e500-replay050-retention001-r5-5k` | 50% | 1% | 0.001775 | 1.0 | 0.0 | 0.0 | 0.0634 | 0.0 | 0.0 | 0/1 | No |

All five checkpoints are rejected local diagnostics with explicit
`--no-sync-checkpoints`; the exact E357 data snapshot, not these checkpoints,
was persisted to the HF bucket.

### E505 replay-loss attribution diagnostic

The single train repeats E504's 50% replay recipe for 5,000 target tokens under
the external 170-second cap. Train v6 adds source-stratified masked-token loss
proxies without changing the objective.

| Run | Decode policy | Primary proxy first→last | Replay proxy first→last | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AgentV | Promote |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e505-e396-e500-replay050-loss-attribution-r1-5k` | Honest, slot-contract bias off | 3.8422→3.3724 | 3.4087→2.9217 | 1.0 | 0.0 | 0.0 | 0.2469 | 0.0833 | 0.0 | 0/1 | No |
| same checkpoint | Constrained slot contract | — | — | 1.0 | 0.0 | 0.1667 | 0.2039 | 0.0833 | 0.2623 | 0/1 | No |

Both source losses improve, while constrained slot-contract decode changes the
fidelity/structure tradeoff without clearing meaningful or AgentV gates. The
checkpoint is a rejected local diagnostic with explicit
`--no-sync-checkpoints`.

### E506 constrained slot-contract multi-suite diagnostic

Both rows evaluate the same rejected E505 checkpoint on all 13 held-out, OOD,
and adversarial records. CPU/frozen local SmolLM2 context, honest contracts,
grammar-constrained LTR decode, four generation steps, one attempt, no fallback,
and a 96-token canvas are matched. No new checkpoint was written.

| Decode | n | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | AgentV | Promote |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Contract off | 13 | 1.0 | 0.0 | 0.0 | 0.1271 | 0.1410 | 0.0 | 0.1524 | 0.0192 | 0/3 | No |
| Contract on | 13 | 1.0 | 0.1538 | 0.2538 | 0.1669 | 0.2654 | 0.5454 | 0.2385 | 0.0 | 0/3 | No |

The constraint improves six semantic/structural metrics, but AgentV remains
red and the 96-token canvas is below every suite's gold p95. This updates the
leading diagnostic inference policy only; it is not checkpoint promotion or a
ship claim.

### E507 length-safe OOD contract-decode diagnostic

Both rows evaluate the same E505 checkpoint on all four OOD records with a
160-token canvas, above gold p95 143. All other settings are matched.

| Decode | n | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | p50 latency | AgentV | Promote |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Contract off | 4 | 1.0 | 0.0 | 0.0 | 0.0729 | 0.0 | 0.0 | 0.0313 | 4,571 ms | 0/1 | No |
| Contract on | 4 | 1.0 | 0.25 | 0.2583 | 0.2281 | 0.3333 | 0.692 | 0.3389 | 6,966 ms | 0/1 | No |

Quality metrics exactly match E506's 96-token OOD rows, ruling out canvas
truncation as the source of the gain. AgentV remains red; no checkpoint was
created or promoted.

### E508 default-generation OOD contract-decode diagnostic

The same E505 checkpoint and length-safe constrained OOD policy use eight
generation steps and four attempts. All four records exactly reproduce E507
quality: syntax 1.0, meaningful 0.25, fidelity 0.2583, structure 0.2281, recall
0.3333, reward 0.692, AST node F1 0.3389, and AgentV 0/1. No checkpoint was
created or promoted.

### E509 honest slot-contract-context diagnostic

The same E505 checkpoint and E508 policy additionally expose the honest request
slot contract in model context. Structure improves 0.2281→0.2406 and
binding-aware coverage 0.75→1.0, but meaningfulness, fidelity, recall, reward,
AST F1, and AgentV 0/1 are unchanged. Inventory visibility does not resolve the
component/placeholder semantic-role failures. No checkpoint was created or
promoted.

### E510 component-plan-decode diagnostic

The same E505 checkpoint activates its trained component-plan head at decode
weight 4. Against E509 on the same four OOD records, meaningful improves
0.25→0.50, fidelity 0.2583→0.6583, structure 0.2406→0.3446, reward
0.6920→0.8405, AST node F1 0.3389→0.4679, and AST edge F1 0→0.1625. Strict
binding-aware meaning and AgentV 0/1 remain red. This is the leading diagnostic
policy, not a checkpoint promotion.

### E511 length-safe three-suite component-plan diagnostic

The E510 policy expands to all 13 held-out/OOD/adversarial records with a
192-token length-safe canvas. Aggregate meaningful is 0.3846, fidelity 0.6718,
structure 0.3440, recall 0.4615, reward 0.6272, AST node F1 0.4654, and AST edge
F1 0.1748. Strict binding-aware meaning remains zero and AgentV remains 0/3.
The policy generalizes diagnostically; the checkpoint is still rejected.

### E512 slot-to-component decode-weight diagnostic

Doubling the E505 checkpoint's slot-to-component decode weight from 4 to 8
reduces OOD placeholder-spam prevalence 3→1 but leaves semantic-role mismatch at
4/4. Meaningful regresses 0.50→0.25, fidelity 0.6583→0.3417, structure
0.3446→0.2869, reward 0.8405→0.7245, and AgentV stays 0/1. Weight 8 is rejected;
the checkpoint and weight 4 policy remain non-promotable.

### E513 durable slot-role supervision continuation

E513 warm-starts E396 on E500 with 50% exact E357 replay, raises
slot-component loss from 1 to 4, adds focal gamma 2, and supplies the honest
slot contract in context. The HF-context CPU run completes 101 steps / 5,000
target tokens in 79.6 seconds under `max_wall_minutes=3`. Its checkpoint SHA
`59253c679477060694370c5e2d8cd9fce5d7accc7d71df3b6d56edf0a88a9548`
and full state are uploaded and verified in the OpenUI bucket.

Matched E514 OOD evaluation under E510's component-plan weight-4 policy
regresses meaningful 0.50→0.00, fidelity 0.6583→0.4917, structure
0.3446→0.2750, recall 0.3958→0.2083, AST node F1 0.4679→0.3500, and AST edge
F1 0.1625→0.0625. Strict binding-aware meaning stays zero and AgentV stays
0/1. The checkpoint is retained as durable diagnostic evidence but rejected
for promotion.

### E515 focal-loss decomposition

E515 is matched to E513 except focal gamma returns from 2 to 0. The CPU
HF-context run completes 101 steps / 5,000 target tokens in 105.8 seconds under
`max_wall_minutes=3`; serving SHA
`97f2e426604e3956f2791398a608b967937ebf548fa7cae0ef59dde324721c1b`
and full state are uploaded and verified in the OpenUI bucket.

Matched E516 OOD evaluation recovers meaningful 0.00→0.25, fidelity
0.4917→0.6583, structure 0.2750→0.3213, recall 0.2083→0.2708, reward
0.7695→0.8270, and AST node F1 0.3500→0.4292 versus E513. It remains below
E510 on meaningfulness and component structure, while strict binding-aware
meaning and AgentV stay zero. Focal gamma 2 is rejected; the focal-zero
checkpoint remains diagnostic and is not promoted.

### E517 slot-loss context control

E517 is matched to E515 except slot-component loss returns from 4 to 1 while
focal gamma stays zero and honest contract context remains enabled during
training. The CPU HF-context run completes 101 steps / 5,000 target tokens in
130.7 seconds under `max_wall_minutes=3`; serving SHA
`2b572a04256db14095e813e146079af9e6f6c948963d60f2bd669855e24b60e3`
and full state are uploaded and verified in the OpenUI bucket.

Matched E518 OOD evaluation regresses meaningful 0.25→0.00, fidelity
0.6583→0.4083, structure 0.3213→0.2250, recall 0.2708→0.2083, reward
0.8270→0.7445, and AST node F1 0.4292→0.2833 versus E515. Strict
binding-aware meaning and AgentV stay zero. The loss and context interact, but
neither context-conditioned checkpoint is promotable.

### E519 honest slot-contract context

E519 adds `train_model --honest-slot-contract`, preventing training-time
context from using gold record placeholders and recording the authority flags
under train harness v7. The clean-source CPU HF-context run completes 101 steps
/ 5,000 target tokens in 103.2 seconds; serving SHA
`d82155b03531c2d852ec8d497d3fdb0878ac1f678c0c5d247e272bc36c91805f`
and full state are uploaded and verified.

E520 exactly matches E518 quality and decoder counts: meaningful 0.0, fidelity
0.4083, structure 0.2250, recall 0.2083, reward 0.7445, AST node F1 0.2833,
AST edge F1 0.0625, and AgentV 0/1. The checkpoint tensors do change, so the
authority path is operational, but it yields no observable quality gain. The
honest harness fix is retained; the checkpoint is rejected.

### E522 visible-inventory continuation

E522 replaces E500 with the 244-row E521 corpus, whose prompts expose every
declared placeholder. Every other E519 train/eval lever is held fixed. The
clean-source CPU HF-context run completes 99 steps / 5,059 target tokens in
120.7 seconds; serving SHA
`97cb10f43d229b1a15403295f71fa425e844ee4865c31761f3e529b24bf420ce`
and full state are uploaded and verified.

Matched E523 OOD fidelity rises 0.4083→0.8667, component recall
0.2083→0.2708, AST node F1 0.2833→0.3437, and AST edge F1
0.0625→0.1007. Structure regresses 0.2250→0.1955, reward
0.7445→0.2093, meaningful and strict meaning remain zero, and AgentV remains
0/1. Visible inventory is retained as a positive slot-grounding lever, but the
checkpoint is rejected.

### E525 visible-component continuation

E525 replaces E521 with membership-identical E524, which appends exact
component type/count inventories to every prompt. All other E522 train/eval
levers remain fixed. The CPU HF-context run completes 99 steps / 5,059 target
tokens in 76.7 seconds; serving SHA
`dbd11811d826fdf7efd8b22557fb3bd48f879e84ec7484bc0a2680198e55e4b9`
and full state are uploaded, independently listed, and verified.

Matched E526 OOD component recall rises 0.2708→0.4167, but fidelity falls
0.8667→0.4667, structure 0.1955→0.1452, AST node F1 0.3437→0.3041, and AST
edge F1 0.1007→0.0774. Meaningful and strict meaning remain zero and AgentV
remains 0/1. The count signal is learned but does not restore hierarchy, so the
checkpoint is rejected.

### E528 visible-component-types continuation

E528 replaces exact type/count inventories with membership-identical E527
type-only contracts. All other E525 train/eval levers remain fixed. The CPU
HF-context run completes 99 steps / 5,059 target tokens in 146.8 seconds;
serving SHA
`6a2180d76c366a282a74d1d27ae2b2fcf4c1b5f2b4d298cf4cef35bc306976d5`
and full state are automatically uploaded, independently listed, and verified.

Matched E529 OOD meaningful rate rises 0.0→0.25, fidelity 0.4667→0.55, and
reward 0.1668→0.5778 versus E525. Component recall falls 0.4167→0.3542,
structure 0.1452→0.1136, and AST node F1 0.3041→0.2270. Strict meaning remains
zero and AgentV remains 0/1. The weaker inventory signal is retained as
diagnostic evidence, but the checkpoint is rejected.

### E531 visible-semantic-role continuation

E531 replaces E527 type-only prompts with membership-identical E530 semantic
role contracts derived exclusively from already-visible slots, already-visible
types, and schema compatibility. All other E528 train/eval levers remain
fixed. The CPU HF-context run completes 99 steps / 5,059 target tokens in
99.72 seconds; serving SHA
`6b8c1abc56a36e8aa15acc373b61d5df033a753907330649e379d9ba374a6154`
and full state are uploaded, independently listed, and verified.

Matched E532 OOD structure rises 0.1136→0.1431 and AST node F1
0.2270→0.2543, but meaningful falls 0.25→0.0, fidelity 0.55→0.4667,
component recall 0.3542→0.2917, reward 0.5778→0.3685, and AST edge F1
0.0801→0.0455. Strict meaning remains zero and AgentV remains 0/1. The
semantic-role signal is retained as negative diagnostic evidence, and the
checkpoint is rejected.

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
| 2026-07-19 | `e552-e544-strict-subset2-lexeme05-r1-24s` | `outputs/runs/e552-e544-strict-subset2-lexeme05-r1-24s/` (local) | OOD fidelity 0.1333, structure 0.2181, recall 0.1250, AgentV 0/1 | Explicit no-sync scratch; midpoint rejected; not promoted |
| 2026-07-19 | `e551-e544-strict-subset2-no-lexeme-r1-24s` | `outputs/runs/e551-e544-strict-subset2-no-lexeme-r1-24s/` (local) | OOD fidelity 0.3000, structure 0.1594, recall 0.1250, AgentV 0/1 | Explicit no-sync scratch; prior removal rejected; not promoted |
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
| 2026-07-16 | `e232-role-component-plan-32step` (E232) | `outputs/autoresearch/e232-role-component-plan/runs/e232-role-component-plan-32step/` (local) | 32 CPU steps; root accuracy 1.0, bound recall 0.7083; syntax 1.0 and meaningful program 0.3333/0/0.5/0/0.6667 | Plan improves adversarial quality, but same 4 frontier thresholds fail; AgentV 1/5; checkpoint SHA `da42b9ea…be208e`; no sync or promotion |
| 2026-07-16 | `e233-component-edges-32step` (E233) | `outputs/autoresearch/e233-component-edges/runs/e233-component-edges-32step/` (local) | 32 CPU steps; edge recall 0→0.50; syntax 1.0 and meaningful program 0.3333/0/0.25/0/0.6667 | Edge on/off aggregates identical; 4 thresholds fail; AgentV 1/5; checkpoint SHA `46141ac1…f2575`; no sync or promotion |
| 2026-07-16 | `e234-edge-decision-alignment-32step` (E234) | `outputs/autoresearch/e234-edge-decision-alignment/runs/e234-edge-decision-alignment-32step/` (local) | 32 CPU steps; decision accuracy 0→0.5714; syntax 1.0 and meaningful program 0.3333/0/0.25/0/0.6667 | Edge on/off aggregates identical despite 5 changes; 4 thresholds fail; AgentV 1/5; checkpoint SHA `350b7c5c…0fc68`; no sync or promotion |
| 2026-07-16 | `e235-binder-instance-plan-32step` (E235) | `outputs/autoresearch/e235-binder-instance-plan/runs/e235-binder-instance-plan-32step/` (local) | 32 CPU steps; binder accuracy 0→0.40 with all 30 bound rows supervised; syntax 1.0 and meaningful program 0.3333/0/0.25/0/0.6667 | Binder on/off aggregates identical despite 4 changes; 9 thresholds fail; AgentV 1/5; checkpoint SHA `83adbccd…73ca8`; no sync or promotion |
| 2026-07-16 | `e236-binder-topology-32step` (E236) | `outputs/autoresearch/e236-binder-topology/runs/e236-binder-topology-32step/` (local) | 32 CPU steps; topology accuracy 0.5455→0.5238; syntax 1.0 but semantic metrics 0 throughout | Decode on/off identical with 0/38 choice changes; 12 thresholds fail; AgentV 0/5; checkpoint SHA `94e1d042…f8c43`; no sync or promotion |
| 2026-07-16 | `e237-detached-topology-32step` (E237) | `outputs/autoresearch/e237-detached-topology/runs/e237-detached-topology-32step/` (local) | 32 CPU steps; detached frozen context reproduces E236 train/eval diagnostics | No-op hypothesis rejected; 12 thresholds fail; AgentV 0/5; checkpoint SHA `edcbad06…4b59d`; no sync or promotion |
| 2026-07-16 | `e238-binder-arity-32step` (E238) | `outputs/autoresearch/e238-binder-arity/runs/e238-binder-arity-32step/` (local) | 32 CPU steps; arity loss 4.4211→2.9895; syntax 0.5–0.75 and meaningful program 0 | Invalidated by optional-head RNG confound; 10 thresholds fail; AgentV 0/5; checkpoint SHA `2f9acccc…3ff7a4`; no sync or promotion |
| 2026-07-16 | `e239d-binder-arity-fully-isolated-32step` (E239) | `outputs/autoresearch/e239-binder-arity-corrected/runs/e239d-binder-arity-fully-isolated-32step/` (local) | 32 CPU steps; arity loss 4.0988→2.4903, accuracy 0→0.4706; 104/104 shared tensors bit-exact | 29 decode changes but meaningful rate 0 on every suite; 11 thresholds fail; AgentV 0/5; checkpoint SHA `677e80ef…674d`; no sync or promotion |
| 2026-07-16 | `qx_e249_local_ce_margin` (E249) | `outputs/autoresearch/e249-local-ce-margin/runs/qx_e249_local_ce_margin/` (local) | 30 CPU exact-event steps; held-out chosen win 0→0.7649 and margin win 0→0.6489; syntax 1.0 on all suites | Structure/reward regress everywhere; 8 thresholds fail; AgentV 0/5; checkpoint SHA `24285bd4…264f32c`; scratch/no sync/no promotion |
| 2026-07-16 | `qx_e252_local_ftpo_set` (E252) | `outputs/autoresearch/e252-ftpo-set/runs/qx_e252_local_ftpo_set/` (local) | 30 CPU set-FTPO steps on 14 judged train events; held-out margin win 0→0.3333 while syntax stays 1.0 | Fidelity 0 and structure/reward regress on every suite; 13 thresholds fail; AgentV 0/5; checkpoint SHA `c01aebc2…088946`; scratch/no sync/no promotion |
| 2026-07-16 | `qx_e262_broad_gold_ast_ftpo_set` (E277; emitted E262 before ID reconciliation) | `outputs/autoresearch/e262-broad-gold-ast-ftpo/runs/qx_e262_broad_gold_ast_ftpo_set/` (local) | 30 CPU set-FTPO steps on 200 committed gold-AST train events; syntax/fidelity match E248 and AgentV publication resumed without retraining | Held-out loss worsens, structure regresses on every suite, 10 thresholds fail, AgentV 0/5; checkpoint SHA `3f6a2eb2…f760831b`; scratch/no sync/no promotion |
| 2026-07-16 | `qx_e278_guarded_gold_ast_ftpo_set` (E278) | `outputs/autoresearch/e278-guarded-gold-ast-ftpo/runs/qx_e278_guarded_gold_ast_ftpo_set/` (local) | 30 CPU set-FTPO steps with validation every 5 steps; no trained step passed the held-out loss/bad-mass/good-mass/mean-margin guard, so step 0 was restored | 374/374 tensors bit-identical to E228; current parent control reproduces all suite metrics and 5 failures; AgentV 2/5; serialized SHA `518d4736…91935ba`; no sync/no promotion |
| 2026-07-17 | `qx_e265_safe_gold_ast_ftpo_set` (E265) | `outputs/autoresearch/e265-safe-gold-ast-ftpo/runs/qx_e265_safe_gold_ast_ftpo_set/` (local) | 3/30 proposals accepted through optimizer-consistent Pareto backtracking; aggregate held-out loss `-0.0471`, bad mass `-0.000420`, good mass `+0.002570`, mean margin `+0.1452` | Per-kind regressions hide behind aggregate gains; fidelity/reward regress on most suites, 5 gates fail, AgentV 2/5; 50m09s stage; checkpoint SHA `44079a8c…a846ab`; no sync/no promotion |
| 2026-07-17 | `qx_e266_stratified_safe_gold_ast_ftpo_set` (E266) | `outputs/autoresearch/e266-stratified-safe-gold-ast-ftpo/runs/qx_e266_stratified_safe_gold_ast_ftpo_set/` (local) | 30 CPU proposals, 150 scales, and per-decision-kind Pareto guards; batched validation cuts the local stage to 79.77s from E265's 3,009.05s | 0/30 accepted; 374/374 tensors match E228; matched current control reproduces all metrics, 5 gates fail, AgentV 2/5; serialized SHA `518d4736…91935ba`; no sync/no promotion |
| 2026-07-17 | `qx_e267_block_stratified_safe_gold_ast_ftpo_set` (E267) | `outputs/autoresearch/e267-block-stratified-safe-ftpo/runs/qx_e267_block_stratified_safe_gold_ast_ftpo_set/` (local) | 30 CPU category-block proposals across 14 decision kinds and 150 scales; batched stage 90.27s | 0/30 accepted; 374/374 tensors match E228 and full evaluation matches current control; 5 gates fail, AgentV 2/5; serialized SHA `518d4736…91935ba`; no sync/no promotion |
| 2026-07-17 | `qx_e268_projected_stratified_safe_gold_ast_ftpo_set` (E268) | `outputs/autoresearch/e268-projected-stratified-safe-ftpo/runs/qx_e268_projected_stratified_safe_gold_ast_ftpo_set/` (local) | 30 CPU steps, 420 task gradients, 2,220/5,460 conflicting ordered pairs projected, and 150 scales; stage 2,338.56s | 0/30 accepted; model tensors match E228 and full evaluation matches current control; 5 gates fail, AgentV 2/5; serialized SHA `518d4736…91935ba`; no sync/no promotion |
| 2026-07-17 | `qx_e269_mgda_stratified_safe_gold_ast_ftpo_set` (E269) | `outputs/autoresearch/e269-mgda-one-step-final/runs/qx_e269_mgda_stratified_safe_gold_ast_ftpo_set/` (local) | One CPU preflight step; 13 active decision-kind gradients; minimum-norm common-descent certificate; stage 219.11s | 0/1 accepted and all five scales regress held-out kinds; 30-step run canceled, parent restored; 5 gates fail, AgentV 2/5; SHA `518d4736…91935ba`; no sync/no promotion |
| 2026-07-17 | `qx_e272_mgda_sgd_stratified_safe_gold_ast_ftpo_set` (E272) | `outputs/autoresearch/e272-mgda-sgd-one-step/runs/qx_e272_mgda_sgd_stratified_safe_gold_ast_ftpo_set/` (local) | One CPU MGDA plus SGD preflight; 13 active loss objectives; stage 214.96s | 0/1 accepted; aggregate loss improves but all scales regress nine per-kind probability/margin guards; parent restored, 5 gates fail, AgentV 2/5; SHA `518d4736…91935ba`; no sync/no promotion |
| 2026-07-16 | A1 emptiness probe (E248, diagnostic — no new checkpoint) | `playground_demo` fixture via `scripts/probe_emptiness.py`; evidence `docs/design/iter-e248-emptiness-probe-20260716.{md,json}` | On the wiring fixture, empty program preferred on total AND per-token NLL (verdict `content_modeling_failure`); probe validated end-to-end | Diagnostic tool for the E224+ wall; real verdict needs the local E224+ checkpoints (gitignored). Fixture result is wiring-only, not a frontier finding |
| 2026-07-16 | `qx_e255_b4_scratch_control` / `qx_e256_b4_ar_adapt` (V11 B4 pair) | `outputs/runs/` (local, not synced); evidence `docs/design/iter-e255-e256-b4-ar-adaptation-20260716.md` + v10 campaign JSONs | 200 CPU steps, fixture v1 corpus, matched pair differing only in `denoiser_backend`; adaptation (SmolLM2-135M, bidirectional 4D-mask) trails the 1.1M scratch control on every signal (loss 8.51 vs 3.75; lr=3e-5 probe worse at 9.72); syntax/meaningful parse 0.0 on both | B4 verdict OPEN — fixture budget can neither confirm nor kill the from-scratch assumption; decisive run needs GPU-scale matched-compute arms with per-arm LR. Wiring-only; no promotion |
| 2026-07-16 | `qx_e257_c1_relative_bind` (V11 C1) | `outputs/runs/` (local, not synced); evidence `docs/design/iter-e257-c1-relative-bind-20260716.md` + `quality-matrix-results-iter-v10-c1-20260716.json` | Matched vs E255 (only `bind_encoding=relative` differs): syntax parse 0.667/0.6/0.25/0.5 vs 0.0, loss 3.27 vs 3.75, decode p50 1–8s vs ~15s; meaningful parse 0.0 on both (failures shift to `empty_root_stack`) | Nameless `<BINDDEF>` + signed `<BINDREL_±k>` refs, scope legality verifier-enforced; fixture wiring evidence only, frontier E-row + C1×A interaction open; no promotion |
| 2026-07-17 | `qx_e280_c3_macro_tokens` (V16 C3) | `outputs/runs/` (local, not synced); evidence `docs/design/iter-e280-c3-macro-tokens-20260717.md` + `quality-matrix-results-iter-v16-c3-20260717.json` | 80 CPU steps, fixture v1 corpus; 16 mined `<MACRO_i>` macros (tokenizer v3), corpus −34.3% tokens incl. table, `seen_target_tokens` −34.4% at the matched recipe; table round-trips through the checkpoint sidecar; syntax/meaningful parse 0.0, loss 5.61 | Deterministic lossless expansion, fixed-vocabulary spans only (alpha-independent); wiring evidence only, no matched no-macro control row in-run; frontier matched pair open; no promotion |
| 2026-07-17 | `qx_e281_c4_anon_control` / `qx_e282_c4_surface_ids` (V17 C4 pair) | `outputs/runs/` (local, not synced); evidence `docs/design/iter-e281-e282-c4-names-disappear-20260717.md` + `quality-matrix-results-iter-v17-c4-20260717.json` | 80 CPU steps, fixture v1 corpus, matched pair differing only in `symbol_anonymization`; both arms unconstrained decode; syntax/meaningful parse 0.0 on both; structural similarity favors the surface arm on 5/5 suites (0.23–0.11 vs 0.12–0.03) at 1.72× target length | C4 verdict OPEN — primary metric never leaves zero at fixture budget; secondary signal is a small adverse data point for the C1–C3 anonymization defense; decisive test needs a frontier-scale replicated pair. Wiring-only; no promotion |
| 2026-07-17 | `gx_x22_kapur_tree_edit` (D3 Kapur baseline, vs matched X9) | `outputs/runs/` (local, not synced); evidence `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` + `grammar-matrix-results-iter-x22-kapur-20260717.json` | 80 CPU steps, seed 0, fixture v1; all-valid tree-edit forward + inverse-edit policy + value-guided search; syntax parse 1.0 by construction; meaningful parse 0.2/0.25/0.25 (held_out/adversarial/ood — first nonzero at this budget; both those gate rows passed) vs 0.0 everywhere for matched X9 | Kapur arXiv:2405.20519 now Faithful (mechanism); observation channel prompt-conditioned (no target render exists); full gate battery still fails (smoke + rico rows); single seed, wiring-only, no promotion |
| 2026-07-17 | `gx_x22_kapur_tree_edit_s0` (EFS0-04 audit-material reproduction) | `outputs/runs/gx_x22_kapur_tree_edit_s0/checkpoints/last.pt` (local, no sync); evidence `docs/design/iter-efs0-04-x22-reproduction-20260717.md` + `grammar-matrix-results-iter-efs0-04-x22-replay-20260717.json` | 80 CPU scratch steps, seed 0; checkpoint SHA `a9cfb450e8146089cb26b6df84e90a5073627c4e59a2933d16f69034ec802ff6`; syntax 1.0; meaningful parse 0.333/0.2/0/0/0.667 across smoke/held/adversarial/OOD/RICO (`n=3/5/4/4/3`) | Reproduces raw X22 material for a blinded study; unchanged gates fail; fixture-grade, single-seed, local-only; no external/human judgment, sync, or promotion |
| 2026-07-17 | G2 recipe-evolution fixture campaign (`g2_fixture_20260717`, no checkpoint kept) | `outputs/experiments/g2_fixture_20260717/` (local); evidence `docs/design/iter-g2-recipe-evolution-20260717.md` + `recipe-evolution-results-iter-g2-20260717.json` | 4 unique recipes, 2 generations, 20 CPU steps each; evolve→train→eval→gate-checked-select loop ran end-to-end; `promotable: false` (no candidate passed the frozen gates) | Gate-locked selection proved load-bearing immediately: NLL fitness monotonically rewards driving `fidelity_loss_weight`→0, exactly the failure the frozen gates block; decode-only genes are NLL-fitness-neutral. Wiring-only; no gate weakened; nothing promoted |
| 2026-07-17 | G4 reasoning bench fixture (`g4_fixture_20260717`, no checkpoint kept) | `outputs/experiments/reasoning_bench/` (local); evidence `docs/design/iter-g4-reasoning-bench-20260717.md` + `reasoning-bench-results-iter-g4-20260717.json` | 96-record arith-sketch corpus, 24 held-out problems, 120 CPU steps per arm; sketch (executed program trace) and direct (bare answer) arms both 0.0 accuracy; loop + single deterministic oracle proven end-to-end | First checkable-answer evaluator in the repo; sketch failures are forward refs + missing `root` (exactly what constrained decode fixes); direct arm collapses to a constant; PAL/PoT-analog comparison unanswered at this budget. Wiring-only; nothing promoted |
| 2026-07-16 | `qx_e240_compiler_tree_control` (V9 campaign E240–E247) | `outputs/runs/qx_e240_compiler_tree_control/` (local scratch control); evidence `docs/design/iter-e240-e247-lattice-campaign-20260716.md` + `quality-matrix-results-iter-v9-lattice-20260716.json` | 800 CPU steps on the 108-record fixture v1 corpus; all 8 V9 rows ran (E241–E247 eval-only from this frozen checkpoint via new `--eval-checkpoint`); syntax/meaningful parse 0.0 everywhere (placeholder-policy rejections); always-on PTRM (E244) = ~3× latency and lower structural similarity; triggered rows byte-identical to greedy | Fixture-grade wiring campaign only — all honest gates fail as expected; no sync or promotion; ship-grade V9 run needs local E224+ checkpoints on a GPU host with full suites (`rico_held` 1500) |
| 2026-07-17 | `qx_e277_a2_asap_decode` (V14 A2, eval-only — no new checkpoint) | eval overlay on frozen `qx_e255_b4_scratch_control` weights; evidence `docs/design/iter-e277-a2-asap-decode-20260717.md` + `quality-matrix-results-iter-v14-a2-20260717.json` | ASAp-style constraint-mass removal (`asap_decode`, GAD/ASAp adapted trie→canvas-position) live during decode: 204–334 penalties across 32–53 positions per suite, deterministic across two runs; structural similarity mixed vs E255 (adversarial +9pts, others −3–9pts at n≤5); syntax/meaningful parse 0.0 on both | Decode-only lever, matched pair vs E255; fixture wiring evidence only — the A1-diagnosed constraint distortion binds at frontier scale, so the A2 verdict needs the local E224+ checkpoints on a GPU host; no promotion |
| 2026-07-17 | `qx_e278_c2_pseudo_embeddings` (V15 C2) | `outputs/runs/` (local, not synced); evidence `docs/design/iter-e278-c2-pseudo-embeddings-20260717.md` + v13 JSON + binding probe JSON | 200 CPU steps, fixture v1, matched vs E255 (only `runtime_symbol_features="replace"` differs): structural similarity 0.19–0.29 vs 0.28–0.37 (honest fixture negative); binding probe on the trained checkpoint: same-surface hidden cosine 0.9998 vs cross 0.9679 (margin +0.032); run flushed + fixed a latent stale-feature leak (training_loss now clears request-local features) | DyVo-style deterministic byte-compositional symbol embeddings via the V8 delta path; fixture wiring + probe evidence only, frontier C2×C1 interaction open; no promotion |
| 2026-07-17 | `capacity_lexer_v1__d64_h2_c1_dn2_t5000_x1__s0` (B3 matched control) | `outputs/ladders/b3-matched-5m-e287-r2/runs/capacity_lexer_v1__d64_h2_c1_dn2_t5000_x1__s0/` (local) | 53 CPU scratch steps / 5,004 target tokens; weighted NLL 13.1800; parse/meaningful/fidelity 0.0 on all five suites | AgentV 0/5; invalid integrity-only promotion removed and gate fixed; scratch/no sync/no promotion |
| 2026-07-17 | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` (B3 matched choice; E288 reevaluation) | `outputs/ladders/b3-matched-5m-e287-r2/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/` (local) | 107 CPU scratch steps / 5,022 target tokens; E287 emitted 19 empty predictions; E288 same SHA restores parse 1.0 on every suite via production-codec/schema state | Meaningful/fidelity/reward remain 0.0, AgentV 0/5; decoder fix confirmed without retraining; scratch/no sync/no promotion |
| 2026-07-17 | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` (E289 exact choice-state cache) | `outputs/ladders/e289-choice-state-cache/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/` (local) | 107 CPU scratch steps / 5,022 target tokens; byte-identical to E288; exact state cache preserves parse 1.0 and zero dead ends while reducing p50 2.65×–5.86× | Cold p95 remains 5.9–8.7s; meaningful/fidelity/reward 0.0 and AgentV 0/5; scratch/no sync/no promotion |
| 2026-07-17 | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` (E290 direct candidates) | `outputs/ladders/e290-choice-direct-candidates/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/` (local) | 107 CPU scratch steps / 5,022 target tokens; byte-identical to E288/E289; grammar-derived candidates avoid 34.8% of cold probes and preserve parse 1.0 / zero dead ends | p95 improves 1.14×–1.19× but p50 regresses; semantic metrics 0.0 and AgentV 0/5 in both repeats; scratch/no sync/no promotion |
| 2026-07-17 | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` (E291 completion cache) | `outputs/ladders/e291-choice-completion-cache/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/` (local) | 107 CPU scratch steps / 5,022 target tokens; byte-identical to E288–E290; 90.7–91.9% exact completion-cache hit rate, p50 1.29×–1.99× and p95 1.51×–1.93× faster than E290 | Parse 1.0 / zero dead ends preserved; semantic metrics 0.0 and AgentV 0/5 in both repeats; scratch/no sync/no model promotion |
| 2026-07-17 | `capacity_choice_v1__d64_h2_c1_dn2_t1250_x1__s0` (E292 budget calibration) | `outputs/ladders/e292-choice-loss-suite-complete/runs/capacity_choice_v1__d64_h2_c1_dn2_t1250_x1__s0/` (local) | 26 CPU scratch steps / 1,263 target tokens; complete weighted NLL 18.5150; SHA `78334790da71535f1b65edd7073b8c66a21bda35d3da87cfd6d33ab1ece11211` | Base budget was divided across ladder cells; retained as calibration evidence only, scratch/no sync/no promotion |
| 2026-07-17 | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0` (E292 complete loss suite) | `outputs/ladders/e292-choice-loss-suite-complete-r2/runs/capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/` (local) | 107 CPU scratch steps / 5,022 target tokens; complete weighted NLL 7.2265, binding NLL 8.0201; byte-identical to E288–E291 | Honest constrained eval: parse 1.0, meaningful 0.0, AgentV 0/5 and 15 gate failures; scratch/no sync/no promotion |
| 2026-07-17 | `e293-choice-component-plan-r2` (E293 E292-matched DESIGN-context arm) | `outputs/runs/e293-choice-component-plan-r2/` (local) | 107 CPU scratch steps / 5,022 target tokens with DESIGN context; plan loss 5.6250→3.4399; SHA `8c6b7595373d623cc821ddfb7a362faaedd4f3aa87bf360133fb167889a9157d` | Bias-off adversarial meaningful 0.5 / AgentV 1/5 versus E292 0.0 / 0/5, but bias 1 erases the gain and no-DESIGN transfer fails; scratch/no sync/no promotion |
| 2026-07-17 | `e293-choice-component-plan-r3` (E293 matched choice-native plan) | `outputs/runs/e293-choice-component-plan-r3/` (local) | 107 CPU scratch steps / 5,022 target tokens, no DESIGN context; plan loss 5.6761→3.2616, root accuracy and bound recall 0→0.5; complete NLL 7.5550; SHA `78b70c81bd16395e22718baa91b50427c205f38136269c6248b85562cdec5308` | Bias reduces failures 17→13 but meaningful remains 0.0 and AgentV 0/5; scratch/no sync/no promotion |
| 2026-07-17 | `e294-choice-no-design-control-r1` (E294 no-plan control) | `outputs/runs/e294-choice-no-design-control-r1/` (local) | 107 CPU scratch steps / 5,022 target tokens, no DESIGN context; complete NLL 7.4977; SHA `df30ca03f8f2bc3313b1b8afff9c40b7ab18c4fd2b0e8ae1b3888ba780d9add0` | Honest metrics exactly match E293 bias-off; meaningful 0.0, AgentV 0/5, 17 failures; scratch/no sync/no promotion |
| 2026-07-17 | `e295-choice-design-dropout-r1` (E295 context interpolation) | `outputs/runs/e295-choice-design-dropout-r1/` (local) | 107 CPU scratch steps / 5,022 target tokens; 240/480 DESIGN contexts omitted; complete NLL 7.3785; SHA `5b4c50467454f7a9dddbc28da2e115c31a8eba8071587e95eda096729a16fb50` | Prompt-only adversarial meaningful 0.25 and AgentV 1/5, but four suites remain 0.0 and 14 failures remain; scratch/no sync/no promotion |
| 2026-07-18 | `e396-balanced-type-head-continuation-r1` (E396 durable diagnostic) | `hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1/` | 427 cumulative CPU steps / 22,044 target tokens in 104.6s; SHA `feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`; E498 current-main smoke structure 0.27057 | Remote artifacts and SHA verified; current `main` now loads and applies the learned head, but smoke semantic gates and AgentV remain red; no promotion or production ship |
| 2026-07-18 | `e499-remediated-control-r1` (invalid lexer control) | `outputs/runs/e499-remediated-control-r1/` (local) | 5 CPU steps / 1,084 target tokens; SHA `0ba703523ed35e262383f745b3f4784c6fb9c739ac210539b70ea61c52a9a040` | Invalid comparison: data-derived lexer vocabulary changed trainable parameter count; no eval, sync, or promotion |
| 2026-07-18 | `e499-strict-r4-candidate-r1` (invalid lexer candidate) | `outputs/runs/e499-strict-r4-candidate-r1/` (local) | 5 CPU steps / 1,082 target tokens; SHA `c089e5e8b3a93b484827b3af32dc186fdc43bac1390a20233caeee421d15d86e` | Invalid comparison: data-derived lexer vocabulary changed trainable parameter count; no eval, sync, or promotion |
| 2026-07-18 | `e499-strict-r4-choice-candidate-r2` (unmatched choice candidate) | `outputs/runs/e499-strict-r4-choice-candidate-r2/` (local) | 9 CPU steps / 1,034 target tokens; SHA `4ae78b53a0b0d548e530a924e4eedd8dfa98a6eb9375fe5fb53a9eaef0fc7569` | Intended control failed before checkpoint on 61 fragment targets; unmatched evidence only, no eval, sync, or promotion |
| 2026-07-18 | `e499-remediated-roots-choice-control-r3` (invalid scratch choice control) | `outputs/runs/e499-remediated-roots-choice-control-r3/` (local) | 10 CPU steps / 1,023 target tokens; SHA `400251045164c9f677fac00c9f14ab526f848fecb9339584da80f4367d1e408c` | Invalid comparison: scratch context vocabulary still changed trainable parameter count; no eval, sync, or promotion |
| 2026-07-18 | `e499-remediated-roots-hf-choice-control-r4` (matched control) | `outputs/runs/e499-remediated-roots-hf-choice-control-r4/` (local) | 10 CPU steps / 1,023 target tokens in 7.35s; SHA `bb4bec5f7565f733cc9b1417916cc13f10d831bdabac025ad45fa9477f359fb6`; smoke structure 0.1542 | Meaningful/fidelity/reward 0.0 and AgentV 0/1; scratch diagnostic, no sync or promotion |
| 2026-07-18 | `e499-strict-r4-hf-choice-candidate-r4` (matched strict-r4 candidate) | `outputs/runs/e499-strict-r4-hf-choice-candidate-r4/` (local) | 9 CPU steps / 1,034 target tokens in 5.76s; SHA `81b2cb669bbd6b5faa3e6fe60caa0fd3a5d9d310463dc11198dfee227bcfbaf1`; smoke structure 0.0375 | Regresses matched control and AgentV 0/1; rejected, no sync or promotion |
| 2026-07-18 | `e499-choice-compatible-strict-hf-choice-candidate-r6` (document-only strict candidate) | `outputs/runs/e499-choice-compatible-strict-hf-choice-candidate-r6/` (local) | 9 CPU steps / 1,091 target tokens in 6.56s; SHA `7230ace958aa61dcc2c11997f12b8cb6ece590205f69676e4123ac8a4b2e2fab`; smoke p50 5.88s | Structure remains 0.0375 with semantic metrics zero and AgentV 0/1; rejected, no sync or promotion |
| 2026-07-18 | `e500-document-control-hf-choice-r1` (1k control) | `outputs/runs/e500-document-control-hf-choice-r1/` (local) | 9 CPU steps / 1,028 target tokens in 7.00s; loss 30.3844; SHA `a40f39a53fe92b298a69fc0727fa55c4ccf32f26b3c48a034378eefb772d6834` | Smoke structure 0.0375 with semantic metrics zero and AgentV 0/1; scratch/no sync/no promotion |
| 2026-07-18 | `e500-documentized-expression-hf-choice-r2` (1k candidate) | `outputs/runs/e500-documentized-expression-hf-choice-r2/` (local) | 11 CPU steps / 1,039 target tokens in 8.95s; loss 27.6250; SHA `f54cea082c57686b7736a5de4058d762de51f97a611d670115f246bd1773d3f0` | Matches control's red smoke metrics and AgentV 0/1; rejected, no sync or promotion |
| 2026-07-18 | `e500-document-control-hf-choice-r3-5k` (5k control) | `outputs/runs/e500-document-control-hf-choice-r3-5k/` (local) | 43 CPU steps / 5,040 target tokens in 10.14s; loss 10.5529; SHA `9f752ae05d0e1bf50fe77cc3133794cf215e518946c04505f9ce25df6e0b2b53` | Smoke structure 0.0375 with semantic metrics zero and AgentV 0/1; scratch/no sync/no promotion |
| 2026-07-18 | `e500-documentized-expression-hf-choice-r4-5k` (5k candidate) | `outputs/runs/e500-documentized-expression-hf-choice-r4-5k/` (local) | 50 CPU steps / 5,062 target tokens in 13.95s; loss 12.6778; SHA `a0ed6a5840304e5b00d815bb895cafdeb10004e08649e09e4a3611553dda5623` | Loss reverses against control; smoke remains red and AgentV 0/1; rejected, no sync or promotion |
| 2026-07-19 | `e501-e396-e500-init-r1` (task-balanced 5k warm-start) | `outputs/runs/e501-e396-e500-init-r1/` (local) | 96 CPU steps / 5,060 target tokens in 44.99s; loss 10.3340; SHA `f86b83d3c2629d311b6c56f618b1f30ab237624fbc9c47c41929882fdcc9cf15` | Structure regresses to 0.1458 with semantic metrics zero and AgentV 0/1; rejected, no sync or promotion |
| 2026-07-19 | `e501-e396-e500-uniform-init-r2` (uniform 5k warm-start) | `outputs/runs/e501-e396-e500-uniform-init-r2/` (local) | 99 CPU steps / 5,019 target tokens in 79.65s; loss 12.8653; SHA `14605459038c8e56d4f797cc359a2ff9e89d261a375b799476855f262736e4e7` | Recall 0.1667 but structure 0.0889 and meaningful/fidelity/reward zero, AgentV 0/1; rejected, no sync or promotion |
| 2026-07-19 | `e501-e396-e500-uniform-init-r3-1k` (uniform 1k warm-start) | `outputs/runs/e501-e396-e500-uniform-init-r3-1k/` (local) | 22 CPU steps / 1,039 target tokens in 21.39s; loss 26.0208; SHA `d84d34c031c6f2a9e017d2566c04e4abb66314ea3814ba3868a8cc7cb5be2ffd` | Structure 0.2317 is +0.0200 vs parent but semantic metrics remain zero, AgentV 0/1; diagnostic only, no sync or promotion |
| 2026-07-19 | `e502-e396-e500-uniform-lr1e4-r1` | `outputs/runs/e502-e396-e500-uniform-lr1e4-r1/` (local) | 22 CPU steps / 1,039 tokens in 22.79s; loss 28.8950; SHA `fcd5126637cb1d0aa206f5b66f37c198d124813038dba2c76eed7101f047255e` | Structure 0.1133 and recall 0.1667; semantic gates zero, AgentV 0/1; rejected, no sync or promotion |
| 2026-07-19 | `e502-e396-e500-uniform-lr3e5-r2` | `outputs/runs/e502-e396-e500-uniform-lr3e5-r2/` (local) | 22 CPU steps / 1,039 tokens in 21.32s; loss 29.5542; SHA `528c86a6677b711ee5a5484fd513faaa3baca11ad3525c007ef6fd383a62677c` | Structure 0.1167 and recall 0.0833; semantic gates zero, AgentV 0/1; rejected, no sync or promotion |
| 2026-07-19 | `e502-e396-e500-prior-retained-lr3e4-r3` | `outputs/runs/e502-e396-e500-prior-retained-lr3e4-r3/` (local) | 22 CPU steps / 1,039 tokens in 21.17s; loss 25.6905; SHA `e1e833cb35f7de656ead215a0d2924f646008e82d3f7b78b94e4569f0746cb6a` | Structure 0.3169 and recall 0.0833, but semantic gates zero and AgentV 0/1; diagnostic only, no sync or promotion |
| 2026-07-19 | `e502-e396-e500-prior-retained-lr3e4-r4-5k` | `outputs/runs/e502-e396-e500-prior-retained-lr3e4-r4-5k/` (local) | 99 CPU steps / 5,019 tokens in 79.48s; loss 12.8937; SHA `6f937374222b7fd0e82f02e603d4315422bd86d8c2728ac65401f3c24a46a726` | Structure collapses to 0.0927; semantic gates zero, AgentV 0/1; rejected, no sync or promotion |
| 2026-07-19 | `e503-e396-e500-retention0-r1-5k` | `outputs/runs/e503-e396-e500-retention0-r1-5k/` (local) | 99 CPU steps / 5,019 tokens in 75.65s; loss 12.8937; SHA `af6e9b1c207f0a6c2b2bfa264e3e0fcaf9f2afceb850cd74314cad278a0af431` | RMS drift 0.003123; structure 0.0927 and recall 0.1667; semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e503-e396-e500-retention001-r2-5k` | `outputs/runs/e503-e396-e500-retention001-r2-5k/` (local) | 99 CPU steps / 5,019 tokens in 74.41s; loss 13.4907; SHA `7c5f016f1baf0dd9eb56df51aae46d3ed2da10d2d956bbd4cc093d021be75711` | RMS drift 0.002071; structure 0.0900 and recall 0.1667; semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e503-e396-e500-retention005-r3-5k` | `outputs/runs/e503-e396-e500-retention005-r3-5k/` (local) | 99 CPU steps / 5,019 tokens in 73.96s; loss 14.7205; SHA `4093f1aa5a1924f729ddb3c1596333215d744a849b44045b6e302bddaf8d2031` | RMS drift 0.000811; structure 0.2029 but recall zero; semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e503-e396-e500-retention003-r4-5k` | `outputs/runs/e503-e396-e500-retention003-r4-5k/` (local) | 99 CPU steps / 5,019 tokens in 74.04s; loss 14.4501; SHA `2dbb52db813530e0ed06af2e7ba144d5988565570bec1f3fed91ae785751455b` | RMS drift 0.001163; structure 0.1667 and recall 0.0833; semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e504-e396-e500-replay000-r1-5k` | `outputs/runs/e504-e396-e500-replay000-r1-5k/` (local) | 99 CPU steps / 5,019 tokens in 88.54s; loss 12.8937; SHA `35cd38e024c4f458eb67c64c1e0877195a5b40122b8cfb3e95697d0b56334c87` | RMS drift 0.003123; structure 0.0927 and recall 0.1667; semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e504-e396-e500-replay0125-r2-5k` | `outputs/runs/e504-e396-e500-replay0125-r2-5k/` (local) | 100 CPU steps / 5,064 tokens in 70.85s; loss 23.7526; SHA `da63b403aae72267751eaf53d83fdf45e8f4596446fd47c657ba9a7581b725d3` | 12.5% replay; structure 0.1558, recall zero, semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e504-e396-e500-replay025-r3-5k` | `outputs/runs/e504-e396-e500-replay025-r3-5k/` (local) | 100 CPU steps / 5,039 tokens in 73.12s; loss 8.8749; SHA `91ab3f73b6a3f74d7b8a19679d5827068be0806cd327a1c3aeaef76c8c3d85b4` | 25% replay; structure 0.0964 and recall 0.0833; semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e504-e396-e500-replay050-r4-5k` | `outputs/runs/e504-e396-e500-replay050-r4-5k/` (local) | 101 CPU steps / 5,000 tokens in 74.70s; loss 9.8487; SHA `7d7e056e9c61ed4ffba53cf2c20e4d6d624d242488ac7f999e1baa05c90294f9` | 50% replay; RMS drift 0.002796, structure 0.2469, recall 0.0833; semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e504-e396-e500-replay050-retention001-r5-5k` | `outputs/runs/e504-e396-e500-replay050-retention001-r5-5k/` (local) | 101 CPU steps / 5,000 tokens in 74.50s; loss 9.5478; SHA `1fc2fc23b7598bffaab0e0beb07c79593ebc9d25221d6441bc924a38ea36036c` | 50% replay + 1% retention; drift 0.001775 but structure 0.0634 and recall zero; semantic gates zero, AgentV 0/1; rejected |
| 2026-07-19 | `e505-e396-e500-replay050-loss-attribution-r1-5k` | `outputs/runs/e505-e396-e500-replay050-loss-attribution-r1-5k/` (local) | 101 CPU steps / 5,000 tokens in 93.82s; loss 9.8487; SHA `8fd11acdcc1e3eaf0585e847c68815190fdc90c9071e30833db40d24525967e8` | Primary/replay proxies both decline; matched structure 0.2469 and recall 0.0833, semantic gates zero, AgentV 0/1; constrained slot-contract ablation still rejected |
| 2026-07-19 | `e513-e396-e500-replay050-slotrole4-focal2-r3-5k` | `hf://buckets/TKendrick/OpenUI/checkpoints/e513-e396-e500-replay050-slotrole4-focal2-r3-5k/` | 101 CPU HF-context steps / 5,000 target tokens in 79.6s; loss 11.1562; SHA `59253c679477060694370c5e2d8cd9fce5d7accc7d71df3b6d56edf0a88a9548` | Bucket upload and resync verification pass; matched E514 OOD meaningful 0.0, fidelity 0.4917, structure 0.2750, AgentV 0/1; durable diagnostic, rejected for promotion |
| 2026-07-19 | `e515-e396-e500-replay050-slotrole4-focal0-r1-5k` | `hf://buckets/TKendrick/OpenUI/checkpoints/e515-e396-e500-replay050-slotrole4-focal0-r1-5k/` | 101 CPU HF-context steps / 5,000 target tokens in 105.8s; loss 11.3045; SHA `97f2e426604e3956f2791398a608b967937ebf548fa7cae0ef59dde324721c1b` | Bucket upload and resync verification pass; removing focal gamma 2 recovers OOD meaningful to 0.25 and fidelity to 0.6583, but strict meaning and AgentV remain zero; rejected for promotion |
| 2026-07-19 | `e517-e396-e500-replay050-slotrole1-context-r1-5k` | `hf://buckets/TKendrick/OpenUI/checkpoints/e517-e396-e500-replay050-slotrole1-context-r1-5k/` | 101 CPU HF-context steps / 5,000 target tokens in 130.7s; loss 9.9594; SHA `2b572a04256db14095e813e146079af9e6f6c948963d60f2bd669855e24b60e3` | Bucket upload and resync verification pass; slot loss 1 with contract context regresses every headline metric versus E515 and AgentV remains 0/1; rejected for promotion |
| 2026-07-19 | `e519-e396-e500-replay050-slotrole1-honest-context-r1-5k` | `hf://buckets/TKendrick/OpenUI/checkpoints/e519-e396-e500-replay050-slotrole1-honest-context-r1-5k/` | 101 CPU HF-context steps / 5,000 target tokens in 103.2s from clean commit `950007f`; loss 9.9594; SHA `d82155b03531c2d852ec8d497d3fdb0878ac1f678c0c5d247e272bc36c91805f` | Bucket upload and resync verification pass; honest authority changes tensors but exactly matches E517 quality and AgentV 0/1; harness fix retained, checkpoint rejected |
| 2026-07-19 | `e522-e396-e521-replay050-slotrole1-honest-context-r2-5k` | `hf://buckets/TKendrick/OpenUI/checkpoints/e522-e396-e521-replay050-slotrole1-honest-context-r2-5k/` | 99 CPU HF-context steps / 5,059 target tokens in 120.7s from clean commit `ba86b71`; loss 17.5728; SHA `97cb10f43d229b1a15403295f71fa425e844ee4865c31761f3e529b24bf420ce` | Bucket upload and resync verification pass; visible inventory improves fidelity and recall but regresses structure/reward, with meaningful 0.0 and AgentV 0/1; checkpoint rejected |
| 2026-07-19 | `e525-e396-e524-replay050-slotrole1-honest-context-r2-5k` | `hf://buckets/TKendrick/OpenUI/checkpoints/e525-e396-e524-replay050-slotrole1-honest-context-r2-5k/` | 99 CPU HF-context steps / 5,059 target tokens in 76.7s from clean commit `f6d7695`; loss 17.4623; SHA `dbd11811d826fdf7efd8b22557fb3bd48f879e84ec7484bc0a2680198e55e4b9` | Rescue upload, report persistence, resync verification, and independent bucket listing pass; recall improves but fidelity/hierarchy regress, meaningful 0.0 and AgentV 0/1; checkpoint rejected |
| 2026-07-19 | `e528-e396-e527-replay050-slotrole1-honest-context-r1-5k` | `hf://buckets/TKendrick/OpenUI/checkpoints/e528-e396-e527-replay050-slotrole1-honest-context-r1-5k/` | 99 CPU HF-context steps / 5,059 target tokens in 146.8s from clean commit `5cbbb5e`; loss 17.6792; SHA `6a2180d76c366a282a74d1d27ae2b2fcf4c1b5f2b4d298cf4cef35bc306976d5` | Automatic upload, resync verification, and independent nine-file listing pass; meaningful/reward recover but hierarchy regresses, strict meaning 0.0 and AgentV 0/1; checkpoint rejected |
| 2026-07-19 | `e531-e396-e530-replay050-slotrole1-honest-context-r1-5k` | `hf://buckets/TKendrick/OpenUI/checkpoints/e531-e396-e530-replay050-slotrole1-honest-context-r1-5k/` | 99 CPU HF-context steps / 5,059 target tokens in 99.72s from clean commit `e74e27c`; loss 17.0918; SHA `6b8c1abc56a36e8aa15acc373b61d5df033a753907330649e379d9ba374a6154` | Canonical direct rescue sync, report reconciliation, resync verification, and independent nine-file listing pass; slight structure gain does not offset semantic/fidelity/reward regressions, strict meaning 0.0 and AgentV 0/1; checkpoint rejected |
| 2026-07-19 | `e542-e531-root-reference-arity1-r1-24s` | `outputs/runs/e542-e531-root-reference-arity1-r1-24s/` (local) | 24 CPU HF-context steps / 1,270 target tokens in 52.93s from clean commit `5cb1d9c`; loss 17.2169; SHA `2d5cd4b3c8c721e8193e06b5aa231bd9ec5009b4bec9cacfeebe842f6854c5d8` | Explicit `--no-sync-checkpoints` scratch diagnostic. OOD `n=4` control improves over E531-era diagnostics, but learned arity weight 1 is exactly quality-neutral, strict meaning 0.0, AgentV 0/1; checkpoint not promoted |
| 2026-07-19 | `e543-e531-root-reference-bounded-r1-24s` | `outputs/runs/e543-e531-root-reference-bounded-r1-24s/` (local) | 24 CPU HF-context steps / 1,270 target tokens in 37.17s from clean commit `0592d81`; loss 15.4532; SHA `c6be3791544def59ad26b8d2b3b605a7efefd93ec83c996371e593a3251d7f90` | Explicit `--no-sync-checkpoints` scratch diagnostic. Bounded loss sharply improves auxiliary calibration, but the matched OOD replay is decision- and quality-identical to E542, strict meaning 0.0, AgentV 0/1; checkpoint not promoted |
| 2026-07-19 | `e544-e543-root-identity1-r2-24s` | `outputs/runs/e544-e543-root-identity1-r2-24s/` (local) | 24 CPU HF-context steps / 1,270 target tokens in 40.96s from clean commit `81c97b3`; loss 15.6135; SHA `3b6e3c00666b8832187a489d6684ce909fff5b3ccaef57965f9cc1975474f20c` | Explicit `--no-sync-checkpoints` scratch diagnostic. Rank-only identity decoding improves meaningful-v1, structure, recall, and AST node F1 on matched OOD `n=4`, but strict meaning and AST edge F1 remain 0.0 and AgentV 0/1; checkpoint not promoted |
| 2026-07-19 | `e545-e544-root-identity-neg1-control-r1-24s` | `outputs/runs/e545-e544-root-identity-neg1-control-r1-24s/` (local) | 24 CPU HF-context steps / 1,270 target tokens in 30.64s from clean commit `7f59a77`; loss 15.9699; SHA `9e54d4700938c2e1feececfa3b952d4188c76873281e54d38f19bcea4cc76fa1` | Explicit `--no-sync-checkpoints` matched scratch control. OOD meaningful 0.0 and AgentV 0/1; extra continuation regresses from E544; checkpoint not promoted |
| 2026-07-19 | `e545-e544-root-identity-neg4-r2-24s` | `outputs/runs/e545-e544-root-identity-neg4-r2-24s/` (local) | 24 CPU HF-context steps / 1,270 target tokens in 28.64s from clean commit `7f59a77`; loss 15.9901; SHA `14dd44043887cfb6b5a14b1a99fee3750dc8f72c2d27f205fe3bdc0506de61ae` | Explicit `--no-sync-checkpoints` matched scratch treatment. Slight sparse negative calibration gain is decode- and quality-neutral; AgentV 0/1; checkpoint not promoted |
| 2026-07-19 | `e546-e544-strict-subset1-control-r1-24s` | `outputs/runs/e546-e544-strict-subset1-control-r1-24s/` (local) | 24 CPU HF-context steps / 1,270 target tokens in 29.10s from clean commit `4e66e46`; loss 15.9699; SHA `46aba9048624f766e6052d202a94b689440baca9f1ab94d8d6c8d48adc40fc55` | Explicit no-sync matched scratch control; meaningful 0.0 and AgentV 0/1; checkpoint not promoted |
| 2026-07-19 | `e546-e544-strict-subset5-r2-24s` | `outputs/runs/e546-e544-strict-subset5-r2-24s/` (local) | 24 CPU HF-context steps / 1,318 target tokens in 30.50s from clean commit `4e66e46`; loss 26.4735; SHA `a1a6bfc94108a8bba9aac18e5570d70e317cdec5bb706f126bf47e67e2b4efe2` | Explicit no-sync matched scratch treatment; mixed fidelity/topology gain with severe recall regression, meaning 0.0 and AgentV 0/1; checkpoint not promoted |
| 2026-07-19 | `e547-e544-strict-subset2-r1-24s` | `outputs/runs/e547-e544-strict-subset2-r1-24s/` (local) | 24 CPU HF-context steps / 1,304 target tokens in 36.48s from clean commit `bad2f230`; loss 12.4980; SHA `37002bfd3c63d1ac58f5fc505bf034805b57eee2415d9e15ec1acbb81620fc57` | Explicit no-sync scratch diagnostic; preferred exposure setting for topology, but fidelity and semantic gates fail; checkpoint not promoted |

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
5. Record the run's version stamp with the eval table: `code_commit` plus the
   `harness.model_build.eval` / `evals.meaningful_program` / `gates.ship`
   versions from `scoreboard.json`
   ([version-stamp-contract.md](design/version-stamp-contract.md)).
6. Commit docs with the checkpoint-producing change.
