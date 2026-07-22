# slm-training

Novel SLM experiments: harnesses for **placeholder OpenUI** layout generation (official `@openuidev/lang-core`), a **TwoTower** masked-diffusion model, plus a **GPU multi-farm MCP**.

## What's included

1. **Training-data harness** — build/validate versioned train corpora
2. **Testing-data harness** — held-out / adversarial / OOD eval suites
3. **Model-building harness** — lineage-first **TwoTower** and causal-LoRA tracks
4. **OpenUI Lang bridge** — Node sidecar over official `@openuidev/lang-core`
5. **GPU multi-farm MCP** — list / launch / cost-project across Vast.ai, RunPod, Lambda

Autonomous experiment campaigns use the fail-closed, evidence-grounded
[`autoresearch` harness](docs/design/autoresearch-autotraining.md), with isolated,
pinned [Open Deep Research](https://github.com/langchain-ai/open_deep_research) and
[OpenResearcher](https://github.com/TIGER-AI-Lab/OpenResearcher) implementations
behind one memo/trajectory contract and trusted hypothesizer. Before execution the
pipeline requires a persisted matrix of at least five distinct, grounded hypotheses,
including categorical candidate-novelty audits adapted from
[Wang and Buehler (2026)](https://arxiv.org/abs/2606.01444). Pre-run audits are not
claims of proven discovery or SOTA. Each matrix names its recommended experiment;
completed outcomes and diagnoses become typed feedback for the next matrix and for
future campaign evidence. The loop improves by evidence, never by rewriting its own
code, frozen cases, or gates. RL remains locked until a model passes the frozen
production readiness contract.

See [docs/design/model-lineage.md](docs/design/model-lineage.md) (canonical two-track cycle), [docs/design/openui-twotower.md](docs/design/openui-twotower.md), [docs/design/grammar-topology-diffusion.md](docs/design/grammar-topology-diffusion.md) (dynamic production-tree diffusion), [docs/design/verified-scope-solver.md](docs/design/verified-scope-solver.md) (VSS0 verified scope-solver contract — prefix legality vs verified support), [docs/design/research-lineage.md](docs/design/research-lineage.md) (papers → code), [docs/design/semantic-planning-valid-state-disposition.md](docs/design/semantic-planning-valid-state-disposition.md) (SPV4-02 final disposition — all plan-aware mechanisms remain default-off diagnostics), [docs/design/research-correction-critics.md](docs/design/research-correction-critics.md) (V4 remask / trust-gate / honest inventory; V6 CoRe/T2M), [docs/design/verifier-stack.md](docs/design/verifier-stack.md) (G0–G12 corpus gates + confidence tiers), [docs/design/abstraction-house-style.md](docs/design/abstraction-house-style.md) (L0–L5 determinacy, grounding, and canonical defaults), [docs/design/verifier-guided-repair.md](docs/design/verifier-guided-repair.md) (PDDL-Instruct / verifier-repair applicability map), [docs/design/quality-experiment-matrix.md](docs/design/quality-experiment-matrix.md) (E0–E75 + X0–X15 matrices; E34 deferred), [docs/design/speculative-denoising.md](docs/design/speculative-denoising.md) (V7 stability / dependency-cluster / survival / successor-cache decode), [docs/design/dsl-native-tokenizer.md](docs/design/dsl-native-tokenizer.md) (V5 lexer alphabet), [docs/design/grammar-fastpath.md](docs/design/grammar-fastpath.md), [docs/design/grammar-backends.md](docs/design/grammar-backends.md), [docs/design/dsl-pack-contract.md](docs/design/dsl-pack-contract.md) (F1 DSL-pack contract; OpenUI first pack), [docs/design/structure-only-eval.md](docs/design/structure-only-eval.md), [docs/design/binding-aware-meaningful-v2.md](docs/design/binding-aware-meaningful-v2.md) (versioned binding-aware metric and gaming audit), [docs/design/judge-independence-audit.md](docs/design/judge-independence-audit.md) (EFS0-04 cross-family/human audit contract), [docs/design/adversarial-review.md](docs/design/adversarial-review.md), [docs/design/runtime-performance.md](docs/design/runtime-performance.md), [docs/design/hf-jobs-train.md](docs/design/hf-jobs-train.md) (HF Jobs full train — not ZeroGPU), [docs/design/gpu-multi-farm-mcp.md](docs/design/gpu-multi-farm-mcp.md), and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

Calculated arity, task rate, neural precision, and physical cost are kept distinct
by the [CAP0 contract](docs/design/calculated-arity-adaptive-precision.md).

## Model card (summary)

Full card: **[docs/MODEL_CARD.md](docs/MODEL_CARD.md)**. Agents update both this
summary and the full card whenever a checkpoint is created or promoted.

**Current compatibility:** output contract v4 requires harness-canonical
`:slot_<ordinal>` markers in persisted train/eval data and nested metadata.
E828 is the first target-inventory-correct v4 scratch checkpoint; it fails quality gates and
is not promoted. Every older checkpoint is incompatible provenance only. See
[the contract](docs/design/symbol-only-output-contract.md).

| Role | Checkpoint | Where | Claim |
| --- | --- | --- | --- |
| E828 target-slot-only v4 baseline | `e828-target-slots-only-v4-scratch120-r1/last.pt` | `outputs/runs/…` (local) | First completion-inventory-correct v4 checkpoint; 120 steps / 14.46s, held-out n=5 parse/meaning/fidelity 0.0 with four bounded timeouts — rejected, not ship |
| E735 full-head root-arity diagnostic | `e735-symbol-only-root-arity-fullhead140-r1/last.pt` | `outputs/runs/…` (local) | Removes impossible class-41 tail prediction, but weight 0/1 smoke quality remains identical and strict-v2 0.0 — fix retained, checkpoint rejected |
| E733 invalid lexer root-identity attempt | `e733-symbol-only-root-identity140-r1/last.pt` | `outputs/runs/…` (local) | Proposed lever has zero reachable decode applications; config now rejects lexer identity before artifacts — checkpoint invalidated |
| E731 lexer root-arity diagnostic | `e731-symbol-only-root-arity140-r1/last.pt` | `outputs/runs/…` (local) | Lexer-native head is executable, but weights 0/1/2 change no choices; smoke strict-v2 0.0 — checkpoint rejected |
| E714 symbol-only baseline | `e714-symbol-only-scratch600-r1/last.pt` | `outputs/runs/…` (local) | Historical v2 CPU scratch checkpoint; now incompatible provenance under opaque-marker v3 |
| E720 component-inventory diagnostic | `e720-symbol-only-component-inventory600-r1/last.pt` | `outputs/runs/…` (local) | Inventory head learned (top-k recall 0.6875), but smoke parse/strict meaning remained 0.0 and weight-4 decode timed out 3/3 — rejected, not ship |
| E721 role/count plan diagnostic | `e721-symbol-only-component-plan190-r4/last.pt` | `outputs/runs/…` (local) | Smoke parse 1.0, but strict meaning 0.0 and plan weight 1 is identical to weight 0; local 190-step syntax diagnostic only, rejected |
| E722 component-edge diagnostic | `e722-symbol-only-component-edge150-r1/last.pt` | `outputs/runs/…` (local) | Parse 1.0 / structure 0.2861 / recall 0.5, but strict meaning 0.0 and edge on/off identical — rejected, not ship |
| E723 slot-owner diagnostic | `e723-symbol-only-slot-owner140-r1/last.pt` | `outputs/runs/…` (local) | Causal smoke + held-out gains; smoke meaning-v1 0.6667 / structure 0.5614, but strict-v2 0.0 — lever retained, checkpoint rejected |
| E725 cumulative inventory diagnostic | `e725-symbol-only-component-inventory130-r1/last.pt` | `outputs/runs/…` (local) | Inventory head learned, but weight 1/0 decode is identical and smoke meaning-v1/strict-v2 0.0 — rejected, not ship |
| E726 invalid root-arity attempt | `e726-symbol-only-root-arity140-r1/last.pt` | `outputs/runs/…` (local) | Choice-only arity lever was unavailable on lexer; tensors match E723 exactly — invalidated, never evaluate/sync/serve |
| E727 binder-arity diagnostic | `e727-symbol-only-binder-arity140-r1/last.pt` | `outputs/runs/…` (local) | Arity head learned, but weights 1/2 change no smoke or held-out choices and strict-v2 remains 0.0 — rejected |
| E729 binder-topology diagnostic | `e729-symbol-only-binder-topology140-r1/last.pt` | `outputs/runs/…` (local) | Topology weights 0.25/1 regress smoke meaning 0.6667→0.3333 and structure 0.5614→0.4642 — rejected |
| Playground demo | `playground_demo/last.pt` | `src/slm_training/resources/checkpoints/playground_demo/` (git) | E497 clean-revision honest smoke: parse/meaningful/fidelity 0.0, structure 0.2203, AgentV 0/5; wiring only |
| Restructure CPU verify | `restructure_cpu_scratch_v0/last.pt` | `outputs/runs/…` (local) | Fixture scratch train OK; smoke parse 0.0 — not ship |
| Local DirectML verify | `local_directml_adreno_20260714/last.pt` | `outputs/runs/…` (local) | Adreno GPU train/checkpoint OK; 5-step wiring run, not evaluated or ship |
| Overnight retrain | `overnight_retrain_200/last.pt` | `/tmp/slm-training-overnight/outputs/runs/…` (local) | 200-step CPU scratch; honest parse 0.0, not ship |
| Overnight retrain extended | `overnight_retrain_1000/last.pt` | `/tmp/slm-training-overnight/outputs/runs/…` (local) | 1,000-step CPU scratch; smoke parse 0.0, not ship |
| E120 singleton diagnostic | `e120_unsandboxed/last.pt` | `outputs/runs/iter-e120-unsandboxed-20260715/…` (local) | 8-step CPU scratch; guarded singleton decode verified, `rico_held n=1` parse 0.0 — not ship |
| E121 judged-corpus E53 iteration | `qx_e53_honest_v5_champion/last.pt` | `outputs/runs/iter-e121d-e53-judged-20260715/…` (local) | 405 judge-approved records; bounded smoke parse 0.0 with decode timeout — not ship |
| E123 judged-corpus 32-step iteration | `e123_judged_32step_b/last.pt` | `outputs/runs/iter-e123b-judged-20260715/…` (local) | 405 judge-approved records; loss 10.97 but smoke parse 0.0 with fallback/canvas cap — not ship |
| E127 schema/slot-contract iteration | `e127_judged_schema_slots/last.pt` | `outputs/runs/iter-e127-schema-slots-20260715/…` (local) | 405 judged records; placeholder validity 0.55 / normalized fidelity 0.25, but parse 0.0 — not ship |
| E128 schema/slot 64-step iteration | `e128_judged_schema_slots_64/last.pt` | `outputs/runs/iter-e128-schema-slots-20260715/…` (local) | Higher LTR/fidelity weights regressed placeholder signals and parse remained 0.0 — not ship |
| E129 schema/slot 64-step low-weight control | `e129_judged_schema_slots_64_lowweights/last.pt` | `outputs/runs/iter-e129-schema-slots-20260715/…` (local) | Lower-weight control also had placeholder/parse 0.0; longer training not justified — not ship |
| E130 schema/slot seed-1 control | `e130_judged_schema_slots_seed1/last.pt` | `outputs/runs/iter-e130-schema-slots-20260715/…` (local) | Seed-1 control had parse and placeholder signals 0.0; E127 not reproducible — not ship |
| E132 generation-focused mixture | `e132_generation_focus/last.pt` | `outputs/runs/iter-e132-generation-focus-20260715/…` (local) | Three-prompt smoke parse/placeholder 0.0; task reweighting rejected — not ship |
| E133 no-fused-LTR path | `e133_no_fuse_ltr/last.pt` | `outputs/runs/iter-e133-no-fuse-ltr-20260715/…` (local) | Three-prompt smoke parse/structure 0.0 with one timeout; fused LTR retained — not ship |
| E135 HF context control | `e135_hf_context_control/last.pt` | `outputs/runs/iter-e135-hf-context-20260715/…` (local) | HF context improves structural/placeholder signals but parse 0.0 with one timeout — not ship |
| E136 HF context 32-step control | `e136_hf_context_32/last.pt` | `outputs/runs/iter-e136-hf-context-20260715/…` (local) | Longer HF run regressed structure/placeholder to 0.0; checkpoint selection next — not ship |
| E137 HF context 16-step midpoint | `e137_hf_context_16/last.pt` | `outputs/runs/iter-e137-hf-context-20260715/…` (local) | Placeholder validity 0.40 and structure 0.2142, parse 0.0; non-monotonic checkpoint trajectory — not ship |
| E138 HF context seed-1 8-step control | `e138_hf_context_seed1_8/last.pt` | `outputs/runs/iter-e138-hf-seed1-20260715/…` (local) | Same recipe as E135 but seed 1: placeholder validity 0.0 and structure 0.1683, parse 0.0 — not ship |
| E139 HF context seed-2 8-step control | `e139_hf_context_seed2_8/last.pt` | `outputs/runs/iter-e139-hf-seed2-20260715/…` (local) | Same recipe as E135 but seed 2: placeholder validity/structure/parse 0.0 with two timeouts — not ship |
| E173 schema-context 32-step control | `e173-schema-context-32step/last.pt` | `outputs/runs/e173-schema-context-32step/…` (local) | Schema/slot context enabled; bounded syntax probe 1.0 but meaningful parse 0.0 — not ship |
| E174 unfrozen-context 8-step control | `e174-unfrozen-context-8step/last.pt` | `outputs/runs/e174-unfrozen-context-8step/…` (local) | Unfrozen context regressed bounded syntax to 0.0; rejected control — not ship |
| E175 retrieval 8-step control | `e175-retrieval-8step/last.pt` | `outputs/runs/e175-retrieval-8step/…` (local) | Retrieval k=4 regressed bounded syntax/parse to 0.0; rejected control — not ship |
| E176 broad-corpus 8-step control | `e176-broad-corpus-8step/last.pt` | `outputs/runs/e176-broad-corpus-8step/…` (local) | 1,417-record corpus regressed bounded syntax/parse to 0.0; rejected control — not ship |
| E177 semantic-judge 32-step control | `e177-semantic-judge-32step/last.pt` | `outputs/runs/e177-semantic-judge-32step/…` (local) | 496 published judge-gated records; E180 bounded decode reaches syntax 1.0 but meaningful parse 0.0 — not ship |
| E181/E184/E191 compiler-alignment diagnostics | `e181-semantic-balanced-32step`, `e184-compiler-aligned-32step`, `e191-full-compiler-aligned-32step` | `outputs/runs/…` (local) | Balanced mixture did not improve quality; component alignment recovered the root, all-branch alignment regressed it; no meaningful parse or promotion — not ship |
| E195/E196 stratified-alignment diagnostics | `e195-stratified-compiler-aligned-32step`, `e196-stratified-compiler-aligned-matched-32step` | `outputs/runs/…` (local) | E195 invalid (mixture unset); matched E196 reaches syntax 1.0 after parser-state fixes but meaningful parse 0.0 — not ship |
| E201 generated-role diagnostic | `e201-role-stratified-compiler-aligned-32step` | `outputs/runs/…` (local) | Grammar/schema role constraints improve component and placeholder signals, but recursive children hit the token cap with parse 0.0 — not ship |
| E205 Lark-terminal diagnostic | `e205-lark-terminal-stratified-32step` | `outputs/runs/…` (local) | Terminal-derived alignment and schema enum paths restore syntax 1.0 without fallback, but empty bound stacks leave meaningful parse 0.0 — not ship |
| E208/E210/E212 contextual-decision diagnostics | `e208-list-occupancy-stratified-32step`, `e210-list-scope-occupancy-stratified-32step`, `e212-contextual-decision-stratified-32step` | `outputs/runs/…` (local) | Contextual root-child supervision recovers a populated root and fidelity signal, but required schema semantics still fail and meaningful parse remains 0.0 — not ship |
| E214/E215 overfiltered schema-judge diagnostic | `e215-schema-role-judged-32step` | `outputs/runs/e215-schema-role-judged-32step/…` (local) | E214 falsely rejected 27 legal optional-null records; E216 syntax 1.0 but meaningful parse 0.0; superseded by E218 — not ship |
| E218/E219 corrected schema-admission diagnostic | `e219-schema-normalized-32step` | `outputs/runs/e219-schema-normalized-32step/…` (local) | Restores 33 valid records and fixes future producers; E220 syntax 1.0, component recall 0.25, meaningful parse 0.0 — not ship |
| E221 task-balanced exposure diagnostic | `e221-canonical-task-balanced` | `outputs/autoresearch/e221-task-balanced-exposure-v4/runs/…` (local) | 32 CPU steps on canonical E218; effective exposure 29.68/128; strict eval failed 9 gates, AgentV 1/5 — not ship |
| E222 capacity-aware exposure diagnostic | `e222-capacity-aware-matched` | `outputs/autoresearch/e222-capacity-aware-exposure/runs/…` (local) | Effective exposure rose to 83.59/128, but strict smoke parse regressed to 0.0 and 10 gates failed — not ship |
| E223 quota-capacity exposure diagnostic | `e223-quota-capacity-matched` | `outputs/autoresearch/e223-quota-capacity-exposure/runs/…` (local) | Task quotas and syntax are deterministic, but semantic metrics are 0.0 and 12 gates failed — not ship |
| E224–E226 semantic alignment + honest tree eval | `e224-semantic-exhaustive-matched` | `outputs/autoresearch/e224-semantic-exhaustive-alignment/runs/…` (local) | Deterministic tree reaches syntax 1.0 on all suites with honest fidelity, but meaningful-program quality fails 5 gates — not ship |
| E227 legal-candidate alignment | `e227-candidate-set-matched` | `outputs/autoresearch/e227-candidate-set-alignment/runs/…` (local) | Candidate loss optimizes, but empty-layout collapse fails 12 gates and AgentV 0/5 — rejected, not ship |
| E228 legal-candidate margin | `e228-candidate-margin-matched` | `outputs/autoresearch/e228-candidate-margin-alignment/runs/…` (local) | Best diagnostic: syntax/contract 1.0, failures reduced to 4, but AgentV 1/5 — not ship |
| E229 64-step margin continuation | `e229-margin-64step` | `outputs/autoresearch/e229-margin-continuation/runs/…` (local) | Syntax restored to 1.0 after generalized literal-frame fix, but the same 4 gates fail — duration rejected, not ship |
| E230 diverse judged roots | `e230-diverse-roots-32step` | `outputs/autoresearch/e230-diverse-judged-roots/runs/…` (local) | Published 126 judge-passed generation roots and verified RICO/human exposure; same 4 gates fail and adversarial regresses — data fix retained, checkpoint rejected, not ship |
| E231 component inventory | `e231-component-inventory-32step` | `outputs/autoresearch/e231-component-inventory/runs/…` (local) | Inventory target learns, but bias-off metrics/component choices are identical; 6 thresholds fail, AgentV 1/5 — rejected, not ship |
| E232 role component plan | `e232-role-component-plan-32step` | `outputs/autoresearch/e232-role-component-plan/runs/…` (local) | Root/count targets learn and improve one adversarial case, but 4 frontier thresholds still fail; stronger calibration has no aggregate gain — rejected, not ship |
| E233 resolved-AST component edges | `e233-component-edges-32step` | `outputs/autoresearch/e233-component-edges/runs/…` (local) | Edge target learns, but edge on/off suite aggregates are identical and 4 thresholds fail — rejected, not ship |
| E234 edge decision alignment | `e234-edge-decision-alignment-32step` | `outputs/autoresearch/e234-edge-decision-alignment/runs/…` (local) | Legal-decision accuracy learns and changes 5 choices, but on/off aggregates are identical and 4 thresholds fail — rejected, not ship |
| E235 binder-instance plan | `e235-binder-instance-plan-32step` | `outputs/autoresearch/e235-binder-instance-plan/runs/…` (local) | Full binder supervision changes 4 legal choices, but on/off aggregates are identical and 9 thresholds fail — rejected, not ship |
| E236 binder topology | `e236-binder-topology-32step` | `outputs/autoresearch/e236-binder-topology/runs/…` (local) | Topology objective fails to learn, changes 0/38 applied choices, and collapses semantic metrics; 12 thresholds fail — rejected, not ship |
| E237 detached topology | `e237-detached-topology-32step` | `outputs/autoresearch/e237-detached-topology/runs/…` (local) | Detaching already-frozen context is a no-op and exactly reproduces E236; 12 thresholds fail — rejected, not ship |
| E238 binder arity (invalidated) | `e238-binder-arity-32step` | `outputs/autoresearch/e238-binder-arity/runs/…` (local) | Optional-head RNG shifted matched training draws; ten thresholds fail and the run is confounded — not ship |
| E239 isolated binder arity | `e239d-binder-arity-fully-isolated-32step` | `outputs/autoresearch/e239-binder-arity-corrected/runs/…` (local) | 104/104 shared tensors match the control; 29 changed choices do not produce meaningful programs; 11 thresholds fail — rejected, not ship |
| E249 exact-event CE plus margin | `qx_e249_local_ce_margin` | `outputs/autoresearch/e249-local-ce-margin/runs/…` (local) | Held-out lexical wins improve sharply, but structure/reward regress on every suite and AgentV is 0/5 — rejected, not ship |
| E252 verifier-backed set FTPO | `qx_e252_local_ftpo_set` | `outputs/autoresearch/e252-ftpo-set/runs/…` (local) | Syntax remains 1.0, but fidelity collapses to 0, structure/reward regress everywhere, and AgentV is 0/5 — rejected, not ship |
| E263 broad gold-AST set FTPO | `qx_e262_broad_gold_ast_ftpo_set` | `outputs/autoresearch/e262-broad-gold-ast-ftpo/runs/…` (local) | Emitted as E262 before ID reconciliation; syntax/fidelity match E248, but held-out loss worsens, structure regresses everywhere, and AgentV is 0/5 — rejected, not ship |
| E264 guarded gold-AST set FTPO | `qx_e264_guarded_gold_ast_ftpo_set` | `outputs/autoresearch/e264-guarded-gold-ast-ftpo/runs/…` (local) | No trained step passed the held-out Pareto guard; restored checkpoint is bit-identical to E228 and current parent control reproduces all metrics — no model gain, not ship |
| E265 safe gold-AST set FTPO | `qx_e265_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e265-safe-gold-ast-ftpo/runs/…` (local) | 3/30 backtracked proposals improve aggregate exact-state metrics, but per-kind regressions are masked and semantic quality falls on most suites — rejected, not ship |
| E266 stratified safe set FTPO | `qx_e266_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e266-stratified-safe-gold-ast-ftpo/runs/…` (local) | Per-decision-kind guard rejects all 30 global FTPO proposals; parent is restored exactly, while batched validation is 37.7× faster — no model gain, not ship |
| E267 block-coordinate safe set FTPO | `qx_e267_block_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e267-block-stratified-safe-ftpo/runs/…` (local) | Averaging gradients within each decision kind still yields 0/30 safe proposals; parent is restored exactly — no model gain, not ship |
| E268 projected safe set FTPO | `qx_e268_projected_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e268-projected-stratified-safe-ftpo/runs/…` (local) | PCGrad projects 2,220 conflicting task pairs but still yields 0/30 safe proposals; parent restored exactly, 38m59s CPU stage — rejected, not ship |
| E269 MGDA safe set FTPO | `qx_e269_mgda_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e269-mgda-one-step-final/runs/…` (local) | One-step MGDA certifies common train descent, but all five scales regress held-out decision kinds; full 30-step run rejected, parent restored — not ship |
| E272 MGDA plus SGD preflight | `qx_e272_mgda_sgd_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e272-mgda-sgd-one-step/runs/…` (local) | Collinear SGD improves aggregate held-out loss, but all scales regress per-kind probability/margin guards; parent restored, no full run — not ship |
| Matrix honest champion | V6 E53 family | `outputs/runs/` + matrix docs | Scratch + limited `rico_held` — not production HF ship |
| P13 matched E50 controls | fixture + integrated E50 | `/tmp/slm17-e50-*-honest/` (local scratch) | Integrated fidelity +0.04 held / +0.0333 RICO; parse 0.0, not ship |
| Frozen X2 baseline | `gx_x2_codec` seeds 0/1/2 | `/tmp/slm-training-fixed-baseline/outputs/topology_baseline/` | Fixed-canvas comparison scored zero on all suites; not ship |
| Topology v2 smoke | `grammar_diffusion_overfit` | pytest temporary checkpoint | n=2 parse/fidelity 0.5, topology composite 0.482; wiring only, not ship |
| Topology X9/X14 confirmation | 6 seed checkpoints | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` | 200-step CPU scratch; all fail multi-suite gates, no promotion/sync |
| ScopeDiff X18/X21 confirmation | 6 seed checkpoints | `outputs/runs/gx_x{18,21}_*_confirm_200/` (local) | 200-step CPU scratch; all-suite median parse/fidelity 0.0, all fail gates, no promotion/sync |
| EFS0-04 X22 reproduction | `gx_x22_kapur_tree_edit_s0/last.pt` | `outputs/runs/gx_x22_kapur_tree_edit_s0/…` (local) | 80-step seed-0 audit-material replay; SHA `a9cfb450…02ff6`; syntax 1.0 but meaningful parse 0.333/0.2/0/0/0.667 on bounded suites; gates fail, no sync/promotion ([results](docs/design/iter-efs0-04-x22-reproduction-20260717.md)) |
| B3 five-minute lexer control | `capacity_lexer_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/b3-matched-5m-e287-r2/…` (local) | 53-step / 5,004-token CPU scratch; five-suite parse/meaningful 0.0, AgentV 0/5 — not promoted or ship |
| B3 five-minute choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/b3-matched-5m-e287-r2/…` (local) | E288 frozen eval: deterministic parse 1.0 on all suites, but meaningful/fidelity 0.0 and AgentV 0/5 — not promoted or ship |
| E289 cached choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/e289-choice-state-cache/…` (local) | Same checkpoint SHA as E288; exact symbolic-state cache preserves parse 1.0 and cuts p50 2.65×–5.86×, but semantic metrics and AgentV remain zero — not promoted or ship |
| E290 direct-candidate choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/e290-choice-direct-candidates/…` (local) | Same checkpoint SHA; exact grammar-derived candidates improve p95 1.14×–1.19× but regress p50, while semantic metrics and AgentV remain zero — not promoted or ship |
| E291 completion-cached choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/e291-choice-completion-cache/…` (local) | Same checkpoint SHA; exact completion caching improves p50 1.29×–1.99× and p95 1.51×–1.93× vs E290, but semantic metrics and AgentV remain zero — not model-promoted or ship |
| E292 complete-loss choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/e292-choice-loss-suite-complete-r2/…` (local) | Same checkpoint SHA; all five frozen loss categories now complete (weighted NLL 7.2265), but honest meaningful rate is 0.0 and AgentV is 0/5 — not promoted or ship |
| E293 choice-native component plan | `e293-choice-component-plan-r3/last.pt` | `outputs/runs/e293-choice-component-plan-r3/…` (local) | Plan target learns and legal bias reduces failures 17→13, but matched no-DESIGN meaningful rate is 0.0 and AgentV 0/5 — not promoted or ship |
| E294 no-DESIGN choice control | `e294-choice-no-design-control-r1/last.pt` | `outputs/runs/e294-choice-no-design-control-r1/…` (local) | No-plan control exactly matches E293 bias-off; meaningful 0.0, AgentV 0/5, 17 failures — not promoted or ship |
| E295 DESIGN-dropout choice arm | `e295-choice-design-dropout-r1/last.pt` | `outputs/runs/e295-choice-design-dropout-r1/…` (local) | 50% deterministic context dropout yields adversarial meaningful 0.25 and AgentV 1/5, but four suites remain 0.0 and 14 gates fail — not promoted or ship |
| E396 durable diagnostic checkpoint | `e396-balanced-type-head-continuation-r1/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1/` | Exact SHA `feefa056…c2f2eee0`; bucket verified. E498 restores current-main loading and learned-head application (smoke structure 0.27057), but semantic gates and AgentV remain red. Diagnostic, not champion or ship |
| E499 bounded strict-corpus checkpoints | `e499-*-r4/r6/last.pt` | `outputs/runs/e499-…/` (local) | Matched strict-r4 and document-only r6 both regress smoke structure 0.1542→0.0375 and recall 0.25→0.0; AgentV 0/1, no sync or promotion. All seven checkpoint SHAs are in the full card |
| E500 documentized-expression checkpoints | `e500-*-r1/r2/r3-5k/r4-5k/last.pt` | `outputs/runs/e500-…/` (local) | The 260-row projected corpus is clean and diverse, but both matched 1k/5k pairs have structure 0.0375, semantic metrics zero, and AgentV 0/1. Four exact SHAs are in the full card; no sync or promotion |
| E501 E396→E500 warm-start checkpoints | `e501-e396-e500-*/last.pt` | `outputs/runs/e501-…/` (local) | Explicit new-corpus initialization works, but 5k arms forget parent structure; the 1k arm reaches structure 0.2317 with semantic metrics still zero. Three exact SHAs are in the full card; no sync or promotion |
| E502 prior-retention checkpoints | `e502-e396-e500-*/last.pt` | `outputs/runs/e502-…/` (local) | Preserving checkpoint serving priors raises 1k structure to 0.3169 with recall 0.0833, but 5k collapses and all semantic gates remain zero. Four exact SHAs are in the full card; no sync or promotion |
| E503 initialized-weight retention checkpoints | `e503-e396-e500-retention*-5k/last.pt` | `outputs/runs/e503-…/` (local) | Retention cuts RMS drift up to 74% and restores structure to 0.2029, but recall falls to zero and semantic gates remain red. Four exact SHAs are in the full card; no sync or promotion |
| E504 parent-replay checkpoints | `e504-e396-e500-replay*-5k/last.pt` | `outputs/runs/e504-…/` (local) | 50% exact E357 replay raises structure to 0.2469 and cuts drift 10.46%, but semantic gates remain zero; replay plus retention regresses structure. Five exact SHAs are in the full card; no checkpoint sync or promotion |
| E505 replay-loss attribution checkpoint | `e505-e396-e500-replay050-loss-attribution-r1-5k/last.pt` | `outputs/runs/e505-…/` (local) | E511 component-plan weight 4 reaches aggregate meaningful 0.3846 and fidelity 0.6718 across 13 records. E512 rejects slot weight 8; strict semantic and AgentV gates remain red, with no promotion |
| E513 durable slot-role checkpoint | `e513-e396-e500-replay050-slotrole4-focal2-r3-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e513-e396-e500-replay050-slotrole4-focal2-r3-5k/` | Bucket-verified SHA `59253c67…a88a9548`; 5,000 target tokens in 79.6s under the three-minute cap. Matched OOD meaningful 0.0, fidelity 0.4917, structure 0.2750, AgentV 0/1; durable diagnostic, rejected for promotion |
| E515 focal-zero slot-role checkpoint | `e515-e396-e500-replay050-slotrole4-focal0-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e515-e396-e500-replay050-slotrole4-focal0-r1-5k/` | Bucket-verified SHA `97f2e426…24721c1b`; 5,000 target tokens in 105.8s under the three-minute cap. Matched OOD meaningful 0.25, fidelity 0.6583, structure 0.3213, AgentV 0/1; focal 2 rejected, checkpoint not promoted |
| E517 slot-loss-1 context checkpoint | `e517-e396-e500-replay050-slotrole1-context-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e517-e396-e500-replay050-slotrole1-context-r1-5k/` | Bucket-verified SHA `2b572a04…e24b60e3`; 5,000 target tokens in 130.7s under the three-minute cap. Matched OOD meaningful 0.0, fidelity 0.4083, structure 0.2250, AgentV 0/1; rejected |
| E519 honest slot-context checkpoint | `e519-e396-e500-replay050-slotrole1-honest-context-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e519-e396-e500-replay050-slotrole1-honest-context-r1-5k/` | Bucket-verified SHA `d82155b0…6c91805f`; 5,000 target tokens in 103.2s from clean harness v7. Exact E517 quality parity (meaningful 0.0, fidelity 0.4083, structure 0.2250, AgentV 0/1); honest path retained, checkpoint rejected |
| E522 visible-inventory checkpoint | `e522-e396-e521-replay050-slotrole1-honest-context-r2-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e522-e396-e521-replay050-slotrole1-honest-context-r2-5k/` | Bucket-verified SHA `97cb10f4…bf420ce`; 5,059 target tokens in 120.7s. E523 fidelity 0.8667 and recall 0.2708 improve, but meaningful stays 0.0, structure falls to 0.1955, and AgentV is 0/1; rejected |
| E525 visible-component checkpoint | `e525-e396-e524-replay050-slotrole1-honest-context-r2-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e525-e396-e524-replay050-slotrole1-honest-context-r2-5k/` | Bucket-verified SHA `dbd11811…e55e4b9`; 5,059 target tokens in 76.7s. E526 recall rises to 0.4167, but fidelity falls to 0.4667, structure to 0.1452, meaningful stays 0.0, and AgentV is 0/1; rejected |
| E528 visible-component-types checkpoint | `e528-e396-e527-replay050-slotrole1-honest-context-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e528-e396-e527-replay050-slotrole1-honest-context-r1-5k/` | Bucket-verified SHA `6a2180d7…306976d5`; 5,059 target tokens in 146.8s. E529 meaningful recovers to 0.25 and reward to 0.5778, but structure falls to 0.1136, strict meaning remains 0.0, and AgentV is 0/1; rejected |
| E616 object-frame slot-bias replay (80-step) | `e616-object-property-slot-bias-scratch80-20260720/last.pt` | `outputs/runs/e616-object-property-slot-bias-scratch80-20260720/` (local) | Fresh 80-step CPU scratch loop on E530, loss 26.5243; matched OOD `n=4` eval now parses 4/4 but stays byte-identical because Gallery's array closes empty before any item opens — not ship |
| E620 required-slot coverage replay (800-step) | `e620-required-slot-coverage-scratch800-20260720/last.pt` | `outputs/runs/e620-required-slot-coverage-scratch800-20260720/` (local) | 800 CPU scratch steps in 80.96s, loss 4.0680; OOD treatment fidelity 0.5500, structure 0.4886, strict-v2 0.0, AgentV 0/1. Lower loss regressed E619 generalization — rejected, not ship |
| E548 fresh TwoTower loop | `e548_training_loop_twotower_scratch_20260720/last.pt` | `outputs/runs/e548_training_loop_twotower_scratch_20260720/` (local) | Fresh 8-step CPU scratch loop on E530, loss 39.4267; no eval/sync, wiring only — not ship |
| E547 fresh TwoTower loop | `e547_training_loop_twotower_scratch_20260720/last.pt` | `outputs/runs/e547_training_loop_twotower_scratch_20260720/` (local) | Fresh 7-step CPU scratch loop on E530, loss 35.7431; no eval/sync, wiring only — not ship |
| E546 fresh TwoTower loop | `e546_training_loop_twotower_scratch_20260720/last.pt` | `outputs/runs/e546_training_loop_twotower_scratch_20260720/` (local) | Fresh 6-step CPU scratch loop on E530, loss 40.3390; no eval/sync, wiring only — not ship |
| E545 fresh TwoTower loop | `e545_training_loop_twotower_scratch_20260719/last.pt` | `outputs/runs/e545_training_loop_twotower_scratch_20260719/` (local) | Fresh 5-step CPU scratch loop on E530, loss 42.1226; no eval/sync, wiring only — not ship |
| E544 fresh TwoTower loop | `e544_training_loop_twotower_scratch_20260719/last.pt` | `outputs/runs/e544_training_loop_twotower_scratch_20260719/` (local) | Fresh 4-step CPU scratch loop on E530 after missing prior local checkpoint, loss 42.3848; no eval/sync, wiring only — not ship |
| E543 resumed TwoTower loop | `e543_training_loop_twotower_resume_scratch_20260719/last.pt` | `outputs/runs/e543_training_loop_twotower_resume_scratch_20260719/` (local) | Resumed E542 full-state to step 3 on E530, loss 39.7476; no eval/sync, wiring only — not ship |
| E542 resumed TwoTower loop | `e542_training_loop_twotower_resume_scratch_20260719/last.pt` | `outputs/runs/e542_training_loop_twotower_resume_scratch_20260719/` (local) | Resumed E541 full-state to step 2 on E530, loss 43.6742; no eval/sync, wiring only — not ship |
| E541 TwoTower training-loop iteration | `e541_training_loop_twotower_scratch_20260719/last.pt` | `outputs/runs/e541_training_loop_twotower_scratch_20260719/` (local) | One-step CPU scratch TwoTower loop on E530, loss 36.9158; no eval/sync, wiring only — not ship |
| E540 training-loop sentinel | `e540_training_loop_scratch_20260719/last.pt` | `outputs/runs/e540_training_loop_scratch_20260719/` (local) | One-step CPU scratch stub loop check on E530, loss 0.5; no eval/sync, wiring only — not ship |
| E531 visible-semantic-role checkpoint | `e531-e396-e530-replay050-slotrole1-honest-context-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e531-e396-e530-replay050-slotrole1-honest-context-r1-5k/` | Bucket-verified SHA `6b8c1abc…74a6154`; 5,059 target tokens in 99.72s. E532 structure improves slightly to 0.1431, but meaningful falls to 0.0, fidelity to 0.4667, reward to 0.3685, strict meaning stays 0.0, and AgentV is 0/1; rejected |
| E542 learned root-arity checkpoint | `e542-e531-root-reference-arity1-r1-24s/last.pt` | `outputs/runs/e542-e531-root-reference-arity1-r1-24s/` (local) | 24-step scratch continuation in 52.93s, SHA `2d5cd4b3…6854c5d8`; OOD `n=4` meaningful 0.50 / fidelity 0.5917 / structure 0.3019, but learned weight 1 is quality-neutral, strict meaning 0.0, AgentV 0/1; no sync or promotion |
| E543 bounded root-arity checkpoint | `e543-e531-root-reference-bounded-r1-24s/last.pt` | `outputs/runs/e543-e531-root-reference-bounded-r1-24s/` (local) | 24-step scratch continuation in 37.17s, SHA `c6be3791…51d7f90`; bounded loss improves calibration, but OOD `n=4` decisions and quality exactly match E542, strict meaning 0.0, AgentV 0/1; no sync or promotion |
| E544 root-identity checkpoint | `e544-e543-root-identity1-r2-24s/last.pt` | `outputs/runs/e544-e543-root-identity1-r2-24s/` (local) | 24-step scratch continuation in 40.96s, SHA `3b6e3c00…474f20c`; rank-only identity decode raises OOD `n=4` meaningful 0.00→0.25, structure 0.1250→0.1688, and recall 0.1458→0.2708, but strict meaning 0.0 and AgentV 0/1; no sync or promotion |
| E545 matched negative-weight checkpoints | `e545-e544-root-identity-neg{1-control,4}-r*/last.pt` | `outputs/runs/e545-…/` (local) | Matched 24-step scratch continuations in 30.64s / 28.64s, SHAs `9e54d470…76fa1` / `14dd4404…61ae`; weight 4 slightly improves sparse late negative accuracy, but predictions and OOD `n=4` metrics are identical, both regress from E544, and AgentV is 0/1; no sync or promotion |
| E546 matched strict-subset checkpoints | `e546-e544-strict-subset{1-control,5}-r*/last.pt` | `outputs/runs/e546-…/` (local) | Multiplier 5 raises strict-negative exposure 7→22 rows and improves OOD `n=4` fidelity 0.4250→0.6083, structure 0.1494→0.2038, reward 0.5078→0.8120, and AST edge F1 0→0.0417, but recall falls 0.2083→0.0625, meaning remains 0, AgentV 0/1; no sync or promotion |
| E547 moderate strict-subset checkpoint | `e547-e544-strict-subset2-r1-24s/last.pt` | `outputs/runs/e547-e544-strict-subset2-r1-24s/` (local) | 24-step multiplier-2 scratch run in 36.48s, SHA `37002bfd…0fc57`; OOD `n=4` structure 0.2248 and AST node F1 0.3270 lead the 1/2/5 ladder while recall stays 0.2083, but fidelity falls to 0.2583, meaning remains 0, AgentV 0/1; no sync or promotion |
| E551 no-lexeme-prior checkpoint | `e551-e544-strict-subset2-no-lexeme-r1-24s/last.pt` | `outputs/runs/e551-e544-strict-subset2-no-lexeme-r1-24s/` (local) | 24-step scratch run in 41.85s, SHA `e7921e66…dac32fc6`; fidelity improves to 0.3000, but structure falls to 0.1594 and recall to 0.1250; meaning 0, AgentV 0/1; no sync or promotion |
| E552 half-strength lexeme-prior checkpoint | `e552-e544-strict-subset2-lexeme05-r1-24s/last.pt` | `outputs/runs/e552-e544-strict-subset2-lexeme05-r1-24s/` (local) | 24-step scratch run in 34.75s, SHA `49a9c111…a151fc04`; fidelity 0.1333, structure 0.2181, recall 0.1250, reward 0.3435; meaning 0, AgentV 0/1; no sync or promotion |
| E553 corpus-local proportional-prior checkpoint | `e553-e544-prior-proportional-r3-24s/last.pt` | `outputs/runs/e553-e544-prior-proportional-r3-24s/` (local) | 24-step scratch run in 34.48s, SHA `510e55cf…e75399d`; fidelity 0.3000, structure 0.1244, recall 0.0625, reward 0.5453; meaning 0, AgentV 0/1; no sync or promotion |
| E554 next-slot-context checkpoint | `e554-e544-slot-next-context-r2-24s/last.pt` | `outputs/runs/e554-e544-slot-next-context-r2-24s/` (local) | 24-step scratch run in 39.91s, SHA `af3cbce7…c67b579`; fidelity 0.2583, structure 0.1594, recall 0.1250, reward 0.5328; meaning 0, AgentV 0/1; no sync or promotion |
| E555 slot-pair-interaction checkpoint | `e555-e544-slot-pair-interaction-r2-24s/last.pt` | `outputs/runs/e555-e544-slot-pair-interaction-r2-24s/` (local) | 24-step scratch run in 50.29s, SHA `af53e161…addf19e`; fidelity 0.3000, structure 0.1594, recall 0.1250, reward 0.5453; Pareto lever retained, meaning 0, AgentV 0/1; no sync or promotion |
| E556 combined-slot-context checkpoint | `e556-e544-slot-context-combined-r1-24s/last.pt` | `outputs/runs/e556-e544-slot-context-combined-r1-24s/` (local) | 24-step scratch run in 68.42s, SHA `139c670c…5831f0a`; fidelity 0.2167, structure 0.1594, recall 0.1250, reward 0.5203; combination rejected, meaning 0, AgentV 0/1 |
| E557 full-balance checkpoint | `e557-e544-slot-pair-balance1-r1-24s/last.pt` | `outputs/runs/e557-e544-slot-pair-balance1-r1-24s/` (local) | 24-step scratch run in 70.09s, SHA `438d9871…b97db05`; metrics exactly match E555; no sync or promotion |
| E558 owner-coverage engineering trial | `e558-e544-owner-coverage-r1-24s/last.pt` | `outputs/runs/e558-e544-owner-coverage-r1-24s/` (local) | 24-step scratch run in 43.31s, SHA `8a572738…de85382`; dirty-tree trial persisted but excluded from decisions |
| E558 owner-coverage checkpoint | `e558-e544-owner-coverage-r2-24s/last.pt` | `outputs/runs/e558-e544-owner-coverage-r2-24s/` (local) | 24-step scratch run in 43.74s, SHA `a45909df…381ede`; fidelity 0.4250 but structure/reward regress and AgentV fails; no sync or promotion |
| E559 twofold owner-coverage checkpoint | `e559-e544-owner-coverage2-r1-24s/last.pt` | `outputs/runs/e559-e544-owner-coverage2-r1-24s/` (local) | 24-step scratch run in 31.14s, SHA `1d11926d…9aac861`; fidelity 0.4417 and recall 0.2708, but reward 0.1643 and AgentV fails; no sync or promotion |
| E560 narrow owner-coverage checkpoint | `e560-e544-owner-threshold4-r1-24s/last.pt` | `outputs/runs/e560-e544-owner-threshold4-r1-24s/` (local) | 24-step scratch run in 42.26s, SHA `dae11cee…d7686a3`; structure 0.2181 and AST-node F1 0.3389, but semantic gates fail; no sync or promotion |
| E561 midpoint owner-coverage checkpoint | `e561-e544-owner-threshold7-r1-24s/last.pt` | `outputs/runs/e561-e544-owner-threshold7-r1-24s/` (local) | 24-step scratch run in 41.47s, SHA `35a4fe6d…3a127f9`; fidelity 0.5750, structure 0.2419, reward 0.5753, but meaning/AgentV fail; no sync or promotion |
| E568 design-context continuation checkpoint | `e568-e561-cont48-r1-48s/last.pt` | `outputs/runs/e568-e561-cont48-r1-48s/` (local) | 48-step scratch run in 116.24s, SHA `8dcc0804…0283a12b`; reward 0.6920 but fidelity/structure regress to 0.2583/0.1375 and meaning/AgentV fail; no sync or promotion |
| E569 matched continuation checkpoint | `e569-e561-matched-cont48-r1-48s/last.pt` | `outputs/runs/e569-e561-matched-cont48-r1-48s/` (local) | 48-step scratch run in 75.20s, SHA `8254fcf7…c6535f73`; meaning-v1 0.25, recall 0.3333, reward 0.6920, but strict meaning/AgentV fail; no sync or promotion |
| E572 fidelity-loss checkpoint | `e572-e569-fidelity2-r1-48s/last.pt` | `outputs/runs/e572-e569-fidelity2-r1-48s/` (local) | 48-step scratch run in 84.26s, SHA `bb6a58ff…cc29efa2`; fidelity 0.6500 and reward 0.8170, but meaning-v1/v2 0 and AgentV fails; no sync or promotion |
| E573 midpoint fidelity checkpoint | `e573-e569-fidelity1-r1-48s/last.pt` | `outputs/runs/e573-e569-fidelity1-r1-48s/` (local) | 48-step scratch run in 109.72s, SHA `ff21fc0c…cf59070d`; meaning-v1 0.25, fidelity 0.4750, reward 0.7570, but strict meaning/AgentV fail; no sync or promotion |
| E574 slot-loss checkpoint | `e574-e569-slotloss2-r1-48s/last.pt` | `outputs/runs/e574-e569-slotloss2-r1-48s/` (local) | 48-step scratch run in 76.23s, SHA `649cf512…3810b7c2`; aggregates exactly match E573 and strict meaning/AgentV fail; no sync or promotion |
| CAP5 evidence package | `cap5-03-evidence` | `docs/design/calculated-arity-adaptive-precision-results.md` | Reproducible evidence package for CAP0–CAP4 exact calculations and controlled fixtures; not a checkpoint or ship claim ([results](docs/design/calculated-arity-adaptive-precision-results.md)) |
| Production HF ship | *(none yet)* | [HF Bucket `TKendrick/OpenUI`](https://huggingface.co/buckets/TKendrick/OpenUI) `checkpoints/<run_id>/` | Register here after first full HF sync + `--ship-gates` |

**Load demo:** `python -m scripts.serve_playground` · **Full train sync:** set
`HF_TOKEN`, then `train_model --context-backend hf` (auto-uploads). Details,
eval tables, and history live in the model card.

## Quick start

```bash
# Node.js 20-22 is required for the locked bridge and browser dependencies.
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hf]"

# Official OpenUI parser + DESIGN.md bridges
cd src/apps/openui_bridge && npm ci && cd ../..
cd src/apps/design_md_bridge && npm ci && cd ../..

# optional MCP server deps
pip install -e ".[mcp]"
# optional live RICO download
pip install -e ".[rico]"
```

## Quick start (train / disjoint test)

Every pipeline phase is also reachable through the unified `slm` CLI
(`slm list` shows the full command map; `slm guide <phase>` prints the
matching operating reference from `.agents/skills/autotrain/references/`). The
`python -m scripts.<name>` forms below remain the direct equivalents.

```bash
# High-quality versioned corpus (default: all sources + quality synthesizer)
python -m scripts.build_train_data --source all --version v1 --synthesizer quality

# Fast fixture-only rebuild
python -m scripts.build_train_data --source fixture --version v0 --synthesizer quality

# Test suites with strict leakage checks against the train manifest
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json

# Full HF-context trains sync checkpoints to the OpenUI bucket
# (https://huggingface.co/buckets/TKendrick/OpenUI). Requires HF_TOKEN.
export HF_TOKEN=hf_...   # or: hf auth login
python -m scripts.train_model \
  --train-dir outputs/data/train/v1 \
  --model twotower \
  --context-backend hf \
  --steps 200 \
  --run-id twotower_v1
# → hf://buckets/TKendrick/OpenUI/checkpoints/twotower_v1/

python -m scripts.evaluate_model \
  --test-dir outputs/data/eval/v1 \
  --model twotower \
  --run-id twotower_v1 \
  --ship-gates
```

### Canonical lever registry

`ModelBuildConfig` is the single source for user-facing model, data, training,
decode, and evaluation lever defaults. Discover the complete machine-readable
set without searching scripts:

```bash
python -m slm_training.levers
python -m slm_training.levers --category decode
```

The catalog identifies intentional checkpoint-vs-harness default differences,
each decode lever's executable tokenizer/compiler configurations, and the
training objective that must exist in a checkpoint before a learned decode
head can be enabled. Invalid, untrained, or inert combinations fail during
config construction or checkpoint override, before run artifacts are made.
The repository-wide run cap is the sole policy lever owned directly by
`src/slm_training/levers.py`; changing `MAX_RUN_MINUTES` updates every Python
consumer. Local compute is the default experiment path. Remote CI and managed
jobs are optional last-resort execution surfaces and are not part of this local
lever registry.

Evaluation uses the [AgentEvals](https://agentevals.io/) JSONL/YAML contract
and the pinned AgentV SDK. Run `npm ci` before Python eval commands; shared
model, loss, task, and diagnostic eval paths automatically write AgentV bundles
beside their domain JSON under `<run-dir>/agentv/`. The existing honest OpenUI
ship gates remain authoritative. See
[the AgentV evaluation contract](docs/design/agentv-evaluation.md).

Local-only / CI scratch: add `--no-sync-checkpoints` (matrix scripts default to
scratch and stay local). Manual sync:
`python -m scripts.sync_checkpoints --run-dir outputs/runs/<id> --ensure-bucket`.
See [docs/design/checkpoint-bucket.md](docs/design/checkpoint-bucket.md).

Checkpoint provenance is fail-closed: each sync emits a verified
`CheckpointReferenceV1`, and `frontier`/`ship_candidate` citations must resolve
from a fresh clone or CI fails (`python -m scripts.verify_checkpoint_references
--check`). See
[docs/design/checkpoint-provenance.md](docs/design/checkpoint-provenance.md).

Honest ship path (V4 inventory-in-prompt / V6 stacked champion):

```bash
python -m scripts.run_quality_matrix --matrix v4 --only E35,E36 \
  --steps 40 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control

# V6: CoRe remask + slot-aware trust + honest V5 alphabet
python -m scripts.run_quality_matrix --matrix v6 --only E53 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control
```

Train artifacts land in `outputs/data/train/<version>/`; eval, preference,
annotation, trajectory, ProgramSpec, and mixture data use sibling typed roots.
Use `slm-data list`, `slm-data resolve train <version>`, and
`slm-data verify train <version>`
instead of memorizing paths. Selected immutable snapshots publish to Git with
`slm-data publish train <version>`.

Every new run writes `outputs/runs/<id>/trace.json` and OTLP JSONL signals under
`outputs/traces/<trace-id>/`. Set `OTEL_EXPORTER_OTLP_ENDPOINT` for an optional
remote OTLP mirror; detailed domain traces remain local and linked by trace ID.

The flush pipeline remains: curated seeds + RICO + Awwwards → deterministic
quality synth → per-record DESIGN.md + OpenUI validate → quality gates → stable
sort by `id` + content fingerprint.

Eval uses **meaningful parse** (rejects empty stacks, missing placeholders, and low gold component-type recall), strict `placeholder_fidelity` for ship gates, `structural_similarity`, and composite `reward_score` (does not credit gold DESIGN.md lint). Suites: smoke/held_out (fixtures), `rico_held`, adversarial, ood. Soft `placeholder_validity` is diagnostic only.

**Fixture demo vs ship:** a tiny upsample + scratch + smoke-only fail-under is wiring only. Readiness requires `--ship-gates` on the full scoreboard (see adversarial review).

Expand `rico_held` with 1500 additional HF RICO screens (cached under `src/slm_training/resources/rico/hf_test_cache.jsonl`):

```bash
python -m scripts.build_test_data \
  --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json \
  --rico-hf-split test --rico-limit 2600 --target-records 1500
```

```bash
# Lightweight unit/integration suite (iterative model training is excluded)
pytest

# Only suites affected by staged + unstaged local changes
.githooks/check-changed

# Repository layout, skill mirrors, and tracked-artifact policy
python -m scripts.repo_policy

# Explicit, compute-intensive model-training tests
pytest -m training
```

Enable the tracked pre-commit hook once per clone with
`git config core.hooksPath .githooks`. Claude Code, Codex, and Copilot CLI
hooks run the same changed-file checker automatically and reject raw `mv` for
tracked paths. See [`docs/repository-organization.md`](docs/repository-organization.md).

## OpenUI Lang

Fixtures and validation use official **`openuiLibrary`** syntax, e.g.:

```
root = Stack([hero], "column")
hero_title = TextContent(":hero.title")
hero_body = TextContent(":hero.body")
hero = Card([hero_title, hero_body])
```

Content props must be placeholder strings. Parsing/serialization/prompt generation come from `@openuidev/lang-core` + `@openuidev/react-ui` — see [`src/apps/openui_bridge/`](src/apps/openui_bridge/).

DESIGN.md conditioning + linter: [`src/apps/design_md_bridge/`](src/apps/design_md_bridge/) and [`src/slm_training/resources/design_md/`](src/slm_training/resources/design_md/).

## Mission Control dashboard

`serve_playground` serves a **control-plane + observability SPA** at `/` — one
pane of glass over the whole lifecycle (data → experiments → smoke →
checkpoints/promotion) — including the annotate playground at `/playground`.

```bash
pip install -e ".[dev,torch,web]"
python -m scripts.serve_playground --port 8765        # full control plane (local)
python -m scripts.serve_playground --no-enable-jobs   # read-only observability
# For network exposure, set SLM_ANNOTATION_TOKEN and add --public.
# open http://127.0.0.1:8765
```

Surfaces (React 19 + Vite SPA, dark-first "mission control" design system):

| Route | What |
| --- | --- |
| `/` Overview | Live jobs, experiment scoreboard, checkpoint roster, corpus health, system status, **remote dispatches** |
| `/data` | Navigate + generate versioned corpora (`build_train_data` / `build_test_data`) |
| `/experiments` | Quality / grammar / perf / phase matrices; run `run_*_matrix`; **dispatch bounded GPU checkpoint smokes** (`hf_jobs_train` / `remote_train`); drill into any run |
| `/smoke` | Smoke canary + perf & telemetry; launch wiring runs |
| `/checkpoints` | Roster + **live configurable ship gates** + promote / deploy + blinded A/B |
| `/runs/<id>` | Per-run detail — gate matrix, telemetry spans, `train_summary` metrics, durable-checkpoint link |
| `/playground` | Full annotate UI (React): staged generation, browser fallback/review, DSL repair, and feedback |

**Read vs execute.** Observability views are pure reads (work on a fresh checkout
and on read-only Vercel, falling back to committed `docs/design/*.json` /
`MODEL_CARD.md` / `src/slm_training/resources/`, tagged with `provenance`). Generate/run/promote
actions execute an **allowlisted** set of scripts as tracked background jobs with
live SSE logs — only when served locally (`--enable-jobs`, default on); Vercel
degrades to read-only automatically. Gate math (`POST /api/gates/evaluate`) is
pure, so the threshold editor stays live even read-only. Backend:
`src/slm_training/web/{observability,jobs,capabilities,routes}.py`; SPA source in
[`src/apps/dashboard/`](src/apps/dashboard/) (built bundle committed under
`web/static/app/`, like the preview lib).

**Compiled ↔ interpreted (dogfooding OpenUI).** The sidebar has a
**◈ Compiled / ◇ Interpreted** toggle. *Compiled* is the hand-written React above.
*Interpreted* renders each page from a committed **OpenUI Lang** program
(`src/slm_training/web/static/openui/<slug>.openui`) run **live** through the official
[`@openuidev`](https://openui.com) `<Renderer>` — same components, live `/api` data via a
tool provider, working nav, reactive selectors, launchers, and the live gate editor — so
the app *is* the DSL. The two are kept at parity (`scripts/validate_page_dsl.py` +
`tests/test_web/test_page_dsl.py` + the `dashboard-openui-parity` skill); interpreted-mode
source lives in [`src/apps/dashboard/src/interpret/`](src/apps/dashboard/src/interpret/).

## Annotate playground (`/playground`)

```bash
python -m scripts.serve_playground --port 8765
# open http://127.0.0.1:8765/playground
```

`/playground` is the React annotate UI inside the SPA shell (shares the dark
design system). It owns the complete annotation flow: bounded server attempts,
browser review/fallback, editable and validated DSL corrections, annotator/model
identity, bearer-token support, activity history, keyboard/swipe grading, and the
diffusion progress canvas. The retired `/playground/classic` URL redirects here.
If both model paths are unavailable, the page shows a clearly labeled wiring
fallback so the renderer/editor/annotation flow remains testable; uncorrected
fallback feedback is excluded from derived training data.

The demo checkpoint lives in `src/slm_training/resources/checkpoints/playground_demo/` (committed
`last.pt` + tokenizer + meta). To regenerate it:

```bash
python -m scripts.bootstrap_playground --force
```

If `last.pt` is missing after a sparse checkout, run the bootstrap command above
before starting the playground.
Annotate mode (default UI): auto-generated prompts, prefetch 1–2 samples ahead, and a live **OpenUI visual preview** (same `@openuidev/react-lang` `Renderer` path as [openui.com/demo](https://www.openui.com/demo/github)).

| Input | Action |
|-------|--------|
| `↑` | Thumbs up (persist, stay on sample) |
| `↓` | Thumbs down (persist, stay on sample) |
| `←` / `→` | Previous / next sample |
| typing | Focus optional note |
| swipe | Mobile: horizontal navigate, vertical grade |

Annotations append to `outputs/data/annotation/feedback.jsonl`. Invalid model outputs are quarantined to `outputs/data/annotation/bad_outputs.jsonl` (never shown in the app). Thumbs-up rows promote into `src/slm_training/resources/annotations/human_train.jsonl` (merged by `build_train_data`). Opposite ratings on the same prompt also write `outputs/data/preference/human_pairs.jsonl`.

```bash
python -m scripts.export_annotations status
python -m scripts.export_annotations export
```

### Rebuild the OpenUI preview bundle

```bash
npm run preview:install
npm run preview:build
# writes src/slm_training/web/static/preview/{preview.js,preview.css}
```

### Rebuild the dashboard bundle

```bash
npm run dashboard:install
npm run dashboard:build
# writes src/slm_training/web/static/app/ (built SPA, committed like the preview lib)
```

### Playwright visual / e2e

```bash
npm ci
npx playwright install chromium
# optional agent skills (already in .agents/skills + .cursor/skills)
playwright-cli install --skills
npm run test:e2e
```

MCP (Cursor): [`.cursor/mcp.json`](.cursor/mcp.json) launches `@playwright/mcp`.


- **Context tower**: scratch TokenEncoder **or** frozen HF model (`--context-backend hf`, default `HuggingFaceTB/SmolLM2-135M`)
- **Denoiser tower**: MaskGIT-style masked token prediction with cross-attention to context ([Chang et al. 2022](https://arxiv.org/abs/2202.04200); adapted)
- **Grammar decode**: DFA force-emit + MaskGIT hole-admit + LTR certify so constrained samples stay valid OpenUI ([research lineage](docs/design/research-lineage.md); `--no-grammar` to disable)
- **Output tokenizer**: dual-mode — default **compositional** `OpenUITokenizer`, or V5 **lexer / DSL-native** `DSLNativeTokenizer` (`output_tokenizer=lexer`; see [dsl-native-tokenizer.md](docs/design/dsl-native-tokenizer.md))
- **Eval**: syntax `parse_rate`, separate `meaningful_program_rate`, placeholder fidelity, and canonical tree match — no hidden gold channel at generate time

```bash
# Optional HF context (requires: pip install -e ".[hf]")
python -m scripts.train_model --model twotower --context-backend hf \
  --hf-model HuggingFaceTB/SmolLM2-135M --steps 200 --run-id twotower_hf --fast-train
```

## Hugging Face Jobs (full GPU train)

ZeroGPU Spaces are for short demos only. Full trains use managed Jobs:

```bash
python -m scripts.hf_jobs_train --dry-run --run-id twotower_jobs_v1 --steps 200
# submit: export HF_TOKEN=… && python -m scripts.hf_jobs_train --run-id … --steps 200
```

Details: [docs/design/hf-jobs-train.md](docs/design/hf-jobs-train.md).

## GPU multi-farm MCP

```bash
cp .env.example .env
pip install -e ".[mcp]"
GPU_MULTI_FARM_MODE=mock python -m scripts.multi_farm_mcp
```

## Agent instructions

All coding agents (Cursor, Claude Code, Codex, Gemini, Copilot / GHCP, …) must
follow **[AGENTS.md](AGENTS.md)**. Canonical skills live in
[`.agents/skills/`](.agents/skills/) (mirrored under `.claude/skills/` and
`.cursor/skills/`).

**Iron law:** after any train / eval / bench / profile / telemetry / matrix /
reproduction (or decision-informing ad-hoc) run, update `docs/design/` JSON
**and** the matching measured-results markdown. Full trigger list and recipe
checklist: [AGENTS.md](AGENTS.md) (skill: `documenting-experiment-results`).
Do not leave results only under `outputs/`.

All eval entrypoints also publish standard AgentEvals cases and AgentV SDK
artifacts. Do not add evaluator-specific envelope formats; extend
`src/slm_training/evals/agentv.py`.

### Token-efficiency stack

Repo ships **ponytail**, **caveman**, **headroom**, and **rtk** under
`.agents/skills/` (plus [`RTK.md`](RTK.md), Cursor rules, and GHCP
`.github/copilot-instructions.md`). Details and refresh commands:
[AGENTS.md — Token-efficiency stack](AGENTS.md).

```bash
# RTK binary (once per machine) — must pass `rtk gain`
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
```

### OpenWiki (code mode)

Repository wiki for agents lives under [`docs/openwiki/`](docs/openwiki/) (start at
[`docs/openwiki/quickstart.md`](docs/openwiki/quickstart.md)). Setup uses
[langchain-ai/openwiki](https://github.com/langchain-ai/openwiki) code mode:
[`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) OpenWiki snippets and
[`.github/workflows/openwiki-update.yml`](.github/workflows/openwiki-update.yml).

```bash
npm install -g openwiki@0.1.2
# needs OPENAI_API_KEY (preferred) or OPENROUTER_API_KEY
python -m scripts.update_openwiki --update --print
```

Add repo secret `OPENAI_API_KEY` to enable scheduled OpenWiki update PRs. The
workflow falls back to `OPENROUTER_API_KEY` when OpenAI is unavailable and
fails clearly when neither secret exists. `LANGSMITH_API_KEY` enables optional
tracing.

### Hugging Face CLI + skills

Agents use the official `hf` CLI and the
[huggingface/skills](https://github.com/huggingface/skills) pack (skill:
`hf-cli` plus datasets / papers / trainers / Spaces / … under
[`.agents/skills/`](.agents/skills/)). Cursor also gets the Hugging Face MCP
server via [`.cursor/mcp.json`](.cursor/mcp.json).

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf skills add --force
hf skills update
hf skills add --claude --force
hf skills add --dest=.cursor/skills --force
```

Optional Cursor UI: [marketplace — Hugging Face](https://cursor.com/marketplace/huggingface).
CLI docs: [huggingface_hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
Tokens: [settings/tokens](https://huggingface.co/settings/tokens).

### Serena MCP

Semantic code tools via [Serena](https://github.com/oraios/serena) (not
marketplace installs). Project is initialised under [`.serena/`](.serena/);
Cursor / Claude / VS Code MCP configs are wired in-repo. See
[AGENTS.md — Serena MCP](AGENTS.md).

```bash
uv tool install -p 3.13 serena-agent
serena init
serena project health-check
```

## Layout

```
AGENTS.md              # cross-tool agent instructions (required reading)
RTK.md                 # Rust Token Killer usage (shell output compression)
docs/MODEL_CARD.md     # checkpoint roster + eval (README holds a summary)
docs/repository-organization.md # tracked-file placement + move policy
.agents/skills/        # canonical agent skills
src/slm_training/
  dsl/                 # OpenUI adapter + design_md + grammar/{backends,fastpath}
  harnesses/           # train_data, test_data, model_build, rl, preference,
                       # distill, quality(+retrieval), experiments, annotations
  models/              # TwoTower, grammar_diffusion, tokenizers, remask
  data/                # RICO / Awwwards adapters + leakage fingerprints
  evals/               # loss suites / denoising NLL
  runtime/             # accel, telemetry, compression, cactus
  web/                 # mission-control API (observability + jobs) + annotate playground + SPA
src/gpu_multi_farm/    # FastMCP server + farm adapters
src/apps/openui_bridge/   # @openuidev/lang-core Node sidecar
src/apps/design_md_bridge/
src/apps/openui_preview/
scripts/               # CLIs
src/slm_training/resources/              # seed pairs + RICO semantic slices
docs/design/           # architecture + research lineage + contracts
tests/
  test_dsl/            # parser, grammar, design_md
  test_harnesses/      # mirrors harnesses/* (rl is its own suite)
  test_runtime/        # accel / cactus / compression
  test_models/ test_data/ test_web/ ...
```
