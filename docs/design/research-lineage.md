# Research lineage

This repo borrows ideas from several papers and systems. This page is the
**source of truth** for what we cite, what we implement, and where the code
lives. Prefer linking here instead of repeating informal “inspired by 2026 …”
notes elsewhere.

Legend:

| Tag | Meaning |
| --- | --- |
| **Faithful** | Core algorithm matches the cited work’s intent |
| **Adapted** | Same idea, specialized / simplified for OpenUI TwoTower |
| **Surrogate** | Named like the paper’s method, but not textbook training |
| **Adjacent** | Related prior art we deliberately did *not* reimplement |

---

## Decode stack (grammar + MaskGIT)

### MaskGIT (masked visual token modeling)

| | |
| --- | --- |
| **Paper** | Chang et al., *MaskGIT: Masked Generative Image Transformer*, CVPR 2022. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200) |
| **Fidelity** | **Adapted** — discrete masked prediction + iterative unmask, applied to OpenUI token sequences instead of image tokens |
| **Code** | `TwoTowerModel` denoiser + `_generate_maskgit_one` in [`src/slm_training/models/twotower.py`](../../src/slm_training/models/twotower.py); unmask policies in [`parallel_decode.py`](../../src/slm_training/models/parallel_decode.py) |
| **Config** | `gen_steps`, `parallel_unmask` ∈ `{topk, confidence, adaptive}` |

Classic MaskGIT unmasks the top‑*k* most confident positions each step
(`parallel_unmask=topk`). Our `confidence` / `adaptive` modes add rising
thresholds and neighbor spacing (mean-field-lite conflict suppression) used in
recent discrete diffusion LLM decode practice — see also the parallel-unmask
row below.

### Constrained decoding for diffusion LLMs (CFG ∩ completions)

| | |
| --- | --- |
| **Paper** | Mündler, Dekoninck, Vechev, *Constrained Decoding of Diffusion LLMs with Context-Free Grammars*, 2025. [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) · site: [constrained-diffusion.ai](https://constrained-diffusion.ai/) |
| **Fidelity** | **Adapted** — we keep the *admit / emptiness* idea for MaskGIT holes, but use a cheap OpenUI specialization (Lark left-span sync + benign `hole` reparse) rather than the paper’s full CFG∩NFA emptiness engine |
| **Code** | `admit_fill` in [`dsl/grammar/fastpath/maskgit_constrain.py`](../../src/slm_training/dsl/grammar/fastpath/maskgit_constrain.py); design note [`grammar-fastpath.md`](grammar-fastpath.md) |
| **Config** | `grammar_fastpath`, `grammar_fastpath_mode` ∈ `{force, mask, hybrid}` |

**What we took:** reject a MaskGIT fill when the partial canvas cannot complete
in the target grammar (CFG ∩ completion language empty ⇒ refuse).

**What we did not take:** the full intersection emptiness algorithm, multi-region
infilling formalism, or C2F⁺ε optimizations from the paper. OpenUI’s LALR
incremental acceptor + hole probe is the practical stand-in.

### Speculative / force-emit constrained decode

| | |
| --- | --- |
| **Papers** | Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). Related: grammar-constrained LTR systems (Outlines / Guidance / SynCode family — **Adjacent**) |
| **Fidelity** | **Adapted** (“pseudo speculative”) — when the OpenUI DFA has a **singleton structural** continuation (`=` `(` `)` `[` `]` `,`), we **force-emit** that token and skip the denoiser forward; otherwise we mask logits / search only DFA-legal ids (`pick_constrained_token`) |
| **Code** | `force_emit_token_id`, `dfa_admits_token`, `pick_constrained_token` in [`models/grammar.py`](../../src/slm_training/models/grammar.py); engine in [`dsl/grammar/fastpath/engine.py`](../../src/slm_training/dsl/grammar/fastpath/engine.py); LTR repair / certify in `TwoTowerModel._ensure_valid_openui` |
| **Config** | `grammar_constrained`, `grammar_ltr_primary`, `grammar_ltr_repair`, `grammar_finalize_validate`, `grammar_top_k` |

This is **not** draft-model speculative decoding (no separate draft LM). It is
speculative in the sense of: propose from the model (or DFA force), **verify**
against the Lark incremental acceptor + stream hard-errors, reject illegal
tokens, repair/finalize until `validate` succeeds.

Playground load forces the certify path so grammar-constrained samples cannot
return invalid OpenUI.

### Adaptive / confidence parallel unmask (dLLM-style)

| | |
| --- | --- |
| **Lineage** | MaskGIT schedules + recent discrete diffusion LLM decode (confidence thresholds, conflict suppression). We do **not** pin a single 2025–2026 paper as a faithful reimplementation |
| **Fidelity** | **Adapted** |
| **Code** | [`models/parallel_decode.py`](../../src/slm_training/models/parallel_decode.py) |
| **Notes** | [`accel-parallel.md`](accel-parallel.md), [`runtime-performance.md`](runtime-performance.md) |

### MDLM continuous-time absorbing mask (training)

| | |
| --- | --- |
| **Paper** | Sahoo et al., *Simple and Effective Masked Diffusion Language Models*, NeurIPS 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524). Related: LLaDA (Nie et al., 2025) |
| **Fidelity** | **Adapted** — log-linear `α(t)=1-t` ⇒ per-row mask rate `t` and CE weight `1/t`; not a full continuous-time ELBO stack |
| **Code** | `_mask_targets` + weighted CE in [`models/twotower.py`](../../src/slm_training/models/twotower.py) |
| **Config** | `mdlm_schedule`, `mdlm_eps` |

### Structure-aware online corruption and variable-length canvases

| | |
| --- | --- |
| **Lineage** | Masked discrete-diffusion denoising specialized to the OpenUI program structure and ProgramSpec edit metadata |
| **Fidelity** | **Adapted** — online token, statement, balanced-subtree, reference-group, edit-local, disjoint, all-mask, reorder, and insert/delete corruption; this is not AST graph diffusion |
| **Code** | [`data/diffusion/`](../../src/slm_training/data/diffusion), TwoTower `_online_diffusion_targets` and target-length head in [`models/twotower.py`](../../src/slm_training/models/twotower.py) |
| **Config** | `mask_pattern=diffusion`, `diffusion_policies`, `diffusion_length_buckets`, `diffusion_overallocate`, `diffusion_length_loss_weight` |
| **Docs** | [`diffusion-data-adapter.md`](diffusion-data-adapter.md) |

### Confidence remasking (self-correction)

| | |
| --- | --- |
| **Lineage** | GIDD / ReMDM / LLaDA remasking — revise low-confidence committed tokens mid-diffusion |
| **Fidelity** | **Adapted** — remask bottom-`q` confidence known tokens each MaskGIT step (`remask_ratio`) |
| **Code** | `select_remask_indices` in [`models/parallel_decode.py`](../../src/slm_training/models/parallel_decode.py); loop in `_generate_maskgit_one` |
| **Config** | `remask_ratio` |

### Slot-contract template fill (DSL-native)

| | |
| --- | --- |
| **Lineage** | Spec/inventory-conditioned structured generation (adjacent to schema-constrained decode) |
| **Fidelity** | **Adapted** — build a valid OpenUI skeleton from `record.placeholders`, seed MaskGIT, remask binder/content positions |
| **Code** | [`models/template_fill.py`](../../src/slm_training/models/template_fill.py); `_generate_maskgit_one` / `_ensure_valid_openui` |
| **Config** | `template_fill_decode` (+ slot contract flags) |

---

## Training stack

### Canonical two-track iteration

Production iteration is lineage-first rather than E-series-first. The
TwoTower track branches from the frozen E53 recipe; the causal track branches
with [LoRA](https://arxiv.org/abs/2106.09685) from one permanently locked Qwen
base selected by an identical bakeoff. Incremental SFT mixes 10% validated
champion history following the [On-Policy Replay](https://arxiv.org/abs/2605.29495)
direction. Compatible sibling deltas may be tested with
[Model Soups](https://arxiv.org/abs/2203.05482) averaging and
[TIES-Merging](https://arxiv.org/abs/2306.01708); merge output is always a new
challenger. Causal grammar mask caching is adapted from the systems direction
of [XGrammar](https://arxiv.org/abs/2411.15100). Implementation and exact gates:
[`model-lineage.md`](model-lineage.md).

### Two-tower conditioning

| | |
| --- | --- |
| **Lineage** | Dual-encoder / conditioned generators (classic two-tower retrieval is **Adjacent**; here “TwoTower” means **context encoder + trainable denoiser**) |
| **Fidelity** | **Adapted** naming — architecture is conditioned MaskGIT, not DSSM retrieval |
| **Code** | [`models/twotower.py`](../../src/slm_training/models/twotower.py), [`models/context.py`](../../src/slm_training/models/context.py), [`models/blocks.py`](../../src/slm_training/models/blocks.py) |

### Preference / “DPO”

| | |
| --- | --- |
| **Paper** | Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| **Fidelity** | **Surrogate** — reference-free loss on **masked denoiser log-probs**, not textbook DPO with a frozen reference policy over autoregressive likelihoods |
| **Code** | [`harnesses/preference/train.py`](../../src/slm_training/harnesses/preference/train.py) (`dpo_loss`), pair builders in [`harnesses/preference/`](../../src/slm_training/harnesses/preference/) |
| **CLI** | `scripts/train_preference.py` |

### GRPO-lite

| | |
| --- | --- |
| **Paper** | Shao et al., *DeepSeekMath* (Group Relative Policy Optimization), 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| **Fidelity** | **Surrogate / lite** — group rollouts + mean/std advantages over a **structure-only** reward; not a full RLHF stack |
| **Code** | [`harnesses/rl/`](../../src/slm_training/harnesses/rl/) (`grpo_loss_for_group`, `train_grpo`) |
| **CLI** | `scripts/train_rl.py`; matrix row E10 in [`quality-experiment-matrix.md`](quality-experiment-matrix.md) |

---

## DSL-native output representation (V5)

| | |
| --- | --- |
| **Papers** | Rabinovich et al., *Abstract Syntax Networks* [arXiv:1704.07535](https://arxiv.org/abs/1704.07535); Kusner et al., *Grammar VAE* [arXiv:1703.01925](https://arxiv.org/abs/1703.01925); Xue et al., *ByT5* [arXiv:2105.13626](https://arxiv.org/abs/2105.13626); CFG-constrained diffusion LMs [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) |
| **Fidelity** | **Adapted** — lexer-native categorical tokens + dynamic symbol table + byte literal channel + kind-factorized embeddings; **not** production-rule sequences or graph diffusion |
| **Code** | [`dsl_tokenizer.py`](../../src/slm_training/models/dsl_tokenizer.py), kind masks in [`dsl/grammar/fastpath/token_map.py`](../../src/slm_training/dsl/grammar/fastpath/token_map.py), structural mask/remask in [`twotower.py`](../../src/slm_training/models/twotower.py) |
| **Docs** | [`dsl-native-tokenizer.md`](dsl-native-tokenizer.md); matrix **E40–E46** |
| **CLI** | `scripts/diagnose_tokenizer.py`; `scripts/run_quality_matrix.py --matrix v5` |

---

## Correction / revision

V3 **Adapted** confidence remasking (`remask_ratio`) and template fill.
**V4** adds critic-guided / trust-head remask and honest inventory contracts —
write-up: [`research-correction-critics.md`](research-correction-critics.md);
levers **E30–E36** in [`quality-experiment-matrix.md`](quality-experiment-matrix.md).

### ReMDM (inference-time remasking)

| | |
| --- | --- |
| **Paper** | *Remasking Discrete Diffusion Models with Inference-Time Scaling*, 2025. [arXiv:2503.00307](https://arxiv.org/abs/2503.00307) |
| **Fidelity** | **Adapted** — V3 bottom-`q` confidence remask; V4 E30 LTR suffix rollback + E33 budgeted remask |
| **Intent** | Let pretrained MaskGIT weights revise committed tokens without GIDD-style retraining |
| **Hook** | [`select_remask_indices`](../../src/slm_training/models/parallel_decode.py) / [`select_remask_policy_indices`](../../src/slm_training/models/parallel_decode.py); `_greedy_ltr_decode_batch` suffix window |

### RemeDi (learned unmasking-policy stream)

| | |
| --- | --- |
| **Paper** | *Don’t Settle Too Early: Self-Reflective Remasking for Diffusion Language Models*, 2025. [arXiv:2509.23653](https://arxiv.org/html/2509.23653v1) |
| **Fidelity** | **Adapted** (lite) — [`FastPathGate`](../../src/slm_training/dsl/grammar/fastpath/gate.py) trained via [`trust_train.py`](../../src/slm_training/dsl/grammar/fastpath/trust_train.py) (E31); gates E33 remask; **E52** extends mining to placeholder/slot-binding errors (`slot_aware_trust_gate`) |
| **Intent** | Cheap per-token reliability head for remask budgets |

### BackPlay (frozen-model correction head)

| | |
| --- | --- |
| **Paper** | *BackPlay: Plug-in Look-Back Self-Correction for Diffusion Language Models*, 2026. [arXiv:2601.06428](https://arxiv.org/html/2601.06428v2) |
| **Fidelity** | **Adapted** — freeze denoiser, mine own errors, train plug-in gate (E31/E52) |
| **Intent** | Model-specific remask scores without joint generator–critic optimization |

### CoRe (context-robust remasking)

| | |
| --- | --- |
| **Paper** | *CoRe: Context-Robust Remasking for Diffusion Language Models*, 2026. [arXiv:2602.04096](https://arxiv.org/abs/2602.04096) |
| **Fidelity** | **Adapted** (lite) — E50: second forward under neighbor-masked context; remask highest support-drop tokens (`select_remask_core_indices`, `core_instability_scores`) |
| **Config** | `remask_policy` ∈ `{confidence, core, combined}`, `core_perturb_frac` |
| **Intent** | Training-free revision targeting context-brittle commitments (not stale confidence alone) |

### T2M (token-to-mask refinement)

| | |
| --- | --- |
| **Paper** | *Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement*, 2026. [arXiv:2605.26436](https://arxiv.org/html/2605.26436v1) |
| **Fidelity** | **Adapted** — E51: remask always resets to `<mask>` (`remask_to_mask=True`); statement-span expansion via `remask_span=statement` |
| **Intent** | Avoid token-edit pollution; re-denoise under cleaner mask noise |

### GIDD / SCDD (revision in the training process)

| | |
| --- | --- |
| **Papers** | *Generalized Interpolating Discrete Diffusion* [arXiv:2503.04482](https://arxiv.org/abs/2503.04482); *Generalized Discrete Diffusion with Self-Correction* [arXiv:2603.02230](https://arxiv.org/html/2603.02230v1) |
| **Fidelity** | **Adapted** (lite) — `visible_corrupt_rate` in `_mask_targets` (E32); full hybrid diffusion still Adjacent |
| **Intent** | Teach the denoiser that visible tokens can be wrong |

### Honest inventory-in-prompt (DSL-native)

| | |
| --- | --- |
| **Lineage** | Schema / inventory-conditioned structured generation |
| **Fidelity** | **Adapted** — E35: `inventory_from_prompt` / `ensure_prompt_inventory` in [`template_fill.py`](../../src/slm_training/models/template_fill.py); `_resolve_slot_contract` ignores silent gold channels when `honest_slot_contract=True` |
| **Config** | `honest_slot_contract` |

### Latent falsification MoE (research design)

| | |
| --- | --- |
| **Lineage** | Coconut continuous latent reasoning [2412.06769](https://arxiv.org/abs/2412.06769); SPC adversarial critics [2504.19162](https://arxiv.org/html/2504.19162v1); sparse MoE reward/critic specialization [2606.04284](https://arxiv.org/abs/2606.04284); LLaDA-MoE (generator experts — do not confuse) [2509.24389](https://arxiv.org/abs/2509.24389); counterfactual MoE routing [2605.07260](https://arxiv.org/abs/2605.07260); PLR parallel latent streams [2601.03153](https://arxiv.org/abs/2601.03153); MIRAGE continuous agent latent CoT [2606.04627](https://arxiv.org/abs/2606.04627) (Adjacent — not a parallel-slot recipe) |
| **Fidelity** | **Adjacent** — long-horizon design only (E34); deferred until residual failures after E33+E35 |
| **Design note** | [`research-correction-critics.md`](research-correction-critics.md) |

### Related decode-order / remask priors

| Idea | Paper | Status here |
| --- | --- | --- |
| Deferred commitment / sliding windows | [arXiv:2601.02076](https://arxiv.org/abs/2601.02076) | **Adapted** (lite) — E30 revisable LTR window |
| Token ordering / “visible ≠ revisable” | [arXiv:2502.06768](https://arxiv.org/html/2502.06768v1) | **Adjacent** — motivates E32 |
| Remask, don’t replace | [arXiv:2604.18738](https://arxiv.org/abs/2604.18738) | **Adapted** — E33/E51 remask→mask constraint (`remask_to_mask`) |
| CoRe context-robust remask | [arXiv:2602.04096](https://arxiv.org/abs/2602.04096) | **Adapted** (lite) — E50 |
| T2M token-to-mask | [arXiv:2605.26436](https://arxiv.org/html/2605.26436v1) | **Adapted** — E51 |

---

## Verifier-guided planning & repair (Adjacent lineage)

These papers motivate **verifier-derived supervision**, candidate–repair loops,
and diffusion-native policy optimization. We do **not** reimplement PDDL
planners or VAL in this repo. The applicability mapping (what already maps to
OpenUI TwoTower, what is a real gap, what is out of scope) lives in
[`verifier-guided-repair.md`](verifier-guided-repair.md).

### PDDL-Instruct (logical CoT instruction tuning for planning)

| | |
| --- | --- |
| **Paper** | *Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning*, LM4Plan @ ICAPS 2025 / preprint. [arXiv:2509.13351](https://arxiv.org/abs/2509.13351) |
| **Fidelity** | **Adjacent** — we adopt the *idea* of verifier-derived process supervision and detailed failure feedback, not the PDDL/VAL training loop or the paper’s underspecified discrete step/plan losses |
| **Takeaway** | Prefer executable traces + localized verifier diagnostics over prose CoT; keep verification at inference time |
| **Do not use** | Unofficial third-party “symbolic-planning” sketches as a trusted parser/executor base |
| **Design map** | [`verifier-guided-repair.md`](verifier-guided-repair.md) |

### LLM-Modulo / CEGIS planning (candidate–verifier–counterexample)

| | |
| --- | --- |
| **Lineage** | LLM proposes candidates; symbolic verifier returns counterexamples; model repairs (LLM-Modulo / neuro-symbolic CEGIS planning family). Related: LLM+P (LLM formalizes, classical planner searches) — we keep the *division of labor* lesson without adopting planners |
| **Fidelity** | **Adjacent** — closest architectural precedent for our certify → remask → re-denoise loop |
| **Code analogue** | `_ensure_valid_openui` + remask policies in [`models/twotower.py`](../../src/slm_training/models/twotower.py), [`models/parallel_decode.py`](../../src/slm_training/models/parallel_decode.py) |
| **Design map** | [`verifier-guided-repair.md`](verifier-guided-repair.md) §3–§4 |

### FoVer (formal tools → process verifiers)

| | |
| --- | --- |
| **Paper** | *Training Step-Level Reasoning Verifiers with Formal Verification Tools*, 2025. [arXiv:2505.15960](https://arxiv.org/abs/2505.15960) |
| **Fidelity** | **Adjacent** — motivates distilling expensive formal checks into a compact process model while retaining the formal tool as authority |
| **Code analogue** | `FastPathGate` + BackPlay-lite mining ([`dsl/grammar/fastpath/gate.py`](../../src/slm_training/dsl/grammar/fastpath/gate.py), [`trust_train.py`](../../src/slm_training/dsl/grammar/fastpath/trust_train.py)); grammar remains legality authority |
| **Proposed** | Calibration / abstention (**E63** in the mapping doc; E53 is the shipped V6 honest champion) |

### MDPO / d1 (trajectory-aligned masked-diffusion RL)

| | |
| --- | --- |
| **Papers** | *MDPO: Overcoming the Training-Inference Divide of Masked Diffusion Language Models*, 2025. [arXiv:2508.13148](https://arxiv.org/abs/2508.13148). Related: d1 masked-diffusion policy optimization [arXiv:2504.12216](https://arxiv.org/abs/2504.12216); PAPO / dOPSD-style dense intermediate rewards (Adjacent) |
| **Fidelity** | **Adjacent** — candidates to replace the GRPO-lite **Surrogate** on final strings |
| **Code today** | [`harnesses/rl/`](../../src/slm_training/harnesses/rl/) GRPO-lite; preference stage in [`harnesses/preference/train.py`](../../src/slm_training/harnesses/preference/train.py) |
| **Proposed** | Trajectory-aligned objective on intermediate MaskGIT states (**E64**; E54/E55 are shipped V6 grammar-honest / process stages) |

### Constrained diffusion decoding (LAVE / EPIC family)

| | |
| --- | --- |
| **Paper** | *Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars*, 2026. [arXiv:2602.00612](https://arxiv.org/abs/2602.00612). Related: Mündler et al. CFG∩completions [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) (already **Adapted** above) |
| **Fidelity** | **Adjacent** — diffusion-native constrained decode beyond our cheap hole-admit stand-in |
| **Code analogue** | `admit_fill` in [`dsl/grammar/fastpath/maskgit_constrain.py`](../../src/slm_training/dsl/grammar/fastpath/maskgit_constrain.py) |

### PlanBench / generalization gap / CoT brittleness

| | |
| --- | --- |
| **Papers** | PlanBench [arXiv:2206.10498](https://arxiv.org/abs/2206.10498); *On the Generalization Gap in LLM Planning* [arXiv:2601.14456](https://arxiv.org/abs/2601.14456); Chain-of-Thoughtlessness / related CoT collapse under complexity |
| **Fidelity** | **Adjacent** — motivates **schema-level** held-out splits (unseen component families, symbol rename), not only held-out instances |
| **Proposed** | **E65** + `toy-layout` transfer stress; see [`verifier-guided-repair.md`](verifier-guided-repair.md) §4 |

### LLM+P (neural formalize, symbolic search)

| | |
| --- | --- |
| **Paper** | LLM+P / related “LLM writes PDDL, classical planner solves” [arXiv:2304.11477](https://arxiv.org/abs/2304.11477) |
| **Fidelity** | **Adjacent** — lesson only: do not force the neural model to relearn exact search when a checker/planner is cheap. Here the “planner” stand-in is the OpenUI grammar stack + optional best-of-N, not Fast Downward |

---

## Speculative denoising (V7)

Design write-up: [`speculative-denoising.md`](speculative-denoising.md).
V7 adapts the AR speculative-decoding program (survival scheduling, outcome
fanout, successor caching) to the TwoTower MaskGIT decode by imposing a
temporary verification order over attention-derived dependency clusters. The
grammar stack remains the verifier; no draft LM is introduced.

### LESS (mutual-stability sampling)

| | |
| --- | --- |
| **Paper** | *LESS Is More: Mutual-Stability Sampling for Diffusion Language Models*, 2026. [arXiv:2606.16908](https://arxiv.org/html/2606.16908v1) |
| **Fidelity** | **Adapted** — E70: training-free top-1 persistence + inter-step Jensen–Shannon divergence as commit/remask signals; not the paper's full mutual-stability sampler |
| **Code** | `StabilityTracker`, `select_remask_stability_indices` in [`models/parallel_decode.py`](../../src/slm_training/models/parallel_decode.py) |
| **Config** | `remask_policy=stability`, `stability_min_persistence`, `stability_jsd_weight` |

### DAPD / DAWN / CLAD (attention-derived dependency structure)

| | |
| --- | --- |
| **Papers** | *DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs*, 2026. [arXiv:2603.12996](https://arxiv.org/html/2603.12996v2); *DAWN: Dependency-Aware Fast Inference for Diffusion LLMs*, 2026. [arXiv:2602.06953](https://arxiv.org/html/2602.06953v1); CLAD span-commit units (**Adjacent**) |
| **Fidelity** | **Adapted** — E71: last-layer attention coupling → greedy commit clusters + anchor-first ordering; we do not reimplement either paper's full dependency-graph scheduler |
| **Code** | `build_dependency_clusters`, `order_clusters` in [`models/speculative_denoise.py`](../../src/slm_training/models/speculative_denoise.py); `return_attn` in [`models/blocks.py`](../../src/slm_training/models/blocks.py) |
| **Config** | `unmask_mode=cluster`, `cluster_attn_threshold`, `cluster_max_size` |

### Self-Speculative Masked Diffusions (draft–verify inside one model)

| | |
| --- | --- |
| **Paper** | *Self-Speculative Masked Diffusions*, 2025. [arXiv:2510.03929](https://arxiv.org/html/2510.03929v2) |
| **Fidelity** | **Adapted** (structural idea only) — E72: noncausal trunk drafts all positions, a small sequential verifier checks them under a temporary order. Our verifier is the **grammar acceptor**, not a trained permutation-causal section; the residualized causal head is **Adjacent** (see deferred table in the design doc) |
| **Code** | `verify_clusters_ordered` in [`models/speculative_denoise.py`](../../src/slm_training/models/speculative_denoise.py); cluster loop in `_generate_maskgit_one` |
| **Config** | `cluster_verify` |

### DSpark (confidence-scheduled speculative decoding)

| | |
| --- | --- |
| **Paper** | *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*, 2026. [arXiv:2607.05147](https://arxiv.org/html/2607.05147v1) |
| **Fidelity** | **Adapted** — E73: survival head predicting whether a committed token survives to the final canvas; cumulative cluster-survival commit budget replaces raw `remaining/steps` scheduling. The low-rank Markov / recurrent draft-repair head and throughput-curve-aware serving scheduler are **Adjacent** |
| **Code** | [`dsl/grammar/fastpath/survival_train.py`](../../src/slm_training/dsl/grammar/fastpath/survival_train.py); `survival_gate` head on `TwoTowerModel`; budget logic in [`models/speculative_denoise.py`](../../src/slm_training/models/speculative_denoise.py) |
| **Config** | `survival_gate`, `survival_gate_train`, `survival_commit_threshold` |

### Saguaro (speculative speculative decoding)

| | |
| --- | --- |
| **Paper** | *Speculative Speculative Decoding*, 2026. [arXiv:2603.03251](https://arxiv.org/html/2603.03251v3) |
| **Fidelity** | **Adapted** — E74: while grammar verification of the current transition resolves, precompute next-pass logits for the top-K likely verifier outcomes in one batched forward; cache hit skips the next forward. Hidden-state adapters (`G_ψ`, resume at layer d+1) are **Adjacent** — at ≤6 layers / d_model ≤192 a full batched forward is as cheap and drift-free |
| **Code** | `enumerate_outcomes`, `SuccessorCache` in [`models/speculative_denoise.py`](../../src/slm_training/models/speculative_denoise.py) |
| **Config** | `speculative_successor`, `speculative_fanout`, `speculative_overlap` |

### Adjacent V7 lineage (documented, deliberately not built)

| Idea | Paper | Why deferred |
| --- | --- | --- |
| Multi-horizon trajectory distillation (Δ∈{1,2,4,8} heads) | T3D [arXiv:2602.12262](https://arxiv.org/html/2602.12262v4) (related: CD4LM trajectory-invariant maps + adaptive decoding effort) | Needs teacher-trajectory infra; fixture-scale trajectories are too short/small to supervise horizon heads |
| Training-free AR-mode self-verification | S2D2 [arXiv:2603.25702](https://arxiv.org/html/2603.25702v1) | Grammar is a stronger, cheaper critic for this DSL than TwoTower's LTR repair mode |
| Draft reference tokens + verification attention mask | SimSD [arXiv:2606.02544](https://arxiv.org/abs/2606.02544) | Requires attention-mask surgery on the trunk; grammar acceptor covers one-pass checking here |
| Off-grid configuration training | Adaptive Block Diffusion [arXiv:2606.29275](https://arxiv.org/html/2606.29275v1) | Partially covered by `mask_pattern=mixed` + `visible_corrupt_rate` + template seeds; full config-support training is future work |
| Continuous latent plan channel | CCDD [arXiv:2510.03206](https://arxiv.org/html/2510.03206v1) | No long-form latent-reasoning phase in short grammar-anchored OpenUI programs |
| Coarse-to-fine block control ("Think Coarse, Critic Fine") | BACD [arXiv:2602.09555](https://arxiv.org/html/2602.09555v1) | Same task-scale reason as CCDD; revisit if programs grow |
| Prefix-cacheability restructuring | WeDLM [arXiv:2512.22737](https://arxiv.org/html/2512.22737v1) | Deployment-scale concern; sequences ≤256 tokens, context KV already cached |

---

## Autoresearch systems and adjacent research directions

### Swappable deep-research systems

| System / paper | Fidelity and role here | Integration |
| --- | --- | --- |
| LangChain, [Open Deep Research](https://github.com/langchain-ai/open_deep_research) | **Adapted system integration** — its configurable LangGraph researcher produces an untrusted cited memo and trajectory; it does not author executable commands or bypass proposal validation | Isolated invocation adapter pinned to `b764481fca7f0dbf00b2c70239bd97cea59d1059` in [`autoresearch/researchers.py`](../../src/slm_training/autoresearch/researchers.py) |
| Li et al., *OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis*, [arXiv:2603.20278](https://arxiv.org/abs/2603.20278), [code](https://github.com/TIGER-AI-Lab/OpenResearcher) | **Adapted system integration** — the released search/browse trajectory runner is another memo producer. We do not adopt its training recipe, model weights, corpus, benchmark claims, or GPU topology | Isolated invocation adapter pinned to `785fd6ba5fcbc068daa4a2f07bbe0964f2983c86`; upstream code is not vendored |

Both implementations share `ResearchRequest` / `ResearcherRun` contracts and the
same trusted `ExperimentSpec` compiler. Setup, pinning, failure limits, and the
paper-reproduction policy are documented in
[`autoresearch-autotraining.md`](autoresearch-autotraining.md).

### Newly tracked papers and applicability

| Paper / code | Status here | Concrete takeaway and boundary |
| --- | --- | --- |
| Fu et al., *Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals* (PUST), [arXiv:2607.11505](https://arxiv.org/abs/2607.11505), [alphaXiv](https://www.alphaxiv.org/abs/2607.11505) | **Adjacent** | Proxy exploration, relative update-signal transfer, and reusable cached guidance motivate separating evidence-producing exploration from the promoted model. Our memo/compiler split is only a systems analogue: it does **not** implement PUST update extraction or policy transfer, and the RL readiness lock still applies. |
| Zhao et al., *UltraX: Refining Pre-Training Data at Scale with Adaptive Programmatic Editing*, [arXiv:2607.08646](https://arxiv.org/abs/2607.08646), [code](https://github.com/openbmb/UltraX) | **Adjacent** | Typed insert/delete/replace programs, confidence filtering, controlled operation mixtures, validation, and deterministic execution are relevant to future immutable data-repair experiments. Current `ExperimentKnobs` and synthesis telemetry do not reproduce UltraX's LAM/DCR pipeline, refinement model, or 20B-token evaluation. |
| Monea et al., *The State-Prediction Separation Hypothesis*, [arXiv:2607.01218](https://arxiv.org/abs/2607.01218), [alphaXiv](https://www.alphaxiv.org/abs/2607.01218) | **Adjacent** | Separating state storage from next-token prediction is an architecture hypothesis worth a matched causal-LM ablation. It is not the same as this repository's context-encoder + denoiser “TwoTower,” so no fidelity claim is made. |
| Korchinski, Favero, Wyart, *Learn from your own latents and not from tokens: A sample-complexity theory*, [arXiv:2605.27734](https://arxiv.org/abs/2605.27734) | **Adjacent** | The theory motivates a future, separately gated own-latent prediction objective for hierarchical program structure. Current masked-token CE, context embeddings, and deferred latent critic are not latent-prediction training and do not inherit the paper's sample-complexity result. |
| Jukić and Titov, *Geometric Self-Distillation for Reasoning Generalization* (GeoSD), [arXiv:2607.06855](https://arxiv.org/abs/2607.06855) | **Adjacent** | Hellinger overlap weighting and a Fisher–Rao checkpoint-proximal term motivate a future self-distillation stability/OOD ablation. Current P2 uses hard accepted traces, trajectory-action loss, and anchor mixing; it has no privileged-context soft teacher, GeoSD loss, or natural-gradient update, so no GeoSD implementation claim is made. |
| Hill, *Saturation Makes Quantization Error Additive: A Coverage Model with a Certificate*, [arXiv:2607.12266](https://arxiv.org/abs/2607.12266) | **Adjacent** | The per-layer additive/coverage model and unexplained-variance certificate motivate a future mixed-precision allocation benchmark under the ≤1 GB deployment envelope. Current exports use uniform dynamic int8 or the separate BF16 codebook sidecar; they do not measure a W4A4 layer-set lattice, fit coverage break-rates, or implement the paper's allocator/certificate. |
| Das et al., *Recover-LoRA for Aggressive Quantization: Reclaiming Accuracy in 2-Bit Language Models via Low-Rank Adaptation with Knowledge Distillation on Synthetic Data*, [arXiv:2606.04238v1](https://arxiv.org/html/2606.04238v1) | **Adjacent** | Selective mixed precision plus synthetic logit-distillation adapters is a candidate recovery stage only if an honestly evaluated causal-LoRA checkpoint later needs ≤2-bit deployment. No current model is quantized or accuracy-recovered by this method. |

These rows are literature intake, not experiment results. Any adoption starts with a
frozen baseline, one bounded lever, full provenance, and the normal quality/perf
matrix and model-card rules. An alphaXiv Autoresearch availability page may help
construct a scratch reproduction, but generated code or reported results do not
enter this repository without review and local evidence.

---

## Systems & data (not papers, but cited lineage)

| System | Role here | Link / path |
| --- | --- | --- |
| OpenUI / `@openuidev/lang-core` | Official grammar, streaming parse, validate | [openui.com](https://www.openui.com/) · [`src/apps/openui_bridge/`](../../src/apps/openui_bridge/) |
| `@openuidev/react-lang` Renderer | Annotate playground visual preview | [`src/apps/openui_preview/`](../../src/apps/openui_preview/) |
| Lark `InteractiveParser` | Incremental LALR acceptor for force-emit / admit | [`dsl/grammar/fastpath/engine.py`](../../src/slm_training/dsl/grammar/fastpath/engine.py) |
| `@google/design.md` | DESIGN.md lint in preference reward | [`src/apps/design_md_bridge/`](../../src/apps/design_md_bridge/) |
| RICO | Mobile UI screens → OpenUI seeds | [`src/slm_training/resources/rico/`](../../src/slm_training/resources/rico/) |
| BF16 exponent codebook | Optional weight-compression sidecar | [brianbell-x weight-compression](https://brianbell-x.github.io/weight-compression/) · [`runtime-performance.md`](runtime-performance.md) |

---

## Quick map: idea → knob → file

| Idea | Primary knob | Primary file |
| --- | --- | --- |
| MaskGIT unmask | `--parallel-unmask` | `models/parallel_decode.py` |
| CFG hole admit | config `grammar_fastpath_mode` ∈ `{force,mask,hybrid}` | `dsl/grammar/fastpath/maskgit_constrain.py` |
| DFA force-emit | config `grammar_fastpath` + mode `force\|hybrid` | `dsl/grammar/fastpath/force_emit.py`, `models/grammar.py` |
| Valid-only certify | config `grammar_ltr_primary` + repair + finalize | `models/twotower.py` (`_ensure_valid_openui`) |
| Length-safe LTR | `grammar_ltr_max_tokens` / `grammar_ltr_stages` | `models/twotower.py` |
| MDLM schedule | `mdlm_schedule` | `models/twotower.py` (`_mask_targets`) |
| Remasking | `remask_ratio` / `remask_use_gate` / `remask_use_entropy` / `remask_policy` | `models/parallel_decode.py` |
| Template fill | `template_fill_decode` | `models/template_fill.py` |
| Honest inventory | `honest_slot_contract` | `models/template_fill.py` (`inventory_from_prompt`) |
| Preference stage | `scripts/train_preference.py` | `harnesses/preference/train.py` |
| GRPO-lite | `scripts/train_rl.py` | `harnesses/rl/` |
| LTR suffix rollback | `suffix_rollback_window` (E30) | `models/twotower.py` |
| Trust head | `trust_gate` / `FastPathGate` (E31) | `dsl/grammar/fastpath/gate.py`, `trust_train.py` |
| Slot-aware trust | `slot_aware_trust_gate` (E52) | `dsl/grammar/fastpath/trust_train.py` |
| Visible-token corruption | `visible_corrupt_rate` (E32) | `models/twotower.py` (`_mask_targets`) |
| Combined remask policy | E33 | `parallel_decode.select_remask_policy_indices` |
| CoRe remask | E50 `remask_policy=core\|combined` | `parallel_decode.select_remask_core_indices` |
| T2M remask-to-mask | E51 `remask_to_mask` | `models/twotower.py` |
| Honest V5 champion | E53 | `scripts/run_quality_matrix.py --matrix v6` |
| Grammar-diffusion honest | E54 / X2–X7 | `models/grammar_diffusion.py` |
| Latent critic MoE (deferred) | E34 | `research-correction-critics.md` |
| Verifier-guided repair map | proposed E60–E65 | `verifier-guided-repair.md` |
| Stability remask / commit gate | E70 `remask_policy=stability`, `stability_min_persistence` | `models/parallel_decode.py` (`StabilityTracker`) |
| Attention dependency clusters | E71 `unmask_mode=cluster` | `models/speculative_denoise.py` |
| Ordered cluster verification | E72 `cluster_verify` | `models/speculative_denoise.py` (`verify_clusters_ordered`) |
| Trajectory-survival head | E73 `survival_gate` | `dsl/grammar/fastpath/survival_train.py` |
| Successor-state cache | E74 `speculative_successor`, `speculative_fanout` | `models/speculative_denoise.py` (`SuccessorCache`) |
| V7 champion | E75 | `scripts/run_quality_matrix.py --matrix v7` |
| Evidence-grounded autoresearch | `--researcher`, typed `ResearcherRun` / `ExperimentKnobs` | `scripts/autoresearch.py`, `src/slm_training/autoresearch/` |

---

## Honesty rules (for docs & claims)

1. Do **not** claim “we implement paper X” unless this page tags it **Faithful**.
2. Prefer **Adapted** / **Surrogate** wording in READMEs, PR descriptions, and eval writeups.
3. When adding a new technique, append a row here **in the same PR** as the code.
4. Keep arXiv / venue links stable; deep-link code paths that reviewers can grep.
