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

### Grammar-topology diffusion (trans-dimensional OpenUI tree)

| | |
| --- | --- |
| **Papers** | Stern et al., *Insertion Transformer* [arXiv:1902.03249](https://arxiv.org/abs/1902.03249); Gu et al., *Levenshtein Transformer* [arXiv:1905.11006](https://arxiv.org/abs/1905.11006); Chen et al., *Diffusion Forcing* [arXiv:2407.01392](https://arxiv.org/abs/2407.01392); *Deletion-Insertion Diffusion* [arXiv:2603.23507](https://arxiv.org/abs/2603.23507); *Multi-Block Diffusion* [arXiv:2606.29215](https://arxiv.org/abs/2606.29215) |
| **Fidelity** | **Adapted** — synchronous typed production-tree expansion/contraction with a bounded active buffer; not a faithful implementation of any cited sequence model |
| **Code** | [`models/grammar_diffusion.py`](../../src/slm_training/models/grammar_diffusion.py), topology metrics in [`harnesses/model_build/eval_runner.py`](../../src/slm_training/harnesses/model_build/eval_runner.py) |
| **Config** | `topology_actions`, `topology_structural_embeddings`, `topology_heterogeneous_noise`, `topology_critic_decode`, `topology_bounded_buffer`, topology budgets |
| **Docs** | [`grammar-topology-diffusion.md`](grammar-topology-diffusion.md), X9-X15 in [`quality-experiment-matrix.md`](quality-experiment-matrix.md) |

Unlike the fixed-canvas MaskGIT/MDLM rows, a topology mask denotes a typed grammar
work item that may materialize zero or many child nodes. Persistent IDs and
tree-relative coordinates avoid assigning a shifted absolute position to every
later production after insertion.

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

### Molt causal RL

| | |
| --- | --- |
| **System** | [NVIDIA Molt](https://github.com/NVIDIA-NeMo/labs-molt), pinned to `0.1.2` / `21c1b8921b73f5c8317b5fc9e359e9a1b7d255d2` |
| **Fidelity** | **Faithful integration** — token-first `Env`/`StepEnvRunner`, vLLM rollouts, GRPO, FSDP actor update, raw rollout dump/replay |
| **Scope** | Causal-LM track only; one-step hardware smoke is wiring evidence, never a quality or ship claim |
| **Code** | [`integrations/molt_rl.py`](../../src/slm_training/integrations/molt_rl.py), agent adapter, and `scripts.model_cycle submit-molt` / `reconcile-molt` |
| **Docs** | [`molt-rl-autoresearch.md`](molt-rl-autoresearch.md) |

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
| Token-tree speculative verification | Miao et al., SpecInfer [arXiv:2305.09781](https://arxiv.org/abs/2305.09781), [code](https://github.com/flexflow/FlexFlow) | **Adapted**: completion paths share trie-parent canvases and one verifier batch; compiler coverage replaces SpecInfer's learned small-model drafters. The paper reports `1.5–2.8×` distributed and `2.6–3.5×` offloaded-inference speedups, but those do not transfer: our grammar/compiler verifier is not SpecInfer's distribution-preserving target-LLM sampler, and the local C-series evidence has an invalid zero-quality anchor with `compiler_decode_mode=off` by default. |
| Static completion tree ranking | TreeRanker [arXiv:2508.02455](https://arxiv.org/abs/2508.02455) | **Adapted**: compiler-valid paths are ranked from target token scores; this implementation uses gathered semantic rows and diffusion canvases |
| Prefix automata + inhabitable-type search | Type-Constrained Code Generation [arXiv:2504.09246](https://arxiv.org/abs/2504.09246) | **Adapted boundary**: Lark reachability, layout capability constraints, and active symbol inventories provide the current OpenUI subset; no claim of reproducing its TypeScript type-search system |
| Hierarchical diffusion verification trees | Self Speculative Decoding for Diffusion LMs [arXiv:2510.04147](https://arxiv.org/abs/2510.04147) | **Adapted**: candidate trie parents use isolated prefix-visible/future-masked canvases; no claim of causal AR probabilities from bidirectional diffusion scores |
| Distribution-preserving speculative sampling | Leviathan et al. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | **Adjacent**: greedy constrained ranking is target-equivalent on complete candidate sets; exact stochastic residual sampling is not implemented |
| Incremental constrained code decoding | Constrained FIM decoding [arXiv:2402.17988](https://arxiv.org/abs/2402.17988) | **Adapted**: incremental parsing rejects invalid prefixes early; OpenUI semantic inventories add a narrower project-specific layer |

---

## Dynamic symbols and constraint-aware diffusion (V8)

The public “Diffusion SLM Optimization” review yielded **50 unique academic
sources**: 49 arXiv records plus ACL `D18-1192`. Canonical metadata,
abstract-derived summaries, authors, applicability, limitations, and fidelity
labels are committed in
[`dynamic-symbol-sources.json`](../../src/slm_training/resources/autoresearch/dynamic-symbol-sources.json).
The manifest is the complete inventory; this section records the cross-paper
synthesis rather than repeating 50 abstracts.

| Research cluster | V8 use | Boundary |
| --- | --- | --- |
| Dynamic vocabulary, pointer/copy models, alpha-equivalence | Fixed reserved rows receive request-local surface/metadata deltas; binder names are gated out; slot permutation is train-only augmentation | **Adapted**; no vocabulary growth, hypernetwork, or retrieval encoder |
| Static/dynamic grammar decomposition and token-space compression | Cache terminal-equivalent token sets, then intersect request-local entity/state rows | **Adapted** systems idea; reported autoregressive speedups do not transfer |
| Future validity and truncation proofs | A deterministic Lark helper returns a conservative completion lower bound or `unknown` | **Adapted** narrow subset; no learned future-validity statistic or full CFG intersection solver |
| Projectional/semantic decoding | A minimal graph joins statement neighbors, repeated symbols, and delimiter pairs and can be unioned with attention edges | **Adapted** scheduling aid, not a typed projectional editor or semantic proof system |
| Diffusion correction, scheduling, caching, and variable length | Existing V7 verification/remasking plus the V8 graph and fixed-vs-compact canvas ablation | Mostly **Adjacent** until matched OpenUI runs exist |

The manifest currently labels 14 sources **Adapted** and 36 **Adjacent**. Those
labels describe code lineage only. No paper result, latency multiplier, quality
gain, or ship status is inherited, and V8 rows E200-E207/C5-C8 remain unrun.

## Lattice-guided recursive compiler search (V9)

The shared tiny-reasoner review and its 25 academic references are normalized in
[`lattice-recursive-search.md`](lattice-recursive-search.md). V9 treats the
existing compiler completion forest as the hard partial-information state, keeps
neural scores soft, and plans bounded rollback plus selectively triggered
PTRM/GRAM-style trajectories. The controller is **Adapted**: it does not reproduce
LDT, TRM, PTRM, or GRAM training, and their reported results do not transfer.

Rows E240-E247 are plan-only hypotheses until the standard quality suites,
AgentEvals, AgentV bundle, measured-results JSON, and markdown scoreboard exist.

## Contract-conditioned scope diffusion (X16-X21)

The public [ScopeDiff discussion](https://chatgpt.com/share/6a583787-8e9c-83ea-94e2-c36b0f4d093e)
cited **19 papers**. Canonical arXiv metadata,
authors, paraphrased summaries, repo mappings, limitations, and fidelity labels
are committed in
[`scope-diffusion-sources.json`](../../src/slm_training/resources/autoresearch/scope-diffusion-sources.json).
The inventory labels 13 sources **Adapted** and 6 **Adjacent**; these are lineage
labels, not reproduced results.

| Research cluster | ScopeDiff use | Boundary |
| --- | --- | --- |
| Insertion, deletion, masked, block, and heterogeneous diffusion | Typed expansion/edit actions, independent scope noise, and a bounded active-scope buffer | **Adapted** to OpenUI topology; none of the paper objectives or benchmarks are reproduced |
| Syntax-directed generation, typed holes, type slicing, and AST decoders | Stable scope contracts with inherited fields, synthesized summaries, and AST failure cones | **Adapted** pragmatic interfaces; not a VAE, formal hole calculus, or mechanized type slicer |
| CFG and semantic constrained decoding | Existing OpenUI parser/verifier remains the hard authority around local wrapper proposals | **Adjacent**; no CFG-intersection, PICARD, or Synchromesh decoder is claimed |
| Concept embeddings and sparse autoencoders | Motivation for explicit contract embeddings and falsification of latent critics | **Adjacent**; no SAE is implemented or treated as a semantic oracle |

The paper-to-experiment audit below makes the derivative claim and its local
falsification boundary explicit. Each linked paper is summarized individually in
the manifest; grouping here avoids treating several adjacent papers as independent
evidence for the same lever.

| Papers | Derived X-row lever | Local falsification |
| --- | --- | --- |
| [Insertion Transformer](https://arxiv.org/abs/1902.03249), [Levenshtein Transformer](https://arxiv.org/abs/1905.11006), [Beyond Masks](https://arxiv.org/abs/2603.23507) | X16/X21 explicit expansion, deletion, contraction | Action macro-F1 and generated topology fail to improve without extra budget failures |
| [MaskGIT](https://arxiv.org/abs/2202.04200), [MDLM](https://arxiv.org/abs/2406.07524) | X16 shared denoising objective over typed scopes | Scope data does not improve unchanged parse, fidelity, or structural gates |
| [Diffusion Forcing](https://arxiv.org/abs/2407.01392), [Block Diffusion](https://arxiv.org/abs/2503.09573), [Multi-Block Diffusion](https://arxiv.org/abs/2606.29215) | X18/X21 independently noised scopes in a bounded active buffer | Heterogeneous noise and buffering do not improve quality per node pass |
| [Syntax-Directed VAE](https://arxiv.org/abs/1802.08786), [Hazelnut](https://arxiv.org/abs/1607.04180), [Abstract Syntax Networks](https://arxiv.org/abs/1704.07535) | X17 inherited contract fields and synthesized summary heads | Local gate accuracy and summary error do not improve over X16 |
| [Bidirectional Type Slicing](https://arxiv.org/abs/2607.12197), [Diffusion on Syntax Trees](https://arxiv.org/abs/2405.20519) | X19 AST failure-cone supervision | Cone precision/recall stays uninformative or downstream topology does not improve |
| [CFG-constrained diffusion](https://arxiv.org/abs/2508.10111), [Synchromesh](https://arxiv.org/abs/2201.11227), [PICARD](https://arxiv.org/abs/2109.05093) | X20 boundary negatives and local wrapper validation | Local decisions do not reduce local-valid/global-invalid disagreement under unchanged global gates |
| [Concept Embedding Models](https://arxiv.org/abs/2209.09056) | X17 learned contract embedding beside explicit fields | Embeddings do not improve contract metrics or downstream quality over X16 |
| [SAIF](https://arxiv.org/abs/2502.11356), [SAE reasoning-feature audit](https://arxiv.org/abs/2601.05679) | No X-row; residual SAE critic remains deferred | Any future SAE must beat explicit contracts under causal counterexamples before it can steer decoding |

The code substrate is `data/progspec/scopes.py`, the conditional heads in
`models/grammar_diffusion.py`, and planned matrix rows X16-X21. No X16-X21 run,
quality gain, or ship status is recorded by this implementation-only change.

## DSL diffusion research program (Tracks A-G)

The 2026-07-16 prior-art sweep for the DSL diffusion SLM research program
(emptiness wall, semantic-choice representation, binding/macros, tree-native
denoising, capacity measurement, DSL packs, self-improvement — Linear project
"DSL Diffusion SLM Research Program", SLM-20..47) is committed in
[`dsl-program-sources.json`](../../src/slm_training/resources/autoresearch/dsl-program-sources.json)
(24 new sources; entries already tracked in the V8/ScopeDiff manifests or on
this page are deliberately omitted). All 24 are labeled **Adjacent** — lineage
only, nothing implemented, no result inherited.

| Research cluster | Program use | Boundary |
| --- | --- | --- |
| Sketch-then-fill and externalized grammar (Coarse-to-Fine [1805.04793](https://arxiv.org/abs/1805.04793), Grammar Prompting [2305.19234](https://arxiv.org/abs/2305.19234), CodeFusion [2310.17680](https://arxiv.org/abs/2310.17680), TinyStories [2305.07759](https://arxiv.org/abs/2305.07759)) | Track B choice-sequence codec + Track E capacity study; direct collisions to differentiate against | **Adjacent** — AR/prompting/large-model settings; no tiny-diffusion externalized-grammar result exists to inherit |
| Span/macro abstraction (Gist tokens [2304.08467](https://arxiv.org/abs/2304.08467), DyVo [2410.07722](https://arxiv.org/abs/2410.07722), DreamCoder [2006.08381](https://arxiv.org/abs/2006.08381), LILO [2310.19791](https://arxiv.org/abs/2310.19791), Stitch [2211.16605](https://arxiv.org/abs/2211.16605)) | Track C macro tokens and dynamic pseudo-embeddings | **Adjacent** — learned/lossy or lambda-calculus settings; our expansion is deterministic and grammar-coupled |
| Binding theory (Smolensky TPR, binding-problem survey [2012.05208](https://arxiv.org/abs/2012.05208), code2seq [1808.01400](https://arxiv.org/abs/1808.01400), open-vocab code [2003.07914](https://arxiv.org/abs/2003.07914)) | Track C design rationale: externalize binding to the verifier | **Adjacent** — theory/motivation only |
| **Negative results engaged head-on** (identifier anonymization degradation [2510.03178](https://arxiv.org/abs/2510.03178); context-sensitive alpha-equivalence hashing pitfall [2401.02948](https://arxiv.org/abs/2401.02948); GAD/ASAp constraint-distortion, already in the V8 manifest) | C4 control experiment; C1/D2 canonicalizer design; A1 emptiness diagnosis | These threats are treated as hypotheses to test locally, not results that transfer either way |
| Diffusion objective/adaptation options (SEDD [2310.16834](https://arxiv.org/abs/2310.16834), DiffuGPT/DiffuLLaMA [2410.17891](https://arxiv.org/abs/2410.17891), constrained discrete diffusion [2503.09790](https://arxiv.org/abs/2503.09790)) | Track B4 adaptation baseline; objective fallback; novelty positioning | **Adjacent** — current stack keeps the MDLM-style schedule (Adapted, above) |
| Reasoning-in-formal-language baselines (PAL [2211.10435](https://arxiv.org/abs/2211.10435), PoT [2211.12588](https://arxiv.org/abs/2211.12588), Sketch-of-Thought [2503.05179](https://arxiv.org/abs/2503.05179)) | Track G4 harness baselines | **Adjacent** — prompting methods on frozen LLMs |
| Self-improvement loops (AI Scientist [2408.06292](https://arxiv.org/abs/2408.06292), AlphaEvolve, ShinkaEvolve) | Track G2 recipe evolution, gated by frozen benchmarks + honest gates + fail-closed RL readiness | **Adjacent** — only the population/evaluator pattern transfers |

Program-level positioning recorded here for honesty: grammar-constrained
diffusion decoding is an actively contested niche
([2508.10111](https://arxiv.org/abs/2508.10111) is already **Adapted** in the
fastpath; [2503.09790](https://arxiv.org/abs/2503.09790) and successors exist),
tree-native denoising collides with *Diffusion on Syntax Trees*
([2405.20519](https://arxiv.org/abs/2405.20519), already cited for X19), and
"reason in an invented DSL" is crowded (DreamCoder/LILO/PAL). The unclaimed
combination this program targets is: tiny from-scratch DSL diffusion + template
markers + deterministic identifier/macro expansion + canonicalizing denoiser,
measured by bits-per-semantic-decision. No novelty is claimed until matched
local runs exist.

## Autoresearch systems and adjacent research directions

### Swappable deep-research systems

| System / paper | Fidelity and role here | Integration |
| --- | --- | --- |
| Wang and Buehler, *Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence*, [arXiv:2606.01444](https://arxiv.org/abs/2606.01444) | **Adapted** — schema transitions, preservation, Kan transport, residuals, and MDL-style worthiness become a typed pre-run audit. We do not claim the paper proves a proposed OpenUI idea globally novel; verified discovery requires accepted post-run evidence that does not yet exist at hypothesis time | `HypothesisMatrix` / `CategoricalNoveltyAudit`, matrix validation, and the execution gate in [`autoresearch/`](../../src/slm_training/autoresearch/) and [`scripts/autoresearch.py`](../../scripts/autoresearch.py) |
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
| Zhang et al., *Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training*, [arXiv:2607.01232v2](https://arxiv.org/abs/2607.01232v2) | **Adjacent** | Across seven Qwen-family 1.5B–8B models and GRPO/GiGPO/Dr. GRPO, the paper defines layer contribution `(single-layer gain)/(full-RL gain)` and reports that the strongest layers cluster around 40–60% depth; some match or exceed full RL, and contribution rankings transfer across data/tasks. After—and only after—the frozen RL-readiness gate passes, this motivates a bounded middle-layer-first profile and a selected-layer control for the causal track. It is not evidence for the ≤6-layer TwoTower, GRPO-lite, or any current checkpoint, and a middle-layer heuristic must remain a hypothesis until matched against a tuned full-update baseline and all frozen suites. |
| Yeh et al., *Tracing Agentic Failure from the Flow of Success* (OAT), [arXiv:2607.12747v1](https://arxiv.org/abs/2607.12747v1) | **Adjacent** | OAT learns a gated Neural-CDE reconstruction of normalized, LLM-embedded **successful** trajectories; reconstruction error becomes a per-step anomaly score, with top-*k* or conformal-calibrated detection. On 103 MCP-Atlas successes / 88 failures and 184 OOD Who&When failures, the paper reports better F1 than GPT-4o/GPT-5 prompting, five-seed results, `<1 GB` deployment, and much lower latency. Existing append-only researcher/decode trajectories make success-only diagnosis attractive, especially conformal thresholds and a clean calibration split, but anomaly is not causal attribution. Do not add a CDE until enough representative successful traces exist and a frozen, step-annotated failure set can show improvement over simple telemetry/rule baselines. |
| Google DeepMind, [*DiffusionGemma model overview*](https://ai.google.dev/gemma/docs/diffusiongemma), [model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card), and [diffusion explanation](https://ai.google.dev/gemma/docs/diffusiongemma/explained) (accessed 2026-07-15; pages updated 2026-06-10) | **Adjacent** | DiffusionGemma uses an autoregressive cached-context encoder plus bidirectional 256-token canvases, uniform-state random-token corruption, self-conditioning, block-autoregressive multi-canvas generation, entropy-bounded token selection, full renoising of unselected tokens, a `0.8→0.4` temperature schedule, and two-part adaptive stopping (mean entropy `<0.005` plus unchanged top-1 predictions across two steps; maximum 48 steps). The closest low-risk borrow is a decode-only entropy/stability stopping and commit-policy ablation using our existing entropy, persistence, remask, and trace hooks. Uniform-state training, probability self-conditioning, and multi-canvas generation are larger architecture/data changes and remain deferred. Google's `>1100 tok/s` claim is H100 FP8 at low batch, while its card trails causal Gemma 4 on most listed quality tasks; neither speed nor quality transfers to this repo. |
| Zhang et al., *Spectral Rewiring for Exploration, Purification, and Model Merging* (SAR), [arXiv:2607.03065v1](https://arxiv.org/abs/2607.03065v1) | **Adjacent** | SAR takes a paired base/RL update `ΔW`, extracts a low-rank component, projects it into the base weight's SVD coordinates, and reconstructs `W_base + U(UᵀΔW_kV)Vᵀ`. The paper reports retaining >99% of measured post-training performance with rewiring matrices as small as about 0.58% of parameters, better high-*k* exploration, reduced mixed-domain interference, and stronger expert merges. This motivates a future post-hoc merge/purification candidate beside `average`/`ties`, not a training recipe. It requires immutable compatible base/child checkpoints and full quality plus Pass@k/diversity evaluation; the authors report weaker projection compatibility when domain knowledge is missing, updates are very strong, or dense critic rewards drive off-manifold behavior. |

These rows are literature intake, not experiment results. Any adoption starts with a
frozen baseline, one bounded lever, full provenance, and the normal quality/perf
matrix and model-card rules. An alphaXiv Autoresearch availability page may help
construct a scratch reproduction, but generated code or reported results do not
enter this repository without review and local evidence.

### 2026-07-15 intake synthesis (documentation only)

No model, training, evaluation, benchmark, reproduction, or remote-provider run was
started for this intake. The sources suggest the following order of operations:

1. **Decode-only, lowest cost:** test DiffusionGemma-style entropy-bounded commits
   and two-signal adaptive stopping by composing existing entropy, persistence,
   remask, and trajectory telemetry. Keep uniform-state corruption and
   self-conditioning out of the first comparison.
2. **Serving-only, rerun before extension:** the SpecInfer-inspired packed
   completion tree is already implemented. Do not add another drafter or tree;
   first rerun its existing C-series controls on an honestly evaluated lexer-native
   checkpoint and require unchanged quality plus lower p50 and neural forwards.
3. **Failure diagnosis after data accrues:** begin with cheap success-trace feature
   baselines and a conformal threshold. Consider OAT's gated Neural CDE only after
   a frozen step-annotated set proves that rules and simple sequence baselines are
   inadequate.
4. **RL only after readiness:** profile a small set of middle and edge layers,
   compute the paper's contribution ratio against a tuned full-update control, and
   proceed to selected-layer training only if the ranking is stable across frozen
   suites. Never use this paper to bypass `RLReadinessReport`.
5. **Post-RL consolidation:** evaluate SAR as a third immutable merge artifact only
   when compatible base and expert checkpoints exist. Compare against `average`
   and TIES on the full scoreboard, Pass@k/diversity, and cross-domain regression;
   projection compactness alone is not a promotion signal.

Cross-source lesson: several reported gains come from **restricting work to a
validated subspace**—a candidate tree, confident canvas positions, normal
trajectory dynamics, selected layers, or the base model's spectral coordinates.
For OpenUI this is a useful experiment-selection prior, not a result: choose the
smallest constrained intervention whose verifier and frozen suites can falsify it.

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
| Molt causal RL | `scripts.model_cycle submit-molt` | `integrations/molt_rl.py` |
| LTR suffix rollback | `suffix_rollback_window` (E30) | `models/twotower.py` |
| Trust head | `trust_gate` / `FastPathGate` (E31) | `dsl/grammar/fastpath/gate.py`, `trust_train.py` |
| Slot-aware trust | `slot_aware_trust_gate` (E52) | `dsl/grammar/fastpath/trust_train.py` |
| Visible-token corruption | `visible_corrupt_rate` (E32) | `models/twotower.py` (`_mask_targets`) |
| Combined remask policy | E33 | `parallel_decode.select_remask_policy_indices` |
| CoRe remask | E50 `remask_policy=core\|combined` | `parallel_decode.select_remask_core_indices` |
| T2M remask-to-mask | E51 `remask_to_mask` | `models/twotower.py` |
| Honest V5 champion | E53 | `scripts/run_quality_matrix.py --matrix v6` |
| Grammar-topology diffusion | E54 / X9–X15 | `models/grammar_diffusion.py` |
| Latent critic MoE (deferred) | E34 | `research-correction-critics.md` |
| Verifier-guided repair map | proposed E60–E65 | `verifier-guided-repair.md` |
| Stability remask / commit gate | E70 `remask_policy=stability`, `stability_min_persistence` | `models/parallel_decode.py` (`StabilityTracker`) |
| Attention dependency clusters | E71 `unmask_mode=cluster` | `models/speculative_denoise.py` |
| Ordered cluster verification | E72 `cluster_verify` | `models/speculative_denoise.py` (`verify_clusters_ordered`) |
| Trajectory-survival head | E73 `survival_gate` | `dsl/grammar/fastpath/survival_train.py` |
| Successor-state cache | E74 `speculative_successor`, `speculative_fanout` | `models/speculative_denoise.py` (`SuccessorCache`) |
| V7 champion | E75 | `scripts/run_quality_matrix.py --matrix v7` |
| Compiler-drafted decode | `compiler_decode_mode=forced|restricted|tree` | `dsl/grammar/fastpath/compiler_draft.py`, `models/twotower.py` |
| Evidence-grounded autoresearch | `--researcher`, typed `ResearcherRun` / `ExperimentKnobs` | `scripts/autoresearch.py`, `src/slm_training/autoresearch/` |
| Five-candidate hypothesis matrix | `hypothesize`, `min_hypotheses>=5`, categorical candidate audit | `scripts/autoresearch.py`, `autoresearch/schemas.py`, `autoresearch/engine.py` |

---

## Honesty rules (for docs & claims)

1. Do **not** claim “we implement paper X” unless this page tags it **Faithful**.
2. Prefer **Adapted** / **Surrogate** wording in READMEs, PR descriptions, and eval writeups.
3. When adding a new technique, append a row here **in the same PR** as the code.
4. Keep arXiv / venue links stable; deep-link code paths that reviewers can grep.
