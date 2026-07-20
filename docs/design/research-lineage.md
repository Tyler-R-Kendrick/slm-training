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

### Shared recursive denoiser tower (SLM-138)

| | |
| --- | --- |
| **Lineage** | Deep equilibrium / weight-tied recurrent transitions; related to recurrent/transformer reasoning towers and the Deep Equilibrium Model (DEQ) idea of running a fixed-depth transition to a steady state. The specific coupled ``y/z`` recurrence with cross-attention to context is an OpenUI-specific design, not a reproduction of any cited DEQ training recipe. |
| **Fidelity** | **Adapted** — same public contract as ``DenoiserTower``, but replaces independent stacked blocks with a small shared transition recursed ``R`` times; the z-state path and deep-supervision weights are new. |
| **Code** | [`models/recursive_denoiser.py`](../../src/slm_training/models/recursive_denoiser.py), routing in [`models/twotower.py`](../../src/slm_training/models/twotower.py) |
| **Config** | `denoiser_arch`, `recursive_steps`, `recursive_transition_layers`, `recursive_depth_supervision_weights` |
| **Fixture** | `scripts/run_slm138_recursive_denoiser_fixture.py` |
| **SLM-139 follow-up** | Closed as `no_supported_probabilistic_regime`: SLM-138 delivered only wiring-only fixture evidence, so the stochastic high-level width campaign did not run. See `docs/design/iter-slm139-stochastic-recursive-width-20260720.md`. |

### Preference / “DPO”

| | |
| --- | --- |
| **Paper** | Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| **Fidelity** | **Surrogate** — reference-free loss on **masked denoiser log-probs**, not textbook DPO with a frozen reference policy over autoregressive likelihoods |
| **Code** | [`harnesses/preference/train.py`](../../src/slm_training/harnesses/preference/train.py) (`dpo_loss`), pair builders in [`harnesses/preference/`](../../src/slm_training/harnesses/preference/) |
| **CLI** | `scripts/train_preference.py` |

### Exact-state local decision preference

| | |
| --- | --- |
| **Papers** | [Unlikelihood](https://arxiv.org/abs/1908.04319), [Token-level DPO](https://arxiv.org/abs/2404.11999), [TIS-DPO](https://arxiv.org/abs/2410.04350), [ConfPO](https://arxiv.org/abs/2506.08712), [TGDPO](https://arxiv.org/abs/2506.14574), [Antislop](https://arxiv.org/abs/2510.15061), [TokenRatio](https://arxiv.org/abs/2605.12288), [TAB-PO](https://arxiv.org/abs/2603.00025) |
| **Fidelity** | **Adapted** — exact masked-token states, verifier-backed good/bad action sets, clipped logit margins, and optional frozen-reference tethering; not sequence DPO or a faithful reproduction of any cited objective |
| **Code** | Existing preference/trace owners; the LDI0-01 architecture contract, named owners, and full 34-source audit are in [`local-decision-interventions.md`](local-decision-interventions.md) |
| **Matrix** | Measured V10 rows E248-E254 and the E265-E286 local-preference ledger (LDI campaign index) in [`quality-experiment-matrix.md`](quality-experiment-matrix.md); the chain is negative (E249/E252 rejected) |

SAE/ReFT discovery, removable LoRA/DoRA/PiSSA/[AdaLoRA](https://arxiv.org/abs/2303.10512)
actuators, adapter routing, iterative remine, and RLVR are **Adjacent**.
Objective-geometry lenses for the current blocker —
[Gradient Surgery/PCGrad](https://arxiv.org/abs/2001.06782) and
[MGDA](https://arxiv.org/abs/1810.04650) — and decoding baselines
([min-p](https://arxiv.org/abs/2407.01082)) are **Adjacent**: diagnostic framing and
baselines to beat, not implemented. [PICARD](https://arxiv.org/abs/2109.05093) and
[Grammar-Aligned Decoding/ASAp](https://arxiv.org/abs/2405.21047) frame the
constraint-legality invariant — constraint shadows certify legality only, never
semantic preference. These are lineage labels, not reproduced results. The first
TwoTower implementation is a localized loss with a global full-parameter update. It
makes no locality or ship claim until the registered controls run under unchanged
gates.

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

The shared tiny-reasoner review and its 25 academic references, plus its video and
community-reproduction context, are normalized as R0-R26 in
[`lattice-recursive-sources.json`](../../src/slm_training/resources/autoresearch/lattice-recursive-sources.json)
and summarized in [`lattice-recursive-search.md`](lattice-recursive-search.md). V9 treats the
existing compiler completion forest as the hard partial-information state, keeps
neural scores soft, and plans bounded rollback plus selectively triggered
PTRM/GRAM-style trajectories. The controller is **Adapted**: it does not reproduce
LDT, TRM, PTRM, or GRAM training, and their reported results do not transfer.

Rows E240-E247 are implemented but remain unrun hypotheses until the standard quality suites,
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
| [Bidirectional Type Slicing](https://arxiv.org/abs/2607.12197), [Diffusion on Syntax Trees](https://arxiv.org/abs/2405.20519) | X19 AST failure-cone supervision; **Kapur now Faithful (mechanism)** — D3/SLM-31 implemented the paper's all-valid tree-edit forward process, inverse-edit policy supervision, and value-guided search as `models/tree_edit_diffusion.py` (X22, `iter-x22-d3-kapur-tree-edit-20260717.md`); observation channel is prompt conditioning instead of rendered-image feedback (this domain has no target render at generation), which remains the stated unreproduced half | Cone precision/recall stays uninformative or downstream topology does not improve; X22 fixture screening: first nonzero meaningful parse at the 80-step budget (0.2–0.25 on held_out/adversarial/ood) vs 0.0 for matched X9 — single seed, no ship claim |
| [CFG-constrained diffusion](https://arxiv.org/abs/2508.10111), [Synchromesh](https://arxiv.org/abs/2201.11227), [PICARD](https://arxiv.org/abs/2109.05093) | X20 boundary negatives and local wrapper validation | Local decisions do not reduce local-valid/global-invalid disagreement under unchanged global gates |
| [Concept Embedding Models](https://arxiv.org/abs/2209.09056) | X17 learned contract embedding beside explicit fields | Embeddings do not improve contract metrics or downstream quality over X16 |
| [SAIF](https://arxiv.org/abs/2502.11356), [SAE reasoning-feature audit](https://arxiv.org/abs/2601.05679) | No X-row; residual SAE critic remains deferred | Any future SAE must beat explicit contracts under causal counterexamples before it can steer decoding |

The code substrate is `data/progspec/scopes.py`, the conditional heads in
`models/grammar_diffusion.py`, and planned matrix rows X16-X21. No X16-X21 run,
quality gain, or ship status is recorded by this implementation-only change.

## Verified scope solving & hybrid realization (VSS0)

[verified-scope-solver.md](verified-scope-solver.md) (VSS0-01, SLM-57) is the
spec-only contract that separates **prefix legality** (already enforced by the
compiler completion forest) from **support** — participation in at least one
bounded, fully verified completion. It adds no code, dependency, experiment, or
checkpoint. The anchors below are lineage labels for that contract, not reproduced
results; because SLM-57 ships no code, none is **Faithful**. TreeDiff and the
Lattice Deduction Transformer reuse their existing R13/R7 labels from the V9
sources ([lattice-recursive-search.md](lattice-recursive-search.md)); grouping the
batch here avoids treating adjacent papers as independent evidence for one lever.

| Paper | Fidelity | Role in the VSS0 contract (spec-only) |
| --- | --- | --- |
| [DeepCoder](https://arxiv.org/abs/1611.01989) (Balog et al., ICLR 2017) | **Adjacent** | Learned search-guidance / candidate ranking; the contract keeps learned scores soft (`rank_forest`) and does not reimplement its attribute predictor or DSL |
| Counterexample-guided (neural) synthesis / CEGIS — see the *LLM-Modulo / CEGIS planning* anchor above and R19/R23 in [lattice-recursive-search.md](lattice-recursive-search.md) | **Adapted boundary** | Deduction-vs-decision split and local nogoods adopt the counterexample → refinement principle; neural synthesis training is not reimplemented |
| [egg / e-graphs](https://arxiv.org/abs/2004.03082) (Willsey et al., POPL 2021) | **Adjacent** | Post-realization canonicalization is motivated by equality saturation, but the contract is not an e-graph / equality-saturation engine (cf. `iter-e252-canonicalizer-20260716.md`) |
| EDLM — energy-based diffusion language model (2024) | **Adjacent** | Energy/score-based candidate ranking is analogous to the soft-scoring layer; no EDLM training or energy head is implemented or assumed |
| [TreeDiff](https://arxiv.org/abs/2508.01473) (Zeng et al., 2025) | **Adapted boundary** | Tree-edit diffusion informs late realization against AST/choice IR; architecture and training remain future empirical work (as R13) |
| [Lattice Deduction Transformer](https://arxiv.org/abs/2605.08605) (Davis et al., 2026) | **Adapted** | Monotone lattice projection plus rollback are the closure/deduction model; LDT architecture, alpha supervision, and training remain future work (as R7) |

No solver, experiment, checkpoint, or ship status follows from this spec-only
anchor set; the contract is implemented behind a feature flag by later VSS issues.

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
| Span/macro abstraction (Gist tokens [2304.08467](https://arxiv.org/abs/2304.08467), DyVo [2410.07722](https://arxiv.org/abs/2410.07722), DreamCoder [2006.08381](https://arxiv.org/abs/2006.08381), LILO [2310.19791](https://arxiv.org/abs/2310.19791), Stitch [2211.16605](https://arxiv.org/abs/2211.16605)) | Track C macro tokens and dynamic pseudo-embeddings. **DyVo now Adapted** as C2 (`runtime_symbol_features="replace"`, row E278): only the dynamic-vocabulary-embedding idea is reused — deterministic byte-compositional vectors cancel the learned symbol-pool rows via the V8 delta path; no learned entity retrieval, no sparse-expansion head (`iter-e278-c2-pseudo-embeddings-20260717.md`). **Stitch/LILO now Adapted** as C3 corpus-mined `<MACRO_i>` tokens (`data/macro_induction.py`, `macro_tokens=true`, row E280): only the greedy-MDL corpus-compression objective transfers (`net_gain = freq*(len-1) - len`), over lexer-token n-grams of fixed-vocabulary kinds only (no `<SYM_i>`/`<BIND_j>` inside macros), sidestepping the alpha-equivalence pitfall [2401.02948](https://arxiv.org/abs/2401.02948); expansion is deterministic + lossless (`iter-e280-c3-macro-tokens-20260717.md`) | **Adjacent** otherwise — learned/lossy or lambda-calculus settings; our expansion is deterministic and grammar-coupled |
| Binding theory (Smolensky TPR, binding-problem survey [2012.05208](https://arxiv.org/abs/2012.05208), code2seq [1808.01400](https://arxiv.org/abs/1808.01400), open-vocab code [2003.07914](https://arxiv.org/abs/2003.07914)) | Track C design rationale: externalize binding to the verifier — implemented as C1 relative binder refs (`bind_encoding=relative`, E257, `iter-e257-c1-relative-bind-20260716.md`) | **Adjacent** — theory/motivation only; no mechanism from these papers is reproduced |
| **Negative results engaged head-on** (identifier anonymization degradation [2510.03178](https://arxiv.org/abs/2510.03178); context-sensitive alpha-equivalence hashing pitfall [2401.02948](https://arxiv.org/abs/2401.02948); GAD/ASAp constraint-distortion, already in the V8 manifest) | C4 control experiment — executed at fixture scale as the E281/E282 matched pair (`symbol_anonymization` lever, `iter-e281-e282-c4-names-disappear-20260717.md`); C1/D2 canonicalizer design; A1 emptiness diagnosis. **GAD/ASAp now Adapted** as A2 (`asap_decode`, `models/parallel_decode.py::AsapLedger`, row E277): only the adaptive removal of observed constraint-violating mass is reused, transplanted from ASAp's prefix trie onto the MaskGIT canvas position; no sampling-until-acceptance loop, no convergence guarantee inherited (the trie→position approximation is documented in the class docstring) | These threats are treated as hypotheses to test locally, not results that transfer either way. C4 fixture verdict: **open** — meaningful parse 0.0 on both arms; structural similarity favored the surface arm on 5/5 suites (small adverse data point for the anonymization defense; decisive test needs a frontier-scale replicated pair) |
| Diffusion objective/adaptation options (SEDD [2310.16834](https://arxiv.org/abs/2310.16834), DiffuGPT/DiffuLLaMA [2410.17891](https://arxiv.org/abs/2410.17891), constrained discrete diffusion [2503.09790](https://arxiv.org/abs/2503.09790)) | Track B4 adaptation baseline; objective fallback; novelty positioning | **DiffuLLaMA now Adapted** — `models/hf_denoiser.py` reuses only the drop-the-causal-mask move (bidirectional 4D attention mask over a pretrained SmolLM2-135M backbone) as V10 rows E255/E256; no annealing/shift/training recipe reproduced, fixture-grade verdict open (`iter-e255-e256-b4-ar-adaptation-20260716.md`). SEDD/constrained-diffusion remain **Adjacent** — the stack keeps the MDLM-style schedule (Adapted, above) |
| Reasoning-in-formal-language baselines (PAL [2211.10435](https://arxiv.org/abs/2211.10435), PoT [2211.12588](https://arxiv.org/abs/2211.12588), Sketch-of-Thought [2503.05179](https://arxiv.org/abs/2503.05179)) | Track G4 — implemented (SLM-36) as `harnesses/reasoning/` + the `arith-sketch` pack (`reasoning-sketch-harness.md`): trace = program in a task DSL, scored by a deterministic evaluator (validity and answer share one code path, fail-closed), matched direct-answer control arm | **PAL/PoT now Adapted** — only the reason-in-formal-language + deterministic-execution split transfers, on a trained tiny model instead of a prompted frozen LLM (the honest baseline at this scale is the matched direct arm, not a frozen-LLM PAL). Sketch-of-Thought remains **Adjacent** — prompt-level NL-symbolic sketching; the trained-tiny + externalized-grammar variant it leaves unclaimed is what G4 instantiates. Fixture verdict open: both arms at zero accuracy at 120 CPU steps (`iter-g4-reasoning-bench-20260717.md`) |
| Grammar Prompting for DSLs ([2305.19234](https://arxiv.org/abs/2305.19234)) | Track G3 latent-DSL generator ([latent-dsl-generator.md](latent-dsl-generator.md)): `synthesize_pack(task→grammar→pack)` — but *instantiating* a full DSL pack (grammar/backend/scope/placeholder/prop-order/engine), not prompting a frozen model | **Adapted (mechanism only)** — the task→grammar move is reused deterministically (LLM step stubbed); pack instantiation is real, the trained-model-per-task rung is deferred to G4 |
| Self-improvement loops (AI Scientist [2408.06292](https://arxiv.org/abs/2408.06292), AlphaEvolve, ShinkaEvolve) | Track G2 recipe evolution — implemented (SLM-35) as `harnesses/experiments/recipe_evolution.py` + `scripts/run_recipe_evolution.py`: bounded typed gene space over corruption/decode/loss-weight knobs, seeded mutation/crossover, unique-gene evaluation cache, selection strictly gated by the unaltered default ship gates (gate-passers always outrank; nothing promotable otherwise); no RL path exists — any future one sits behind `rl_gate.assert_rl_ready` | **AlphaEvolve/ShinkaEvolve now Adapted** — only the population/evaluator pattern transfers; no LLM-guided program mutation, no evolved code, evaluator is the existing frozen train/eval/gate stack. AI Scientist remains **Adjacent** |

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

## Calculated arity and adaptive precision (CAP0)

[calculated-arity-adaptive-precision.md](calculated-arity-adaptive-precision.md)
is the specification-only contract separating exact symbolic capacity,
task-relevant rate, neural precision, and measured deployment cost. SLM-77 runs no
experiment and implements none of the mechanisms below, so all new anchors are
**Adjacent**. Grammar-Aligned Decoding and Diffusion on Syntax Trees retain their
existing A2/X22 fidelity records above; their inclusion here is a cross-reference,
not a second or stronger implementation claim.

| Paper / result | Fidelity | CAP use and transfer boundary |
| --- | --- | --- |
| Ma et al., [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/abs/2402.17764) (BitNet b1.58) | **Adjacent** | Native ternary weight training motivates a `K_w=3` control. It does not show that post-hoc choice codes or latent quantizers preserve this tiny OpenUI model. |
| Liu et al., [*ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization*](https://arxiv.org/abs/2502.02631) | **Adjacent** | Its unified low-bit QAT comparison motivates matched bit-width controls. Its empirical transition between at-most-two and at-least-three bits is model/recipe specific and does not transfer locally. |
| Wang et al., [*CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs*](https://arxiv.org/abs/2606.26650) | **Adjacent** | Learnable modulation and softened ternarization motivate a ternary PTQ control distinct from native training. Large-model/calibration results do not establish local arity or system optima. |
| Mentzer et al., [*Finite Scalar Quantization: VQ-VAE Made Simple*](https://arxiv.org/abs/2309.15505) | **Adjacent** | Product scalar grids motivate explicit `K_z,d_z` accounting. FSQ is a latent quantizer, not an error-correcting code or weight-PTQ result. |
| Zhu et al., [*Robust Residual Finite Scalar Quantization for Neural Compression*](https://arxiv.org/abs/2508.15860) | **Adjacent** | Learned scaling and invertible normalization motivate a residual-scale control. Audio/image neural-compression evidence does not establish an OpenUI rate-distortion gain. |
| Dong et al., [*HAWQ-V2*](https://arxiv.org/abs/1911.03852) | **Adjacent** | Hessian-trace sensitivity and Pareto allocation motivate `CAP-H8`. They do not prove an optimal local `K_w`, `K_a`, or semantic-quality frontier. |
| Maletti, [*Minimizing deterministic weighted tree automata*](https://doi.org/10.1016/j.ic.2009.01.004); Rabusseau et al., [*Low-Rank Approximation of Weighted Tree Automata*](https://arxiv.org/abs/1511.01442) | **Adjacent** | Exact minimization over deterministic WTA and approximate Hankel/SVD reduction motivate keeping exact `Q` separate from estimated task compression. Their algebraic assumptions do not automatically cover this compiler, CFG, or learned score algebra. |
| Shin et al., [*Grammar-Aligned Decoding*](https://arxiv.org/abs/2405.21047) | Existing **Adapted** A2 boundary | The existing ASAp transplant concerns positionwise constraint-mass removal. It supplies no convergence, exact quotient, or precision optimum here. |
| Kapur et al., [*Diffusion on Syntax Trees*](https://arxiv.org/abs/2405.20519) | Existing **Faithful (mechanism)** X22 boundary | The tree-edit mechanism and its missing rendered-observation half remain recorded above. Tree arity does not imply weight, activation, or deployment arity. |
| Bogdanova and Kapralov, [*Bounds for Codes over Small Alphabets*](https://www.math.bas.bg/smb/2000_PK/tom_2000/pdf/149-154.pdf) | **Adjacent theorem provenance** | The computer-assisted `A_3(6,3) <= 39` bound proves the 41-message ternary length-six robust arm infeasible. It is a coding bound, not model evidence or a general ternary-optimality result. |

No paper in this table supplies a model, checkpoint, local benchmark result, or
ship evidence. Future CAP rows must use meaningful parse as primary quality,
retain frozen multi-suite gates, and measure physical packing/kernel costs rather
than infer them from nominal arity.

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
| ASAp constraint-mass removal (A2) | `asap_decode` | `models/parallel_decode.py` (`AsapLedger`), `models/twotower.py` |
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

## VSS3-02 cost-to-go energy scorer (adapted EDLM/EBM)

**Fidelity label: adapted.** The VSS3-02 scorer
([`models/solver_energy.py`](../../src/slm_training/models/solver_energy.py))
borrows the *energy-as-ranking* idea from energy-based models (EBM) and
energy-guided discrete/latent decoding (EDLM-style residual guidance), but
deliberately **narrows** their scope: the classical EBM defines an unnormalized
density over sequences and can shape the sample space, whereas this energy has no
authority over legality or membership at all. It only orders the exact, already-
certified live candidate set (`CandidateRanker` seam), and its learned quantity is
**search cost-to-go** (expected remaining exact solver work), not a likelihood or
correctness score. Certified deductions, `UNKNOWN`, and the final verifier remain
the sole authorities; a scorer defect falls back to deterministic order. So the
lineage is "EBM/EDLM ranking signal, subordinated to an exact solver" — an
adaptation, not a faithful reproduction of a generative EBM.

---

## Semantic planning & valid-state learning (SPV0)

**Fidelity label: adapted / adjacent.** SPV0 introduces a `SemanticPlanV1`
contract that separates learned semantic hypotheses from compiler-owned legality.
The contract is documented in [`semantic-planning-valid-state.md`](semantic-planning-valid-state.md)
and the source manifest is [`src/slm_training/resources/autoresearch/semantic-planning-sources.json`](../../src/slm_training/resources/autoresearch/semantic-planning-sources.json).

The plan IR borrows structured-prediction ideas from Pointer Networks, Abstract
Syntax Networks, Set Transformer, DETR, Diffusion On Syntax Trees, Retrieve-and-
Edit, Structured Prediction Energy Networks, Sequence-Level Knowledge
Distillation, DAgger, CDC, Discrete Flow Matching, and FS-DFM. All are applied
only as **soft** candidates, seeds, scorer features, or oracle-diagnostic
controls; the compiler's exact legal action set remains the sole hard authority.
No plan predictor, X22 change, energy model, or training run is added by SPV0-01.

| Source | Fidelity | SPV0 use |
| --- | --- | --- |
| Pointer Networks | Adjacent | Binding pointer candidates |
| Abstract Syntax Networks | Adapted | Factored topology/terminal plan IR |
| Set Transformer | Adjacent | Set-structured plan factors |
| DETR | Adjacent | Variable-cardinality role-slot matching |
| Diffusion On Syntax Trees | Adjacent | Plan-conditioned topology edits |
| Retrieve-and-Edit | Adapted | Retrieved prototype plans |
| Structured Prediction Energy Networks | Adjacent | Global plan energy / ranker regret |
| Sequence-Level Knowledge Distillation | Adapted | Dense teacher plan supervision |
| DAgger | Adjacent | On-policy plan aggregation |
| CDC | Adjacent | Constraint-aware plan localization |
| Discrete Flow Matching | Adjacent | Plan-conditioned valid-edit trajectories |
| FS-DFM | Adjacent | Few-step latency-sensitive plan editing |

## SPV1-04 retrieve-and-edit prototype initialization (SLM-147)

**Fidelity label: wiring / fixture only.** SLM-147 wires the retrieve-and-edit
mechanism from Hashimoto et al., *Retrieve-and-Edit: A Practical Approach to
Large-Scale Domain-Specific Natural Language Generation* (EMNLP 2018,
[arXiv:1812.01194](https://arxiv.org/abs/1812.01194)), specialized to the Kapur
tree-edit diffusion baseline (X22). A train-only local index of hard-valid
canonical AST prototypes is built from the fixture plan corpus; retrieval
strategies include prompt similarity, AST-sketch similarity, SemanticPlanV1
factor similarity, a hybrid, and an oracle-nearest diagnostic control. Retrieved
prototypes are canonicalized, leakage-audited, and hygienically remapped onto
the query placeholder inventory before being used as the initial X22 state.

This is a wiring campaign only: no X22 checkpoint is trained or decoded, no
AgentV evaluation is run, and no ship-gate claim is made. It isolates whether
retrieved prototypes shorten the valid-state search path relative to the generic
minimal X22 seed, leaving the full quality/cost frontier to a later frontier run.
Evidence: [`iter-slm147-x22-retrieval-20260720.md`](iter-slm147-x22-retrieval-20260720.md).

## SPV1-05 plan-conditioned X22 × conflict-slice campaign (SLM-148)

**Fidelity label: wiring / fixture only.** SLM-148 wires the staged seed ×
recovery factorial requested by SPV1-05. It combines the SLM-144/145 plan
predictor, the SLM-146 `OpenUISemanticPlanCompiler`, the SLM-147 leakage-safe
retrieved-prototype index, and the SLM-113 conflict-slice repair policies into a
single campaign harness. Seed strategies cover minimal X22, frequency/archetype
prior, learned archetype+role-set, fully learned plan, a gold-factor diagnostic,
full gold-plan oracle, retrieved prototype, and plan-reranked retrieval.
Recovery arms include no recovery, full remask, suffix rollback, conflict-slice
localized revision, and an oracle-conflict diagnostic.

This is a wiring campaign only: no X22 checkpoint is trained or decoded, no live
conflict analyzer exists yet, no AgentV evaluation is run, and no ship-gate claim
is made. It validates that every preregistered seed source can compile to a
hard-valid initial state, that recovery policies can be applied to those states
with deterministic bookkeeping, and that gold/oracle arms are correctly flagged
non-promotable. The full causal bottleneck classification requires a trained X22
checkpoint, SLM-111 beam/depth points, a real analyzer, and AgentV evaluation.
Evidence: [`iter-slm148-x22-conflict-campaign-20260720.md`](iter-slm148-x22-conflict-campaign-20260720.md).

## SPV1 plan-predictor factor gates (SLM-145) — closed

SLM-145 asked for learned topology, cardinality, and live-symbol pointer heads
under the SPV1 plan-predictor contract.  Its authorization gate required SPV0-02
(SLM-142) to demonstrate, via factor-wise oracle substitution, a downstream
ceiling for each candidate factor.  SLM-142 wired extraction, canonicalization,
oracle substitution, and seed construction, but never ran the factor-wise
gold-substitution experiments, so the gate returned
`blocked_pending_spv0_02_ceiling_evidence`.  No learned head was implemented.
The closeout report is at
`outputs/runs/slm145-plan-predictor-factors-20260720/` with mirrored design
artifacts `docs/design/iter-slm145-plan-predictor-factors-20260720.json` and
`.md`.

## SPV4-02 causal architecture disposition (SLM-160)

**Fidelity label: disposition audit / no new experiment.** SLM-160 closes the
Semantic Planning & Valid-State Learning program by aggregating the preregistered
hypotheses, oracle ceilings, matched experiments, semantic/cost results, cross-DSL
evidence, and guarantee-boundary audits into explicit per-mechanism dispositions.
It does not run a new architecture experiment and it does not override EFS/VSS/CAP/LDI
dispositions.

The audit finds that all SPV evidence up through SLM-159 is wiring/fixture,
blocked, or measured-not-promotable. No mechanism satisfies the criteria for
`adopt_primary` or `adopt_optional`. The canonical architecture remains the
existing honest-slot-contract TwoTower decoder with all plan-aware mechanisms
retained as default-off diagnostics.

Evidence and machine-readable disposition artifact:
[`iter-slm160-spv-disposition-20260720.md`](iter-slm160-spv-disposition-20260720.md)
/ [`iter-slm160-spv-disposition-20260720.json`](iter-slm160-spv-disposition-20260720.json).

## Calculated arity, adaptive precision, and quantization (CAP0–CAP4)

**Fidelity label: adapted / adjacent.** The CAP campaign uses coding-theoretic and
information-theoretic tools as *reference benchmarks and falsifiers*, not as
production algorithms. Exact constructions are verified only for their declared
parameters; estimated quantities carry declared uncertainty and are never treated
as exact.

| Source | Fidelity | CAP use |
| --- | --- | --- |
| BitNet b1.58 (Ma et al., 2024) [arXiv:2402.02764](https://arxiv.org/abs/2402.02764) and BitNet 2B4T (Ma et al., 2025) [arXiv:2501.15308](https://arxiv.org/abs/2501.15308) | Adjacent | Motivates ternary/low-bit baseline arms; our quantizers are reference fake-quant/STE only (`src/slm_training/models/quantization/`) |
| ParetoQ (Zhang et al.) | Adjacent | Motivates mixed-precision sensitivity allocation; our allocation is a grammar-group knapsack (`src/slm_training/harnesses/quantization/allocation.py`) |
| CAT-Q / LLM-QAT-style PTQ | Adjacent | Reference category for calibration and low-bit adaptation (`src/slm_training/harnesses/quantization/calibration.py`) |
| FSQ/LFQ/VQ (e.g. Mentzer et al. FSQ 2024; Yu et al. LFQ 2024) | Adjacent | Latent-codec control arms (`src/slm_training/models/mixed_radix_fsq.py`, `binary_lfq.py`, `learned_vq.py`, `continuous_latent.py`) |
| Weighted tree automata / Hankel methods | Adjacent | Inspiration for exact-state and predictive-rank discussion; no Hankel learning is implemented |
| Grammar/AST-native decoding (e.g. Lark, SynCode, Outlines, Guidance) | Adapted | Hard legality and incremental acceptor (`src/slm_training/dsl/grammar/fastpath/`) |
| Grammar-aligned diffusion / constrained diffusion LLMs | Adapted | Constrained MaskGIT fill and grammar-fastpath admit/reject (`dsl/grammar/fastpath/maskgit_constrain.py`) |
| Structured energy / dynamic programming for legal actions | Adapted | Local-action energy quantizer and exact lattice comparison (`src/slm_training/evals/quantized_energy_inference.py`) |
| AST/tree diffusion (Stern et al. Insertion Transformer; Gu et al. Levenshtein Transformer; Chen et al. Diffusion Forcing) | Adapted | Grammar-topology diffusion v2 (`src/slm_training/models/grammar_diffusion.py`) |
| Mixed-precision sensitivity allocation | Adapted | Group sensitivity profiling + knapsack allocation (`src/slm_training/harnesses/quantization/sensitivity.py`, `allocation.py`) |
| Residual quantization / adaptive compute (e.g. BitNet b1.58 residual approximations, adaptive mixed-precision) | Adapted | Residual ternary planes and adaptive-plane routing (`src/slm_training/models/quantization/residual_planes.py`, `adaptive_planes.py`) |

## External constrained-decoding semantic ceiling (EFS1-01 / SLM-108)

**Fidelity label: adapted / adjacent.** SLM-108 treats publicly available
HuggingFace causal/instruct models as a **control**, not as a deployable
replacement. The adapter loads a pinned model revision and scores the exact live
legal action set produced by the OpenUI compiler (`ExternalLegalActionScorer`);
it never adds or removes candidates. This design borrows the general idea of
using a strong pretrained model as a semantic ceiling (common in distillation and
coverage-evaluation work) but is deliberately narrowed to the compiler-owned
action space so the comparison isolates learned semantic competence from
constraint-layer correctness.

| | |
| --- | --- |
| **Lineage** | External-model baselines for semantic coverage / ceiling estimation; constrained decoding with pretrained LMs (Outlines, Guidance, SynCode family — **Adjacent**) |
| **Fidelity** | **Adapted** — external scorer is legality-agnostic; constrained decode remains compiler-owned |
| **Code** | `src/slm_training/models/external_scorer.py`, `src/slm_training/harnesses/experiments/external_ceiling_matrix.py`, `scripts/run_external_ceiling.py` |
| **Config** | `--matrix-set external-ceiling`, `--checkpoint-reference-uri`, `--mode fixture|frontier` |

**What we took:** a provider-neutral interface for scoring compiler-legal actions
and complete candidates with a pinned HF causal/instruct model, plus an
`ExternalScorePolicy` adapter for the existing eval-only score-policy path.

**What we did not take:** any claim that the external model replaces the tiny
SLM, any fine-tuning of external weights, or any bypass of the compiler in the
constrained arms. Frontier execution requires durable checkpoint provenance
(SLM-103) and a GPU host.

## Exposure scaling as a falsifier (EFS1-02 / SLM-109)

**Fidelity label: adapted / adjacent.** SLM-109 applies the classic machine-
learning diagnostic of scaling compute/data exposure to falsify a specific claim
(“the tiny SLM is simply underexposed”). The ladder design is borrowed from
scaling-law and capacity-ladder practice, but narrowed to a **frozen recipe**:
no architecture, objective, decoder, corpus admission, or evaluation change is
allowed inside the main ladder. Continuation uses bit-exact resume from
`last_full_state.pt` so cumulative target-token exposure is the only experimental
axis.

| | |
| --- | --- |
| **Lineage** | Scaling-law / exposure-threshold experiments; capacity ladders; continuation learning |
| **Fidelity** | **Adapted** — frozen-recipe exposure ladder with bit-exact resume and recipe-hash enforcement |
| **Code** | `src/slm_training/harnesses/experiments/e228_exposure_ladder.py`, `scripts/run_e228_exposure_ladder.py`; continuation via `scripts/train_model.py --resume-from .../last_full_state.pt --target-token-budget` |
| **Config** | `--mode fixture|plan-only|frontier`, `--parent-checkpoint-uri`, `--checkpoint-bucket` |

**What we took:** a preregistered ladder over `target_token_budget` (1×, 4×,
16×, 64×, 128× the original E228 exposure) with recipe-hash freeze and durable
checkpoint provenance.

**What we did not take:** any claim that the current recipe is sufficient until
a 128× run completes with paired confidence intervals. Fixture/planning evidence
is not treated as a result.

## Near-solved semantic corruption curriculum (EFS3-02 / SLM-120)

**Fidelity label: adapted.** SLM-120 tests whether a controlled share of
one- and two-error states improves recovery and fixed-point stability. The
curriculum idea is borrowed from curriculum learning and targeted repair
literature, but narrowed to a **frozen base recipe** and a representation-
independent severity taxonomy so that the intervention is isolated from the
model architecture and corruption policy.

| | |
| --- | --- |
| **Lineage** | Curriculum learning; targeted repair corpora; corruption severity taxonomies |
| **Fidelity** | **Adapted** — frozen-recipe factorial over near-solved share (0%, 5%, 10%, 15%, 30%) with S0–S4 severity levels and CorruptionTraceV2 provenance |
| **Code** | `src/slm_training/data/corrupt/trace.py`, `src/slm_training/harnesses/experiments/corruption_curriculum.py`, `scripts/run_corruption_curriculum.py` |
| **Config** | `--mode fixture|plan-only|frontier`, `--parent-checkpoint-uri`, `--near-solved-shares` |

**What we took:** a preregistered factorial over near-solved share, a severity
schema (S0 clean, S1 one semantic error, S2 two semantic errors, S3 medium,
S4 heavy), and a trace dataclass that can be produced by any representation
(topology diffusion, tree-edit diffusion, token diffusion, or the formal
corruption oracle).

**What we did not take:** any claim that a particular near-solved share helps
until matched A–D arm trains complete with S0 stability, S1/S2 recovery, and
end-to-end binding-aware meaningful v2 metrics. Fixture/planning evidence is
not treated as a result.

## Causal PEFT FTPO adapters (LDI1-02 / SLM-121)

**Fidelity label: adapted.** SLM-121 tests whether small removable PEFT
adapters trained on exact-state causal decision events with FTPO objectives can
improve binding-aware meaningful-program rate while preserving base-model
legality. The FTPO idea is borrowed from preference optimization and targeted
repair literature, but narrowed to **adapter-only updates** on a frozen causal
base recipe so the intervention is isolated from base-weight changes.

| | |
| --- | --- |
| **Lineage** | Parameter-efficient fine-tuning (LoRA/DoRA/PiSSA/AdaLoRA); preference optimization; targeted adapter repair |
| **Fidelity** | **Adapted** — frozen-recipe factorial over FTPO objectives (`unlikelihood`, `ftpo_single`, `ftpo_set`, `legal_set_mass`) and adapter methods, with recipe-hash freeze and removable adapter constraint |
| **Code** | `src/slm_training/harnesses/experiments/causal_peft_ftpo.py`, `scripts/run_causal_peft_ftpo.py` |
| **Config** | `--mode fixture|plan-only|frontier`, `--parent-checkpoint-uri`, `--checkpoint-bucket`, `--objectives`, `--adapter-methods` |

**What we took:** a preregistered factorial over FTPO objectives and adapter
methods, a frozen base recipe extended with adapter/FTPO hyperparameters, and a
torch-free fixture runner that validates the manifest and emits a plan.

**What we did not take:** any claim that a particular FTPO objective or adapter
method improves quality until matched arm trains complete with binding-aware
meaningful v2 metrics and reference-locality drift. Fixture/planning evidence is
not treated as a result.

## TwoTower removable low-rank delta adapter (LDI2-01 / SLM-123)

**Fidelity label: adapted.** SLM-123 adds a small repository-owned LoRA-style
low-rank delta actuator for selected TwoTower denoiser projections. The approach
is borrowed from parameter-efficient fine-tuning literature, but narrowed to a
**removable adapter** that leaves parent weights untouched, can be disabled to
restore the exact parent map, and can be merged one-way into a wrapper-free copy.

| | |
| --- | --- |
| **Lineage** | LoRA / parameter-efficient fine-tuning; removable adapter actuators |
| **Fidelity** | **Adapted** — repository-owned low-rank delta with frozen parent, zero-init identity, deterministic target resolution, save/load/merge, and compatibility fingerprint checks |
| **Code** | `src/slm_training/models/adapters/`, `src/slm_training/models/twotower.py`, `src/slm_training/harnesses/model_build/config.py`, `src/slm_training/harnesses/model_build/factory.py`, `scripts/train_model.py` |
| **Config** | `--adapter-spec`, `--adapter-frozen`, `TwoTowerAdapterSpec` |

**What we took:** a deterministic target resolver, a `LowRankAdapter` wrapper
with frozen parent and zero-initialized `B`, explicit enable/disable/merge
semantics, adapter-only `trainable_parameters()`, and compatibility-fingerprint
guards on load.

**What we did not take:** any claim that a low-rank adapter improves OpenUI
quality until matched adapter-only vs full-update trains complete with
binding-aware meaningful v2 metrics and merge-parity tests on the target device.
Fixture/planning evidence is not treated as a result.

## Dense legal-set knowledge distillation (SPV2-03 / SLM-151)

**Fidelity label: adapted / surrogate.** SLM-151 wires a dense KL-distillation
objective that normalizes both student and teacher to the compiler-owned legal
action set. It borrows the sequence-level knowledge-distillation framing (Hinton
et al., 2015; Kim & Rush, 2016) but deliberately narrows it to a **legal-set**
surrogate: the teacher is a probability vector over exactly the actions the
compiler admits, not a full-vocabulary next-token distribution, and the loss is
a wiring fixture over synthetic logits rather than a trained sequence model.

| | |
| --- | --- |
| **Paper** | Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*, NIPS 2014 Workshop ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)); Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016 ([arXiv:1606.07947](https://arxiv.org/abs/1606.07947)) |
| **Fidelity** | **Surrogate** — legal-set KL over compiler-legal actions only; fixture-only, no full sequence distillation |
| **Code** | `src/slm_training/harnesses/distill/legal_set_kl.py`, `src/slm_training/harnesses/distill/legal_set_teacher_trace.py` |
| **Config** | `temperature`, `teacher_is_prob`, `kl_weight` |

**What we took:** a deterministic teacher-trace manifest, a legal-set-masked KL
objective, and a tiny fixture trainer that can later consume external scorer
outputs (SLM-108).

**What we did not take:** a claim that this surrogate matches sequence-level KD,
that it improves OpenUI quality, or that it is ready for production. Real teacher
scoring requires the SLM-108 external scorer and is not run here. Evidence:
[`iter-spv2-03-legal-set-distillation-20260720.md`](iter-spv2-03-legal-set-distillation-20260720.md).

## Evidence-First Semantic SLM causal synthesis (EFS4-04 / SLM-140)

**Fidelity label: adapted.** SLM-140 publishes a preregistered campaign manifest,
a fail-closed result-loader, an evidence DAG, causal diagnosis, and explicit
architecture dispositions for the Evidence-First Semantic SLM campaign. The
preregistration / registered-report idea is borrowed from meta-science and
causal evidence-synthesis practice, but narrowed to a **versioned, machine-
readable manifest** that is validated against committed `docs/design/iter-*`
result JSON and never treats plan/fixture evidence as a frontier result.

| | |
| --- | --- |
| **Lineage** | Preregistered hypothesis testing; registered reports; evidence synthesis; causal DAGs |
| **Fidelity** | **Adapted** — versioned Pydantic manifest, deterministic evidence graph, honest `NOT_RUN_BY_GATE`/`MISSING` states, and architecture dispositions that require supporting experiments before `ADOPT`/`PROMOTE_EXPERIMENTAL` |
| **Code** | `src/slm_training/harnesses/experiments/efs4_04_causal_synthesis.py`, `scripts/synthesize_efs_campaign.py` |
| **Config** | `docs/design/evidence-first-semantic-slm-campaign-v1.json`, `--docs-design`, `--validate-only`, `--graph-output` |

**What we took:** a preregistered campaign with falsifiers, activation gates,
allowed terminal decisions, and result-ref globs; a loader that infers terminal
states from committed result manifests; causal diagnosis that returns
`insufficient_valid_evidence` when core measurement branches are unresolved;
and explicit architecture dispositions (`ADOPT_AS_SAFETY_ONLY` for correctness
infrastructure, `CONDITIONAL_RESEARCH`/`REJECT`/`NOT_RUN_BY_GATE` for everything
else).

**What we did not take:** any claim that a branch improves semantic quality or
is ready for production until it clears its activation gate and produces a
`POSITIVE` result under ship-gates with durable checkpoints. The synthesis
renderer labels the report as wiring-grade and does not promote a champion.

## Honesty rules (for docs & claims)

1. Do **not** claim “we implement paper X” unless this page tags it **Faithful**.
2. Prefer **Adapted** / **Surrogate** wording in READMEs, PR descriptions, and eval writeups.
3. When adding a new technique, append a row here **in the same PR** as the code.
4. Keep arXiv / venue links stable; deep-link code paths that reviewers can grep.
