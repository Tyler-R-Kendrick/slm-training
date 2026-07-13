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
| **Code** | `admit_fill` in [`grammar_fastpath/maskgit_constrain.py`](../../src/slm_training/grammar_fastpath/maskgit_constrain.py); design note [`grammar-fastpath.md`](grammar-fastpath.md) |
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
| **Code** | `force_emit_token_id`, `dfa_admits_token`, `pick_constrained_token` in [`models/grammar.py`](../../src/slm_training/models/grammar.py); engine in [`grammar_fastpath/engine.py`](../../src/slm_training/grammar_fastpath/engine.py); LTR repair / certify in `TwoTowerModel._ensure_valid_openui` |
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
| **Code** | [`preference/train.py`](../../src/slm_training/preference/train.py) (`dpo_loss`), pair builders in [`preference/`](../../src/slm_training/preference/) |
| **CLI** | `scripts/train_preference.py` |

### GRPO-lite

| | |
| --- | --- |
| **Paper** | Shao et al., *DeepSeekMath* (Group Relative Policy Optimization), 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| **Fidelity** | **Surrogate / lite** — group rollouts + mean/std advantages over a **structure-only** reward; not a full RLHF stack |
| **Code** | [`rl/`](../../src/slm_training/rl/) (`grpo_loss_for_group`, `train_grpo`) |
| **CLI** | `scripts/train_rl.py`; matrix row E10 in [`quality-experiment-matrix.md`](quality-experiment-matrix.md) |

---

## DSL-native output representation (V5)

| | |
| --- | --- |
| **Papers** | Rabinovich et al., *Abstract Syntax Networks* [arXiv:1704.07535](https://arxiv.org/abs/1704.07535); Kusner et al., *Grammar VAE* [arXiv:1703.01925](https://arxiv.org/abs/1703.01925); Xue et al., *ByT5* [arXiv:2105.13626](https://arxiv.org/abs/2105.13626); CFG-constrained diffusion LMs [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) |
| **Fidelity** | **Adapted** — lexer-native categorical tokens + dynamic symbol table + byte literal channel + kind-factorized embeddings; **not** production-rule sequences or graph diffusion |
| **Code** | [`dsl_tokenizer.py`](../../src/slm_training/models/dsl_tokenizer.py), kind masks in [`token_map.py`](../../src/slm_training/grammar_fastpath/token_map.py), structural mask/remask in [`twotower.py`](../../src/slm_training/models/twotower.py) |
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
| **Fidelity** | **Adapted** (lite) — [`FastPathGate`](../../src/slm_training/grammar_fastpath/gate.py) trained via [`trust_train.py`](../../src/slm_training/grammar_fastpath/trust_train.py) (E31); gates E33 remask |
| **Intent** | Cheap per-token reliability head for remask budgets |

### BackPlay (frozen-model correction head)

| | |
| --- | --- |
| **Paper** | *BackPlay: Plug-in Look-Back Self-Correction for Diffusion Language Models*, 2026. [arXiv:2601.06428](https://arxiv.org/html/2601.06428v2) |
| **Fidelity** | **Adapted** — freeze denoiser, mine own errors, train plug-in gate (E31) |
| **Intent** | Model-specific remask scores without joint generator–critic optimization |

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
| Remask, don’t replace | [arXiv:2604.18738](https://arxiv.org/abs/2604.18738) | **Adapted** — E33 remask policy constraint |

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
| **Code analogue** | `FastPathGate` + BackPlay-lite mining ([`grammar_fastpath/gate.py`](../../src/slm_training/grammar_fastpath/gate.py), [`trust_train.py`](../../src/slm_training/grammar_fastpath/trust_train.py)); grammar remains legality authority |
| **Proposed** | Calibration / abstention (**E53** in the mapping doc) |

### MDPO / d1 (trajectory-aligned masked-diffusion RL)

| | |
| --- | --- |
| **Papers** | *MDPO: Overcoming the Training-Inference Divide of Masked Diffusion Language Models*, 2025. [arXiv:2508.13148](https://arxiv.org/abs/2508.13148). Related: d1 masked-diffusion policy optimization [arXiv:2504.12216](https://arxiv.org/abs/2504.12216); PAPO / dOPSD-style dense intermediate rewards (Adjacent) |
| **Fidelity** | **Adjacent** — candidates to replace the GRPO-lite **Surrogate** on final strings |
| **Code today** | [`rl/`](../../src/slm_training/rl/) GRPO-lite; preference stage in [`preference/train.py`](../../src/slm_training/preference/train.py) |
| **Proposed** | Trajectory-aligned objective on intermediate MaskGIT states (**E54**) |

### Constrained diffusion decoding (LAVE / EPIC family)

| | |
| --- | --- |
| **Paper** | *Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars*, 2026. [arXiv:2602.00612](https://arxiv.org/abs/2602.00612). Related: Mündler et al. CFG∩completions [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) (already **Adapted** above) |
| **Fidelity** | **Adjacent** — diffusion-native constrained decode beyond our cheap hole-admit stand-in |
| **Code analogue** | `admit_fill` in [`grammar_fastpath/maskgit_constrain.py`](../../src/slm_training/grammar_fastpath/maskgit_constrain.py) |

### PlanBench / generalization gap / CoT brittleness

| | |
| --- | --- |
| **Papers** | PlanBench [arXiv:2206.10498](https://arxiv.org/abs/2206.10498); *On the Generalization Gap in LLM Planning* [arXiv:2601.14456](https://arxiv.org/abs/2601.14456); Chain-of-Thoughtlessness / related CoT collapse under complexity |
| **Fidelity** | **Adjacent** — motivates **schema-level** held-out splits (unseen component families, symbol rename), not only held-out instances |
| **Proposed** | **E55** + `toy-layout` transfer stress; see [`verifier-guided-repair.md`](verifier-guided-repair.md) §4 |

### LLM+P (neural formalize, symbolic search)

| | |
| --- | --- |
| **Paper** | LLM+P / related “LLM writes PDDL, classical planner solves” [arXiv:2304.11477](https://arxiv.org/abs/2304.11477) |
| **Fidelity** | **Adjacent** — lesson only: do not force the neural model to relearn exact search when a checker/planner is cheap. Here the “planner” stand-in is the OpenUI grammar stack + optional best-of-N, not Fast Downward |

---

## Systems & data (not papers, but cited lineage)

| System | Role here | Link / path |
| --- | --- | --- |
| OpenUI / `@openuidev/lang-core` | Official grammar, streaming parse, validate | [openui.com](https://www.openui.com/) · [`tools/openui_bridge/`](../../tools/openui_bridge/) |
| `@openuidev/react-lang` Renderer | Annotate playground visual preview | [`tools/openui_preview/`](../../tools/openui_preview/) |
| Lark `InteractiveParser` | Incremental LALR acceptor for force-emit / admit | [`grammar_fastpath/engine.py`](../../src/slm_training/grammar_fastpath/engine.py) |
| `@google/design.md` | DESIGN.md lint in preference reward | [`tools/design_md_bridge/`](../../tools/design_md_bridge/) |
| RICO | Mobile UI screens → OpenUI seeds | [`fixtures/rico/`](../../fixtures/rico/) |
| BF16 exponent codebook | Optional weight-compression sidecar | [brianbell-x weight-compression](https://brianbell-x.github.io/weight-compression/) · [`runtime-performance.md`](runtime-performance.md) |

---

## Quick map: idea → knob → file

| Idea | Primary knob | Primary file |
| --- | --- | --- |
| MaskGIT unmask | `--parallel-unmask` | `models/parallel_decode.py` |
| CFG hole admit | config `grammar_fastpath_mode` ∈ `{force,mask,hybrid}` | `grammar_fastpath/maskgit_constrain.py` |
| DFA force-emit | config `grammar_fastpath` + mode `force\|hybrid` | `grammar_fastpath/force_emit.py`, `models/grammar.py` |
| Valid-only certify | config `grammar_ltr_primary` + repair + finalize | `models/twotower.py` (`_ensure_valid_openui`) |
| Length-safe LTR | `grammar_ltr_max_tokens` / `grammar_ltr_stages` | `models/twotower.py` |
| MDLM schedule | `mdlm_schedule` | `models/twotower.py` (`_mask_targets`) |
| Remasking | `remask_ratio` / `remask_use_gate` / `remask_use_entropy` | `models/parallel_decode.py` |
| Template fill | `template_fill_decode` | `models/template_fill.py` |
| Honest inventory | `honest_slot_contract` | `models/template_fill.py` (`inventory_from_prompt`) |
| Preference stage | `scripts/train_preference.py` | `preference/train.py` |
| GRPO-lite | `scripts/train_rl.py` | `rl/` |
| LTR suffix rollback | `suffix_rollback_window` (E30) | `models/twotower.py` |
| Trust head | `trust_gate` / `FastPathGate` (E31) | `grammar_fastpath/gate.py`, `trust_train.py` |
| Visible-token corruption | `visible_corrupt_rate` (E32) | `models/twotower.py` (`_mask_targets`) |
| Combined remask policy | E33 | `parallel_decode.select_remask_policy_indices` |
| Latent critic MoE (deferred) | E34 | `research-correction-critics.md` |
| Verifier-guided repair map | proposed E50–E55 | `verifier-guided-repair.md` |

---

## Honesty rules (for docs & claims)

1. Do **not** claim “we implement paper X” unless this page tags it **Faithful**.
2. Prefer **Adapted** / **Surrogate** wording in READMEs, PR descriptions, and eval writeups.
3. When adding a new technique, append a row here **in the same PR** as the code.
4. Keep arXiv / venue links stable; deep-link code paths that reviewers can grep.
