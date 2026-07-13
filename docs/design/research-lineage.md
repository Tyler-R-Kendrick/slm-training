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
| CFG hole admit | `--grammar-fastpath-mode mask\|hybrid` | `grammar_fastpath/maskgit_constrain.py` |
| DFA force-emit | `--grammar-fastpath` / mode `force\|hybrid` | `grammar_fastpath/force_emit.py` |
| Valid-only certify | `--grammar-ltr-primary` + repair + finalize | `models/twotower.py` (`_ensure_valid_openui`) |
| Length-safe LTR | `grammar_ltr_max_tokens` / stages | `models/twotower.py` |
| MDLM schedule | `mdlm_schedule` | `models/twotower.py` (`_mask_targets`) |
| Remasking | `remask_ratio` | `models/parallel_decode.py` |
| Template fill | `template_fill_decode` | `models/template_fill.py` |
| Preference stage | `scripts/train_preference.py` | `preference/train.py` |
| GRPO-lite | `scripts/train_rl.py` | `rl/` |

---

## Honesty rules (for docs & claims)

1. Do **not** claim “we implement paper X” unless this page tags it **Faithful**.
2. Prefer **Adapted** / **Surrogate** wording in READMEs, PR descriptions, and eval writeups.
3. When adding a new technique, append a row here **in the same PR** as the code.
4. Keep arXiv / venue links stable; deep-link code paths that reviewers can grep.
