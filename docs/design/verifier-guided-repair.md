# Verifier-guided repair — applicability mapping

This page maps an executive assessment of **PDDL-Instruct–style
verifier-derived supervision** and related diffusion-repair ideas onto
**this** repository (`slm-training`). It is the honesty contract for how far
those ideas apply, what we already do, and which future levers are worth
implementing.

Companion pages:

- Papers → code fidelity tags: [research-lineage.md](research-lineage.md)
- V4 remask / trust gate: [research-correction-critics.md](research-correction-critics.md)
- Experiment matrix (E0–E55, X0–X8): [quality-experiment-matrix.md](quality-experiment-matrix.md)
- Grammar backends: [grammar-backends.md](grammar-backends.md)

**Scope of this document:** design mapping. No PDDL planner, VAL, or Fast
Downward integration is planned for this repo.

**ID collision note:** Matrix IDs **E50–E55** are taken by the shipped **V6**
CoRe / T2M / slot-trust / honest-champion levers. Remaining assessment gaps
are reserved as **E60–E65** (proposed). Do not reuse E50–E55 for new meanings.

---

## 1. What this repo actually is

`slm-training` trains a **masked discrete diffusion SLM** (TwoTower /
optional `GrammarDiffusionModel`) to emit **grammar-valid OpenUI layout
programs** with placeholder content slots. The authoritative “reasoning”
artifact is the DSL program itself — not free-form chain-of-thought prose.

```text
NL prompt (+ DESIGN.md)
        → context tower (scratch | frozen HF SmolLM2)
        → MaskGIT / block-diffusion denoiser
        → DFA force-emit + hole admit + LTR certify
        → OpenUI program (placeholders only)
```

The assessment’s planning stack (PDDL domain/problem → VAL → Fast Downward)
does **not** describe this product. The transferable pattern is narrower:

> Train and decode a diffusion model as a **bidirectional, verifier-guided
> repair operator** over a **typed executable representation**, keeping the
> symbolic checker at inference time.

That pattern already matches large parts of our architecture. The rest of
this page says exactly which parts.

---

## 2. Three correctness layers (translated)

The assessment separates formalization, planning validity, and execution.
For OpenUI generation those layers become:

| Layer | Planning (assessment) | This repo | Authority today |
| --- | --- | --- | --- |
| **Formalization** | Does the PDDL capture the user’s request? | Does the layout / slot inventory capture the prompt + DESIGN.md intent? | Human annotation, DESIGN.md lint, inventory-in-prompt (E35); **not** fully automatic |
| **Structural validity** | Is the plan valid under the PDDL? | Is the program parseable / typed / placeholder-legal OpenUI? | `@openuidev/lang-core` + Lark DFA (`dsl/grammar/backends/`, `dsl/grammar/fastpath/`) |
| **Execution** | Do tools behave as the PDDL effects say? | Does the React `Renderer` preview render the intended UI? | Preview island (`src/apps/openui_preview/`); not a training gate |

A program can be fully grammar-valid while solving the wrong layout intent.
Grammar verification only covers the middle layer — same caveat VAL has for
PDDL. Do not treat parse success as semantic success.

---

## 3. Already implemented (honest analogues)

These are **Adapted** or **Surrogate** analogues of the assessment’s design,
not Faithful reimplementations of PDDL-Instruct.

| Assessment idea | Analogue here | Primary code |
| --- | --- | --- |
| Typed executable trace (not prose CoT) | OpenUI DSL + compositional / lexer-native tokenization | [`dsl/`](../../src/slm_training/dsl/), [`models/dsl_tokenizer.py`](../../src/slm_training/models/dsl_tokenizer.py), [`models/tokenizer.py`](../../src/slm_training/models/tokenizer.py) |
| Inference-time verification | `_ensure_valid_openui` certify / repair / finalize | [`models/twotower.py`](../../src/slm_training/models/twotower.py) |
| Constrained decoding during denoising | DFA force-emit + MaskGIT hole admit (`force` / `mask` / `hybrid`) | [`dsl/grammar/fastpath/`](../../src/slm_training/dsl/grammar/fastpath/), [`models/grammar.py`](../../src/slm_training/models/grammar.py) |
| Process / value head | `FastPathGate` trust scores; BackPlay-lite + slot-aware mining (E31/E52) | [`dsl/grammar/fastpath/gate.py`](../../src/slm_training/dsl/grammar/fastpath/gate.py), [`trust_train.py`](../../src/slm_training/dsl/grammar/fastpath/trust_train.py) |
| Remasking low-confidence commitments | V3 confidence remask + V4 E33 + **V6 CoRe (E50)** + **T2M (E51)** | [`models/parallel_decode.py`](../../src/slm_training/models/parallel_decode.py) |
| Skeleton / landmarks before fill | Template fill from prompt-visible slot inventory | [`models/template_fill.py`](../../src/slm_training/models/template_fill.py) |
| Valid-over-invalid preference | Structure-first `composite_reward` + human preference pairs + E55 process stage | [`harnesses/preference/`](../../src/slm_training/harnesses/preference/), [`harnesses/rl/`](../../src/slm_training/harnesses/rl/) |
| Training-time process signal | Masked CE / MDLM, fidelity aux, visible corruption (E32), GRPO-lite | [`models/twotower.py`](../../src/slm_training/models/twotower.py) |

**Important honesty note:** remask is driven by **confidence, trust-gate,
entropy, grammar legality, and CoRe support-drop** — still not by a localized
“first hard parse error + structural dependents” cone. That remaining gap is
**E61** below.

---

## 4. Real gaps → proposed levers (E60+)

These assessment recommendations still map cleanly onto OpenUI generation and
are **not** fully implemented. IDs **E60–E65** are reserved (docs only until
code lands). Do **not** reuse E50–E55.

| Proposed ID | Gap | Why it matters here | Suggested shape |
| --- | --- | --- | --- |
| **E60** | Differential validation | `OpenUIHybridBackend` **falls back** lang-core ↔ Lark; it does not dual-parse or quarantine disagreement → verifier monoculture | Parse with both backends when available; on disagreement, quarantine sample from train/eval claims; log disagreement rate as a reliability metric |
| **E61** | Failure-cone remasking | Remask is score-/CoRe-based, not error-localized | On first hard parse / stream error at span `k`, remask `k` plus structural dependents (containing statement, binder refs, child list); freeze verified spans outside the cone |
| **E62** | Minimal hard negatives | **Wired and P13-verified:** the deterministic `data/corrupt` taxonomy emits verified clean-target repair rows across lexical, grammar, schema, reference-graph, dataflow, and patch failures; the integrated corpus contains the `corruption_repair` family and passes the final verifier | Use [`data/corrupt`](../../src/slm_training/data/corrupt/) cases for repair / preference / RL inputs; ambiguous repairs carry multiple accepted targets and are excluded from exact-repair claims; [P13 evidence](data-synthesis.md) |
| **E63** | Calibration / abstention | Trust gate is BCE-trained but not calibrated for selective decode | ECE / Brier on gate; selective accuracy vs coverage; abstain or escalate to best-of-N / longer remask when gate is unreliable |
| **E64** | Trajectory-aligned diffusion RL | E55 / GRPO-lite still score final strings; schedule mismatch remains | Collect on-policy intermediate MaskGIT states; label with grammar + reward; prefer MDPO / d1-style objectives over AR PPO/GRPO imports |
| **E65** | Schema-level generalization | Held-out instances ≠ held-out component schemas | Leave-one-schema-family-out; symbol rename / reordered decls; use `toy-layout` backend as a deliberately alien grammar for transfer stress |

Priority intuition (not a schedule): **E60** and **E61** are the highest
reliability leverage relative to another generic SFT round; **E62** cleans
credit assignment; **E63–E65** harden claims before calling the system
“schema-general.”

---

## 5. Deliberately not applicable

Do **not** port these assessment pieces into this codebase:

| Assessment component | Why it does not apply |
| --- | --- |
| PDDL / VAL / Fast Downward / Unified Planning | OpenUI programs are declarative layout trees, not action sequences over a transition system |
| State–action–state process traces | No world state, preconditions, or add/delete effects |
| Forward reachability ∩ backward goal regression | No goal fluents or causal support links between actions |
| Exact planner as search partner / UNSAT certificates | No classical planning problem; unsatisfiable layouts are not a first-class formal object here |
| Linked `symbolic-planning` GitHub sketch | Not an official PDDL-Instruct implementation; regex PDDL parser and broken state traces make it unsafe as a trusted base — and irrelevant to OpenUI |

If a future product adds tool-using agents with real world models, revisit
this section. Until then, keep symbolic authority on **grammar + placeholder
policy + DESIGN.md lint**, not on a planner.

---

## 6. Threats restated for this repo

| Threat (assessment) | Failure mode here | Defense today / needed |
| --- | --- | --- |
| **Verifier monoculture** | One parser quirk poisons train labels and eval | Need **E60** differential validation; today hybrid is fallback-only |
| **Parser exploitation** | Model learns to pass a buggy acceptor | Fuzz OpenUI strings; keep lang-core as primary authority; quarantine on backend disagreement |
| **Single-reference bias** | Alternate valid layouts penalized | Multi-plan positives via preference pairs / best-of-N; score with `composite_reward`, not exact string match |
| **Locally valid wandering** | Legal but intent-irrelevant trees | Fidelity / inventory contract (E35); human thumbs; still weak on open-ended intent |
| **High-confidence early freeze** | Wrong early tokens locked | V3–V6 remask + E30 suffix rollback + CoRe/T2M; deepen with **E61** cone remask |
| **Oscillatory repair** | Same spans flip forever | Remask budgets (E33); add repair memory / decreasing remask budget if oscillation appears |
| **Reasoning rationalization** | Prose justifies an invalid tree | We already forbid prose as the substrate; keep explanations (if any) rendered *from* the verified program |
| **Reward hacking** | Model exploits reward / lint quirks | Structure-only eval path ([structure-only-eval.md](structure-only-eval.md)); adversarial suites; do not credit gold DESIGN.md lint at ship time |
| **Domain-schema memorization** | Strong on known components, collapse on new schemas | Need **E65**; `toy-layout` is the ready held-out grammar |

---

## 7. Evaluation additions (when implementing E60+)

Keep existing ship gates (`parse_rate`, `placeholder_fidelity`,
`structural_similarity`, structure-only `reward_score`). Add when the
corresponding lever lands:

| Metric family | Examples |
| --- | --- |
| Reliability | Backend disagreement rate; ECE / Brier on `FastPathGate`; selective accuracy vs coverage |
| Repair | % slots remasked; repair cycles; oscillatory remask rate; cone size distribution (E61) |
| Negatives | First-error-class confusion on minimal counterexamples (E62) |
| Generalization | Unseen schema / renamed symbols / `toy-layout` transfer (E65) |
| Efficiency | Diffusion NFEs, verifier calls, planner-fallback N/A |

Ablation ladder for attributing gains (mirror the assessment, OpenUI-flavored):

1. Free-form / unconstrained decode (baseline)
2. Typed valid-program SFT
3. + minimal hard negatives
4. + detailed verifier feedback / process heads
5. + trajectory-aligned diffusion optimization
6. + grammar/type constrained decoding (largely done)
7. + inference-time verify with localized remask (partially done via V4/V6; deepen via E61)
8. + multi-candidate / preference ranking (partially done)

---

## 8. Relation to V4 / V5 / V6 work

| Ship cycle | What it contributed | Relation to this page |
| --- | --- | --- |
| **V3** | Length-safe LTR, remask, template fill | Inference verify + remask skeleton |
| **V4** | Trust gate, combined remask policy, honest inventory | Process-head analogue; closes silent-gold leakage |
| **V5** | DSL-native / lexer tokenizer (E40–E46) | Stronger typed executable representation |
| **V6** | CoRe remask (E50), T2M (E51), slot-aware trust (E52), honest champion (E53), grammar-honest (E54), process (E55) | Stronger revision policy; **not** differential validation or failure-cone remask |
| **V7** | Speculative denoising (E70–E75) | Stability signals, dependency clusters, ordered cluster verify, survival head, successor cache — see [speculative-denoising.md](speculative-denoising.md) |
| **Proposed** | E60–E65 above | Remaining verifier-guided gaps from the assessment |

E34 (latent MoE critics) remains deferred research-grade; see
[research-correction-critics.md](research-correction-critics.md). Prefer
landing E60–E62 before reopening E34.

---

## 9. Inspiration / references (short list)

Full fidelity tags and code maps live in [research-lineage.md](research-lineage.md)
under **Verifier-guided planning & repair (Adjacent lineage)**. Headline sources:

- **PDDL-Instruct** — [arXiv:2509.13351](https://arxiv.org/abs/2509.13351) — verifier-derived process supervision framing; losses underspecified; workshop/preprint evidence; **do not** use the linked unofficial repo as a base
- **LLM-Modulo / CEGIS planning** — candidate–verifier–counterexample loop (closest architectural precedent for certify + repair)
- **FoVer** — [arXiv:2505.15960](https://arxiv.org/abs/2505.15960) — distill formal checks into a compact process model (motivates strengthening `FastPathGate`)
- **MDPO** — [arXiv:2508.13148](https://arxiv.org/abs/2508.13148); **d1** — [arXiv:2504.12216](https://arxiv.org/abs/2504.12216) — trajectory-aligned diffusion policy optimization
- **Constrained diffusion decoding (LAVE / related)** — [arXiv:2602.00612](https://arxiv.org/abs/2602.00612) — relates to existing `admit_fill`
- **PlanBench / CoT brittleness / generalization gap** — motivates schema-level splits, not only held-out instances

---

## 10. Honesty rules

1. Do not claim we “implement PDDL-Instruct” or run VAL. Tag that lineage **Adjacent**.
2. Do not treat grammar validity as formalization or UX correctness.
3. Do not redefine **E50–E55**; those IDs belong to shipped V6 levers.
4. When implementing E60–E65, update this page’s “proposed” rows to “wired”
   and append matching rows to `research-lineage.md` + the quality matrix **in
   the same PR**.
5. Prefer **remask, don’t replace** and keep the deterministic grammar stack
   as the legality authority.
