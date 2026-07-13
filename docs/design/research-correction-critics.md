# Correction, remasking, and latent critics

Inspiration and candidate research for improving TwoTower OpenUI quality via
**revision** (undoing premature commitments) and **semantic critique**, beyond
the confidence remask already wired in V3. Nothing in the V4 levers is
implemented yet — tags in [research-lineage.md](research-lineage.md) are
**Adjacent** for critic/trust-head extensions. Concrete levers are **E30–E34**
in [quality-experiment-matrix.md](quality-experiment-matrix.md).

V3 already **Adapted** confidence remasking (`remask_ratio` / `select_remask_indices`)
and template fill; those clear fixture `--ship-gates`. This note targets what
comes *after* that baseline remask.

---

## Why this matters for us

Current setup (see [research-lineage.md](research-lineage.md)):

| Layer | Today | Gap |
| --- | --- | --- |
| Training | MaskGIT / optional MDLM schedule ([`_mask_targets`](../../src/slm_training/models/twotower.py)) | Visible tokens are still treated as clean except when MDLM masks them — the denoiser does not systematically learn to revise a *wrong but visible* token (GIDD-style) |
| Decode order | Length-safe LTR + MaskGIT (+ template fill) | LTR-primary commitments remain mostly permanent unless MaskGIT remask path is used |
| Remasking | Grammar reject + **V3 confidence remask** (`remask_ratio`) | No semantic / fidelity remask policy or learned trust head |
| Trust head | [`FastPathGate`](../../src/slm_training/grammar_fastpath/gate.py) defined, **unwired** | No learned token-correctness signal beyond softmax confidence |

Grammar/DFA verification already catches illegal OpenUI. Learned correction
targets failures the grammar cannot see: wrong but legal trees, placeholder
namespace drift, slot-contract violations, brittle structure that passes stream
checks.

---

## Core decomposition

Keep three concerns separate:

```text
decode ordering          +  revision transition  +  revision policy
(LTR / MaskGIT / block)     (ReMDM / GIDD)           (confidence / trust / latent critic MoE)
```

| Concern | Role | Maps into this repo |
| --- | --- | --- |
| **Decode ordering** | Which positions open first | `parallel_decode.select_unmask_indices`, `_greedy_ltr_decode_batch`, MaskGIT / template-fill paths in `twotower.py` |
| **Revision transition** | How a committed token becomes uncertain again (`token → [MASK]` or learned edit) | V3: confidence remask + grammar remask. Candidate: critic-/trust-gated remask; later GIDD/SCDD training |
| **Revision policy** | *Whether* and *where* to revise | Today: confidence quantile + stream/DFA reject. Candidate: BackPlay-lite gate → RemeDi-like UPS → latent falsification MoE |

**Rule:** critics should primarily replace/augment the **revision policy**, not
replace ReMDM/GIDD or the existing LTR/MaskGIT construction order. Prefer
**remask, don't replace**: reset suspect tokens to `[MASK]` and regenerate from
cleaned context rather than overwrite while neighbors may still be wrong
([arXiv:2604.18738](https://arxiv.org/abs/2604.18738)).

---

## Paper inventory

| Paper | arXiv | Core idea | Relevance here |
| --- | --- | --- | --- |
| **Coconut** | [2412.06769](https://arxiv.org/abs/2412.06769) | Recurrent continuous latent reasoning (thoughts fed back through the decoder) | Inspiration for multi-step *latent* critique without emitting critique text. Serial full-backbone rollouts are likely too expensive for our decode loop — prefer small recurrent modules / parallel streams |
| **BackPlay** | [2601.06428](https://arxiv.org/html/2601.06428v2) | Plug-in correction head on a **frozen** DLM, trained on that model’s own error distribution; drives remasking | Strongest near-term template. Maps to wiring `FastPathGate` (E31) on frozen TwoTower weights |
| **RemeDi** | [2509.23653](https://arxiv.org/html/2509.23653v1) | Separate unmasking-policy stream (UPS): learn token quality; remask via SFT/RL; train on masked *and* randomly replaced visible tokens | Semantic upgrade path from BackPlay-lite: cheap local confidence always-on; heavier critic gated. Training signal informs E32 |
| **ReMDM** | [2503.00307](https://arxiv.org/abs/2503.00307) | Inference-time remasking sampler for pretrained masked diffusion — send revealed tokens back to `[MASK]` without retraining | Conceptual parent of V3 confidence remask; E30 extends to LTR suffix rollback and critic-gated budgets |
| **GIDD** | [2503.04482](https://arxiv.org/abs/2503.04482) | Hybrid absorbing masks + uniform token noise so the denoiser sees wrong *visible* tokens and learns revision transitions | Future foundation objective; wrong visible-token supervision. Do **not** replace MaskGIT/MDLM early (E32 is a lite subset) |
| **SCDD** | [2603.02230](https://arxiv.org/html/2603.02230v1) | Explicit discrete self-correction transitions (alternative to GIDD’s hard-to-tune interpolation); evidence at GPT-2 scale | Adjacent alternative if GIDD-style pretraining is revisited |
| **SPC** | [2504.19162](https://arxiv.org/html/2504.19162v1) | Self-play: sneaky generator produces hard reasoning errors; critic detects them | Curriculum for specialized critics (adversarial corruptions per failure family) |
| **Sparse MoE reward / PRISM-style** | [2606.04284](https://arxiv.org/abs/2606.04284) | Sparse MoE critics/discriminators with interpretable specialization under sparsity + diversity objectives | Evidence that expert specialization is *induced*, not automatic — needed if we build E34 |
| **LLaDA-MoE** | [2509.24389](https://arxiv.org/abs/2509.24389) | Sparse MoE inside a diffusion LM (generator experts) | Shows MoE + discrete diffusion is viable; not a critic MoE — do not confuse generator experts with falsification experts |
| **Counterfactual MoE routing** | [2605.07260](https://arxiv.org/abs/2605.07260) | Standard routers are least informative on fragile reasoning tokens; router-only updates can help | Motivation to train routers on **corrective utility**, not token perplexity alone; avoid brittle top-1 token routers |
| **PLR (parallel latent streams)** | [2601.03153](https://arxiv.org/abs/2601.03153) | Width-level latent reasoning: multiple parallel continuous streams + aggregation, instead of deeper serial latent chains | Preferred *execution* for latent critics vs serial Coconut: several streams / few refinement rounds, no repeated full denoiser backbone |
| **MIRAGE** | [2606.04627](https://arxiv.org/abs/2606.04627) | Continuous latent reasoning for mobile agents (compress CoT into hidden states + world-model objective) | Adjacent prior for *internal* continuous critique without decoding long rationale text — not a parallel-slot recipe |
| **Deferred commitment / sliding windows** | [2601.02076](https://arxiv.org/abs/2601.02076) | Confidence-aware sliding windows vs fixed block boundaries for DLMs | Supports blockwise LTR + revisable window (E30) over strict irreversible LTR |
| **Token ordering in masked diffusions** | [2502.06768](https://arxiv.org/html/2502.06768v1) | Masked training teaches fill-in given visible context; does not teach repair of wrong visible tokens | Explains remaining “visible ≠ revisable” gap after V3 confidence remask |
| **Remask, don’t replace** | [2604.18738](https://arxiv.org/abs/2604.18738) | Prefer token→mask→token refinement over direct substitution while context is still wrong | Remask policy design constraint for E30/E33 |

---

## Mixture of Falsification Experts (long-horizon design)

**Do not** insert standard MoE layers throughout the diffusion backbone. Build a
separate sparse critic over the frozen denoiser’s hidden state:

```text
Frozen denoiser
    │
    ├── penultimate / multi-layer Hₜ, uncertainty, grammar signals
    │
    ▼
Shared general correction head  (always on)
    ├── Is correction needed?
    ├── Which spans look suspicious?
    └── Which failure modes are plausible?
    │
    ▼
Top-2 (or Top-K under router uncertainty) specialist latent critics
    ├── parallel latent streams + few recurrent refinement rounds
    ├── mechanism-specific falsification
    └── risk / remask / revision conditioning
    │
    ▼
Critique controller → remask budget + continue/stop
```

The **shared head matters**: a purely sparse router that must already recognize a
blind spot to pick the expert that recognizes that blind spot fails before
criticism begins.

### Expert taxonomy (OpenUI-specific)

Specialize by **failure mechanism**, not broad domain (“UI expert”):

1. **Grammar / structure violation** — illegal nesting, unfinished constructs (mostly owned by Lark/DFA today; critic decides *when* to trust grammar vs soft risk).
2. **Placeholder / namespace consistency** — `:acme.*` drift, invalid slot names (fidelity failure family).
3. **Slot-contract compliance** — inventory/schema vs emitted tree (ties to E12 / template fill).
4. **Component-hierarchy plausibility** — legal but nonsensical trees (depth/type co-occurrence).

Defer generic reasoning experts (counterexample, causal reversal, etc.) unless
we expand beyond OpenUI.

### Routing and training principles

- Route at **candidate / span** level, not per-token top-1.
- Combine shared + sparse: \(z = z_{\text{shared}} + \sum_{e \in \mathrm{TopK}} w_e z_e\).
- Train the router on **counterfactual repair utility** (does remasking guided by expert \(e\) improve `composite_reward` / parse / fidelity?) — not only CE correlation.
- Adversarial curricula per expert (\(A_e\): subtly corrupt gold so the frozen model and weak critics fail); reward detection + localization + successful remask repair, with penalties for over-remasking and false positives.
- Force specialization: diversity/decorrelation losses, ablations, abstention, periodic re-clustering of residual failure modes (do not assume MoE automatically specializes).

### Independence from the generator

Critics that only read final hidden states can inherit the backbone’s blind spots.
Prefer multi-view inputs (early + late layers, \(\Delta H\), entropy, grammar verifier
signals, deliberately corrupted trajectories), stop-gradient into the backbone,
and critic-specific projections.

### Latent recurrence cost

Prefer **parallel latent streams** (PLR-style width scaling; small fixed rounds)
over serial Coconut decode through the full TwoTower backbone each thought step.

---

## ReMDM vs GIDD vs RemeDi

| Property | ReMDM | RemeDi | GIDD / SCDD |
| --- | --- | --- | --- |
| Works with existing MaskGIT weights | Yes | Partial (needs UPS training) | No (needs training-process change) |
| Requires training changes | No | Yes (UPS + optional visible corruption) | Yes (hybrid / self-correction transitions) |
| Revision mechanism | Token → mask → token at inference | Learned remask policy + regenerate | Learned edit / hybrid noise |
| Best use here | Extend V3 confidence remask + LTR rollback (E30) | Cheap trust + remask head (E31/E33) | Later foundation when we own pretraining again |
| Integration risk | Low | Medium | High |

**Progression we recommend:**

1. Keep MaskGIT / MDLM training (V3).
2. Keep V3 confidence remask; add LTR suffix / window rollback where LTR-primary is used (**E30**).
3. Wire a BackPlay-lite / RemeDi-lite trust head on frozen model errors (**E31**).
4. Add light visible-token corruption supervision (**E32**) and a combined remask budget (**E33**).
5. Only then compare GIDD/SCDD-style objectives or latent MoE critics (**E34**).

### Recommended decoder shape (target)

```text
1. LTR (or blockwise LTR) within a sliding active window
2. Revisable suffix / block behind the frontier
3. Cheap token-quality head every step (FastPathGate / RemeDi-lite)
4. Heavier critics only at block boundaries, uncertainty spikes,
   projection disagreement, or grammar/tool failure
5. Combine critic + entropy + grammar into a remask distribution
6. Remask implicated spans (+ dependency neighborhood); re-denoise
7. Commit when expected repair value < compute threshold
```

Variants if needed later: critic-triggered global rollback (expensive cache
invalidations); blockwise LTR with bidirectional remask inside the active block.

---

## What moves the needle *now*

Honest priority given [quality-experiment-matrix.md](quality-experiment-matrix.md)
results:

1. **Ship baseline:** V3 template fill (E20) and champion (E29) already clear fixture
   `--ship-gates`. Prioritize full `rico_held` + HF context before heavy critic MoE.
2. **Next needle on LTR-primary paths:** suffix / window rollback (**E30**) so
   permanence is not total when MaskGIT remask is not the active loop.
3. **Next needle beyond confidence remask:** model-specific trust training (**E31**)
   and visible-corruption aux (**E32**) so remasks target placeholder/slot errors the DFA and softmax confidence miss.
4. **Deterministic stack stays primary** for verifiable legality — Lark/DFA/force-emit/admit. Latent critics **select and interpret** repairs; they do not replace validators.
5. **Latent MoE falsification (E34)** is research-grade: only justified after E30–E33 show that confidence + trust remask policies are insufficient on residual semantic failures.

Bottom line for this codebase:

```text
MaskGIT / MDLM training
  + length-safe LTR / MaskGIT / template fill (V3)
  + confidence remask (V3) → critic-gated remask (V4)
```

is the defensible next experiment path. The novel long-run claim would be
**model-specific sparse falsification experts that perform recurrent continuous
critique and directly control revision inside a diffusion trajectory** — not
“MoE layers inside the backbone.”
