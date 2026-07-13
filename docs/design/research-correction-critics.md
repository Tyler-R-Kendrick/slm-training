# Correction, remasking, and latent critics

Inspiration and candidate research for improving TwoTower OpenUI quality via
**revision** (undoing premature commitments) rather than only better first-pass
decode. Nothing here is implemented yet — tags in
[research-lineage.md](research-lineage.md) are **Adjacent**. Concrete levers
are **E18–E22** in [quality-experiment-matrix.md](quality-experiment-matrix.md).

---

## Why this matters for us

Current setup (see [research-lineage.md](research-lineage.md)):

| Layer | Today | Gap |
| --- | --- | --- |
| Training | MaskGIT-style random masking ([`_mask_targets`](../../src/slm_training/models/twotower.py)) | Visible tokens are always clean — the denoiser never learns to revise a wrong visible token |
| Decode order | LTR primary + MaskGIT fallback | Once an LTR token commits, it is effectively permanent |
| Remasking | Rule-based grammar reject ([`filter_ids_by_stream`](../../src/slm_training/models/grammar.py)) | No semantic / fidelity remask policy |
| Trust head | [`FastPathGate`](../../src/slm_training/grammar_fastpath/gate.py) defined, **unwired** | No learned token-correctness signal |

Ship gates still fail primarily on **fidelity ≈ 0** and **held_out / adversarial parse**
(see experiment matrix). Grammar/DFA verification already catches illegal OpenUI.
Learned correction targets failures the grammar cannot see: wrong but legal trees,
placeholder namespace drift, slot-contract violations, brittle structure that
passes stream checks.

---

## Core decomposition

Keep three concerns separate:

```text
decode ordering          +  revision transition  +  revision policy
(LTR / MaskGIT / block)     (ReMDM / GIDD)           (confidence / latent critic MoE)
```

| Concern | Role | Maps into this repo |
| --- | --- | --- |
| **Decode ordering** | Which positions open first | `parallel_decode.select_unmask_indices`, `_greedy_ltr_decode_batch`, MaskGIT path in `twotower.py` |
| **Revision transition** | How a committed token becomes uncertain again (`token → [MASK]` or learned edit) | Today: only grammar remask. Candidate: ReMDM-style sampler; later GIDD/SCDD training |
| **Revision policy** | *Whether* and *where* to revise | Today: binary stream/DFA reject. Candidate: BackPlay-lite gate → RemeDi-like UPS → latent falsification MoE |

**Rule:** critics should primarily replace/augment the **revision policy**, not
replace ReMDM/GIDD or the existing LTR/MaskGIT construction order. Prefer
**remask, don't replace**: reset suspect tokens to `[MASK]` and regenerate from
cleaned context rather than overwrite while neighbors may still be wrong
([arXiv:2604.18738](https://arxiv.org/abs/2604.18738)).

---

## Paper inventory

| Paper | arXiv | Core idea | Relevance here |
| --- | --- | --- | --- |
| **Coconut** | [2412.06769](https://arxiv.org/abs/2412.06769) | Recurrent continuous latent reasoning (thoughts fed back through the decoder) | Inspiration for multi-step *latent* critique without emitting critique text. Serial full-backbone rollouts are likely too expensive for our decode loop — prefer small recurrent modules / parallel slots |
| **BackPlay** | [2601.06428](https://arxiv.org/html/2601.06428v2) | Plug-in correction head on a **frozen** DLM, trained on that model’s own error distribution; drives remasking | Strongest near-term template. Maps to wiring `FastPathGate` (E19) on frozen TwoTower weights |
| **RemeDi** | [2509.23653](https://arxiv.org/html/2509.23653v1) | Separate unmasking-policy stream (UPS): learn token quality; remask via SFT/RL; train on masked *and* randomly replaced visible tokens | Semantic upgrade path from BackPlay-lite: cheap local confidence always-on; heavier critic gated. Training signal informs E20 |
| **ReMDM** | [2503.00307](https://arxiv.org/abs/2503.00307) | Inference-time remasking sampler for pretrained masked diffusion — send revealed tokens back to `[MASK]` without retraining | Immediate transport layer for critique on existing MaskGIT weights (E18) |
| **GIDD** | [2503.04482](https://arxiv.org/abs/2503.04482) | Hybrid absorbing masks + uniform token noise so the denoiser sees wrong *visible* tokens and learns revision transitions | Future foundation objective; wrong visible-token supervision. Do **not** replace MaskGIT early (E20 is a lite subset) |
| **SCDD** | [2603.02230](https://arxiv.org/html/2603.02230v1) | Explicit discrete self-correction transitions (alternative to GIDD’s hard-to-tune interpolation); evidence at GPT-2 scale | Adjacent alternative if GIDD-style pretraining is revisited |
| **SPC** | [2504.19162](https://arxiv.org/html/2504.19162v1) | Self-play: sneaky generator produces hard reasoning errors; critic detects them | Curriculum for specialized critics (adversarial corruptions per failure family) |
| **Sparse MoE reward / PRISM-style** | [2606.04284](https://arxiv.org/abs/2606.04284) | Sparse MoE critics/discriminators with interpretable specialization under sparsity + diversity objectives | Evidence that expert specialization is *induced*, not automatic — needed if we build E22 |
| **LLaDA-MoE** | [2509.24389](https://arxiv.org/abs/2509.24389) | Sparse MoE inside a diffusion LM (generator experts) | Shows MoE + discrete diffusion is viable; not a critic MoE — do not confuse generator experts with falsification experts |
| **Counterfactual MoE routing** | [2605.07260](https://arxiv.org/abs/2605.07260) | Standard routers are least informative on fragile reasoning tokens; router-only updates can help | Motivation to train routers on **corrective utility**, not token perplexity alone; avoid brittle top-1 token routers |
| **MIRAGE (parallel latent slots)** | [2606.04627](https://arxiv.org/html/2606.04627v2) | Multiple latent slots updated in synchronous rounds instead of one serial Coconut step per slot | Preferred execution for latent critics: 4–16 slots, 2–4 refinement rounds, no repeated full denoiser backbone |
| **Deferred commitment / sliding windows** | [2601.02076](https://arxiv.org/abs/2601.02076) | Confidence-aware sliding windows vs fixed block boundaries for DLMs | Supports blockwise LTR + revisable window (E18) over strict irreversible LTR |
| **Token ordering in masked diffusions** | [2502.06768](https://arxiv.org/html/2502.06768v1) | Masked training teaches fill-in given visible context; does not teach repair of wrong visible tokens | Explains our LTR-permanence / no-self-correction gap |
| **Remask, don’t replace** | [2604.18738](https://arxiv.org/abs/2604.18738) | Prefer token→mask→token refinement over direct substitution while context is still wrong | Remask policy design constraint for E18/E21 |

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
    ├── parallel latent slots + few recurrent refinement rounds
    ├── mechanism-specific falsification
    └── risk / remask / revision conditioning
    │
    ▼
Critique controller → ReMDM remask budget + continue/stop
```

The **shared head matters**: a purely sparse router that must already recognize a
blind spot to pick the expert that recognizes that blind spot fails before
criticism begins.

### Expert taxonomy (OpenUI-specific)

Specialize by **failure mechanism**, not broad domain (“UI expert”):

1. **Grammar / structure violation** — illegal nesting, unfinished constructs (mostly owned by Lark/DFA today; critic decides *when* to trust grammar vs soft risk).
2. **Placeholder / namespace consistency** — `:acme.*` drift, invalid slot names (fidelity failure family).
3. **Slot-contract compliance** — inventory/schema vs emitted tree (ties to E12).
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

Prefer **parallel slot refinement** (small fixed rounds) over serial Coconut
decode through the full TwoTower backbone each thought step.

---

## ReMDM vs GIDD vs RemeDi

| Property | ReMDM | RemeDi | GIDD / SCDD |
| --- | --- | --- | --- |
| Works with existing MaskGIT weights | Yes | Partial (needs UPS training) | No (needs training-process change) |
| Requires training changes | No | Yes (UPS + optional visible corruption) | Yes (hybrid / self-correction transitions) |
| Revision mechanism | Token → mask → token at inference | Learned remask policy + regenerate | Learned edit / hybrid noise |
| Best use here | Prototype critic-guided rollback (E18) | Cheap trust + remask head (E19/E21) | Later foundation when we own pretraining again |
| Integration risk | Low | Medium | High |

**Progression we recommend:**

1. Keep MaskGIT training.
2. Add ReMDM-style suffix / window rollback on existing checkpoints (**E18**).
3. Wire a BackPlay-lite / RemeDi-lite trust head on frozen model errors (**E19**).
4. Add light visible-token corruption supervision (**E20**) and a combined remask budget (**E21**).
5. Only then compare GIDD/SCDD-style objectives or latent MoE critics (**E22**).

### Recommended decoder shape (target)

```text
1. LTR (or blockwise LTR) within a sliding active window
2. Revisable suffix / block behind the frontier
3. Cheap token-quality head every step (FastPathGate / RemeDi-lite)
4. Heavier critics only at block boundaries, uncertainty spikes,
   projection disagreement, or grammar/tool failure
5. Combine critic + entropy + grammar into a ReMDM remask distribution
6. Remask implicated spans (+ dependency neighborhood); re-denoise
7. Commit when expected repair value < compute threshold
```

Variants if needed later: critic-triggered global rollback (expensive cache
invalidations); blockwise LTR with bidirectional remask inside the active block.

---

## What moves the needle *now*

Honest priority given [quality-experiment-matrix.md](quality-experiment-matrix.md)
results:

1. **Still the biggest lever:** more train steps / capacity / slot contract (E12, E15, E16) and HF context when available. Correction machinery does not fix underfit fidelity-0 memorizer behavior.
2. **Next needle for parse/held_out:** LTR permanence. E18 suffix-rollback + grammar/entropy remask can undo locally bad commitments without new weights.
3. **Next needle for fidelity:** model-specific error training (E19/E20) so remasks target placeholder and slot errors the DFA cannot see.
4. **Deterministic stack stays primary** for verifiable legality — Lark/DFA/force-emit/admit. Latent critics **select and interpret** repairs; they do not replace validators.
5. **Latent MoE falsification (E22)** is research-grade: only justified after E18–E21 show that a single shared remask policy is insufficient on residual semantic failures.

Bottom line for this codebase:

```text
MaskGIT training
  + LTR / blockwise decode
  + critic-gated ReMDM-style rollback
```

is the defensible first experiment. The novel long-run claim would be
**model-specific sparse falsification experts that perform recurrent continuous
critique and directly control revision inside a diffusion trajectory** — not
“MoE layers inside the backbone.”
