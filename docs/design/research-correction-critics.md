# Correction, remasking, and latent critics

Inspiration and research for improving TwoTower OpenUI quality via
**revision** (undoing premature commitments) and **semantic critique**, beyond
the confidence remask wired in V3. Concrete levers **E30–E36** live in
[quality-experiment-matrix.md](quality-experiment-matrix.md); implementation
status and fidelity tags are in [research-lineage.md](research-lineage.md).

V3 **Adapted** confidence remasking (`remask_ratio` / `select_remask_indices`)
and template fill; those cleared fixture `--ship-gates` but used a silent
`gold.placeholders` channel. V4 adds revision policy + honest inventory.

---

## Status (what is wired)

| Layer | Today | Notes |
| --- | --- | --- |
| Training | MaskGIT / optional MDLM + **E32** `visible_corrupt_rate` | Wrong-visible recovery is learnable |
| Decode order | Length-safe LTR + MaskGIT + template fill + **E30** suffix rollback | LTR-primary can remask a revisable window |
| Remasking | V3 confidence remask + **E33** grammar/gate/entropy policy | `select_remask_policy_indices` |
| Trust head | **E31** `FastPathGate` via `grammar_fastpath/trust_train.py` | Freezes denoiser; BCE on own errors |
| Slot inventory | **E35** inventory-in-prompt (`honest_slot_contract`) | No silent gold channel; clears ship gates |
| Latent MoE | **E34** deferred | Runner skips unless `--force-e34` |

Grammar/DFA verification still catches illegal OpenUI. Learned correction
targets failures the grammar cannot see: wrong but legal trees, placeholder
namespace drift, slot-contract violations.

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
| **Revision transition** | How a committed token becomes uncertain again | V3 confidence remask; V4 E30 suffix remask; E32 trains revise-visible |
| **Revision policy** | *Whether* and *where* to revise | E31 gate + E33 budgeted remask; E34 reserved for residual semantics |

**Rule:** critics primarily replace/augment the **revision policy**, not
replace ReMDM/GIDD or the existing LTR/MaskGIT construction order. Prefer
**remask, don't replace** ([arXiv:2604.18738](https://arxiv.org/abs/2604.18738)).

---

## Paper inventory

| Paper | arXiv | Core idea | Status here |
| --- | --- | --- | --- |
| **Coconut** | [2412.06769](https://arxiv.org/abs/2412.06769) | Recurrent continuous latent reasoning | Adjacent — E34 only |
| **BackPlay** | [2601.06428](https://arxiv.org/html/2601.06428v2) | Plug-in correction head on frozen DLM | **Adapted** — E31 |
| **RemeDi** | [2509.23653](https://arxiv.org/html/2509.23653v1) | Learned token quality / remask UPS | **Adapted** (lite) — E31/E33 |
| **ReMDM** | [2503.00307](https://arxiv.org/abs/2503.00307) | Inference remask without retrain | **Adapted** — V3 remask + E30/E33 |
| **GIDD** | [2503.04482](https://arxiv.org/abs/2503.04482) | Wrong visible tokens in training | **Adapted** (lite) — E32 |
| **SCDD** | [2603.02230](https://arxiv.org/html/2603.02230v1) | Explicit self-correction transitions | Adjacent alternative to full GIDD |
| **SPC** | [2504.19162](https://arxiv.org/html/2504.19162v1) | Self-play hard errors for critics | Adjacent — E34 curricula |
| **Sparse MoE reward / PRISM-style** | [2606.04284](https://arxiv.org/abs/2606.04284) | Specialist critics under sparsity | Adjacent — E34 |
| **LLaDA-MoE** | [2509.24389](https://arxiv.org/abs/2509.24389) | MoE inside a diffusion LM | Adjacent — generator MoE ≠ critic MoE |
| **Counterfactual MoE routing** | [2605.07260](https://arxiv.org/abs/2605.07260) | Route on repair utility | Adjacent — E34 routing principle |
| **PLR** | [2601.03153](https://arxiv.org/abs/2601.03153) | Parallel latent streams | Adjacent — preferred E34 execution |
| **MIRAGE** | [2606.04627](https://arxiv.org/abs/2606.04627) | Continuous latent CoT for agents | Adjacent |
| **Deferred commitment** | [2601.02076](https://arxiv.org/abs/2601.02076) | Confidence-aware sliding windows | **Adapted** (lite) — E30 |
| **Token ordering** | [2502.06768](https://arxiv.org/html/2502.06768v1) | Visible ≠ revisable without noise | Motivates E32 |
| **Remask, don’t replace** | [2604.18738](https://arxiv.org/abs/2604.18738) | Prefer token→mask→token | E33 design constraint |

---

## Progression (executed)

1. Keep MaskGIT / MDLM training (V3).
2. Keep V3 confidence remask; add LTR suffix / window rollback (**E30**).
3. Wire BackPlay-lite / RemeDi-lite trust head (**E31**).
4. Add light visible-token corruption (**E32**) and combined remask budget (**E33**).
5. Close the eval leakage hole with honest inventory-in-prompt (**E35**) + decode scaling (**E36**).
6. Only then revisit GIDD/SCDD-scale objectives or latent MoE critics (**E34**).

### Recommended decoder shape (now)

```text
1. LTR (or blockwise LTR) within a sliding active window
2. Revisable suffix / block behind the frontier (E30)
3. Cheap token-quality head every MaskGIT step (E31 FastPathGate)
4. Combine critic + entropy + grammar into a remask distribution (E33)
5. Remask implicated spans; re-denoise (remask, don't replace)
6. Template fill seeded from prompt-visible inventory (E35)
7. Optional best-of-N ranking (E36)
```

---

## What moves the needle *now*

1. **Ship baseline:** E35 / E36 clear fixture `--ship-gates` **without** silent
   gold placeholder leakage. Prioritize full `rico_held` + HF context next.
2. **LTR-primary paths:** E30 suffix rollback is wired; needs longer train or an
   E35 seed to show gate deltas alone.
3. **Trust remask:** E31/E33 lift adversarial parse but need E35 inventory for
   fidelity.
4. **Deterministic stack stays primary** for verifiable legality.
5. **E34** remains research-grade until residual semantic failures after E33+E35.

Bottom line for this codebase:

```text
MaskGIT / MDLM training
  + length-safe LTR / MaskGIT / template fill (V3)
  + confidence remask (V3) → critic-gated remask (V4)
  + honest inventory-in-prompt (E35)
```
