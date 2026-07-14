# Speculative denoising (V7) — outcome-conditioned decode for TwoTower

> Status: **V7 shipped levers E70–E75** (see
> [quality-experiment-matrix.md](quality-experiment-matrix.md)). Paper tags in
> [research-lineage.md](research-lineage.md) §"Speculative denoising (V7)".

This document maps the "speculative speculative decoding for diffusion SLMs"
research program onto the OpenUI TwoTower model, records exactly which parts we
**Adapted** (built, in lite form) versus which stay **Adjacent** (documented,
deliberately deferred), and defines the V7 experiment family.

The transferable abstraction, in one line:

> **propose a denoising transition → verify the transition → precompute from
> likely verified successor states.**

For an AR model the state is a token prefix and the speculative outcome is
"accept *j* tokens, emit recovery token *z*". For our masked-diffusion decode
the state is `(canvas, unknown-mask, step)` and a verification result decides
which positions are accepted, remasked, or deferred. V7 makes that outcome
space **prefix-like again** by imposing a temporary order over dependency
clusters, then reuses the AR speculation machinery (survival scheduling,
outcome fanout, successor caching) at cluster granularity.

---

## 1. AR → TwoTower mapping

Specialization of the generic AR ↔ diffusion table to this repo:

| AR speculative concept | Diffusion-SLM equivalent | TwoTower realization |
| --- | --- | --- |
| Generated prefix | Partial canvas + mask state | `ids` / `unknown` in `_generate_maskgit_one` |
| Draft token block | Candidate denoising transition | One MaskGIT step's proposed commits |
| Draft length | #positions / span / time jump | `select_unmask_indices` budget per step |
| Accepted prefix | Accepted token/span clusters | Clusters admitted by the grammar acceptor |
| First rejection | First rejected cluster under a temporary order | `verify_clusters_ordered` → outcome `(j, repair)` |
| Recovery / bonus token | Repaired or remasked cluster | Rejected cluster remasked to `<mask>` (T2M) |
| Prefix-survival probability | Trajectory-survival probability | `survival_gate` head (E73) |
| Markov draft head | Micro-causal / graph-local consistency | Grammar acceptor + attention clusters (we do not train a causal corrector; see §6) |
| Speculation cache | Successor-state cache | `SuccessorCache` batched next-pass logits (E74) |
| Target verification | Grammar / stronger verification | Lark incremental acceptor + `admit_fill` + `validate` |
| KV reuse | Stable-prefix / hidden-state reuse | Context tower KV is already cached (`cache_context`); denoiser is bidirectional → full recompute on miss |
| SSD fanout | Fanout over commit/remask patterns | `speculative_fanout` K ∈ {1, 2, 4} outcomes |

Two AR assumptions that do **not** transfer, and how V7 handles them:

1. **Diffusion outcomes are arbitrary subsets, not prefixes.** V7 imposes a
   temporary verification order over attention-derived clusters (anchors
   first), so an outcome collapses to `(j, repair)`: first `j` clusters
   accepted, cluster `j+1` remasked. Fanout over `2^m` subsets never happens.
2. **Bidirectional hidden states invalidate on any canvas change.** We do not
   pretend to cache denoiser activations across canvas edits. Speculation is at
   the **logits-of-a-concrete-successor-canvas** level: we materialize the K
   most likely next canvases and run one batched forward, which is cheap at
   this model size (d_model ≤ 192, ≤ 6 layers).

## 2. Why this maps well here

- **The verifier is a genuinely separate resource.** Target "verification" in
  this repo is the CPU grammar stack (Lark incremental acceptor, `admit_fill`
  hole probe, lang-core `validate`), not another neural forward. Overlapping
  or amortizing it against denoiser compute is the SSD economics without
  needing a second model.
- **The trunk is tiny.** A K-way batched denoiser forward costs roughly one
  forward. That makes "predict the next state for likely outcomes" nearly free
  where it would be prohibitive on a 7B dLLM — the SLM lesson from the
  proposal ("speculate states, not whole branches") holds, and here even full
  next-pass logits are affordable.
- **We already have the draft–verify loop.** MaskGIT proposes all unresolved
  positions each pass; grammar stream-check rejects some; remask policies
  (confidence / CoRe / combined) revise commitments. V7 restructures that loop
  around clusters + survival + successor caching rather than inventing it.

## 3. The five V7 components

### 3.1 Mutual-stability signals (E70 — LESS-lite, training-free)

Ordinary confidence decoding trusts one pass's top-1 probability. LESS shows
that **top-1 persistence across passes** and **inter-step distributional
stability (Jensen–Shannon divergence)** are stronger, training-free signals.

`StabilityTracker` in
[`models/parallel_decode.py`](../../src/slm_training/models/parallel_decode.py)
records, per position, across MaskGIT steps:

- `persistence[t]` — consecutive steps the argmax stayed the same;
- `jsd[t]` — JS divergence between the current and previous step's
  distribution at `t` (0 when no history yet).

Uses:

- **Remask** (`remask_policy=stability`): rank committed tokens by
  `score = jsd_weight * jsd − persistence` (high score → remask). Composes
  with the existing grammar-hard-error / gate / entropy budget via
  `select_remask_stability_indices`.
- **Commit gating** (`stability_min_persistence > 0`): a masked position may
  only be committed once its argmax has persisted for N model passes
  (positions seen fewer times than N are exempt so decode always progresses).

### 3.2 Attention dependency clusters (E71 — DAPD/DAWN-lite)

Committing correlated token marginals independently is the classic parallel
decoding failure. DAPD/DAWN/CLAD use attention-derived dependency structure to
avoid committing mutually incompatible positions together and to pick
**anchors** whose commitment de-risks their dependents.

`DenoiserTower.forward(..., return_attn=True)` exposes the last layer's
self-attention map (explicit softmax path, only when requested; the SDPA fast
path is untouched). `build_dependency_clusters` in
[`models/speculative_denoise.py`](../../src/slm_training/models/speculative_denoise.py)
then greedily groups candidate commit positions whose symmetric attention
coupling exceeds `cluster_attn_threshold` (capped at `cluster_max_size`).

Cluster ordering uses the anchor score from the proposal:

```
a(C) = mean_survival(C) * (1 + coupling_centrality(C))
```

where survival falls back to confidence when no survival head is loaded, and
centrality is the summed attention mass from *other* candidate positions into
the cluster. High-anchor clusters commit first; ambiguous, strongly-coupled
clusters commit last (or wait for the next pass).

This subsumes the old fixed `min_spacing` heuristic ("mean-field-lite") with
measured coupling.

### 3.3 Temporary verification order (E72)

With `m` proposed clusters there are `2^m` accept/remask patterns — the
combinatorial blocker for outcome-conditioned speculation. V7 verifies
clusters **in the anchor order**: commit cluster 1, stream-check/admit it,
then cluster 2, … The first rejected cluster ends the transaction:

```
outcome o = (j, repair)   # clusters 1..j accepted; cluster j+1 remasked
```

Later clusters stay masked for the next pass (their logits were conditioned on
a canvas that no longer exists). Rejected clusters are always remasked to
`<mask>` — the T2M / remask-don't-replace discipline from V6 is preserved.

The verifier is the existing grammar stack at cluster granularity:
`admit_fill` (canvas completability) plus `filter_ids_by_stream` (hard
stream errors). This is the "any-order autoregressive micro-process on top of
a parallel trunk" from the proposal, with the grammar playing the causal
verifier role.

### 3.4 Trajectory-survival head (E73 — DSpark-lite)

DSpark schedules speculation with a calibrated head predicting whether a
drafted token **survives target verification**, rather than raw confidence.
The diffusion analogue: will this committed token remain unchanged through the
rest of the denoising trajectory (and match the final answer)?

[`grammar_fastpath/survival_train.py`](../../src/slm_training/dsl/grammar/fastpath/survival_train.py)
mirrors the BackPlay-lite trust-gate recipe (`trust_train.py`): freeze the
denoiser, run partial-canvas forwards on train records, label each visible
position by whether the model's current commitment agrees with gold (a cheap,
well-defined surrogate for "survives the teacher trajectory"), and train a
separate `survival_gate` head (same `FastPathGate` architecture) with BCE.

Uses at decode time (`survival_gate=True`):

- Cluster anchor scores use mean survival instead of confidence.
- The per-step commit budget stops at the cluster where the **cumulative
  product of cluster survival** drops below `survival_commit_threshold` —
  the DSpark cumulative-acceptance schedule, at cluster granularity. The
  joint cluster survival is intentionally *not* modeled as an independent
  product over long spans: clusters are small (≤ `cluster_max_size`) and the
  cross-cluster ordering handles the rest.

### 3.5 Outcome-conditioned successor cache (E74 — Saguaro-SSD-lite)

While the current transition is being verified, prepare the next state for
the most likely verifier outcomes:

1. `enumerate_outcomes` ranks outcomes by cluster survival:
   `accept-all`, `reject-weakest-cluster`, `reject-two-weakest`, … up to
   `speculative_fanout` K.
2. For each outcome, materialize the concrete successor canvas (accepted
   clusters committed, rejected clusters remasked).
3. Run **one batched denoiser forward** over the K canvases → `SuccessorCache`.
4. After verification resolves, look up the actual outcome's canvas. Hit →
   next step's logits come from the cache (no new forward). Miss → normal
   forward (fallback).

Ideal latency moves from `T_verify + T_next_state` toward
`max(T_verify, T_speculate) + P_miss · T_fallback`. On CPU with a tiny trunk
the win is mostly **amortization** (batched forward ≈ one forward) rather
than parallel overlap; `speculative_overlap` (threaded verify) exists but
defaults off for determinism. Telemetry counts `denoiser_forwards`,
`successor_hits`, `successor_misses` per generate call so the hit rate and
net forward savings are measurable, not asserted.

Speculation **auto-abstains** when the active remask policy needs extra model
forwards (trust gate / CoRe perturbation): those remasks cannot be predicted
without paying the remask cost, so the cache would miss. Measured on V7:
E74 (deterministic remask) hit rate 1.0; E75 (trust-gate remask) skips
speculation and pays only the remask-forward cost.

## 4. Decode loop (V7 flags on)

```python
while unknown.any():
    logits, attn = denoiser(ids, ctx, return_attn=True)      # 1 forward (or cache hit)
    stability.update(probs)                                   # E70
    clusters = build_dependency_clusters(attn, candidates)    # E71
    order = order_clusters(clusters, survival or confidence)  # E71/E73
    successors = speculate_successors(order, K)               # E74 (batched)
    outcome = verify_clusters_ordered(order)                  # E72 (grammar)
    apply(outcome)                                            # commit 1..j, remask j+1
    logits = successors.get(outcome) or None                  # E74 hit/miss
```

Every lever is opt-in; with all V7 knobs at defaults the decode path is
byte-identical to V6.

## 5. Config surface

| Knob | Default | Lever |
| --- | --- | --- |
| `remask_policy="stability"` | `confidence` | E70 remask ranking |
| `stability_min_persistence` | `0` (off) | E70 commit gating |
| `stability_jsd_weight` | `1.0` | E70 score mix |
| `unmask_mode="cluster"` | `positions` | E71 cluster commits |
| `cluster_attn_threshold` | `0.08` | E71 coupling cut |
| `cluster_max_size` | `4` | E71 cluster cap |
| `cluster_verify` | `False` | E72 ordered verification |
| `survival_gate` | `False` | E73 decode-time survival |
| `survival_gate_train` | `False` | E73 head training stage |
| `survival_commit_threshold` | `0.3` | E73 cumulative budget |
| `speculative_successor` | `False` | E74 successor cache |
| `speculative_fanout` | `2` | E74 K outcomes |
| `speculative_overlap` | `False` | E74 threaded verify |

## 6. Deferred (Adjacent) — and why

These parts of the proposal are documented but deliberately **not** built:

| Idea | Source | Why deferred here |
| --- | --- | --- |
| Multi-horizon transition heads `q^(Δ)`, Δ∈{1,2,4,8} | T3D / CD4LM | Needs teacher-trajectory distillation infra; our trajectories are 8–16 steps on a fixture-scale corpus, so horizon heads would train on almost no signal. Revisit after longer trains on real GPUs. |
| Hidden-state adapters `G_ψ` (feature + logit losses, resume at layer d+1) | SSD §SLM | With ≤ 6 denoiser layers and d_model ≤ 192, a full batched forward costs about as much as an adapter + partial forward. The cache in E74 stores **full next-pass logits** instead — same benefit, no drift risk, no new training stage. |
| Micro-causal cluster corrector (permutation-causal block / low-rank Markov head) | DSpark / Self-Spec MD | Our grammar acceptor already provides exact sequential dependency repair for the structural channel, which dominates OpenUI errors. A learned corrector would target intra-cluster *content* dependencies; worth revisiting if E71/E72 plateau with content errors. |
| Unordered graphical-model outcomes (`log q(o) = Σ u_C + Σ v_CD`) | proposal §4-alt | Ordered clusters make outcomes prefix-like with zero calibration burden; the graph variant is strictly harder to make fast and distribution-preserving. Start ordered (as the proposal itself recommends). |
| AR-mode / stronger-diffusion self-verification | S2D2 / SimSD | TwoTower has an LTR mode, but it is a *repair* path, not a calibrated critic; the grammar is a stronger and cheaper verifier for this DSL. |
| Arbitrary-configuration training support | Adaptive Block Diffusion | Partially covered: `mask_pattern=mixed`, `visible_corrupt_rate`, template-fill seeds already diversify canvas configurations. Full off-grid configuration training is future work if pre-speculation states show distribution shift. |
| Latent plan slots / coarse-to-fine block control | CCDD / BACD | OpenUI programs are short, structured, and grammar-anchored; there is no long-form reasoning phase to plan latently. Not applicable at this task scale. |
| Prefix-cacheability restructuring (causal/topological reorder) | WeDLM | The denoiser is bidirectional by design and sequences are ≤ 256 tokens; prefix KV restructuring is a deployment-scale concern, not a fixture-scale one. Context-tower KV is already cached. |

## 7. V7 experiment family

| ID | Lever | Measures |
| --- | --- | --- |
| E70 | Stability remask + persistence commit gate | parse/fidelity vs E50; remask precision |
| E71 | Attention clusters (`unmask_mode=cluster`) | parse/fidelity; commits per forward |
| E72 | Ordered cluster verification | false-commit rate; grammar remask volume |
| E73 | Survival head + cumulative commit budget | calibration; forwards per generate |
| E74 | Successor cache K=2 | hit rate; forwards saved; wall-clock |
| E75 | Champion: E53 stack + E70–E74 | honest ship gates |

Success metrics beyond parse/fidelity (all recorded in telemetry):

- `denoiser_forwards` per generate (function evaluations),
- `successor_hit_rate`,
- remask volume and false-commit rate (grammar rejections after commit),
- wall-clock per generate.

Commands and measured results live in
[quality-experiment-matrix.md](quality-experiment-matrix.md) §V7.
