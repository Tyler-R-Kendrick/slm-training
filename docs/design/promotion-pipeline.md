# Promotion pipeline: mixture search, scaling ladders, self-distillation, trajectory RL

Status: **implemented** (P0–P3). P0 primitives remain the climbing substrate;
P1–P3 modules below are wired and covered by unit tests. Large-scale mixture /
ladder sweeps are CLI-driven and intentionally cheap in CI (`--dry-fit`,
`--limit-probes`).

| Stage | Where |
| --- | --- |
| P0 denoising-NLL / token budget / full-state resume / catalog / preference / traces | `evals/`, `train_loop.py`, `full_state.py`, `catalog.py`, `preference/`, `distill/trace_store.py` |
| P1a fuzzy + semantic dedup | `src/slm_training/data/dedup.py` (`--fuzzy-dedup`, `--semantic-cluster-cap`) |
| P1b mixture search + online sampling | `src/slm_training/data/mixture.py`, `scripts/run_mixture_search.py`, `ModelBuildConfig.mixture_manifest` |
| P1c scaling ladders + EG + promotion protocol | `src/slm_training/harnesses/experiments/`, `scripts/run_scaling_ladder.py` |
| P1d midtraining anchor | `register_promoted` → `promoted.pt` / `promoted.json` |
| P2 self-distillation | `distill/select.py`, `distill/sft.py`, `distill/repair.py`, `scripts/self_distill.py` |
| P3 trajectory RL (E64) | `rl/trajectory.py`, `scripts/resume_climb.py`, `--record-support` on collect |

The target loop:

```text
source catalog → fixed denoising-NLL suites → mixture search
  → model/token scaling ladders → base + midtraining
  → generated ship gates → trajectory store
  → self-distillation SFT reset → short trajectory-aligned RL → repeat
```

## Promotion protocol (applies to every stage below)

A candidate (data mixture, architecture, training change) is **promotable**
only when:

1. Data-integrity checks pass (existing leakage manifest + disjoint suites).
2. No hard loss category (`binding`, `structural`, `repair`) regresses beyond
   tolerance (default 2% relative) at any ladder point.
3. The aggregate weighted denoising NLL improves **and the improvement
   persists at the largest two ladder points** (rank stability, not just
   small-scale wins — mixture rankings can reverse with scale).
4. The lower confidence bound of `EG_Time` (3 seeds) is ≥ 1.
5. Finalists pass the existing generated ship gates
   (`ship_gates.DEFAULT_SHIP_GATES`) with **frozen decoder settings**
   (tokenizer, grammar backend, `gen_steps`, remask policy, `best_of_n`,
   seeds). Decoder sweeps are a separate system-level matrix; a model
   promotion must never be entangled with an inference-search change. The
   trace store's `decode_config_hash` is the enforcement handle.

Scratch-context and frozen-HF-context runs are separate tracks with separate
baselines and curves (`summary["track"]`); never pool them on one fit.

## P1a: fuzzy + semantic dedup (extend `catalog.py` → `data/dedup.py`)

Current dedup is exact-pair only (`fingerprint_pair`). Add two layers behind
`TrainDataConfig` flags, applied after the existing exact/structural checks:

1. **Fuzzy lexical** — character 4-gram MinHash (64 permutations) over
   `norm_text(prompt) + serialized program`; Jaccard ≥ 0.92 within the same
   `structure_cluster` collapses to the highest-priority copy. Pure Python,
   no new dependencies at fixture scale; revisit datasketch if RICO-full is
   slow.
2. **Semantic cluster caps** — cluster key = (`prompt_semantic_cluster`,
   `structure_cluster`, `binding_pattern_cluster`):
   * `structure_cluster` = existing `fingerprint_openui_structure` (already
     isomorphism-normalized).
   * `binding_pattern_cluster` = multiset of (component type → slot arity)
     pairs, namespace-erased.
   * `prompt_semantic_cluster` = content-word bag (stopword-stripped,
     template-prefix-stripped) hashed; upgrade to embeddings only if the bag
     proves too coarse on RICO.
   Cap representatives per cluster (default 8) with the same
   root-parent-first, sorted-id-order rule as `apply_parent_cap`.

Cross-source duplicates resolve by versioned priority:
`human_feedback > human_curated > rico_real > awwwards_real > synth families`
(never input order). Manifest gains `p50/p95 cluster exposure` next to the
existing parent-exposure stats.

**Memorization diagnostic:** per source family, the fraction of held-out
positions with NLL < 0.05 nats (`evals.denoising_nll` already produces
per-record means; add a per-family split keyed by `source_family`). A family
dominated by near-certain tokens is exhausted or templated regardless of its
average loss.

## P1b: mixture search (`data/mixture.py`, `scripts/run_mixture_search.py`)

A v2 mixture manifest separates task-group targets from within-task family
priors:

```json
{"mixture_id": "m03", "task_weights": {"generation": 0.2,
 "repair_completion_inpaint": 0.2, "patch_edit": 0.2,
 "state_behavior": 0.2, "noop_adversarial": 0.2},
 "weights": {"rico_real": 0.12, "programspec_generated": 0.12,
 "language_contract": 0.08, "corruption_repair": 0.06}}
```

Sampling is **online**: draw a task group, then a source family available for
that task, then a row. This makes task mass independent of raw row counts. All
draws use the loop RNG so full-state resume stays bit-exact; legacy v1
family-only manifests retain their old behavior. The manifest hash and both
weight maps go into `train_summary.json` and the full-state checkpoint.

Search hierarchy, adapted from RegMix but sized for this repo:

1. **Local probes**: vary every configured family with non-identity scales;
   bounded runs select one probe per family before second scales. One small model
   (`d_model=64`), one token budget, 1 seed, scored on weighted denoising NLL.
2. **Global probes**: vary organic vs synthetic vs feedback totals with local
   composition frozen.
3. **Fit + propose**: linear regression NLL ← weights over the probe set;
   propose 3–5 candidates.
4. **Ladder validation** (P1c) for the finalists; select only after the top
   candidate is rank-stable at the two largest points.

Per-source learning curves (NLL of a family's *own* validation slice over
training) classify sources: low-from-start = duplicated/trivial; steadily
decreasing = useful; high and flat = noisy or out of capacity; early
saturation = exposure-limited. Never use per-example low NLL as a keep
criterion (it selects duplicates).

## P1c: scaling ladders + efficiency gain (`experiments/ladder.py`, `scaling_fit.py`, `efficiency_gain.py`, `scripts/run_scaling_ladder.py`)

Two ladders, never pooled:

* **Scratch track**: `d_model ∈ {64, 96, 128, 192}` with proportional depths,
  constant target-tokens per trainable parameter (token budgets via
  `target_token_budget`, not steps), same tokenizer + frozen decoder.
* **Frozen-HF track**: scale only the denoiser (`d_model`, `denoiser_layers`)
  against the fixed SmolLM2 tower; report trainable vs frozen params (already
  in `summary["track"]`) and total system cost including frozen inference.

Each surviving candidate additionally runs token horizons `{0.5×, 1×, 2×}`
target budget to expose early-helping / fast-saturating sources — the local
analogue of the STEM-vs-code rank reversal.

Fit the baseline family with `L(C) = A·C^(−α) + E` (least squares on log
residuals; C = FLOPs or wall-clock seconds or NFE) and report

```text
EG_x = f_x^{-1}(L_candidate) / C_candidate   for x ∈ {FLOPs, Time, NFE}
```

optionally `EG_Verifier` including parser/`stream_check` call counts (the
trace store already counts NFE and repair rounds per generation). Reuse the
three-seed successive-halving runner (`scripts/run_grammar_matrix.py`)
as the execution engine: ladder points and horizons become new matrix
dimensions; halving on weighted NLL first, generated suites last.

## P1d: midtraining

After the base mixture is selected: a second `target_token_budget` phase on
complex layouts, long targets, schema variation, `visible_corrupt_rate > 0`,
and verifier-localized repair examples, retaining a 20–40% anchor slice of
base data. The mid-trained checkpoint becomes the **immutable anchor** for
P2 and is registered as `promoted.pt` alongside `best_weighted_nll.pt` /
`best_ship_score.pt` (the divergence between the last two is itself evidence).

## P2: self-distillation as a first-class stage (`distill/select.py`, `distill/sft.py`, `scripts/self_distill.py`)

Inputs come exclusively from the trace store. Three corpora stay separate
end-to-end (`pair_corpus` / trace `labels` already distinguish them):

* `self_distilled_success` — policy-generated, verifier-accepted final
  programs (`labels.accepted`, not `labels.exact_gold`-only).
* `self_distilled_repair` — intermediate failing canvases paired with the
  verifier-localized correction (failure-cone spans from E61).
* `gold_correction` — anything where the gold target was injected
  (`include_gold=True` collection). Never mixed silently into the first two.

**Selection (`distill/select.py`)** — coverage over score:

* Sample from several strong checkpoints (policy sha per trace), not only the
  final one.
* Stratify by prompt-intent cluster, structure cluster, binding pattern,
  target-length bin, empirical pass rate, and failure/repair type.
* Prefer one or a few traces from many prompts over many traces from one
  prompt; random sampling within strata (clever trace heuristics
  underperformed in MAI's ablations). Budget: thousands, not millions.

**SFT (`distill/sft.py`)** — start from the immutable mid-trained anchor:

```text
L = L_final_programs + λ_traj · L_next_denoising_action + λ_anchor · L_anchor_data
```

`L_next_denoising_action` teacher-forces the *recorded* intermediate canvases
(trace `steps[i].canvas` → commits at `steps[i].commits`), which is exactly
the trajectory-state loss the NLL suite should eventually adopt as a sixth
category. Mix 20–40% original base/mid data (`λ_anchor` ablation), and run a
small dropout ladder {0.0, 0.05, 0.10, 0.15} judged on *resumed-RL*
performance, not immediate SFT loss.

**Triggers** (promotion checkpoints, not per-update): RL slope flattened,
candidate diversity collapsed, reward hacking rising, numerical instability,
new base/mid checkpoint, tokenizer/grammar/context change, or consolidation
of several experiment branches.

**First required ablation** (identical rollouts):
(1) GRPO-lite direct update; (2) no update — harvest + self-distill;
(3) GRPO-lite then self-distill reset; (4) gold-correction SFT control.
This decides whether the current direct policy update adds value beyond
harvesting its outputs.

## P3: trajectory-aligned RL (E61–E65 + E64 core)

Order: E61 failure-cone remask → E62 minimal hard negatives → E63 gate
calibration → **E64** → E65 schema transfer → OPSD-lite last (student on its
own masked states, teacher = same weights + verifier-localized diagnosis,
KL only on the failure cone; only after trajectory logging is trusted).

E64 replaces the current GRPO-lite scoring (group-relative advantages applied
to a random-remask one-step score) with trajectory likelihoods:

* Rollouts come from `scripts/collect_trajectories.py` with
  `--samples-per-prompt N` and sample decode; every step already persists
  canvas, chosen ids, and rollout log-probs.
* The learner replays the *same grammar support* used at rollout (extend the
  recorder to persist `allowed_id_set` per commit when `record_support` is
  enabled) — otherwise learner and rollout probabilities live on different
  supports, the diffusion analogue of MAI's top-p mismatch.
* Loss per step: importance-weighted policy gradient over the recorded
  unmask/remask actions with the group-relative advantage of the final
  reward; clip per-token ratios; skip stale traces (policy sha ≠ current
  tranche's base sha — traces from different checkpoints are never mixed).

**Reward composition is lexicographic, not weighted:** (1) parser/type
validity gate → (2) slot-contract gate → (3) structural quality → (4) style →
(5) efficiency (`R_cost = ρ_q · NFE/NFE_max`, difficulty-scaled by empirical
pass rate ρ_q). An invalid program gets minimum reward regardless of style;
store the full reward vector in the trace even when the optimizer consumes a
rank.

**Group sizing:** start with 4 rollouts; expand to 8/16 only when the early
pass rate is in ~[0.1, 0.8]. All-fail groups route into the repair pipeline
(failure-cone examples) instead of being discarded.

**Policy management at this scale:** synchronous rollouts from one immutable
checkpoint per tranche, bounded update, versioned rollout buffer, rollback to
the previous promoted checkpoint on collapse. No async fleet, stale-rollout
ratios, or adaptive entropy control until rollout production and learning are
actually decoupled — and before adaptive entropy, first log entropy by
denoising step and token kind, candidate diversity per prompt, remask
frequency, legal-support size, and probability mass removed by grammar
constraints (the recorder carries the raw material).

## Module map (target layout)

```text
src/slm_training/data/dedup.py            # P1a fuzzy + semantic layers
src/slm_training/data/mixture.py          # P1b mixture manifests + online sampling
src/slm_training/harnesses/experiments/ladder.py    # P1c ladder definitions
src/slm_training/harnesses/experiments/scaling_fit.py
src/slm_training/harnesses/experiments/efficiency_gain.py
src/slm_training/harnesses/experiments/promotion.py # promotion-protocol evaluation
src/slm_training/harnesses/distill/select.py        # P2 stratified trace selection
src/slm_training/harnesses/distill/sft.py           # P2 anchor-mixed self-distillation
src/slm_training/harnesses/distill/repair.py        # P2/P3 failure-cone repair data
src/slm_training/harnesses/rl/trajectory.py         # P3 E64 objective

scripts/run_mixture_search.py
scripts/run_scaling_ladder.py
scripts/self_distill.py
scripts/resume_climb.py                   # tranche loop: rollouts → RL → gates
```
