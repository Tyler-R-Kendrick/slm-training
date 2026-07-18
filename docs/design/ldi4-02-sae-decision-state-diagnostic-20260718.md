# LDI4-02 — SAE decision-state diagnostic harness (SLM-136)

**Date:** 2026-07-18. **Status:** torch-free capture/config/arm schema + SAE module
(reconstruction + L1 sparsity, dead-feature policy, metrics, fail-closed artifacts) +
causal-intervention primitives + the matched S0–S7 fixture matrix — implemented and
tested. **Fixture/wiring evidence only.** Real activation capture from a checkpoint, the
site/position hook integration into `twotower.py`, arm training, and multi-suite/AgentV
evaluation are deferred to the GPU run; no steering, interpretability, or superiority
claim is made here, and no SAE feature is promoted from reconstruction/correlation.

Unblocked by the merged SLM-129 (structural-slop profiles) and SLM-134 (ReFT/DiffMean
matched baselines), both on `main`.

## Objective

A sparse-autoencoder track for **decision-state diagnosis and causal feature testing** —
not a semantic oracle, not a default actuator. It must answer whether sparse features add
causal selectivity/efficiency *over* the direct baselines (DiffMean, linear probe, ReFT,
weight-space adapter), using the same DecisionEventV2 evidence, prompt-group splits,
protected metrics, and compute accounting. It is eligible only after the
structural-forensics and ReFT/direct baselines exist (they do).

## What this delivers

- `src/slm_training/harnesses/representations/spec.py` — **torch-free** schema:
  - `CaptureRow` / `CaptureManifest` — the activation-capture contract keyed to exact V2
    states (site/position/checkpoint/tokenizer/decode/verifier identity, dtype/shape,
    content hash, target/preservation/unlabeled role). Train and held groups must stay
    disjoint; a changed identity invalidates the cached activation; the tensor is stored
    out of band (content-addressed), never in Git.
  - `SAEConfig` — the declared architecture + sparsity objective (fail-closed,
    fingerprinted).
  - `matched_sae_arms` — the **S0–S7** matched matrix (parent, random, DiffMean, linear
    probe, ReFT-r1, direct adapter, top SAE feature, sparse SAE set); every supervised
    steering arm is `train_only`.
  - `select_features_train_only` — fails closed if held-out scores would drive feature /
    sign / threshold / dose selection.
- `src/slm_training/harnesses/representations/sae.py` — the transparent baseline SAE
  (`z = act(encoder(h − bias_dec)); h_hat = decoder(z) + bias_dec; loss = recon + λ·L1`),
  ReLU/JumpReLU codes, unit-norm decoder, dead-feature accounting, `sae_metrics`
  (reconstruction MSE, explained variance, cosine, L0/L1, dead / ultra-dense rates), and
  fail-closed `save_sae`/`load_sae` (`torch.save` + JSON manifest, width/kind/version and
  config-fingerprint checks — a mismatched site or activation width cannot silently load).
- `src/slm_training/harnesses/representations/interventions.py` — bounded interventions
  (ablate a feature, dose a decoder/probe direction, a **wrong-site negative control**),
  the DiffMean/linear-probe direction helpers, honest `classify_arm` (`diagnostic_only`
  / `causal_but_inferior` / `competitive` / `rejected`), and `run_fixture_matrix` over
  synthetic activations.

## Honesty invariants (tested)

- Feature / sign / threshold / dose selection uses **train groups only**; supplying
  held-out scores for selection raises.
- A feature is `competitive` only when it is **localized** (wrong-site control null),
  within the **preservation** budget, and **not** beaten by the direct baselines. In the
  committed fixture the SAE arms come back `diagnostic_only`/`rejected` — the harness
  proves wiring and makes **no** SAE superiority claim.
- A low reconstruction loss is reported as a diagnostic, never as interpretability or
  steering evidence. No SAE acts as a verifier or training-label source.

## Verification

```bash
python -m pytest tests/test_harnesses/representations -q                 # 17 passed
python -m ruff check src/slm_training/harnesses/representations tests/test_harnesses/representations
python -m scripts.repo_policy
```

Tests cover: capture-row split agreement + role validation; manifest train/held
disjointness + width consistency + stable identity fingerprint; SAE config fail-closed +
derived width + fingerprint; SAE forward shapes + nonnegative codes; loss finite and
reconstruction learns; metric keys/ranges; dead-feature mask; JumpReLU sparser than
ReLU; save/load round-trip + width/kind fail-closed; DiffMean/probe directions; ablation
changes the reconstruction; the S0–S7 fixture matrix (parent null, DiffMean moves the
target, honest per-arm classification); and `classify_arm` rejecting non-localized or
preservation-damaging arms. Committed fixture:
`docs/design/ldi4-02-sae-decision-state-diagnostic-fixture-20260718.json`.

## Scope / deferred to the GPU run

The real activation capture from a pinned checkpoint (the `DenoiserTower.encode` residual
seam / per-block `register_forward_hook`), training the S0–S7 arms under the shared local
objective with held-out guards, the group/seed replication, and the multi-suite/AgentV
evaluation are the GPU run — which then publishes a quality-bearing memo with matched
S0–S7 results, target/preservation dose curves, seed/retrain stability, and an explicit
classification (diagnostic only, causal but inferior, competitive, or rejected). Positive
SAE claims require ≥3 seeds/retrains and held-group replication, and remain gated on the
SLM-126 authorization decision. Out of scope: SAE-generated labels, automatic feature
naming, and any adapter/SAE composition or routing.
