# LDI1-03 — Causal adapter & objective campaign matrix (harness)

**Issue:** SLM-122 (LDI1-03). **Date:** 2026-07-18. **Status:** orchestration
harness + eligibility gates only. **No training or evaluation run was executed
and no quality claim is made.**

## What this delivers

A Torch-free orchestration layer that plans the matched causal exact-state
intervention campaign over the merged LDI1-02 trainer
(`causal_trainer.train_causal_local`) and classifies every arm before any compute:

- `src/slm_training/harnesses/preference/causal_adapter_matrix.py` — arm-matrix
  builders, fail-closed eligibility/falsification gates, per-arm status, and a
  guarded executor.
- `scripts/run_causal_adapter_matrix.py` — thin dry-run/classify CLI.
- `tests/test_harnesses/preference/test_causal_adapter_matrix.py` — 14 invariant
  tests.

The matrix is generated from one canonical `CampaignConfig`; each arm differs
from its stage baseline only in its **declared** levers (checked by
`only_declared_levers_differ`).

| Stage | Arms | Varied lever(s) |
| ----- | ---- | --------------- |
| 0 — objective controls | C0 parent · C1 unlikelihood · C2 ftpo_single · C3 ftpo_set · C4 legal_set_mass · C5 best+tether · C6 C5+balanced | objective (+ tether/sampler) |
| 1 — rank × placement | rank {16,32,64} × {all, last-k} on the best eligible objective, `alpha == rank` | rank, alpha, layer_pattern |
| 2 — actuator method | {lora, dora, pissa} at fixed map & matched rank | method |

## Fail-closed gates (the honesty-critical part)

- **Set-valued objectives** (`ftpo_set`, `legal_set_mass`) `block` with
  `blocked_by_corpus` when the corpus lacks multi-alternative support — they are
  never silently narrowed to single pairs.
- **Unsupported / experimental methods** (`adalora` without explicit opt-in, or
  any unknown method) return `not_supported` — never an implicit LoRA fallback.
- An **unadmitted corpus** blocks every trainable arm; the C0 parent baseline
  (no update) stays admissible.
- **`expired`** is the honest outcome for an admissible arm with no executable
  policy + admitted corpus in the environment — distinct from `completed`, and
  it carries no metrics. Only positive `completed` arms trigger the ≥3-seed
  replication rule.

## Commands (this environment)

```bash
python -m pytest tests/test_harnesses/preference/test_causal_adapter_matrix.py -q   # 14 passed
PYTHONPATH=src python -m scripts.run_causal_adapter_matrix --describe                # 16 arms (7/6/3)
PYTHONPATH=src python -m scripts.run_causal_adapter_matrix --stage 0 \
    --corpus-admitted --has-pairs --has-set-valued                                   # -> status_counts {'expired': 7}
python -m scripts.repo_policy                                                        # ok
```

The classify run above returns **7 `expired`** arms (no GPU policy / no admitted
corpus here) — i.e. the harness refuses to invent results.

## What remains (the ship-grade run — deferred)

The quality-bearing matrix run is **out of scope for this environment** and is not
performed here. It requires:

- GPU + the repo's remote-training/HF-Jobs owner;
- a pinned causal base checkpoint and revision from the lineage contract;
- a DecisionEventV2 semantic corpus that passes the LDI0-03 (SLM-117)
  objective-support gate;
- a globally unique E-ID from the live allocator (intentionally **not**
  hard-coded here, since no run occurred) and the five-suite evaluation policy.

At that point the executor path (`run_arm` with a real `policy_factory` +
`train_items`) produces `completed` arms with pre/post held-out metrics, and the
existing quality-matrix / model-card / research-lineage evidence is written per
`documenting-experiment-results`. This note is wiring evidence only.
