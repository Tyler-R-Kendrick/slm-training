---
name: train
description: Operate the OpenUI SLM training pipeline end to end — build train/test data, SFT, distillation, preference, RL, evaluation with honest ship gates, experiment matrices, checkpoints/promotion, benchmarks, annotations, and bounded autoresearch self-improvement. Use when RUNNING any pipeline phase; to CHANGE harness code use improve-openui-harnesses instead.
---

# Train OpenUI SLMs

Facade for **operating** the training pipeline. It routes to one per-phase skill
at a time; it never duplicates harness logic. To modify a harness, use
`improve-openui-harnesses`.

## Workflow

1. Pick the phase from the routing table below.
2. Open and follow that `phase-<slug>` skill — only the one you need.
3. Run its commands; keep artifacts in the canonical roots it names.
4. Close out: docs + model-card duties per the phase skill
   (`documenting-experiment-results`).
5. Hand off: ship claims → `honest-ship-eval`; matrix methodology →
   `running-experiment-matrices`; campaign methodology → `openui-autoresearch`.

## Phase routing

| Phase | Skill |
| --- | --- |
| Build/publish training corpora | `phase-train-data` |
| Build held-out/adversarial/OOD suites | `phase-test-data` |
| SFT / model build (Phase A; local, pod, HF Jobs) | `phase-sft` |
| Evaluate + ship gates | `phase-eval` |
| Distillation / P1–P3 climb | `phase-distill` |
| Preference / surrogate-DPO (Phase B) | `phase-preference` |
| RL / GRPO-lite (Phase C; NeMo/MOLT) | `phase-rl` |
| Experiment matrices, scaling, recipes | `phase-experiments` |
| Checkpoint sync, lineage, promotion | `phase-checkpoints` |
| Annotation export → preference inputs | `phase-annotations` |
| Benchmarks + generation profiling | `phase-bench` |
| Autoresearch self-improvement + RL gate | `phase-autoresearch` |

Quality/retrieval is library-only (`src/slm_training/harnesses/quality/`) —
consumed by the phases above, no direct invocation.

## Non-negotiable contracts

- **Iron law**: no train / eval / bench / matrix / telemetry run without the
  matching `docs/design/` JSON + markdown update.
- **Model card**: every created/synced/promoted checkpoint updates
  `docs/MODEL_CARD.md` + the README summary.
- **Honesty**: fixture/scratch evidence is wiring only; readiness needs
  `--ship-gates` on full scoreboards.
- **RL is fail-closed**: an approved `RLReadinessReport` or no RL — no override.
- **No shadow paths**: reuse the canonical scripts/harnesses; never build a
  parallel trainer or artifact tree.
