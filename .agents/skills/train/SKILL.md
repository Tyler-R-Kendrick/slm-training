---
name: train
description: Operate the OpenUI SLM training pipeline end to end — build train/test data, SFT, distillation, preference, RL, evaluation with honest ship gates, experiment matrices, checkpoints/promotion, benchmarks, annotations, and bounded autoresearch self-improvement. Use when RUNNING any pipeline phase; to CHANGE harness code use improve-openui-harnesses instead.
---

# Train OpenUI SLMs

Facade for **operating** the training pipeline with progressive disclosure:
this file routes; each phase's full instructions live in `references/` and are
read only when that phase is being run. To modify a harness, use
`improve-openui-harnesses`.

## Workflow

1. Pick the phase from the routing table below.
2. Read `references/<slug>.md` — only the one you need — plus
   [references/contracts.md](references/contracts.md) once per session.
3. Run its commands; keep artifacts in the canonical roots it names.
4. Close out: docs + model-card duties per contracts
   (`documenting-experiment-results`).
5. Hand off: ship claims → `honest-ship-eval`; matrix methodology →
   `running-experiment-matrices`; campaign methodology → `openui-autoresearch`.

## Phase routing

| Phase | Reference |
| --- | --- |
| Build/publish training corpora | [references/train-data.md](references/train-data.md) |
| Build held-out/adversarial/OOD suites | [references/test-data.md](references/test-data.md) |
| SFT / model build (Phase A; local, pod, HF Jobs) | [references/sft.md](references/sft.md) |
| Evaluate + ship gates | [references/eval.md](references/eval.md) |
| Distillation / P1–P3 climb | [references/distill.md](references/distill.md) |
| Preference / surrogate-DPO (Phase B) | [references/preference.md](references/preference.md) |
| RL / GRPO-lite (Phase C; NeMo/MOLT) | [references/rl.md](references/rl.md) |
| Experiment matrices, scaling, recipes | [references/experiments.md](references/experiments.md) |
| Checkpoint sync, lineage, promotion | [references/checkpoints.md](references/checkpoints.md) |
| Annotation export → preference inputs | [references/annotations.md](references/annotations.md) |
| Benchmarks + generation profiling | [references/bench.md](references/bench.md) |
| Autoresearch self-improvement + RL gate | [references/autoresearch.md](references/autoresearch.md) |

## Non-negotiable contracts

Digest — full versions in [references/contracts.md](references/contracts.md):

- **Iron law**: no run without the matching `docs/design/` JSON + markdown.
- **Model card**: every checkpoint updates `docs/MODEL_CARD.md` + README summary.
- **Honesty**: fixture/scratch evidence is wiring only; readiness needs
  `--ship-gates` on full scoreboards.
- **RL is fail-closed**: approved `RLReadinessReport` or no RL — no override.
- **No shadow paths**: reuse canonical scripts/harnesses and artifact roots.
