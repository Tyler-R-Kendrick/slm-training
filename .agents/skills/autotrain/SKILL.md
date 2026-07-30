---
name: autotrain
description: Operate the OpenUI SLM training pipeline end to end, including a continuous evidence-driven model and harness improvement loop. Bare /autotrain is continuous; an explicit phase or --once is finite.
---

# Autotrain OpenUI SLMs

Facade for **operating and continuously improving** the training pipeline with progressive disclosure:
this file routes; each phase's full instructions live in `references/` and are
read only when that phase is being run. To modify a harness, use
`improve-openui-harnesses`. For the higher-level, knowledge-driven research
loop that *coordinates* this pipeline with brains / OpenWiki / literature
discovery / Linear, use `autoresearch`.

## Workflow

1. A bare `/autotrain` invocation enters the continuous loop in
   [references/continuous.md](references/continuous.md). An explicit phase or
   `--once` performs one finite pass.
2. Pick the phase from the routing table below (`slm list` shows every
   command; `slm guide <slug>` prints a reference from the terminal).
3. Read `references/<slug>.md` — only the one you need — plus
   [references/contracts.md](references/contracts.md) once per session.
4. Run its `slm` commands (each ≡ `python -m scripts.<module>`); keep
   artifacts in the canonical roots it names.
5. Close out: docs + model-card duties per contracts
   (`documenting-experiment-results`).
6. Hand off: ship claims → `honest-ship-eval`; matrix methodology →
   `running-experiment-matrices`; campaign methodology → `openui-autoresearch`;
   certified metric bands and Lean proof/assumption repairs →
   `improve-lean-optimums`;
   knowledge-driven research orchestration (brains/OpenWiki/Linear) →
   `autoresearch`.

## Phase routing

| Phase | Command | Reference |
| --- | --- | --- |
| Continuous model + harness improvement | `slm autoresearch status --loop-id <id> --matrix` | [references/continuous.md](references/continuous.md) |
| Build/publish training corpora | `slm data build-train` / `publish-train` / `store` | [references/train-data.md](references/train-data.md) |
| Build held-out/adversarial/OOD suites | `slm data build-test` | [references/test-data.md](references/test-data.md) |
| SFT / model build (Phase A) | `slm sft train` / `remote` / `hf-jobs` | [references/sft.md](references/sft.md) |
| Evaluate + ship gates | `slm eval model` / `diagnose` / `loss-suites` / `tasks` | [references/eval.md](references/eval.md) |
| Distillation / P1–P3 climb | `slm distill collect` / `self` / `resume-climb` | [references/distill.md](references/distill.md) |
| Preference / surrogate-DPO (Phase B) | `slm preference <subcommand>` | [references/preference.md](references/preference.md) |
| RL / GRPO-lite (Phase C; NeMo/MOLT) | `slm rl train` / `nemo` / `molt` | [references/rl.md](references/rl.md) |
| Experiment matrices, scaling, recipes | `slm experiments <matrix>` | [references/experiments.md](references/experiments.md) |
| Checkpoint sync, lineage, promotion | `slm checkpoints sync` / `migrate`; `slm cycle <sub>` | [references/checkpoints.md](references/checkpoints.md) |
| Annotation export → preference inputs | `slm annotations export` | [references/annotations.md](references/annotations.md) |
| Benchmarks + generation profiling | `slm bench telemetry` / `accel` / `cactus` / `profile` | [references/bench.md](references/bench.md) |
| Model/weight spectral inspection | `slm inspect <subcommand>` | [references/inspect.md](references/inspect.md) |
| Autoresearch self-improvement + RL gate | `slm autoresearch <subcommand>` | [references/autoresearch.md](references/autoresearch.md) |

## Non-negotiable contracts

Digest — full versions in [references/contracts.md](references/contracts.md):

- **Iron law**: no run without the matching `docs/design/` JSON + markdown.
- **Model card**: every checkpoint updates `docs/MODEL_CARD.md` + README summary.
- **Honesty**: fixture/scratch evidence is wiring only; readiness needs
  `--ship-gates` on full scoreboards.
- **RL is fail-closed**: approved `RLReadinessReport` or no RL — no override.
- **No shadow paths**: reuse canonical scripts/harnesses and artifact roots.
- **Decode invariants**: constrained decoding is the product. Deterministic
  singleton bypass outranks any learned score; unconstrained arms are
  diagnostic controls, never defaults, serving paths, or gate inputs. Full
  law in `AGENTS.md` §Non-negotiable architecture invariants +
  [decode-invariants.md](../../../docs/design/decode-invariants.md).
