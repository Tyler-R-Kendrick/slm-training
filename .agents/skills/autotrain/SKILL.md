---
name: autotrain
description: >
  Operate the OpenUI SLM training pipeline end to end, including a continuous
  hands-off model and harness improvement loop. Bare /autotrain is non-terminating
  and must not stop for user confirmation; an explicit phase or --once is finite.
  Code fixes during training use incremental commits every cycle and stacked
  PRs only after positive-result runs (sdlc autotrain-iteration-delivery);
  when training stops, full bottom-up SDLC closeout of open positive layers.
---

# Autotrain OpenUI SLMs

Facade for **operating and continuously improving** the training pipeline with progressive disclosure:
this file routes; each phase's full instructions live in `references/` and are
read only when that phase is being run. To modify a harness, use
`improve-openui-harnesses`. For the higher-level, knowledge-driven research
loop that *coordinates* this pipeline with brains / OpenWiki / literature
discovery / Linear, use `autoresearch`.

**Delivery process** for code and durable docs during training is owned by
`sdlc` — read
[`../sdlc/references/autotrain-iteration-delivery.md`](../sdlc/references/autotrain-iteration-delivery.md)
before continuous or multi-run work.

## Workflow

1. **Bare `/autotrain` (default) is continuous and hands-off.** Immediately
   enter [references/continuous.md](references/continuous.md). Keep chaining
   bounded campaigns until the session is preempted or the repeated-blocker
   rule fires. **Do not stop to ask the user to continue.** Do not end a turn
   with only a resume recipe. Print the result matrix between cycles, then
   start the next cycle without waiting.
2. An explicit phase name or `--once` performs **one finite pass** only.
3. For finite phase work: pick the phase from the routing table below
   (`slm list` / `slm guide <slug>`), read `references/<slug>.md` plus
   [references/contracts.md](references/contracts.md) once per session, run
   the `slm` commands, and close out docs/model-card duties.
4. Hand off (when those claims appear): ship → `honest-ship-eval`; matrices →
   `running-experiment-matrices`; campaigns → `openui-autoresearch`; Lean
   bands → `improve-lean-optimums`; brains/OpenWiki/Linear → `autoresearch`;
   multi-layer land → `sdlc`.

## Continuous mode (bare `/autotrain`) — non-negotiable

| Rule | Required behavior |
| --- | --- |
| Hands-off | No confirmation prompts; no “say continue” |
| Non-terminating | Cycle N finish → cycle N+1 start immediately |
| Self-heal | Fix path/knob/harness failures from evidence; re-run |
| Soft failures | Fixture ship-gate fails / null deltas / single timeouts → next cycle |
| Hard block only | Same unrecoverable blocker 3× with no new info → report blocked |
| Incremental commits | Commit green code/docs units every cycle while working on an iteration or fix |
| Stacked PR (positive only) | Open/update a `gh stack` layer **only** after a positive-result run (metric win, ship-quality win, or proven executable unblock) |
| Non-positive cycles | Docs + local commits only — **no** new stack layer for fixture fails / null deltas |
| Get latest between runs | `git fetch` + `gh stack sync` / merge `origin/main`; resolve conflicts |
| Remote compute default | No paid GPU / HF write without prior user authority |
| Training stopped | Full `sdlc` bottom-up closeout of open positive layers (review → CI → squash-merge) — not a resume paste |
| **Matrix to the user** | After every cycle (and whenever reporting status): paste the three-table matrix from `status --loop-id <id> --matrix --last 5` (or `/tmp/autotrain-report.sh` / `/tmp/autotrain-loop-dashboard.md`) into the chat. **Never** claim progress without it |
| **Liveness proof** | Prove RUNNING with `driver_state` + PID + top child + latest campaign from `/tmp/autotrain-loop-status.txt` (or `autotrain-report.sh`). **Never** use Grok “background ops” UI as liveness |
| **Never kill the loop** | Do **not** `kill`/`pkill`/`kill -9` `run_autotrain_continuous`, its children, or the continuous worktree processes to ship skills, fix CI, merge PRs, or “restart cleanly.” Side work uses another worktree/branch |

Full procedure: [references/continuous.md](references/continuous.md).  
Delivery: [`../sdlc/references/autotrain-iteration-delivery.md`](../sdlc/references/autotrain-iteration-delivery.md).  
User-facing report (preferred): `bash /tmp/autotrain-report.sh` → updates `/tmp/autotrain-loop-dashboard.md`.

## Phase routing

| Phase | Command | Reference |
| --- | --- | --- |
| **Continuous model + harness improvement (default)** | keep chaining campaigns; `status --loop-id <id> --matrix` between cycles | [references/continuous.md](references/continuous.md) |
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
