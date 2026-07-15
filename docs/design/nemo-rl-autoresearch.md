# NeMo RL autoresearch pipeline

**Status:** infrastructure implemented; paid hardware acceptance not yet run  
**Scope:** causal-LM track, one-step LoRA GRPO hardware smoke, then separately
approved autoresearch campaigns

## Decision

Use NVIDIA NeMo RL as an external RL execution engine under the existing
immutable `model_cycle` lineage system. Hugging Face Jobs replaces the Brev
machine used in NVIDIA's walkthrough; the OpenUI HF Bucket remains the durable
artifact store. NeMo does not become a third model lineage and a smoke run does
not become a champion candidate.

NVIDIA's July 14, 2026 workflow identifies three reusable layers: environment
etiquette, durable session memory, and an autoresearch loop. This repository
vendors the upstream `nemo-rl-auto-research` and `nemo-rl-session-memory` skills
and adds `nemo-rl-hf-jobs-etiquette` for the local compute/storage policy.

Sources:

- [NVIDIA autoresearch workflow](https://developer.nvidia.com/blog/how-to-run-an-autoresearch-workflow-with-rl-agent-skills-and-nvidia-nemo/)
- [NeMo RL v0.6.0](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.6.0)
- [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)

## Pinned supply chain

| Input | Pin |
| --- | --- |
| NeMo RL | `0.6.0` / `c339070fa3bfa83a5ac58ff80d73518911e14b81` |
| Container | `nvcr.io/nvidia/nemo-rl:v0.6.0` |
| Initial base | `Qwen/Qwen3-0.6B` / `c1899de289a04d12100db370d81485cdf75e47ca` |
| NVIDIA skills | `NVIDIA/skills` / `1700ebd31ba862df5b6992403764c989f2bda66b` |
| Durable store | `hf://buckets/TKendrick/OpenUI` |

The job verifies the NeMo git SHA inside the image, checks out an exact
`slm-training` revision, and downloads the exact Hub model revision before
training. Secrets are passed by name with `--secrets HF_TOKEN`; values are not
placed in manifests, commands, or summaries.

## Runtime flow

```text
causal_lm branch
  -> model-cycle submit-nemo --dry-run
  -> explicit approval + --ack-paid-gpu
  -> pinned NeMo container on HF Jobs
  -> OpenUI processor -> Ray reward environment -> one GRPO step
  -> checkpoint + train_summary.json in HF Bucket
  -> model-cycle reconcile-nemo
  -> immutable lineage revision marked hardware_smoke/screened
```

The reward uses only information intentionally exposed to the learner:

- parse validity from the production OpenUI validator;
- placeholder recall against the prompt's visible slot inventory;
- style-stripped structural similarity against the training target.

The composite is `0.45 parse + 0.30 placeholder + 0.25 structure`, with a hard
zero for invalid syntax. It does not read hidden `gold.placeholders`, frozen
evaluation records, or DESIGN.md styling.

## Smoke recipe

The initial recipe is deliberately small and diagnostic:

- one A10G, one node, one GRPO step;
- two prompts, two generations per prompt, global batch four;
- DTensor v2 LoRA, rank 16, alpha 32, dropout 0.05;
- Qwen projection targets from the frozen causal recipe;
- two committed wiring-only prompts under `fixtures/nemo_rl/`;
- safetensors/consolidated checkpoint, no W&B, bucket sync required.

These fixtures are not held-out evaluation data. A completed run proves model
access, dependency compatibility, rollout/reward wiring, one optimizer step,
checkpoint writing, and durable sync. It does not prove reward improvement,
generalization, ship-gate quality, or reload fidelity.

## Commands

Create a dedicated causal branch run using the normal snapshot and lineage
commands. Then preview the exact paid command:

```bash
python -m scripts.model_cycle --lineage-root outputs/lineage \
  submit-nemo --run-id <causal-run-id> --dry-run
```

Only after reviewing cost, revision, recipe, data, and command:

```bash
python -m scripts.model_cycle --lineage-root outputs/lineage \
  submit-nemo --run-id <causal-run-id> --ack-paid-gpu
```

Reconciliation reads HF Jobs status and the durable summary:

```bash
python -m scripts.model_cycle --lineage-root outputs/lineage \
  reconcile-nemo --run-id <causal-run-id> --job-id <job-id>
```

For offline testing, `reconcile-nemo` accepts `--status-json` and `--summary`.

## Acceptance ladder

| Gate | Evidence | Current state |
| --- | --- | --- |
| Static integration | Ruff, compile, unit tests | Implemented |
| Dispatch safety | exact dry-run, explicit paid flag, named secret | Implemented |
| Runtime identity | image git SHA, code SHA, model revision checks | Implemented |
| Hardware smoke | one optimizer step on HF Jobs | Not run |
| Durability | checkpoint + valid summary in HF Bucket | Not run |
| Reload | load emitted LoRA artifact on pinned base | Not run |
| Quality | full honest eval snapshot and ship gates | Out of smoke scope |

After the first real run, use `documenting-experiment-results`: commit a JSON
record and matching measured-results markdown, including the failing result if
the job fails. If the smoke writes a checkpoint, update `docs/MODEL_CARD.md` and
the README model-card summary before treating the work as complete. Hardware
smoke runs are permanently barred from promotion by `model_cycle`.

## Autoresearch phase after acceptance

Start a campaign only after the smoke and reload gates pass and the user sets a
metric plus a time/GPU-hour/experiment budget. Use a dedicated branch per
hypothesis, the session-memory handoff, and a TSV experiment ledger. Baseline
first; then vary one bounded lever at a time. Full candidates must train on a
real versioned data snapshot and run the frozen production evaluation suites.
Only normal validated causal runs—not hardware-smoke lineage—may enter the
existing promote/merge/deploy path.

