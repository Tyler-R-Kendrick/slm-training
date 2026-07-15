---
name: nemo-rl-hf-jobs-etiquette
description: Safely plan, submit, and reconcile pinned NVIDIA NeMo RL training on Hugging Face Jobs with durable OpenUI artifacts. Use for NeMo RL Jobs commands, recipes, credentials, checkpoint handling, or hardware-smoke claims.
---

# NeMo RL on Hugging Face Jobs

Use the repository's `scripts.model_cycle submit-nemo` and `reconcile-nemo`
commands. Do not invent a second dispatch path.

## Before submission

1. Inspect git status and preserve unrelated changes.
2. Create or branch a `causal_lm` lineage run with the pinned base model revision.
3. Dry-run the exact Jobs command first.
4. Keep the NeMo RL release git SHA, container tag, model revision, recipe, data
   snapshot, seed, and code revision explicit.
5. Require explicit human approval before adding `--ack-paid-gpu` or launching
   any paid flavor.

## Secrets and storage

- Pass `HF_TOKEN` with `--secrets HF_TOKEN`; never serialize its value.
- Pass optional tracker credentials as secret names, never command values.
- Mount `hf://buckets/TKendrick/OpenUI` at `/mnt/openui-bucket`.
- Put caches and clones on ephemeral `/workspace` storage.
- Put checkpoints, summaries, and failure evidence under the mounted bucket.

## Claims and completion

- A one-step run is `hardware_smoke`, never a quality result or promotion
  candidate.
- Reconcile the HF Job and validate `train_summary.json` before recording the
  durable artifact URI in lineage.
- Preserve failed-job logs and metadata; do not call failure success.
- After an actual run, follow `documenting-experiment-results`. If a reusable
  checkpoint was created, update `docs/MODEL_CARD.md` and the README summary.
- Do not assume Brev, Spaces ZeroGPU, or local GPUs are part of this workflow.
