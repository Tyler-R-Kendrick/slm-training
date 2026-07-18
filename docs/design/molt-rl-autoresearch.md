# Molt RL causal pipeline and trace observability

**Status:** integration and dry-run contract implemented; no paid job, training,
checkpoint, or quality result was produced by this change.

## Decision and applicability

[NVIDIA Molt](https://github.com/NVIDIA-NeMo/labs-molt) is a second external RL
engine for the canonical `causal_lm` lineage track. It is parallel to NeMo RL:
neither is an implicit fallback or a new model lineage. The local TwoTower
grammar-diffusion model keeps its purpose-built RL harness because Molt consumes
autoregressive Hugging Face models through an AutoModel actor and vLLM rollouts.

The integration uses Molt's supported token-first `Env`/`StepEnvRunner` contract,
GRPO, rollout dump/replay surface, and final Hugging Face export. Molt `0.1.2`
does not expose a LoRA/PEFT actor option, so this smoke uses a one-step
full-parameter FSDP update on the small pinned causal base. NeMo RL remains the
LoRA smoke engine. Claiming Molt LoRA without upstream support would be false.

Sources:

- [Molt README at the pinned revision](https://github.com/NVIDIA-NeMo/labs-molt/blob/21c1b8921b73f5c8317b5fc9e359e9a1b7d255d2/README.md)
- [Molt agent contract](https://github.com/NVIDIA-NeMo/labs-molt/blob/21c1b8921b73f5c8317b5fc9e359e9a1b7d255d2/molt/agents/base.py)
- [Molt rollout dump/replay implementation](https://github.com/NVIDIA-NeMo/labs-molt/blob/21c1b8921b73f5c8317b5fc9e359e9a1b7d255d2/molt/trainer/rl_trainer.py)
- [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)

## Pinned supply chain and smoke recipe

| Input | Pin |
| --- | --- |
| Molt | `0.1.2` / `21c1b8921b73f5c8317b5fc9e359e9a1b7d255d2` |
| Container | `hijkzzz/molt:0.1.2@sha256:b9c82365b0c65e9cd4daf0addc34c9a5eba89cfc4593fa2e480246dc7c1dfcd2` |
| Initial base | lineage manifest model ID plus exact Hub revision |
| Hardware | HF Jobs `h200x2`: one actor GPU and one vLLM GPU |
| Durable store | `hf://buckets/TKendrick/OpenUI/checkpoints/<run_id>/molt_rl/` |

The job checks out the exact application and Molt revisions, downloads the exact
base-model revision, revalidates the embedded `RLReadinessReport`, converts the
shared two-row OpenUI smoke data into Molt `input`/`label` rows, and performs one
GRPO update over two prompts with two generations each. KL/reference workers are
disabled. The smoke is diagnostic only and is permanently recorded as
`hardware_smoke`; it cannot support a quality, readiness, or promotion claim.

## Honest reward and readiness

Molt and NeMo call the same reward implementation:

```text
0.45 parse + 0.30 visible-placeholder fidelity + 0.25 structural similarity
```

Invalid syntax receives zero composite reward. Slot inventory comes from the
learner-visible prompt record; the adapter does not read hidden evaluation
placeholders or DESIGN.md style information. Submission is fail-closed behind the
same frozen five-suite, full-RICO, AgentV, ship-gate, and reward-variance readiness
report as every other RL path. There is no override.

## Durable trace contract

Molt writes trusted replay batches under `traces/raw/rollout_step<N>.pt`. An exit
trap syncs partial raw dumps and logs to the bucket even if training fails. On a
successful run, the pinned container also converts those objects to
`rl_traces.jsonl`; the web server never loads PyTorch/pickle data.

Normalized schema version 1 records:

- engine, run ID, step, prompt-group ID, and rollout ID;
- prompt, decoded completion, gold OpenUI, and visible slot inventory;
- parse, placeholder, structure, and composite rewards;
- prompt/completion token counts, truncation, and action token IDs;
- rollout log-probabilities when Molt collected them.

Reconciliation validates the summary and JSONL before copying the normalized file
to `outputs/runs/<run_id>/rl_traces.jsonl`. The dashboard reads it through
`GET /api/runs/<run_id>/rl-traces?offset=0&limit=20`; malformed individual rows
are counted and skipped. Raw dumps remain available only through the bucket for
trusted replay.

## Commands and acceptance

Preview the exact paid command without submitting it:

```bash
python -m scripts.model_cycle --lineage-root outputs/lineage \
  submit-molt --run-id <causal-run-id> --dry-run \
  --rl-readiness-report outputs/runs/<causal-run-id>/rl_readiness.json
```

A future paid smoke requires a separate explicit decision and
`--ack-paid-gpu`. Submission is fixed to a three-minute HF Jobs timeout; a
timeout is a failed hardware smoke, never partial RL or checkpoint evidence.
After completion:

```bash
python -m scripts.model_cycle --lineage-root outputs/lineage \
  reconcile-molt --run-id <causal-run-id> --job-id <job-id>
```

Static tests and dry-run command generation are the only acceptance evidence in
this change. The first real job must follow `documenting-experiment-results`,
including failure evidence. Any emitted checkpoint also requires
`docs/MODEL_CARD.md` and the README model-card summary before the work is complete.
