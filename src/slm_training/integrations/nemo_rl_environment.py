"""NeMo RL adapter for the OpenUI reward contract.

This module is imported only inside the pinned NeMo RL container.
"""

from __future__ import annotations

from typing import Any, TypedDict

import ray
import torch
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from slm_training.integrations.nemo_rl import score_openui


class OpenUIMetadata(TypedDict):
    gold_openui: str
    slot_inventory: list[str]


def openui_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: Any,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    prompt = str(datum_dict["messages"][0]["content"])
    messages: list[dict[str, str]] = []
    if task_data_spec.system_prompt:
        messages.append({"role": "system", "content": task_data_spec.system_prompt})
    messages.append({"role": "user", "content": prompt})
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    token_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    limit = max_seq_length or len(token_ids) + 1
    loss_multiplier = 1.0
    if len(token_ids) >= limit:
        token_ids = token_ids[:4]
        loss_multiplier = 0.0
    return {
        "message_log": [
            {"role": "user", "content": formatted, "token_ids": token_ids}
        ],
        "length": len(token_ids),
        "extra_env_info": {
            "gold_openui": str(datum_dict["gold_openui"]),
            "slot_inventory": list(datum_dict.get("slot_inventory") or ()),
        },
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }


@ray.remote(max_restarts=-1, max_task_retries=-1)
class OpenUIEnvironment(EnvironmentInterface[OpenUIMetadata]):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def shutdown(self) -> None:
        return None

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[OpenUIMetadata],
    ) -> EnvironmentReturn[OpenUIMetadata]:
        predictions = [
            "".join(
                str(message["content"])
                for message in conversation
                if message["role"] == "assistant"
            )
            for conversation in message_log_batch
        ]
        scores = [
            score_openui(
                prediction,
                gold_openui=item["gold_openui"],
                slot_inventory=item["slot_inventory"],
            )
            for prediction, item in zip(predictions, metadata, strict=True)
        ]
        rewards = torch.tensor([score.composite for score in scores]).cpu()
        return EnvironmentReturn(
            observations=[
                {
                    "role": "environment",
                    "content": (
                        f"parse={score.parse:.3f} "
                        f"slots={score.placeholder_fidelity:.3f} "
                        f"structure={score.structural_similarity:.3f}"
                    ),
                }
                for score in scores
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(scores),
            rewards=rewards,
            terminateds=torch.ones_like(rewards).cpu(),
            answers=predictions,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        rewards = batch["rewards"] * batch["is_end"]
        return batch, {
            "reward": rewards.float().mean().item(),
            "fraction_of_samples_properly_ended": batch["is_end"]
            .float()
            .mean()
            .item(),
            "num_samples": int(rewards.shape[0]),
        }

