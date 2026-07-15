"""Molt agent adapter for the shared honest OpenUI reward contract."""

from __future__ import annotations

from molt.agents import Env, Result, StepEnvRunner

from slm_training.integrations.molt_rl import decode_label
from slm_training.integrations.openui_rl import score_openui


class OpenUIEnv(Env):
    async def step(self, state: dict, **kwargs) -> Result:
        label = decode_label(str(state["label"]))
        reward = score_openui(
            str(state["action_text"]),
            gold_openui=label["gold_openui"],
            slot_inventory=label["slot_inventory"],
        )
        return Result(
            reward=reward.composite,
            score=reward.composite,
            terminated=True,
            info=reward.to_dict(),
        )


class AgentRunner(StepEnvRunner):
    def __init__(self) -> None:
        super().__init__(OpenUIEnv)
