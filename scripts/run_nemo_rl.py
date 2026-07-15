#!/usr/bin/env python3
"""Register the OpenUI environment, run NeMo RL, and write a run summary."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

from slm_training.integrations.nemo_rl import NEMO_RL_ROOT, write_train_summary


def main() -> int:
    actor_fqn = (
        "slm_training.integrations.nemo_rl_environment.OpenUIEnvironment"
    )
    from nemo_rl.data.processors import register_processor
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import register_env
    from slm_training.integrations.nemo_rl_environment import (
        openui_hf_data_processor,
    )

    ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.SYSTEM
    register_env("openui", actor_fqn)
    register_processor("openui_hf_data_processor", openui_hf_data_processor)
    nemo_root = Path(os.environ.get("NEMO_RL_ROOT", NEMO_RL_ROOT))
    runpy.run_path(str(nemo_root / "examples/run_grpo.py"), run_name="__main__")

    run_id = os.environ["SLM_NEMO_RUN_ID"]
    output_root = Path(os.environ["SLM_NEMO_OUTPUT_ROOT"])
    write_train_summary(
        output_root / "train_summary.json",
        run_id=run_id,
        bucket_uri=os.environ["SLM_NEMO_BUCKET_URI"],
        metadata={
            "base_model_id": os.environ["SLM_NEMO_BASE_MODEL_ID"],
            "base_model_revision": os.environ["SLM_NEMO_BASE_MODEL_REVISION"],
            "code_revision": os.environ["SLM_NEMO_CODE_REVISION"],
            "data_path": os.environ["SLM_NEMO_DATA_PATH"],
            "seed": int(os.environ["SLM_NEMO_SEED"]),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
