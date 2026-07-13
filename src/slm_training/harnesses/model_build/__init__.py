"""Model-building harness: train/eval shell with TwoTower + stub plug-ins."""

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.eval_runner import evaluate, evaluate_suites
from slm_training.harnesses.model_build.factory import build_model
from slm_training.harnesses.model_build.plugin import ModelPlugin, StubModel
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
    write_ship_gates,
)
from slm_training.harnesses.model_build.train_loop import train

__all__ = [
    "DEFAULT_SHIP_GATES",
    "ModelBuildConfig",
    "ModelPlugin",
    "StubModel",
    "build_model",
    "evaluate",
    "evaluate_ship_gates",
    "evaluate_suites",
    "train",
    "write_ship_gates",
]
