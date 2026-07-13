"""Model-building harness: train/eval shell with stub plug-in (no TwoTower)."""

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.eval_runner import evaluate
from slm_training.harnesses.model_build.plugin import ModelPlugin, StubModel
from slm_training.harnesses.model_build.train_loop import train

__all__ = [
    "ModelBuildConfig",
    "ModelPlugin",
    "StubModel",
    "evaluate",
    "train",
]
