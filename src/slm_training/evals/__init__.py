"""Deterministic teacher-forced evaluation suites (inner-loop signals).

These evaluations are candidate-invariant: fixed held-out records, fixed
hash-derived masks, fixed mask rates, no stochastic augmentation, no MDLM /
LTR / fidelity weighting, no grammar repair, no best-of-N. They exist so that
data / architecture / training changes can be compared on a low-noise signal
before running the expensive generated scoreboard.
"""

# Lazy re-exports (PEP 562): denoising_nll/loss_suites import torch, but this
# package also hosts torch-free modules (record_schema, agentv) that the web
# entrypoint must import on cold start without torch installed.
_LAZY_EXPORTS = {
    "DEFAULT_MASK_RATES": "denoising_nll",
    "DenoisingNLLConfig": "denoising_nll",
    "evaluate_denoising_nll": "denoising_nll",
    "fixed_mask_positions": "denoising_nll",
    "CATEGORY_WEIGHTS": "loss_suites",
    "LOSS_SUITE_VERSION": "loss_suites",
    "evaluate_loss_suites": "loss_suites",
    "evaluate_repair_nll": "loss_suites",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module_name}"), name)

__all__ = [
    "CATEGORY_WEIGHTS",
    "DEFAULT_MASK_RATES",
    "DenoisingNLLConfig",
    "LOSS_SUITE_VERSION",
    "evaluate_denoising_nll",
    "evaluate_loss_suites",
    "evaluate_repair_nll",
    "fixed_mask_positions",
]
from slm_training.evals.generalization import (
    generalization_report,
    train_generalization_profile,
)
from slm_training.evals.task_scoreboard import build_task_scoreboard, score_case

__all__ = [
    "build_task_scoreboard",
    "generalization_report",
    "score_case",
    "train_generalization_profile",
]
