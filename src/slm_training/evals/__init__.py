"""Deterministic teacher-forced evaluation suites (inner-loop signals).

These evaluations are candidate-invariant: fixed held-out records, fixed
hash-derived masks, fixed mask rates, no stochastic augmentation, no MDLM /
LTR / fidelity weighting, no grammar repair, no best-of-N. They exist so that
data / architecture / training changes can be compared on a low-noise signal
before running the expensive generated scoreboard.
"""

from slm_training.evals.denoising_nll import (
    DEFAULT_MASK_RATES,
    DenoisingNLLConfig,
    evaluate_denoising_nll,
    fixed_mask_positions,
)
from slm_training.evals.loss_suites import (
    CATEGORY_WEIGHTS,
    LOSS_SUITE_VERSION,
    evaluate_loss_suites,
    evaluate_repair_nll,
)

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
