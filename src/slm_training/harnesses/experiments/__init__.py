"""Experiment runners: ladders, scaling fits, promotion protocol."""

from slm_training.harnesses.experiments.efficiency_gain import (
    efficiency_gain,
    efficiency_gain_lcb,
)
from slm_training.harnesses.experiments.ladder import (
    LadderPoint,
    ScalingLadder,
    hf_ladder_default,
    ladder_run_id,
    model_build_config_for_point,
    proportional_depths,
    scratch_ladder_default,
)
from slm_training.harnesses.experiments.promotion import (
    PromotionCriteria,
    check_category_regression,
    check_data_integrity,
    check_rank_stability,
    evaluate_promotion,
    register_promoted_checkpoint,
)
from slm_training.harnesses.experiments.recipe_evolution import (
    CandidateResult,
    EvolutionConfig,
    RecipeGene,
    crossover,
    gate_check,
    mutate,
    rank_candidates,
    run_evolution,
    train_eval_evaluator,
)
from slm_training.harnesses.experiments.scaling_fit import (
    ScalingObservation,
    fit_power_law,
    invert_loss,
    observation_from_summary,
    predict_loss,
)

try:
    from slm_training.harnesses.experiments.cap2_bottleneck import (
        BottleneckArm,
        BottleneckMatrixReport,
        BottleneckResult,
        build_matrix,
        evaluate_arm,
        run_matrix,
    )
except Exception:  # pragma: no cover - optional if torch unavailable
    BottleneckArm = None  # type: ignore[misc,assignment]
    BottleneckMatrixReport = None  # type: ignore[misc,assignment]
    BottleneckResult = None  # type: ignore[misc,assignment]
    build_matrix = None  # type: ignore[misc,assignment]
    evaluate_arm = None  # type: ignore[misc,assignment]
    run_matrix = None  # type: ignore[misc,assignment]

__all__ = [
    "BottleneckArm",
    "BottleneckMatrixReport",
    "BottleneckResult",
    "LadderPoint",
    "PromotionCriteria",
    "ScalingLadder",
    "ScalingObservation",
    "check_category_regression",
    "check_data_integrity",
    "CandidateResult",
    "EvolutionConfig",
    "RecipeGene",
    "build_matrix",
    "check_rank_stability",
    "crossover",
    "efficiency_gain",
    "efficiency_gain_lcb",
    "evaluate_arm",
    "evaluate_promotion",
    "fit_power_law",
    "gate_check",
    "hf_ladder_default",
    "invert_loss",
    "ladder_run_id",
    "model_build_config_for_point",
    "mutate",
    "observation_from_summary",
    "predict_loss",
    "proportional_depths",
    "rank_candidates",
    "register_promoted_checkpoint",
    "run_evolution",
    "run_matrix",
    "scratch_ladder_default",
    "train_eval_evaluator",
]
