"""Experiment runners: ladders, scaling fits, promotion protocol."""

from slm_training.harnesses.experiments.cap5_02_campaign import (
    CAMPAIGN_ID as CAP5_CAMPAIGN_ID,
    CampaignArm,
    Cap5CampaignManifest,
    build_cap5_campaign_manifest,
    validate_cap5_campaign_manifest,
)
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
from slm_training.harnesses.experiments.teacher_paraphrase_activation import (
    ACTIVATION_VERDICTS,
    CAMPAIGN_VERDICTS,
    ActivationGate,
    BudgetCap,
    CanonicalRequest,
    TeacherParaphraseActivationManifest,
    TeacherParaphraseArm,
    TeacherProviderConfig,
    build_teacher_paraphrase_activation_manifest,
    render_canonical_request,
    validate_teacher_paraphrase_activation_manifest,
)
from slm_training.harnesses.experiments.verified_solver_matrix import (
    MATRIX_SET,
    MATRIX_VERSION,
    VerifiedSolverMatrixReport,
    VerifiedSolverRow,
    build_matrix_rows,
    describe_matrix,
    evaluate_hard_gates,
    render_markdown,
    run_fixture_matrix,
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

try:
    from slm_training.harnesses.experiments.cap2_04_state_ablation import (
        ArmConfig,
        ArmResult,
        StateAblationReport,
        build_arms,
        evaluate_arm,
        fixture_decisions,
        match_active_parameters,
        run_matrix,
    )
except Exception:  # pragma: no cover - optional if torch unavailable
    ArmConfig = None  # type: ignore[misc,assignment]
    ArmResult = None  # type: ignore[misc,assignment]
    StateAblationReport = None  # type: ignore[misc,assignment]
    build_arms = None  # type: ignore[misc,assignment]
    evaluate_arm = None  # type: ignore[misc,assignment]
    fixture_decisions = None  # type: ignore[misc,assignment]
    match_active_parameters = None  # type: ignore[misc,assignment]
    run_matrix = None  # type: ignore[misc,assignment]

__all__ = [
    "ACTIVATION_VERDICTS",
    "CAMPAIGN_VERDICTS",
    "ActivationGate",
    "BudgetCap",
    "CanonicalRequest",
    "TeacherParaphraseActivationManifest",
    "TeacherParaphraseArm",
    "TeacherProviderConfig",
    "CAP5_CAMPAIGN_ID",
    "CampaignArm",
    "Cap5CampaignManifest",
    "build_cap5_campaign_manifest",
    "validate_cap5_campaign_manifest",
    "build_teacher_paraphrase_activation_manifest",
    "render_canonical_request",
    "validate_teacher_paraphrase_activation_manifest",
    "ArmConfig",
    "ArmResult",
    "BottleneckArm",
    "BottleneckMatrixReport",
    "BottleneckResult",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "StateAblationReport",
    "VerifiedSolverMatrixReport",
    "VerifiedSolverRow",
    "build_arms",
    "build_matrix_rows",
    "describe_matrix",
    "evaluate_hard_gates",
    "evaluate_arm",
    "fixture_decisions",
    "match_active_parameters",
    "render_markdown",
    "run_fixture_matrix",
    "run_matrix",
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
    "scratch_ladder_default",
    "train_eval_evaluator",
]
