"""Experiment runners: ladders, scaling fits, promotion protocol."""

from slm_training.harnesses.experiments.cap5_02_campaign import (
    CAMPAIGN_ID as CAP5_CAMPAIGN_ID,
    CampaignArm,
    Cap5CampaignManifest,
    build_cap5_campaign_manifest,
    validate_cap5_campaign_manifest,
)
from slm_training.harnesses.experiments.constraint_backend_benchmark import (
    MANIFEST_SCHEMA as CONSTRAINT_BACKEND_BENCHMARK_SCHEMA,
    BackendAdapter,
    BenchmarkArm,
    ConstraintBackendBenchmarkManifest,
    ActivationGate as ConstraintBackendActivationGate,
    BudgetCap as ConstraintBackendBudgetCap,
    build_constraint_backend_benchmark_manifest,
    validate_constraint_backend_benchmark_manifest,
)
from slm_training.harnesses.experiments.efficiency_gain import (
    efficiency_gain,
    efficiency_gain_lcb,
)
from slm_training.harnesses.experiments.external_ceiling_matrix import (
    MATRIX_SET as EXTERNAL_CEILING_MATRIX_SET,
    MATRIX_VERSION as EXTERNAL_CEILING_MATRIX_VERSION,
    ExternalCeilingArm,
    ExternalCeilingManifest,
    ExternalCeilingReport,
    build_external_ceiling_manifest,
    render_markdown as render_external_ceiling_markdown,
    run_fixture_matrix as run_external_ceiling_fixture_matrix,
    validate_external_ceiling_manifest,
)
from slm_training.harnesses.experiments.causal_peft_ftpo import (
    CAUSAL_PEFT_FTPO_ID,
    MATRIX_SET as CAUSAL_PEFT_FTPO_MATRIX_SET,
    MATRIX_VERSION as CAUSAL_PEFT_FTPO_MATRIX_VERSION,
    CausalPeftFtpoManifest,
    CausalPeftFtpoReport,
    FtpoArmResult,
    build_causal_peft_ftpo_manifest,
    render_markdown as render_causal_peft_ftpo_markdown,
    run_fixture_ftpo,
    validate_manifest as validate_causal_peft_ftpo_manifest,
)
from slm_training.harnesses.experiments.corruption_curriculum import (
    CORRUPTION_CURRICULUM_ID,
    MATRIX_SET as CORRUPTION_CURRICULUM_MATRIX_SET,
    MATRIX_VERSION as CORRUPTION_CURRICULUM_MATRIX_VERSION,
    CorruptionCurriculumManifest,
    CorruptionCurriculumReport,
    CurriculumArmResult,
    build_corruption_curriculum_manifest,
    render_markdown as render_corruption_curriculum_markdown,
    run_fixture_curriculum,
    validate_manifest as validate_corruption_curriculum_manifest,
)
from slm_training.harnesses.experiments.e228_exposure_ladder import (
    LADDER_ID as E228_EXPOSURE_LADDER_ID,
    MATRIX_SET as E228_EXPOSURE_MATRIX_SET,
    MATRIX_VERSION as E228_EXPOSURE_MATRIX_VERSION,
    E228ExposureLadderManifest,
    E228ExposureReport,
    build_e228_exposure_ladder,
    build_e228_recipe_config,
    render_markdown as render_e228_exposure_markdown,
    run_fixture_ladder as run_e228_fixture_ladder,
    validate_manifest as validate_e228_exposure_manifest,
)
from slm_training.harnesses.experiments.pretrained_denoiser_activation import (
    DEFAULT_ACTIVATION_GATES,
    DEFAULT_ARMS,
    HYPOTHESIS_ID,
    MANIFEST_SCHEMA,
    ActivationGate as PretrainedDenoiserActivationGate,
    BudgetCap as PretrainedDenoiserBudgetCap,
    LicenseTerms,
    PretrainedDenoiserActivationManifest,
    PretrainedDenoiserArm,
    PretrainedDenoiserCandidate,
    build_pretrained_denoiser_activation_manifest,
    validate_pretrained_denoiser_activation_manifest,
)
from slm_training.harnesses.experiments.promotion import (
    PromotionCriteria,
    check_category_regression,
    check_data_integrity,
    check_rank_stability,
    evaluate_promotion,
    register_promoted_checkpoint,
)
from slm_training.harnesses.experiments.proxy_metric_calibration import (
    ActivationGate as ProxyMetricCalibrationActivationGate,
    BudgetCap as ProxyMetricCalibrationBudgetCap,
    CalibrationArm,
    ProxyFeatureSet,
    ProxyMetricCalibrationManifest,
    build_proxy_metric_calibration_manifest,
    validate_proxy_metric_calibration_manifest,
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
from slm_training.harnesses.experiments.scaffold_distillation_activation import (
    ActivationGate as ScaffoldDistillationActivationGate,
    BudgetCap as ScaffoldDistillationBudgetCap,
    ScaffoldDistillationActivationManifest,
    ScaffoldDistillationArm,
    TeacherTraceContract,
    build_scaffold_distillation_activation_manifest,
    validate_scaffold_distillation_activation_manifest,
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


_LAZY_LADDER_EXPORTS = {
    "LadderPoint",
    "ScalingLadder",
    "hf_ladder_default",
    "ladder_run_id",
    "model_build_config_for_point",
    "proportional_depths",
    "scratch_ladder_default",
}


def __getattr__(name: str):
    if name in _LAZY_LADDER_EXPORTS:
        from slm_training.harnesses.experiments import ladder

        value = getattr(ladder, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "CAUSAL_PEFT_FTPO_ID",
    "CAUSAL_PEFT_FTPO_MATRIX_SET",
    "CAUSAL_PEFT_FTPO_MATRIX_VERSION",
    "CausalPeftFtpoManifest",
    "CausalPeftFtpoReport",
    "FtpoArmResult",
    "CORRUPTION_CURRICULUM_ID",
    "CORRUPTION_CURRICULUM_MATRIX_SET",
    "CORRUPTION_CURRICULUM_MATRIX_VERSION",
    "CorruptionCurriculumManifest",
    "CorruptionCurriculumReport",
    "CurriculumArmResult",
    "E228_EXPOSURE_LADDER_ID",
    "E228_EXPOSURE_MATRIX_SET",
    "E228_EXPOSURE_MATRIX_VERSION",
    "E228ExposureLadderManifest",
    "E228ExposureReport",
    "EXTERNAL_CEILING_MATRIX_SET",
    "EXTERNAL_CEILING_MATRIX_VERSION",
    "ExternalCeilingArm",
    "ExternalCeilingManifest",
    "ExternalCeilingReport",
    "ACTIVATION_VERDICTS",
    "CAMPAIGN_VERDICTS",
    "CONSTRAINT_BACKEND_BENCHMARK_SCHEMA",
    "CAP5_CAMPAIGN_ID",
    "DEFAULT_ACTIVATION_GATES",
    "DEFAULT_ARMS",
    "HYPOTHESIS_ID",
    "MANIFEST_SCHEMA",
    "ActivationGate",
    "BackendAdapter",
    "BenchmarkArm",
    "BudgetCap",
    "CampaignArm",
    "CanonicalRequest",
    "Cap5CampaignManifest",
    "ConstraintBackendActivationGate",
    "ConstraintBackendBenchmarkManifest",
    "ConstraintBackendBudgetCap",
    "LicenseTerms",
    "ProxyFeatureSet",
    "ProxyMetricCalibrationActivationGate",
    "ProxyMetricCalibrationBudgetCap",
    "ProxyMetricCalibrationManifest",
    "CalibrationArm",
    "PretrainedDenoiserActivationGate",
    "PretrainedDenoiserActivationManifest",
    "PretrainedDenoiserArm",
    "PretrainedDenoiserBudgetCap",
    "PretrainedDenoiserCandidate",
    "ScaffoldDistillationActivationGate",
    "ScaffoldDistillationActivationManifest",
    "ScaffoldDistillationArm",
    "ScaffoldDistillationBudgetCap",
    "TeacherParaphraseActivationManifest",
    "TeacherParaphraseArm",
    "TeacherProviderConfig",
    "TeacherTraceContract",
    "build_cap5_campaign_manifest",
    "build_constraint_backend_benchmark_manifest",
    "build_causal_peft_ftpo_manifest",
    "build_corruption_curriculum_manifest",
    "build_e228_exposure_ladder",
    "build_e228_recipe_config",
    "build_external_ceiling_manifest",
    "build_pretrained_denoiser_activation_manifest",
    "build_proxy_metric_calibration_manifest",
    "build_scaffold_distillation_activation_manifest",
    "build_teacher_paraphrase_activation_manifest",
    "render_causal_peft_ftpo_markdown",
    "render_corruption_curriculum_markdown",
    "render_e228_exposure_markdown",
    "render_external_ceiling_markdown",
    "run_e228_fixture_ladder",
    "run_fixture_curriculum",
    "run_fixture_ftpo",
    "run_external_ceiling_fixture_matrix",
    "validate_causal_peft_ftpo_manifest",
    "validate_corruption_curriculum_manifest",
    "validate_e228_exposure_manifest",
    "validate_external_ceiling_manifest",
    "validate_cap5_campaign_manifest",
    "validate_constraint_backend_benchmark_manifest",
    "validate_pretrained_denoiser_activation_manifest",
    "validate_proxy_metric_calibration_manifest",
    "validate_scaffold_distillation_activation_manifest",
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
    "render_canonical_request",
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
