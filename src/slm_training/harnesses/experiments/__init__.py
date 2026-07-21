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
from slm_training.harnesses.experiments.ast_sketch_retrieval_factorial import (
    AST_SKETCH_RETRIEVAL_ID,
    DATA_SAMPLING_ARMS,
    MATRIX_SET as AST_SKETCH_RETRIEVAL_MATRIX_SET,
    MATRIX_VERSION as AST_SKETCH_RETRIEVAL_MATRIX_VERSION,
    RETRIEVAL_MODES,
    AstSketchRetrievalArm,
    AstSketchRetrievalManifest,
    AstSketchRetrievalReport,
    AstSketchRetrievalRow,
    AstTrainingSketchV1,
    ChoiceRetrievalExemplarV1,
    DataSampling,
    RetrievalMode,
    build_ast_sketch_retrieval_manifest,
    build_ast_training_sketch,
    build_choice_exemplar_bank,
    build_choice_retrieval_exemplar,
    format_choice_exemplar_context,
    nearest_choice_exemplars,
    random_choice_exemplars,
    render_markdown as render_ast_sketch_retrieval_markdown,
    run_fixture_matrix as run_ast_sketch_retrieval_fixture_matrix,
    validate_manifest as validate_ast_sketch_retrieval_manifest,
)
from slm_training.harnesses.experiments.slm147_x22_retrieval import (
    MATRIX_SET as SLM147_MATRIX_SET,
    MATRIX_VERSION as SLM147_MATRIX_VERSION,
    X22_RETRIEVAL_ID,
    PrototypeCandidate,
    PrototypeIndexEntry,
    RetrievalStrategy,
    Slm147Arm,
    Slm147Manifest,
    Slm147Record,
    Slm147Report,
    Slm147Row,
    ValidPrototypeIndex,
    ValidPrototypeRetriever,
    build_manifest as build_slm147_manifest,
    build_prototype_index,
    render_markdown as render_slm147_markdown,
    run_fixture_matrix as run_slm147_fixture_matrix,
    validate_manifest as validate_slm147_manifest,
)
from slm_training.harnesses.experiments.slm148_x22_conflict_campaign import (
    MATRIX_SET as SLM148_MATRIX_SET,
    MATRIX_VERSION as SLM148_MATRIX_VERSION,
    X22_CONFLICT_CAMPAIGN_ID,
    SeedStrategy,
    Slm148Manifest,
    Slm148Record,
    Slm148RecoveryArm,
    Slm148Report,
    Slm148Row,
    Slm148SearchConfig,
    Slm148SeedArm,
    build_manifest as build_slm148_manifest,
    render_markdown as render_slm148_markdown,
    run_fixture_matrix as run_slm148_fixture_matrix,
    validate_manifest as validate_slm148_manifest,
)
from slm_training.harnesses.experiments.efs4_04_causal_synthesis import (
    CAMPAIGN_ID as EFS_CAMPAIGN_ID,
    CampaignHypothesisSpec,
    CampaignManifestV1,
    EvidenceFirstSemanticSynthesisV1,
    build_default_campaign_manifest,
    load_manifest as load_efs_campaign_manifest,
    render_dot as render_efs_evidence_dot,
    render_markdown as render_efs_synthesis_markdown,
    render_mermaid as render_efs_evidence_mermaid,
    save_manifest as save_efs_campaign_manifest,
    save_synthesis as save_efs_synthesis,
    synthesize_campaign as synthesize_efs_campaign,
    validate_synthesis as validate_efs_synthesis,
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

_LAZY_B3_EXPORTS = {
    "B3_CAPACITY_V2_ID": "B3_CAPACITY_V2_ID",
    "B3_CAPACITY_V2_MATRIX_SET": "MATRIX_SET",
    "B3_CAPACITY_V2_MATRIX_VERSION": "MATRIX_VERSION",
    "B3CapacityV2Arm": "B3CapacityV2Arm",
    "B3CapacityV2Manifest": "B3CapacityV2Manifest",
    "B3CapacityV2Report": "B3CapacityV2Report",
    "B3CapacityV2Row": "B3CapacityV2Row",
    "build_b3_capacity_v2_manifest": "build_b3_capacity_v2_manifest",
    "render_b3_capacity_v2_markdown": "render_markdown",
    "run_b3_capacity_v2_fixture_ladder": "run_fixture_ladder",
    "validate_b3_capacity_v2_manifest": "validate_manifest",
}

_LAZY_SDE4_02_EXPORTS = {
    "SDE4_02_COMPETENCE_TARGET": "COMPETENCE_TARGET",
    "SDE4_02_MATRIX_SET": "MATRIX_SET",
    "SDE4_02_MATRIX_VERSION": "MATRIX_VERSION",
    "ControllerCapacityArm": "ControllerCapacityArm",
    "ControllerCapacityManifest": "ControllerCapacityManifest",
    "ControllerCapacityReport": "ControllerCapacityReport",
    "ControllerCapacityRow": "ControllerCapacityRow",
    "ControllerCapacityRung": "ControllerCapacityRung",
    "build_sde4_02_manifest": "build_manifest",
    "render_sde4_02_markdown": "render_markdown",
    "run_sde4_02_fixture_ladder": "run_fixture_ladder",
}

_LAZY_SLM135_EXPORTS = {
    "SLM135_MATRIX_SET": "MATRIX_SET",
    "SLM135_MATRIX_VERSION": "MATRIX_VERSION",
    "Slm135Arm": "Slm135Arm",
    "Slm135Manifest": "Slm135Manifest",
    "Slm135Report": "Slm135Report",
    "Slm135Row": "Slm135Row",
    "build_slm135_manifest": "build_manifest",
    "render_slm135_markdown": "render_markdown",
    "run_slm135_fixture_matrix": "run_fixture_matrix",
}

_LAZY_SLM222_EXPORTS = {
    "SLM222_EXPERIMENT_ID": "EXPERIMENT_ID",
    "SLM222_MATRIX_SET": "MATRIX_SET",
    "SLM222_MATRIX_VERSION": "MATRIX_VERSION",
    "MuonBaselineArm": "MuonBaselineArm",
    "MuonBaselineReport": "MuonBaselineReport",
    "render_slm222_markdown": "render_markdown",
    "run_slm222_fixture": "run_muon_baseline_fixture",
}

_LAZY_SLM212_EXPORTS = {
    "SLM212_EXPERIMENT_ID": "EXPERIMENT_ID",
    "SLM212_MATRIX_SET": "MATRIX_SET",
    "SLM212_MATRIX_VERSION": "MATRIX_VERSION",
    "SLM212_ARM_NAMES": "ARM_NAMES",
    "DebtRoutingExample": "DebtRoutingExample",
    "DebtRoutingArmResult": "DebtRoutingArmResult",
    "DebtRoutingMatrixManifest": "DebtRoutingMatrixManifest",
    "build_slm212_synthetic_routing_examples": "build_synthetic_routing_examples",
    "build_slm212_matrix_manifest": "build_matrix_manifest",
    "run_slm212_fixture_matrix": "run_fixture_matrix",
    "render_slm212_markdown": "render_markdown",
    "validate_slm212_manifest": "validate_manifest",
}

_LAZY_SLM189_EXPORTS = {
    "SLM189_EXPERIMENT_ID": "EXPERIMENT_ID",
    "SLM189_MATRIX_SET": "MATRIX_SET",
    "SLM189_MATRIX_VERSION": "MATRIX_VERSION",
    "SLM189_ARM_NAMES": "ARM_NAMES",
    "BridgePlannerCase": "BridgePlannerCase",
    "BridgePlannerArmSummary": "BridgePlannerArmSummary",
    "BridgePlannerManifest": "BridgePlannerManifest",
    "build_slm189_exact_fixture_targets": "build_exact_fixture_targets",
    "build_slm189_synthetic_scale_targets": "build_synthetic_scale_targets",
    "run_slm189_fixture": "run_bridge_planner_fixture",
    "render_slm189_markdown": "render_markdown",
    "validate_slm189_manifest": "validate_manifest",
}

_LAZY_SLM190_EXPORTS = {
    "SLM190_EXPERIMENT_ID": "EXPERIMENT_ID",
    "SLM190_MATRIX_SET": "MATRIX_SET",
    "SLM190_MATRIX_VERSION": "MATRIX_VERSION",
    "SLM190_ARM_NAMES": "ARM_NAMES",
    "ExactFlowCase": "ExactFlowCase",
    "ObjectiveComparisonRow": "ObjectiveComparisonRow",
    "LumpabilityCase": "LumpabilityCase",
    "ExactFlowReport": "ExactFlowReport",
    "build_toy_layout_adapter": "build_toy_layout_adapter",
    "build_choice_sequence_adapter": "build_choice_sequence_adapter",
    "build_canonical_edit_adapter": "build_canonical_edit_adapter",
    "run_slm190_fixture": "run_exact_flow_fixture",
    "render_slm190_markdown": "render_markdown",
    "validate_slm190_report": "validate_report",
}

_LAZY_SLM191_EXPORTS = {
    "SLM191_EXPERIMENT_ID": "EXPERIMENT_ID",
    "SLM191_MATRIX_SET": "MATRIX_SET",
    "SLM191_MATRIX_VERSION": "MATRIX_VERSION",
    "SLM191_ARM_NAMES": "ARM_NAMES",
    "TerminationTargetRowV1": "TerminationTargetRowV1",
    "TerminationCase": "TerminationCase",
    "TerminationArmSummary": "TerminationArmSummary",
    "TerminationManifestV1": "TerminationManifestV1",
    "run_slm191_fixture": "run_termination_matrix",
    "render_slm191_markdown": "render_markdown",
    "validate_slm191_manifest": "validate_manifest",
}

_LAZY_SLM262_EXPORTS = {
    "SLM262_EXPERIMENT_ID": "EXPERIMENT_ID",
    "SLM262_MATRIX_SET": "MATRIX_SET",
    "SLM262_MATRIX_VERSION": "MATRIX_VERSION",
    "AcceleratorRunManifestV1": "AcceleratorRunManifestV1",
    "build_slm262_default_manifest": "build_default_manifest",
    "run_slm262_local_smoke": "run_local_smoke",
}


def __getattr__(name: str):
    if name in _LAZY_B3_EXPORTS:
        from slm_training.harnesses.experiments import b3_capacity_v2

        value = getattr(b3_capacity_v2, _LAZY_B3_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _LAZY_SDE4_02_EXPORTS:
        from slm_training.harnesses.experiments import sde4_02_min_controller_capacity

        value = getattr(sde4_02_min_controller_capacity, _LAZY_SDE4_02_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _LAZY_SLM135_EXPORTS:
        from slm_training.harnesses.experiments import slm135_trailed_assumptions_ablation

        value = getattr(
            slm135_trailed_assumptions_ablation, _LAZY_SLM135_EXPORTS[name]
        )
        globals()[name] = value
        return value
    if name in _LAZY_SLM222_EXPORTS:
        from slm_training.harnesses.experiments import slm222_muon_baseline

        value = getattr(slm222_muon_baseline, _LAZY_SLM222_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _LAZY_SLM212_EXPORTS:
        from slm_training.harnesses.experiments import slm212_debt_routing

        value = getattr(slm212_debt_routing, _LAZY_SLM212_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _LAZY_SLM189_EXPORTS:
        from slm_training.harnesses.experiments import slm189_bridge_planner

        value = getattr(slm189_bridge_planner, _LAZY_SLM189_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _LAZY_SLM190_EXPORTS:
        from slm_training.harnesses.experiments import slm190_exact_flow

        value = getattr(slm190_exact_flow, _LAZY_SLM190_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _LAZY_SLM191_EXPORTS:
        from slm_training.harnesses.experiments import slm191_termination_matrix

        value = getattr(slm191_termination_matrix, _LAZY_SLM191_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _LAZY_SLM262_EXPORTS:
        from slm_training.harnesses.experiments import slm262_gpu_reference

        value = getattr(slm262_gpu_reference, _LAZY_SLM262_EXPORTS[name])
        globals()[name] = value
        return value
    if name in _LAZY_LADDER_EXPORTS:
        from slm_training.harnesses.experiments import ladder

        value = getattr(ladder, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "AST_SKETCH_RETRIEVAL_ID",
    "AST_SKETCH_RETRIEVAL_MATRIX_SET",
    "AST_SKETCH_RETRIEVAL_MATRIX_VERSION",
    "B3_CAPACITY_V2_ID",
    "B3_CAPACITY_V2_MATRIX_SET",
    "B3_CAPACITY_V2_MATRIX_VERSION",
    "B3CapacityV2Arm",
    "B3CapacityV2Manifest",
    "B3CapacityV2Report",
    "B3CapacityV2Row",
    "SDE4_02_COMPETENCE_TARGET",
    "SDE4_02_MATRIX_SET",
    "SDE4_02_MATRIX_VERSION",
    "ControllerCapacityArm",
    "ControllerCapacityManifest",
    "ControllerCapacityReport",
    "ControllerCapacityRow",
    "ControllerCapacityRung",
    "SLM135_MATRIX_SET",
    "SLM135_MATRIX_VERSION",
    "Slm135Arm",
    "Slm135Manifest",
    "Slm135Report",
    "Slm135Row",
    "SLM147_MATRIX_SET",
    "SLM147_MATRIX_VERSION",
    "X22_RETRIEVAL_ID",
    "PrototypeCandidate",
    "PrototypeIndexEntry",
    "RetrievalStrategy",
    "Slm147Arm",
    "Slm147Manifest",
    "Slm147Record",
    "Slm147Report",
    "Slm147Row",
    "ValidPrototypeIndex",
    "ValidPrototypeRetriever",
    "build_slm147_manifest",
    "build_prototype_index",
    "render_slm147_markdown",
    "run_slm147_fixture_matrix",
    "validate_slm147_manifest",
    "SLM148_MATRIX_SET",
    "SLM148_MATRIX_VERSION",
    "X22_CONFLICT_CAMPAIGN_ID",
    "SeedStrategy",
    "Slm148Manifest",
    "Slm148Record",
    "Slm148RecoveryArm",
    "Slm148Report",
    "Slm148Row",
    "Slm148SearchConfig",
    "Slm148SeedArm",
    "build_slm148_manifest",
    "render_slm148_markdown",
    "run_slm148_fixture_matrix",
    "validate_slm148_manifest",
    "SLM222_EXPERIMENT_ID",
    "SLM222_MATRIX_SET",
    "SLM222_MATRIX_VERSION",
    "MuonBaselineArm",
    "MuonBaselineReport",
    "render_slm222_markdown",
    "run_slm222_fixture",
    "SLM212_EXPERIMENT_ID",
    "SLM212_MATRIX_SET",
    "SLM212_MATRIX_VERSION",
    "SLM212_ARM_NAMES",
    "DebtRoutingExample",
    "DebtRoutingArmResult",
    "DebtRoutingMatrixManifest",
    "build_slm212_synthetic_routing_examples",
    "build_slm212_matrix_manifest",
    "run_slm212_fixture_matrix",
    "render_slm212_markdown",
    "validate_slm212_manifest",
    "SLM189_EXPERIMENT_ID",
    "SLM189_MATRIX_SET",
    "SLM189_MATRIX_VERSION",
    "SLM189_ARM_NAMES",
    "BridgePlannerCase",
    "BridgePlannerArmSummary",
    "BridgePlannerManifest",
    "build_slm189_exact_fixture_targets",
    "build_slm189_synthetic_scale_targets",
    "run_slm189_fixture",
    "render_slm189_markdown",
    "validate_slm189_manifest",
    "SLM190_EXPERIMENT_ID",
    "SLM190_MATRIX_SET",
    "SLM190_MATRIX_VERSION",
    "SLM190_ARM_NAMES",
    "ExactFlowCase",
    "ObjectiveComparisonRow",
    "LumpabilityCase",
    "ExactFlowReport",
    "build_toy_layout_adapter",
    "build_choice_sequence_adapter",
    "build_canonical_edit_adapter",
    "run_slm190_fixture",
    "render_slm190_markdown",
    "validate_slm190_report",
    "SLM191_EXPERIMENT_ID",
    "SLM191_MATRIX_SET",
    "SLM191_MATRIX_VERSION",
    "SLM191_ARM_NAMES",
    "TerminationTargetRowV1",
    "TerminationCase",
    "TerminationArmSummary",
    "TerminationManifestV1",
    "run_slm191_fixture",
    "render_slm191_markdown",
    "validate_slm191_manifest",
    "SLM262_EXPERIMENT_ID",
    "SLM262_MATRIX_SET",
    "SLM262_MATRIX_VERSION",
    "AcceleratorRunManifestV1",
    "build_slm262_default_manifest",
    "run_slm262_local_smoke",
    "DATA_SAMPLING_ARMS",
    "RETRIEVAL_MODES",
    "AstSketchRetrievalArm",
    "AstSketchRetrievalManifest",
    "AstSketchRetrievalReport",
    "AstSketchRetrievalRow",
    "AstTrainingSketchV1",
    "ChoiceRetrievalExemplarV1",
    "DataSampling",
    "RetrievalMode",
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
    "build_ast_sketch_retrieval_manifest",
    "build_ast_training_sketch",
    "build_b3_capacity_v2_manifest",
    "build_choice_exemplar_bank",
    "build_choice_retrieval_exemplar",
    "build_causal_peft_ftpo_manifest",
    "build_corruption_curriculum_manifest",
    "build_sde4_02_manifest",
    "format_choice_exemplar_context",
    "nearest_choice_exemplars",
    "random_choice_exemplars",
    "render_ast_sketch_retrieval_markdown",
    "render_sde4_02_markdown",
    "run_ast_sketch_retrieval_fixture_matrix",
    "run_sde4_02_fixture_ladder",
    "build_slm135_manifest",
    "render_slm135_markdown",
    "run_slm135_fixture_matrix",
    "validate_ast_sketch_retrieval_manifest",
    "build_e228_exposure_ladder",
    "build_e228_recipe_config",
    "build_external_ceiling_manifest",
    "build_pretrained_denoiser_activation_manifest",
    "build_proxy_metric_calibration_manifest",
    "build_scaffold_distillation_activation_manifest",
    "build_teacher_paraphrase_activation_manifest",
    "render_b3_capacity_v2_markdown",
    "render_causal_peft_ftpo_markdown",
    "render_corruption_curriculum_markdown",
    "render_e228_exposure_markdown",
    "render_external_ceiling_markdown",
    "run_e228_fixture_ladder",
    "run_b3_capacity_v2_fixture_ladder",
    "run_fixture_curriculum",
    "run_fixture_ftpo",
    "run_external_ceiling_fixture_matrix",
    "validate_b3_capacity_v2_manifest",
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
    "EFS_CAMPAIGN_ID",
    "CampaignHypothesisSpec",
    "CampaignManifestV1",
    "EvidenceFirstSemanticSynthesisV1",
    "build_default_campaign_manifest",
    "load_efs_campaign_manifest",
    "render_efs_evidence_dot",
    "render_efs_synthesis_markdown",
    "render_efs_evidence_mermaid",
    "save_efs_campaign_manifest",
    "save_efs_synthesis",
    "synthesize_efs_campaign",
    "validate_efs_synthesis",
]
