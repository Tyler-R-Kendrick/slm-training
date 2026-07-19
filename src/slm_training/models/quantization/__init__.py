"""Reference low-bit quantizers and an honest physical-cost ledger (CAP3-01)."""

from __future__ import annotations

from slm_training.models.quantization.convert import (
    ConversionRecord,
    QuantizationPolicy,
    convert_twotower,
    select_parameter_groups,
)
from slm_training.models.quantization.cost import (
    FormatCostReport,
    PhysicalCostLedger,
    TensorCost,
    build_model_ledger,
    compute_tensor_cost,
    physical_cost_evidence,
)
from slm_training.models.quantization.diagnostics import TensorDiagnostics, diagnose_tensor
from slm_training.models.quantization.fake_quant import fake_quantize, fake_quantize_weight
from slm_training.models.quantization.formats import (
    KERNEL_REGISTRY,
    QuantFormat,
    binary_format,
    binary_plus_mask_format,
    bf16_format,
    block_sparse_ternary_format,
    fp16_format,
    int4_format,
    int8_format,
    learned_four_level_zero_format,
    residual_binary_plane_format,
    residual_ternary_plane_format,
    state_family_expert_format,
    symmetric_four_level_format,
    ternary_format,
)
from slm_training.models.quantization.observers import (
    observe_asymmetric_scale,
    observe_symmetric_scale,
)
from slm_training.models.quantization.adaptive_planes import (
    AdaptivePlaneRouteResult,
    AdaptivePlaneRoutingContext,
    PlaneRouter,
    PlaneScheduleSpec,
    PlaneScheduler,
    RuntimeDiagnostics,
    make_schedule_spec,
    oracle_min_planes,
)
from slm_training.models.quantization.block_sparsity import (
    BlockMaskedLinear,
    CompilerRoutedDenoiserTower,
    CompilerRoutedMLP,
    CompilerRoutedTransformerBlock,
    StateFamilyExpert,
    StateFamilyKey,
    StateFamilyRouter,
    compute_block_sparse_cost,
)
from slm_training.models.quantization.residual_planes import PlaneOutput, ResidualTritStack

__all__ = [
    "QuantFormat",
    "fp16_format",
    "bf16_format",
    "int8_format",
    "int4_format",
    "binary_format",
    "ternary_format",
    "symmetric_four_level_format",
    "learned_four_level_zero_format",
    "binary_plus_mask_format",
    "residual_binary_plane_format",
    "residual_ternary_plane_format",
    "PlaneOutput",
    "ResidualTritStack",
    "PlaneScheduleSpec",
    "PlaneScheduler",
    "PlaneRouter",
    "AdaptivePlaneRoutingContext",
    "AdaptivePlaneRouteResult",
    "RuntimeDiagnostics",
    "make_schedule_spec",
    "oracle_min_planes",
    "block_sparse_ternary_format",
    "state_family_expert_format",
    "KERNEL_REGISTRY",
    "StateFamilyKey",
    "StateFamilyRouter",
    "BlockMaskedLinear",
    "StateFamilyExpert",
    "CompilerRoutedMLP",
    "CompilerRoutedTransformerBlock",
    "CompilerRoutedDenoiserTower",
    "compute_block_sparse_cost",
    "observe_symmetric_scale",
    "observe_asymmetric_scale",
    "fake_quantize",
    "fake_quantize_weight",
    "TensorCost",
    "FormatCostReport",
    "PhysicalCostLedger",
    "build_model_ledger",
    "compute_tensor_cost",
    "physical_cost_evidence",
    "TensorDiagnostics",
    "diagnose_tensor",
    "ConversionRecord",
    "QuantizationPolicy",
    "select_parameter_groups",
    "convert_twotower",
]
