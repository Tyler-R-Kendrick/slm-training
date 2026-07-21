"""Exact finite-state CTMC reference for compiler-certified legal edits."""

from __future__ import annotations

from slm_training.flow.reference.adapter import (
    ActionRef,
    StateAdapter,
    StateRef,
)
from slm_training.flow.reference.enumerate import ExactEnumerator, StateGraph
from slm_training.flow.reference.generator import (
    GeneratorBuilder,
    RateFn,
    apply_matrix_exp_col,
    apply_matrix_exp_row,
    build_bridge_rate_fn,
    build_distance_rate_fn,
    build_doob_bridge_rate_fn,
    build_uniform_rate_fn,
    check_generator,
    endpoint_distribution,
    forward_equation,
    matrix_exponential,
)
from slm_training.flow.reference.lumpability import (
    LUMPABLE,
    NOT_LUMPABLE,
    UNKNOWN_NUMERIC,
    build_quotient_matrix,
    classify_partition,
    is_ordinary_lumpable,
    is_strongly_lumpable,
)
from slm_training.flow.reference.row import FlowTargetRowV1
from slm_training.flow.reference.sampler import (
    FixedGridSampler,
    GillespieSampler,
)
from slm_training.flow.reference.trajectory import (
    FlowSampleV1,
    FlowTrajectoryV1,
)

__all__ = [
    "ActionRef",
    "StateAdapter",
    "StateRef",
    "ExactEnumerator",
    "StateGraph",
    "GeneratorBuilder",
    "RateFn",
    "build_bridge_rate_fn",
    "build_distance_rate_fn",
    "build_doob_bridge_rate_fn",
    "build_uniform_rate_fn",
    "check_generator",
    "endpoint_distribution",
    "forward_equation",
    "matrix_exponential",
    "LUMPABLE",
    "NOT_LUMPABLE",
    "UNKNOWN_NUMERIC",
    "apply_matrix_exp_col",
    "apply_matrix_exp_row",
    "build_quotient_matrix",
    "classify_partition",
    "is_ordinary_lumpable",
    "is_strongly_lumpable",
    "FlowTargetRowV1",
    "FixedGridSampler",
    "GillespieSampler",
    "FlowSampleV1",
    "FlowTrajectoryV1",
]
