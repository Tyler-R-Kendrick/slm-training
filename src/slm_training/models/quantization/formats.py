"""Quantization format descriptors and kernel-capability registry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantFormat:
    """Versioned descriptor for a weight/activation representation."""

    format_id: str
    weight_levels: tuple[float, ...] | Literal["learned"]
    nominal_symbol_bits: float
    physical_slot_bits: int
    group_size: int
    scale_dtype: str
    zero_point_dtype: str | None
    bias_dtype: str | None
    activation_dtype: str
    accumulation_dtype: str
    packing_layout: str
    supports_exact_zero: bool
    entropy_coding: str | None
    kernel_id: str | None
    # Learned codebooks only: final reconstruction levels and zero constraint.
    learned_levels: tuple[float, ...] | None = None
    learned_zero_constrained: bool = False

    def __post_init__(self) -> None:
        if self.weight_levels == "learned":
            if self.learned_levels is None or len(self.learned_levels) < 2:
                raise ValueError("learned formats must provide at least two learned_levels")
        if self.nominal_symbol_bits < 0:
            raise ValueError("nominal_symbol_bits must be non-negative")
        if self.physical_slot_bits <= 0:
            raise ValueError("physical_slot_bits must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")

    @property
    def is_learned(self) -> bool:
        return self.weight_levels == "learned"

    @property
    def level_count(self) -> int:
        if self.is_learned:
            return len(self.learned_levels) if self.learned_levels else 0
        return len(self.weight_levels)

    @property
    def ideal_symbol_bits(self) -> float:
        """Analytical information-theoretic lower bound."""
        if self.is_learned and self.learned_levels:
            return math.log2(max(1, len(self.learned_levels)))
        if isinstance(self.weight_levels, tuple):
            return math.log2(max(1, len(self.weight_levels)))
        return self.nominal_symbol_bits


@dataclass(frozen=True)
class KernelCapability:
    """Which execution paths exist for a format."""

    reference_pytorch: bool = True
    cpu_optimized: bool = False
    cuda: bool = False
    zero_skipping: bool = False
    packed_inference: bool = False
    batch_constraints: tuple[int, ...] = ()
    supported_group_sizes: tuple[int, ...] = ()
    notes: str = ""


KERNEL_REGISTRY: dict[str, KernelCapability] = {
    "fp16": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=True,
        cuda=True,
        packed_inference=True,
        notes="Baseline control; no quantization.",
    ),
    "bf16": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=True,
        cuda=True,
        packed_inference=True,
        notes="Baseline control; no quantization.",
    ),
    "int8": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=True,
        cuda=True,
        zero_skipping=False,
        packed_inference=True,
        supported_group_sizes=(128, 256, 512),
        notes="INT8 per-group symmetric reference path.",
    ),
    "int4": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=False,
        cuda=True,
        zero_skipping=False,
        packed_inference=True,
        supported_group_sizes=(128, 256),
        notes="INT4 reference path only on CPU; CUDA kernel exists but not claimed here.",
    ),
    "binary": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=False,
        cuda=False,
        zero_skipping=False,
        packed_inference=True,
        supported_group_sizes=(128,),
        notes="Binary {-1,+1} reference path; optimized kernel not in scope.",
    ),
    "ternary": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=False,
        cuda=False,
        zero_skipping=True,
        packed_inference=True,
        supported_group_sizes=(128,),
        notes="Ternary {-1,0,+1} reference path; zero skipping is semantic, not kernel.",
    ),
    "symmetric_four_level": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=False,
        cuda=False,
        packed_inference=True,
        supported_group_sizes=(128,),
        notes="Fixed four-level grid reference path.",
    ),
    "learned_four_level_zero": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=False,
        cuda=False,
        zero_skipping=True,
        packed_inference=True,
        supported_group_sizes=(128,),
        notes="Learned four-level codebook containing exact zero.",
    ),
    "binary_plus_mask": KernelCapability(
        reference_pytorch=True,
        cpu_optimized=False,
        cuda=False,
        zero_skipping=True,
        packed_inference=True,
        supported_group_sizes=(128,),
        notes="Sign bit + explicit zero mask; 2 physical bits per weight.",
    ),
    "residual_binary_plane": KernelCapability(
        reference_pytorch=False,
        cpu_optimized=False,
        cuda=False,
        packed_inference=False,
        notes="Descriptor-only residual plane; execution in CAP4.",
    ),
    "residual_ternary_plane": KernelCapability(
        reference_pytorch=False,
        cpu_optimized=False,
        cuda=False,
        packed_inference=False,
        notes="Descriptor-only residual plane; execution in CAP4.",
    ),
}


def _int_symmetric_levels(bits: int) -> tuple[float, ...]:
    # Exclude the saturated -2^(bits-1) bucket for symmetric quantization.
    return tuple(float(i) for i in range(-(2 ** (bits - 1)) + 1, 2 ** (bits - 1)))


def fp16_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="fp16",
        weight_levels=(),
        nominal_symbol_bits=16.0,
        physical_slot_bits=16,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="fp16_dense",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="fp16",
    )


def bf16_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="bf16",
        weight_levels=(),
        nominal_symbol_bits=16.0,
        physical_slot_bits=16,
        group_size=group_size,
        scale_dtype="bf16",
        zero_point_dtype=None,
        bias_dtype="bf16",
        activation_dtype="bf16",
        accumulation_dtype="fp32",
        packing_layout="bf16_dense",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="bf16",
    )


def int8_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="int8",
        weight_levels=_int_symmetric_levels(8),
        nominal_symbol_bits=8.0,
        physical_slot_bits=8,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="int8_packed",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="int8",
    )


def int4_format(group_size: int = 128) -> QuantFormat:
    # Symmetric INT4: -7 .. 7 (15 levels), one code point unused for scale symmetry.
    return QuantFormat(
        format_id="int4",
        weight_levels=tuple(float(i) for i in range(-7, 8)),
        nominal_symbol_bits=4.0,
        physical_slot_bits=4,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="int4_packed",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="int4",
    )


def binary_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="binary",
        weight_levels=(-1.0, 1.0),
        nominal_symbol_bits=1.0,
        physical_slot_bits=1,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="binary_sign_packed",
        supports_exact_zero=False,
        entropy_coding=None,
        kernel_id="binary",
    )


def ternary_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="ternary",
        weight_levels=(-1.0, 0.0, 1.0),
        nominal_symbol_bits=math.log2(3),
        physical_slot_bits=2,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="ternary_two_bit_packed",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="ternary",
    )


def symmetric_four_level_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="symmetric_four_level",
        weight_levels=(-3.0, -1.0, 1.0, 3.0),
        nominal_symbol_bits=2.0,
        physical_slot_bits=2,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="four_level_two_bit_packed",
        supports_exact_zero=False,
        entropy_coding=None,
        kernel_id="symmetric_four_level",
    )


def learned_four_level_zero_format(
    levels: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0),
    group_size: int = 128,
) -> QuantFormat:
    if 0.0 not in levels:
        raise ValueError("learned_four_level_zero must contain an exact zero level")
    if len(levels) != 4:
        raise ValueError("learned_four_level_zero requires exactly four levels")
    return QuantFormat(
        format_id="learned_four_level_zero",
        weight_levels="learned",
        nominal_symbol_bits=math.log2(4),
        physical_slot_bits=2,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="learned_four_level_two_bit_packed",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="learned_four_level_zero",
        learned_levels=tuple(sorted(set(levels))),
        learned_zero_constrained=True,
    )


def binary_plus_mask_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="binary_plus_mask",
        weight_levels=(-1.0, 0.0, 1.0),
        nominal_symbol_bits=2.0,
        physical_slot_bits=2,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="binary_sign_plus_mask_packed",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="binary_plus_mask",
    )


def residual_binary_plane_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="residual_binary_plane",
        weight_levels=(-1.0, 1.0),
        nominal_symbol_bits=1.0,
        physical_slot_bits=1,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="residual_binary_plane",
        supports_exact_zero=False,
        entropy_coding=None,
        kernel_id=None,
    )


def residual_ternary_plane_format(group_size: int = 128) -> QuantFormat:
    return QuantFormat(
        format_id="residual_ternary_plane",
        weight_levels=(-1.0, 0.0, 1.0),
        nominal_symbol_bits=math.log2(3),
        physical_slot_bits=2,
        group_size=group_size,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="residual_ternary_plane",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id=None,
    )
