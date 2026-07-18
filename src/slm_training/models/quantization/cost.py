"""Physical-cost ledger for quantized models."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from slm_training.dsl.analysis.arity.certificate import (
    ConstraintFrame,
    EstimatedEvidence,
    EvidenceKind,
)
from slm_training.models.quantization.observers import dtype_bits

if TYPE_CHECKING:
    from slm_training.models.quantization.formats import QuantFormat


@dataclass
class TensorCost:
    """Cost breakdown for one tensor or parameter group."""

    name: str
    shape: tuple[int, ...]
    format_id: str
    group_size: int
    numel: int
    level_count: int
    ideal_bits: float
    empirical_entropy_bits: float | None
    physical_slot_bits: int
    physical_weight_bytes: int
    scale_bytes: int
    zero_point_bytes: int
    bias_bytes: int
    other_bytes: int
    metadata_overhead_bytes: int
    total_bytes: int
    activation_bytes: int = 0
    scratch_bytes: int = 0
    exclusion_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "format_id": self.format_id,
            "group_size": self.group_size,
            "numel": self.numel,
            "level_count": self.level_count,
            "ideal_bits": self.ideal_bits,
            "empirical_entropy_bits": self.empirical_entropy_bits,
            "physical_slot_bits": self.physical_slot_bits,
            "physical_weight_bytes": self.physical_weight_bytes,
            "scale_bytes": self.scale_bytes,
            "zero_point_bytes": self.zero_point_bytes,
            "bias_bytes": self.bias_bytes,
            "other_bytes": self.other_bytes,
            "metadata_overhead_bytes": self.metadata_overhead_bytes,
            "total_bytes": self.total_bytes,
            "activation_bytes": self.activation_bytes,
            "scratch_bytes": self.scratch_bytes,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass
class FormatCostReport:
    """Aggregated cost for one quantization format across a model."""

    format_id: str
    tensors: list[TensorCost] = field(default_factory=list)

    @property
    def numel(self) -> int:
        return sum(t.numel for t in self.tensors)

    @property
    def ideal_bits(self) -> float:
        return sum(t.ideal_bits for t in self.tensors)

    @property
    def physical_weight_bytes(self) -> int:
        return sum(t.physical_weight_bytes for t in self.tensors)

    @property
    def scale_bytes(self) -> int:
        return sum(t.scale_bytes for t in self.tensors)

    @property
    def zero_point_bytes(self) -> int:
        return sum(t.zero_point_bytes for t in self.tensors)

    @property
    def bias_bytes(self) -> int:
        return sum(t.bias_bytes for t in self.tensors)

    @property
    def other_bytes(self) -> int:
        return sum(t.other_bytes for t in self.tensors)

    @property
    def metadata_overhead_bytes(self) -> int:
        return sum(t.metadata_overhead_bytes for t in self.tensors)

    @property
    def total_bytes(self) -> int:
        return sum(t.total_bytes for t in self.tensors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_id": self.format_id,
            "numel": self.numel,
            "ideal_bits": self.ideal_bits,
            "physical_weight_bytes": self.physical_weight_bytes,
            "scale_bytes": self.scale_bytes,
            "zero_point_bytes": self.zero_point_bytes,
            "bias_bytes": self.bias_bytes,
            "other_bytes": self.other_bytes,
            "metadata_overhead_bytes": self.metadata_overhead_bytes,
            "total_bytes": self.total_bytes,
            "tensors": [t.as_dict() for t in self.tensors],
        }


@dataclass
class PhysicalCostLedger:
    """Whole-model physical-cost ledger."""

    formats: dict[str, FormatCostReport] = field(default_factory=dict)
    unquantized_bytes: int = 0
    metadata_overhead_bytes: int = 0
    alignment_overhead_bytes: int = 0
    activation_bytes: int = 0
    kv_bytes: int = 0
    scratch_bytes: int = 0
    checkpoint_bytes: int = 0
    resident_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    def add_tensor(self, tensor_cost: TensorCost) -> None:
        fmt_id = tensor_cost.format_id
        if fmt_id not in self.formats:
            self.formats[fmt_id] = FormatCostReport(format_id=fmt_id)
        self.formats[fmt_id].tensors.append(tensor_cost)

    def total(self) -> int:
        return sum(f.total_bytes for f in self.formats.values()) + self.unquantized_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "formats": {k: v.as_dict() for k, v in self.formats.items()},
            "unquantized_bytes": self.unquantized_bytes,
            "metadata_overhead_bytes": self.metadata_overhead_bytes,
            "alignment_overhead_bytes": self.alignment_overhead_bytes,
            "activation_bytes": self.activation_bytes,
            "kv_bytes": self.kv_bytes,
            "scratch_bytes": self.scratch_bytes,
            "checkpoint_bytes": self.checkpoint_bytes,
            "resident_bytes": self.resident_bytes,
            "total_bytes": self.total(),
            "notes": self.notes,
        }


def _packing_bytes(numel: int, slot_bits: int, group_size: int) -> int:
    """Packed weight bytes with per-group padding."""
    if numel == 0:
        return 0
    groups = math.ceil(numel / group_size)
    slots_per_group = group_size
    bytes_per_group = math.ceil(slots_per_group * slot_bits / 8)
    return groups * bytes_per_group


def _scale_bytes(
    numel: int,
    group_size: int,
    scale_dtype: str,
    zero_point_dtype: str | None,
) -> tuple[int, int]:
    groups = max(1, math.ceil(numel / group_size))
    scale_bits = dtype_bits(scale_dtype)
    scale_bytes = groups * scale_bits // 8
    zp_bytes = 0
    if zero_point_dtype:
        zp_bytes = groups * dtype_bits(zero_point_dtype) // 8
    return scale_bytes, zp_bytes


def _empirical_entropy_bits(tensor: torch.Tensor, levels: tuple[float, ...] | None) -> float | None:
    """Empirical Shannon entropy of the quantized symbol distribution."""
    if tensor.numel() == 0 or levels is None:
        return None
    flat = tensor.detach().cpu().flatten()
    level_t = torch.tensor(levels, dtype=flat.dtype)
    idx = flat.sub(level_t.view(-1, 1)).abs().argmin(dim=0)
    counts = torch.bincount(idx, minlength=len(levels)).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    if probs.numel() == 0:
        return None
    return float(-(probs * torch.log2(probs)).sum().item())


def compute_tensor_cost(
    name: str,
    tensor: torch.Tensor,
    fmt: QuantFormat,
    group_size: int | None = None,
    bias: torch.Tensor | None = None,
    activation_shape: tuple[int, ...] | None = None,
    exclusion_reason: str | None = None,
    metadata_overhead_bytes: int = 32,
) -> TensorCost:
    """Compute a physical-cost breakdown for a single tensor."""
    gsize = group_size if group_size is not None else fmt.group_size
    numel = tensor.numel()
    level_count = fmt.level_count
    ideal_bits = numel * fmt.ideal_symbol_bits

    levels = fmt.learned_levels if fmt.is_learned else fmt.weight_levels
    empirical_bits = _empirical_entropy_bits(tensor, levels) if levels else None

    if fmt.format_id in ("fp16", "bf16") or exclusion_reason:
        physical_weight_bytes = numel * fmt.physical_slot_bits // 8
        scale_bytes = 0
        zp_bytes = 0
    else:
        physical_weight_bytes = _packing_bytes(numel, fmt.physical_slot_bits, gsize)
        scale_bytes, zp_bytes = _scale_bytes(
            numel,
            gsize,
            fmt.scale_dtype,
            fmt.zero_point_dtype,
        )

    bias_bytes = 0
    if bias is not None and fmt.bias_dtype:
        bias_bytes = bias.numel() * dtype_bits(fmt.bias_dtype) // 8

    other_bytes = 0
    activation_bytes = 0
    if activation_shape:
        activation_bytes = math.prod(activation_shape) * dtype_bits(fmt.activation_dtype) // 8

    total = (
        physical_weight_bytes
        + scale_bytes
        + zp_bytes
        + bias_bytes
        + other_bytes
        + metadata_overhead_bytes
    )

    return TensorCost(
        name=name,
        shape=tuple(tensor.shape),
        format_id=fmt.format_id,
        group_size=gsize,
        numel=numel,
        level_count=level_count,
        ideal_bits=ideal_bits,
        empirical_entropy_bits=empirical_bits,
        physical_slot_bits=fmt.physical_slot_bits,
        physical_weight_bytes=physical_weight_bytes,
        scale_bytes=scale_bytes,
        zero_point_bytes=zp_bytes,
        bias_bytes=bias_bytes,
        other_bytes=other_bytes,
        metadata_overhead_bytes=metadata_overhead_bytes,
        total_bytes=total,
        activation_bytes=activation_bytes,
        exclusion_reason=exclusion_reason,
    )


def build_model_ledger(
    model: Any,
    format_map: dict[str, QuantFormat],
    default_format: QuantFormat,
    activation_batch_seq: tuple[int, int] = (1, 256),
    d_model: int = 128,
    alignment_bytes: int = 64,
    metadata_overhead_per_tensor: int = 32,
) -> PhysicalCostLedger:
    """Walk ``model.parameters()`` and build a whole-model physical-cost ledger."""
    ledger = PhysicalCostLedger()
    seen: dict[int, str] = {}
    for name, param in model.named_parameters():
        storage_id = param.untyped_storage().data_ptr() if hasattr(param, "untyped_storage") else id(param)
        if storage_id in seen:
            ledger.notes.append(
                f"Tied/shared parameter detected: {name} shares storage with {seen[storage_id]}"
            )
            continue
        seen[storage_id] = name

        fmt = format_map.get(name, default_format)
        exclusion = None
        if any(excluded in name.lower() for excluded in ("embed", "lm_head", "norm", "bias")):
            fmt = fp16_format(group_size=default_format.group_size)
            exclusion = "embedding/norm/head/bias excluded by policy"

        bias = None
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        try:
            parent = model.get_submodule(parent_name)
            if isinstance(parent, torch.nn.Linear) and parent.bias is not None:
                bias = parent.bias
        except Exception:
            pass

        activation_shape = activation_batch_seq + (d_model,)
        cost = compute_tensor_cost(
            name,
            param,
            fmt,
            group_size=fmt.group_size,
            bias=bias,
            activation_shape=activation_shape,
            exclusion_reason=exclusion,
            metadata_overhead_bytes=metadata_overhead_per_tensor,
        )
        ledger.add_tensor(cost)
        if exclusion:
            ledger.unquantized_bytes += cost.total_bytes

    # Activation + KV + scratch estimates (analytical placeholders, not measurements).
    ledger.activation_bytes = math.prod(activation_batch_seq) * d_model * dtype_bits(default_format.activation_dtype) // 8
    ledger.kv_bytes = 2 * math.prod(activation_batch_seq) * d_model * dtype_bits(default_format.activation_dtype) // 8
    ledger.scratch_bytes = ledger.total() // 8
    ledger.metadata_overhead_bytes = len(list(model.named_parameters())) * metadata_overhead_per_tensor
    ledger.alignment_overhead_bytes = alignment_bytes
    ledger.checkpoint_bytes = ledger.total() + ledger.metadata_overhead_bytes + ledger.alignment_overhead_bytes
    ledger.resident_bytes = ledger.total() + ledger.activation_bytes + ledger.kv_bytes + ledger.scratch_bytes
    return ledger


def physical_cost_evidence(
    ledger: PhysicalCostLedger,
    grammar_hash: str,
    dataset_ids: tuple[str, ...] = (),
    checkpoint_ids: tuple[str, ...] = (),
    sample_count: int = 1,
) -> EstimatedEvidence:
    """Return a CAP0-04 EstimatedEvidence object validating the ledger."""
    digest = hashlib.sha256(
        str(ledger.as_dict()).encode("utf-8")
    ).hexdigest()[:16]
    frame = ConstraintFrame(
        grammar_hash=grammar_hash,
        parser_version="cap3-01",
        codec_version="cap3-01",
        state_signature_version="cap3-01",
        generation_order="none",
        ast_bounds={},
        scope_bounds={},
        template_classes=(),
        latent_role=None,
        dimensions=None,
        noise_model="reference_quantization",
        packing_assumption="per_group_pad",
    )
    return EstimatedEvidence(
        evidence_kind=EvidenceKind.ESTIMATED,
        constraints=frame,
        dataset_ids=dataset_ids,
        trace_ids=(),
        checkpoint_ids=checkpoint_ids,
        sample_count=sample_count,
        sampling_design="analytical_physical_cost",
        coverage={"formats": len(ledger.formats), "total_bytes": ledger.total()},
        estimator="physical_cost_ledger",
        confidence_interval=None,
        tail_metric={"digest": digest},
    )


# Circular imports: keep this local import.
def fp16_format(group_size: int = 128) -> QuantFormat:
    from slm_training.models.quantization.formats import fp16_format as _fp16

    return _fp16(group_size=group_size)
