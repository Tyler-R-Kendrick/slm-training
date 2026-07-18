"""Disabled-by-default model conversion to reference quantizers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from slm_training.models.quantization.diagnostics import diagnose_tensor
from slm_training.models.quantization.fake_quant import fake_quantize_weight
from slm_training.models.quantization.formats import QuantFormat


@dataclass
class QuantizationPolicy:
    """Policy for applying quantization to a model."""

    default_format: QuantFormat
    # Module-path patterns -> format overrides.
    path_overrides: dict[str, QuantFormat] = field(default_factory=dict)
    # Module-type patterns -> format overrides (checked after path overrides).
    type_overrides: dict[str, QuantFormat] = field(default_factory=dict)
    # Explicit exclusion regex patterns for module paths.
    exclude_paths: list[str] = field(default_factory=list)
    # Exclude by type name substring.
    exclude_types: list[str] = field(
        default_factory=lambda: ["Embedding", "LayerNorm", "RMSNorm", "BatchNorm"]
    )
    # Exclude modules whose name contains these substrings.
    exclude_name_fragments: list[str] = field(
        default_factory=lambda: ["embed", "lm_head", "head", "norm", "bias"]
    )
    group_size: int | None = None
    symmetric: bool = True

    def format_for(self, name: str, module: torch.nn.Module) -> QuantFormat | None:
        for pattern, fmt in self.path_overrides.items():
            if re.search(pattern, name):
                return fmt
        for type_name, fmt in self.type_overrides.items():
            if type_name.lower() in type(module).__name__.lower():
                return fmt
        return self.default_format

    def exclusion_reason(self, name: str, module: torch.nn.Module) -> str | None:
        for pattern in self.exclude_paths:
            if re.search(pattern, name):
                return f"matches exclusion path {pattern!r}"
        type_name = type(module).__name__
        for excluded in self.exclude_types:
            if excluded.lower() in type_name.lower():
                return f"excluded type {type_name}"
        for fragment in self.exclude_name_fragments:
            if fragment.lower() in name.lower():
                return f"excluded name fragment {fragment!r}"
        return None


@dataclass
class ConversionRecord:
    """Record of what was quantized or excluded during conversion."""

    module_path: str
    param_name: str
    original_shape: tuple[int, ...]
    format_id: str
    group_size: int
    scale_shape: tuple[int, ...] | None
    mse: float
    max_error: float
    excluded_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "param_name": self.param_name,
            "original_shape": self.original_shape,
            "format_id": self.format_id,
            "group_size": self.group_size,
            "scale_shape": self.scale_shape,
            "mse": self.mse,
            "max_error": self.max_error,
            "excluded_reason": self.excluded_reason,
        }


def select_parameter_groups(
    model: torch.nn.Module,
    policy: QuantizationPolicy,
) -> list[tuple[str, torch.nn.Module, QuantFormat | None, str | None]]:
    """Return (path, module, format, exclusion_reason) for every quantizable module."""
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        reason = policy.exclusion_reason(name, module)
        fmt = None if reason else policy.format_for(name, module)
        selected.append((name, module, fmt, reason))
    return selected


def _detect_shared_storage(model: torch.nn.Module) -> dict[int, list[str]]:
    """Map storage pointers to Linear weight names to detect tied weights."""
    buckets: dict[int, list[str]] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            param = module.weight
            ptr = param.untyped_storage().data_ptr() if hasattr(param, "untyped_storage") else id(param)
            buckets.setdefault(ptr, []).append(f"{name}.weight")
    return buckets


def convert_twotower(
    model: torch.nn.Module,
    policy: QuantizationPolicy,
    *,
    fail_on_tied: bool = True,
    in_place: bool = False,
) -> tuple[torch.nn.Module, list[ConversionRecord]]:
    """Convert a TwoTower-compatible model using ``policy``.

    Returns the (possibly new) model and a list of conversion records.  When
    ``fail_on_tied`` is True and tied/shared storage is detected among
    quantizable Linear weights, conversion raises ``ValueError`` instead of
    silently duplicating storage.
    """
    records: list[ConversionRecord] = []

    if fail_on_tied:
        shared = _detect_shared_storage(model)
        tied = {ptr: names for ptr, names in shared.items() if len(names) > 1}
        if tied:
            msg = "; ".join(f"{names} share storage" for names in tied.values())
            raise ValueError(f"Refusing to quantize tied/shared weights: {msg}")

    target = model if in_place else _shallow_clone_model(model)
    groups = select_parameter_groups(target, policy)

    for path, module, fmt, reason in groups:
        if reason or fmt is None:
            records.append(
                ConversionRecord(
                    module_path=path,
                    param_name="weight",
                    original_shape=tuple(module.weight.shape),
                    format_id="fp16",
                    group_size=policy.default_format.group_size,
                    scale_shape=None,
                    mse=0.0,
                    max_error=0.0,
                    excluded_reason=reason,
                )
            )
            continue

        gsize = policy.group_size if policy.group_size is not None else fmt.group_size
        quantized, scale, _ = fake_quantize_weight(
            module.weight.data,
            fmt,
            group_size=gsize,
            symmetric=policy.symmetric,
        )
        orig = module.weight.data.clone()
        module.weight.data = quantized.to(module.weight.dtype)

        diag = diagnose_tensor(
            orig,
            quantized,
            fmt.format_id,
            name=f"{path}.weight",
            levels=fmt.learned_levels if fmt.is_learned else fmt.weight_levels,
            scale=scale,
        )
        records.append(
            ConversionRecord(
                module_path=path,
                param_name="weight",
                original_shape=tuple(orig.shape),
                format_id=fmt.format_id,
                group_size=gsize,
                scale_shape=tuple(scale.shape) if scale.numel() > 1 else (),
                mse=diag.mse,
                max_error=diag.max_error,
            )
        )

    return target, records


def _shallow_clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return a model whose parameters are copies but architecture is shared."""
    import copy

    clone = copy.deepcopy(model)
    return clone


def restore_original_weights(
    model: torch.nn.Module,
    original_state: dict[str, torch.Tensor],
) -> None:
    """Restore model weights from an unquantized state dict."""
    for name, param in model.named_parameters():
        if name in original_state:
            param.data.copy_(original_state[name])


def format_override_map(
    model: torch.nn.Module,
    policy: QuantizationPolicy,
) -> dict[str, QuantFormat]:
    """Return a map from parameter name to the format that would apply."""
    mapping: dict[str, QuantFormat] = {}
    for path, module, fmt, reason in select_parameter_groups(model, policy):
        if reason or fmt is None:
            from slm_training.models.quantization.formats import fp16_format

            mapping[f"{path}.weight"] = fp16_format(group_size=policy.default_format.group_size)
        else:
            mapping[f"{path}.weight"] = fmt
    return mapping
