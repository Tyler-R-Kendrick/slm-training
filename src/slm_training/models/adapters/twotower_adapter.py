"""Deterministic resolution + attachment of low-rank adapters onto TwoTower (LDI2-01).

Maps a :class:`TwoTowerAdapterSpec`'s target-module names onto the concrete linear
projections inside the denoiser's transformer blocks, wraps each with a
:class:`LowRankAdapter`, and reports which modules were resolved. Only the denoiser is
adapted here; the context tower is never touched. Missing, duplicate, or unsupported
targets fail closed with an actionable error.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from slm_training.models.adapters.low_rank import LowRankAdapter
from slm_training.models.adapters.spec import TwoTowerAdapterSpec

__all__ = [
    "TARGET_MODULE_PATHS",
    "attach_low_rank_adapters",
    "resolve_adapter_targets",
]

# Canonical target-module name -> attribute path within one denoiser TransformerBlock.
TARGET_MODULE_PATHS: dict[str, tuple[Any, ...]] = {
    "attn_q": ("self_attn", "q_proj"),
    "attn_k": ("self_attn", "k_proj"),
    "attn_v": ("self_attn", "v_proj"),
    "attn_out": ("self_attn", "out_proj"),
    "cross_attn_q": ("cross_attn", "q_proj"),
    "cross_attn_k": ("cross_attn", "k_proj"),
    "cross_attn_v": ("cross_attn", "v_proj"),
    "cross_attn_out": ("cross_attn", "out_proj"),
    "mlp_in": ("mlp", 0),
    "mlp_out": ("mlp", 2),
}


def _child(module: Any, key: Any) -> Any:
    """Index (int key) or attribute-access (str key) one step into ``module``, or ``None``."""
    if isinstance(key, int):
        try:
            return module[key]
        except (IndexError, KeyError, TypeError):
            return None
    return getattr(module, key, None)


def _resolve_leaf(block: nn.Module, path: tuple[Any, ...]) -> tuple[Any, Any, Any]:
    """Walk ``path`` to its leaf, returning ``(parent, leaf_key, leaf)`` or a ``None`` triple.

    Any missing intermediate short-circuits to ``(None, None, None)`` so the caller can
    fail closed on an unresolved target.
    """
    parent: Any = block
    for key in path[:-1]:
        parent = _child(parent, key)
        if parent is None:
            return None, None, None
    leaf_key = path[-1]
    return parent, leaf_key, _child(parent, leaf_key)


def resolve_adapter_targets(
    denoiser: nn.Module, spec: TwoTowerAdapterSpec
) -> dict[str, tuple[nn.Module, Any, nn.Linear]]:
    """Resolve a spec's targets to (parent, leaf-key, linear) triples, deterministically.

    Fails closed when a requested target names an unknown module, resolves to a
    non-linear, or references a layer index outside the denoiser.
    """
    layers = getattr(denoiser, "layers", None)
    if layers is None:
        raise ValueError("denoiser exposes no `layers` to adapt")
    count = len(layers)
    indices = (
        tuple(range(count))
        if spec.target_layer_indices is None
        else spec.target_layer_indices
    )
    resolved: dict[str, tuple[nn.Module, Any, nn.Linear]] = {}
    for index in indices:
        if index < 0 or index >= count:
            raise ValueError(
                f"adapter target layer index {index} is out of range [0, {count})"
            )
        block = layers[index]
        for name in spec.target_modules:
            path = TARGET_MODULE_PATHS.get(name)
            if path is None:
                raise ValueError(
                    f"unsupported adapter target module {name!r}; "
                    f"expected one of {sorted(TARGET_MODULE_PATHS)}"
                )
            parent, leaf_key, linear = _resolve_leaf(block, path)
            if not isinstance(linear, nn.Linear):
                raise ValueError(
                    f"adapter target {name!r} at denoiser layer {index} did not "
                    "resolve to an nn.Linear (unsupported module type or absent)"
                )
            resolved[f"denoiser.layers.{index}.{name}"] = (parent, leaf_key, linear)
    if not resolved:
        raise ValueError("adapter spec matched no modules")
    return resolved


def attach_low_rank_adapters(
    denoiser: nn.Module, spec: TwoTowerAdapterSpec, *, seed: int
) -> dict[str, LowRankAdapter]:
    """Wrap each resolved target with a LowRankAdapter, in place, and return the map.

    Adapter factors are initialized under a forked RNG so attaching an adapter never
    shifts the model's training RNG stream (a fresh adapter is output-identical anyway,
    since B is zero-initialized).
    """
    import torch

    resolved = resolve_adapter_targets(denoiser, spec)
    wrappers: dict[str, LowRankAdapter] = {}
    for offset, (key, (parent, leaf_key, linear)) in enumerate(sorted(resolved.items())):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 4096 + offset)
            wrapper = LowRankAdapter(
                linear, rank=spec.rank, alpha=spec.alpha, dropout=spec.dropout
            )
        if isinstance(leaf_key, int):
            parent[leaf_key] = wrapper
        else:
            setattr(parent, leaf_key, wrapper)
        wrappers[key] = wrapper
    return wrappers
