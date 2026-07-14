"""Sibling delta averaging and TIES merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from slm_training.lineage.records import RunManifest


def validate_merge_manifests(
    parent: RunManifest, children: Iterable[RunManifest]
) -> list[RunManifest]:
    rows = list(children)
    if len(rows) < 2:
        raise ValueError("merge requires at least two children")
    for child in rows:
        if child.parent_ids != (parent.run_id,):
            raise ValueError("merge candidates must be siblings with one common parent")
        if child.compatibility_sha != parent.compatibility_sha:
            raise ValueError(
                "merge candidates have incompatible track/base/architecture/tokenizer/shapes"
            )
    return rows


def merge_checkpoints(
    parent_path: Path | str,
    child_paths: Iterable[Path | str],
    output_path: Path | str,
    *,
    method: str,
    density: float = 0.2,
) -> Path:
    import torch

    if method not in {"average", "ties"}:
        raise ValueError(f"unknown merge method {method!r}")
    if not 0 < density <= 1:
        raise ValueError("TIES density must be in (0, 1]")
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    parent_state = _state_dict(parent_payload)
    children = [
        torch.load(path, map_location="cpu", weights_only=True) for path in child_paths
    ]
    child_states = [_state_dict(payload) for payload in children]
    if len(child_states) < 2:
        raise ValueError("merge requires at least two child checkpoints")
    _same_shapes(parent_state, child_states)
    merged: dict[str, Any] = {}
    for name, base in parent_state.items():
        deltas = [
            state[name].to(dtype=torch.float32) - base.to(dtype=torch.float32)
            for state in child_states
        ]
        delta = torch.stack(deltas)
        combined = delta.mean(dim=0) if method == "average" else _ties(delta, density)
        merged[name] = (base.to(dtype=torch.float32) + combined).to(dtype=base.dtype)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"merge output already exists: {output}")
    payload = dict(parent_payload) if isinstance(parent_payload, dict) else {}
    payload["state_dict"] = merged
    torch.save(payload, output)
    return output


def _state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return payload["state_dict"]
    if (
        isinstance(payload, dict)
        and payload
        and all(hasattr(value, "shape") for value in payload.values())
    ):
        return payload
    raise ValueError("checkpoint does not contain a tensor state_dict")


def _same_shapes(parent: dict[str, Any], children: list[dict[str, Any]]) -> None:
    expected = {name: tuple(value.shape) for name, value in parent.items()}
    for state in children:
        shapes = {name: tuple(value.shape) for name, value in state.items()}
        if shapes != expected:
            raise ValueError("checkpoint parameter names or shapes differ")


def _ties(deltas: Any, density: float) -> Any:
    import torch

    flat = deltas.abs().flatten(1)
    keep = max(1, int(flat.shape[1] * density))
    thresholds = flat.topk(keep, dim=1).values[:, -1]
    trimmed = torch.where(
        deltas.abs() >= thresholds.reshape((-1,) + (1,) * (deltas.ndim - 1)),
        deltas,
        torch.zeros_like(deltas),
    )
    elected = torch.sign(trimmed.sum(dim=0))
    selected = torch.where(
        torch.sign(trimmed) == elected, trimmed, torch.zeros_like(trimmed)
    )
    counts = (selected != 0).sum(dim=0).clamp_min(1)
    return selected.sum(dim=0) / counts
