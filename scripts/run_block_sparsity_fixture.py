"""CAP4-04 fixture: compiler-routed block sparsity and state-family experts.

Builds small ``CompilerRoutedDenoiserTower`` variants, runs synthetic
state-family routed traces, and reports active-parameter savings versus a dense
control.  This is a wiring fixture only; no ship gate or wall-clock claim is made.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.models.local_action_head import StateContext
from slm_training.models.quantization import (
    CompilerRoutedDenoiserTower,
    StateFamilyRouter,
    block_sparse_ternary_format,
    compute_block_sparse_cost,
    state_family_expert_format,
)
from slm_training.versioning import build_version_stamp


def _make_state_contexts(
    n: int,
    families: list[str],
    *,
    seed: int = 13,
) -> list[StateContext]:
    rng = random.Random(seed)
    contexts: list[StateContext] = []
    for _ in range(n):
        family = rng.choice(families)
        coverage = rng.choice(["complete", "partial", "unknown"])
        contexts.append(
            StateContext(
                state_family_id=family,
                state_signature=(family, coverage),
                branch_count=rng.randint(2, 8),
                forced=False,
            )
        )
    return contexts


def _route_indices(contexts: list[StateContext], router: StateFamilyRouter) -> torch.Tensor:
    return torch.tensor(
        [router.route_for_context(ctx, coverage=ctx.state_signature[1]) for ctx in contexts],
        dtype=torch.long,
    )


def _build_model(
    *,
    expert_rank: int | None = None,
    seed: int = 7,
) -> CompilerRoutedDenoiserTower:
    torch.manual_seed(seed)
    return CompilerRoutedDenoiserTower(
        vocab_size=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_len=32,
        n_routes=5,
        block_size=16,
        expert_rank=expert_rank,
    )


def _set_random_block_masks(model: CompilerRoutedDenoiserTower, density: float, *,
                            seed: int = 11) -> None:
    torch.manual_seed(seed)
    for layer in model.layers:
        for linear in (layer.mlp.fc, layer.mlp.proj):
            for route in range(linear.n_routes):
                mask = torch.rand(linear.out_blocks, linear.in_blocks)
                mask = (mask < density).float()
                # Ensure unknown route 0 stays all-active (dense fallback).
                if route == 0:
                    mask = torch.ones_like(mask)
                linear.set_route_mask(route, mask)


def _active_costs(model: CompilerRoutedDenoiserTower, route_indices: torch.Tensor) -> dict[str, Any]:
    fmt = block_sparse_ternary_format()
    expert_fmt = state_family_expert_format()
    total = 0
    active = 0
    for layer in model.layers:
        for linear in (layer.mlp.fc, layer.mlp.proj):
            cost = compute_block_sparse_cost(
                linear,
                route_indices,
                fmt=expert_fmt if model.layers[0].mlp.use_experts else fmt,
            )
            total += cost["total_bytes"]
            active += cost["active_bytes"]
    return {"total_mlp_bytes": total, "active_mlp_bytes": active, "active_ratio": active / max(1, total)}


def _evaluate(
    name: str,
    model: CompilerRoutedDenoiserTower,
    noisy: torch.Tensor,
    context: torch.Tensor,
    routes: torch.Tensor,
    dense_logits: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(noisy, context, pad_id=0, route_indices=routes)
    diff = (logits - dense_logits).reshape(logits.shape[0], -1)
    l2 = float(torch.norm(diff, dim=-1).mean().item())
    max_diff = float(diff.abs().max().item())
    costs = _active_costs(model, routes)
    return {
        "variant": name,
        "mean_l2_vs_dense": l2,
        "max_abs_diff_vs_dense": max_diff,
        "mlp_costs": costs,
    }


def main() -> int:
    n = 64
    seq = 8
    ctx_len = 4
    families = ["card_root", "bind_arg0", "bind_arg1", "literal_text"]

    torch.manual_seed(0)
    noisy = torch.randint(0, 32, (n, seq))
    context = torch.randn(n, ctx_len, 64)
    contexts = _make_state_contexts(n, families)
    router = StateFamilyRouter()
    routes = _route_indices(contexts, router)

    # Dense control: all routes use all-active masks.
    dense_model = _build_model(expert_rank=None, seed=1)
    _set_random_block_masks(dense_model, density=1.0, seed=1)
    dense_model.eval()
    with torch.no_grad():
        dense_logits = dense_model(noisy, context, pad_id=0, route_indices=routes)

    # Block-mask variant at 50% block density.
    block_model = _build_model(expert_rank=None, seed=1)
    _set_random_block_masks(block_model, density=0.5, seed=2)
    block_result = _evaluate(
        "block_mask_50pct", block_model, noisy, context, routes, dense_logits
    )

    # Low-rank expert variant.
    expert_model = _build_model(expert_rank=8, seed=1)
    expert_result = _evaluate(
        "state_family_expert_r8", expert_model, noisy, context, routes, dense_logits
    )

    result: dict[str, Any] = {
        "recipe": {
            "n": n,
            "seq": seq,
            "ctx_len": ctx_len,
            "families": families,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_routes": 5,
            "block_size": 16,
        },
        "routing": {
            "n_distinct_routes": router.n_routes,
            "route_counts": {
                str(int(r)): int((routes == r).sum().item()) for r in routes.unique().tolist()
            },
        },
        "variants": [block_result, expert_result],
        "version_stamp": build_version_stamp("model.quantization"),
        "caveats": [
            "wiring fixture only; no ship gate, checkpoint, or wall-clock claim",
            "block-mask path is dense math with zeroed blocks; no optimized sparse kernel is used",
            "expert path loops over routes; no optimized gather kernel is used",
            "costs are analytical packing estimates, not measured on-device",
            "models are randomly initialized, so semantic quality is not meaningful",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = Path("outputs/runs/cap4-04-block-sparsity")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"block_sparsity_fixture_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps({v["variant"]: v["mlp_costs"] for v in result["variants"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
