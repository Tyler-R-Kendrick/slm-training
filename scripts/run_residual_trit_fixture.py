"""CAP4-01 fixture: compare FP16, ternary-direct, and residual-trit local scorers.

Generates synthetic ``(hidden, teacher_scores)`` pairs, trains three small
scorers, and reports test MSE plus estimated byte cost.  This is a wiring
fixture only; no ship gate or checkpoint claim is made.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.models.quantization import (
    build_model_ledger,
    fake_quantize_weight,
)
from slm_training.models.quantization.formats import (
    fp16_format,
    residual_ternary_plane_format,
    ternary_format,
)
from slm_training.models.quantization.residual_planes import ResidualTritStack


def _make_data(
    n: int,
    hidden_dim: int,
    num_actions: int,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    hidden = torch.randn(n, hidden_dim)
    teacher = nn.Linear(hidden_dim, num_actions, bias=False)
    teacher.weight.requires_grad = False
    with torch.no_grad():
        teacher.weight.normal_(0.0, 0.3)
    target = teacher(hidden)
    return hidden, target


def _train_fp16(
    hidden: torch.Tensor,
    target: torch.Tensor,
    steps: int = 100,
    lr: float = 5e-2,
) -> tuple[nn.Linear, list[float]]:
    model = nn.Linear(hidden.shape[1], target.shape[1], bias=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(hidden), target)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return model, history


def _train_ternary_direct(
    hidden: torch.Tensor,
    target: torch.Tensor,
    steps: int = 100,
    lr: float = 5e-2,
) -> tuple[nn.Linear, list[float]]:
    model = nn.Linear(hidden.shape[1], target.shape[1], bias=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    fmt = ternary_format()
    for _ in range(steps):
        opt.zero_grad()
        # Project weights to ternary for the forward, but keep gradients flowing
        # through the full-precision shadow weight (straight-through).
        w = model.weight
        q, _, _ = fake_quantize_weight(w, fmt)
        projection = q + (w - w.detach())
        out = F.linear(hidden, projection, model.bias)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return model, history


def _train_residual_trit(
    hidden: torch.Tensor,
    target: torch.Tensor,
    R: int = 2,
    base_steps: int = 60,
    plane_steps: int = 40,
    base_lr: float = 5e-2,
    plane_lr: float = 1e-1,
) -> tuple[ResidualTritStack, list[float]]:
    model = ResidualTritStack(
        hidden.shape[1],
        target.shape[1],
        R=R,
        scale_mode="geometric_balanced",
        residual_normalization="none",
    )
    # Train the base module first.
    opt = torch.optim.Adam(model.base_module.parameters(), lr=base_lr)
    history: list[float] = []
    for _ in range(base_steps):
        opt.zero_grad()
        loss = F.mse_loss(model(hidden, max_planes=0), target)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    # Fit planes sequentially on the residual.
    fit_result = model.fit_planes_sequential(
        hidden, target, steps=plane_steps, lr=plane_lr, freeze_previous=True
    )
    history.extend(fit_result["loss_histories"][0] if fit_result["loss_histories"] else [])
    return model, history


def _byte_cost(
    model: nn.Module,
    fmt: Any,
    residual_plane_fmt: Any | None = None,
) -> int:
    ledger = build_model_ledger(
        model,
        format_map={},
        default_format=fmt,
        residual_plane_format=residual_plane_fmt,
    )
    return ledger.total()


def _evaluate(model: nn.Module, hidden: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        if isinstance(model, ResidualTritStack):
            pred = model(hidden)
        else:
            pred = model(hidden)
        return float(F.mse_loss(pred, target).item())


def main() -> int:
    hidden_dim = 16
    num_actions = 8
    n_train = 256
    n_test = 64

    train_hidden, train_target = _make_data(n_train, hidden_dim, num_actions, seed=7)
    test_hidden, test_target = _make_data(n_test, hidden_dim, num_actions, seed=8)

    fp16_model, fp16_history = _train_fp16(train_hidden, train_target)
    ternary_model, ternary_history = _train_ternary_direct(train_hidden, train_target)
    residual_model, residual_history = _train_residual_trit(train_hidden, train_target)

    fp16_mse = _evaluate(fp16_model, test_hidden, test_target)
    ternary_mse = _evaluate(ternary_model, test_hidden, test_target)
    residual_mse = _evaluate(residual_model, test_hidden, test_target)

    fp16_bytes = _byte_cost(fp16_model, fp16_format())
    ternary_bytes = _byte_cost(ternary_model, ternary_format())
    residual_bytes = _byte_cost(
        residual_model,
        fp16_format(),
        residual_plane_fmt=residual_ternary_plane_format(),
    )

    result: dict[str, Any] = {
        "recipe": {
            "hidden_dim": hidden_dim,
            "num_actions": num_actions,
            "n_train": n_train,
            "n_test": n_test,
            "R": residual_model.R,
            "scale_mode": residual_model.scale_mode,
        },
        "results": {
            "fp16": {
                "test_mse": fp16_mse,
                "bytes": fp16_bytes,
                "final_train_mse": fp16_history[-1] if fp16_history else None,
            },
            "ternary_direct": {
                "test_mse": ternary_mse,
                "bytes": ternary_bytes,
                "final_train_mse": ternary_history[-1] if ternary_history else None,
            },
            "residual_trit": {
                "test_mse": residual_mse,
                "bytes": residual_bytes,
                "final_train_mse": residual_history[-1] if residual_history else None,
            },
        },
        "caveats": [
            "wiring fixture only; no ship gate or checkpoint claim",
            "byte costs are analytical packing estimates, not measured on-device",
            "synthetic teacher linear may favor the FP16 baseline",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = Path("outputs/runs/cap4-01-residual-trit")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"residual_trit_fixture_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(result["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
