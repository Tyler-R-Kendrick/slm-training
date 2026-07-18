#!/usr/bin/env python3
"""Dry-run quantization format inspection ledger (CAP3-01).

Produces a per-format physical-cost ledger and diagnostics without writing a
converted checkpoint unless ``--write-converted`` is explicit.  This is a
reference / wiring tool; it does not claim speedup or quality retention.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.models.quantization import (
    QuantFormat,
    binary_format,
    binary_plus_mask_format,
    build_model_ledger,
    int4_format,
    int8_format,
    learned_four_level_zero_format,
    symmetric_four_level_format,
    ternary_format,
)
from slm_training.models.quantization.convert import QuantizationPolicy, convert_twotower
from slm_training.models.quantization.diagnostics import diagnose_tensor
from slm_training.models.quantization.fake_quant import fake_quantize_weight
from slm_training.models.quantization.formats import KERNEL_REGISTRY


FORMAT_FACTORIES: dict[str, Any] = {
    "fp16": lambda gs=None: QuantFormat(
        format_id="fp16",
        weight_levels=(),
        nominal_symbol_bits=16.0,
        physical_slot_bits=16,
        group_size=gs or 128,
        scale_dtype="fp16",
        zero_point_dtype=None,
        bias_dtype="fp16",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        packing_layout="fp16_dense",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="fp16",
    ),
    "bf16": lambda gs=None: QuantFormat(
        format_id="bf16",
        weight_levels=(),
        nominal_symbol_bits=16.0,
        physical_slot_bits=16,
        group_size=gs or 128,
        scale_dtype="bf16",
        zero_point_dtype=None,
        bias_dtype="bf16",
        activation_dtype="bf16",
        accumulation_dtype="fp32",
        packing_layout="bf16_dense",
        supports_exact_zero=True,
        entropy_coding=None,
        kernel_id="bf16",
    ),
    "int8": int8_format,
    "int4": int4_format,
    "binary": binary_format,
    "ternary": ternary_format,
    "symmetric_four_level": symmetric_four_level_format,
    "learned4zero": learned_four_level_zero_format,
    "learned_four_level_zero": learned_four_level_zero_format,
    "binary_plus_mask": binary_plus_mask_format,
}


def _build_toy_model(d_model: int = 64, vocab: int = 32) -> torch.nn.Module:
    """Fixture model for dry-run ledger wiring."""

    class ToyTwoTowerLike(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embed = torch.nn.Embedding(vocab, d_model)
            self.norm = torch.nn.LayerNorm(d_model)
            self.denoiser = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model, 2, dim_feedforward=4 * d_model, batch_first=True),
                num_layers=2,
            )
            self.action_head = torch.nn.Linear(d_model, vocab)

        def named_parameters(self, prefix: str = "", recurse: bool = True):  # noqa: ANN001, ANN201
            # Keep the generator signature compatible.
            yield from super().named_parameters(prefix=prefix, recurse=recurse)

    return ToyTwoTowerLike()


def _load_checkpoint(checkpoint: str | None) -> torch.nn.Module:
    if checkpoint is None:
        return _build_toy_model()
    path = Path(checkpoint)
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)
    try:
        from slm_training.models.twotower import TwoTowerModel

        return TwoTowerModel.from_pretrained(checkpoint)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not load checkpoint {checkpoint!r}: {exc}") from exc


def _format_diagnostics(model: torch.nn.Module, fmt: QuantFormat) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        q, scale, _ = fake_quantize_weight(module.weight.data, fmt, group_size=fmt.group_size)
        levels = fmt.learned_levels if fmt.is_learned else fmt.weight_levels
        diag = diagnose_tensor(
            module.weight.data,
            q,
            fmt.format_id,
            name=f"{name}.weight",
            levels=levels,
            scale=scale,
        )
        rows.append(diag.as_dict())
    return rows


def inspect_format(
    model: torch.nn.Module,
    fmt: QuantFormat,
    write_converted: bool,
    out_dir: Path,
) -> dict[str, Any]:
    """Build ledger + diagnostics for a single format."""
    policy = QuantizationPolicy(default_format=fmt)
    converted = model
    records: list[dict[str, Any]] = []
    if fmt.format_id not in ("fp16", "bf16"):
        converted, recs = convert_twotower(model, policy, fail_on_tied=False, in_place=False)
        records = [r.as_dict() for r in recs]

    ledger = build_model_ledger(converted, {}, default_format=fmt)
    diagnostics = _format_diagnostics(converted, fmt)

    if write_converted:
        ckpt_path = out_dir / f"converted_{fmt.format_id}.pt"
        torch.save(converted.state_dict(), ckpt_path)
        ledger.notes.append(f"wrote converted checkpoint: {ckpt_path}")

    cap = KERNEL_REGISTRY.get(fmt.format_id)
    return {
        "format_id": fmt.format_id,
        "group_size": fmt.group_size,
        "ledger": ledger.as_dict(),
        "diagnostics": diagnostics,
        "conversion_records": records,
        "kernel_capability": asdict(cap) if cap else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--formats",
        type=str,
        default="binary,ternary,learned4zero,int4,int8",
        help="Comma-separated format ids",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("outputs/runs/quantization"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/design/quantization-results.json"))
    parser.add_argument("--write-converted", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=32)
    args = parser.parse_args(argv)

    model = _load_checkpoint(args.checkpoint)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in requested if f not in FORMAT_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown formats: {unknown}")

    rows = []
    for fid in requested:
        fmt = FORMAT_FACTORIES[fid](group_size=args.group_size)
        row = inspect_format(model, fmt, args.write_converted, out_dir)
        rows.append(row)

    scoreboard = {
        "run_id": out_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "group_size": args.group_size,
        "formats": requested,
        "rows": rows,
        "caveat": "Reference fake-quantization ledger only; not a speed or quality claim.",
    }

    board_path = out_dir / "format_ledger.json"
    board_path.write_text(json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8")
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scoreboard, indent=2))
    print(f"wrote {board_path} and {args.docs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
