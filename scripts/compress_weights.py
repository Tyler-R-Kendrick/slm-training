#!/usr/bin/env python3
"""Lossless BF16 exponent-codebook compress / verify for TwoTower checkpoints.

Reference: https://brianbell-x.github.io/weight-compression/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.compression import (
    LAYOUT_BYTESPLIT,
    LAYOUT_REGROUP,
    decompress_state_dict,
    write_compressed_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Compressed JSON path (default: <checkpoint>.wc.json)",
    )
    parser.add_argument(
        "--layout",
        choices=("regroup", "bytesplit"),
        default="regroup",
        help="regroup = headline ~11.3 b/w; bytesplit = GPU-validated layout",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="Decompress and confirm BF16 view matches original state_dict.",
    )
    args = parser.parse_args(argv)

    layout = LAYOUT_REGROUP if args.layout == "regroup" else LAYOUT_BYTESPLIT
    out = args.out or args.checkpoint.with_suffix(args.checkpoint.suffix + ".wc.json")
    summary = write_compressed_checkpoint(args.checkpoint, out, layout=layout)

    report = {
        "checkpoint": str(args.checkpoint),
        "compressed": str(out),
        "layout": layout,
        **summary,
        "reference": "https://brianbell-x.github.io/weight-compression/",
        "kernel_separate": True,
    }

    if args.verify_load:
        import torch

        from slm_training.compression import _to_bf16_u16

        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = raw.get("state_dict") or raw
        payload = json.loads(out.read_text(encoding="utf-8"))
        restored = decompress_state_dict(payload["weights"])
        ok = True
        checked = 0
        for name, tensor in state.items():
            if name not in restored or not torch.is_floating_point(tensor):
                continue
            if tensor.numel() < 64:
                continue
            a = _to_bf16_u16(tensor.detach().cpu().float().numpy())
            b = _to_bf16_u16(restored[name].detach().cpu().float().numpy())
            if a.shape != b.shape or not (a == b).all():
                ok = False
                report["first_mismatch"] = name
                break
            checked += 1
        report["bit_exact_bf16_roundtrip"] = ok
        report["tensors_checked"] = checked

    print(json.dumps(report, indent=2))
    if args.verify_load and not report.get("bit_exact_bf16_roundtrip"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
