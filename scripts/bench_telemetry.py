#!/usr/bin/env python3
"""Profile train + generate cycle spans to find bottlenecks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.runtime.telemetry import CycleTelemetry, bind_telemetry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("outputs/data/train/v1"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gen-prompts", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/runs/cycle_telemetry.json"),
    )
    args = parser.parse_args(argv)

    records = load_jsonl(args.train_dir / "records.jsonl")[:64]
    if not records:
        raise SystemExit("no train records")

    tel = CycleTelemetry(
        enabled=True,
        meta={"device": args.device, "purpose": "cycle_bottleneck_profile"},
    )
    cfg = TwoTowerConfig(
        context_backend="scratch",
        d_model=128,
        n_heads=4,
        context_layers=2,
        denoiser_layers=4,
        grammar_ltr_primary=True,
        cache_context=True,
        fuse_ltr_loss=True,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device=args.device)
    batch = records[: args.batch_size]
    prompts = [r.prompt for r in records[: args.gen_prompts]]

    import torch

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-4)
    with bind_telemetry(tel):
        for _ in range(2):
            loss = model.training_loss(batch)
            loss.backward()
            opt.zero_grad(set_to_none=True)
        for _ in range(args.train_steps):
            opt.zero_grad(set_to_none=True)
            loss = model.training_loss(batch)
            loss.backward()
            opt.step()
        _ = model.generate_batch(prompts)

    path = tel.write(args.out)
    also = Path("docs/design/cycle-telemetry.json")
    also.parent.mkdir(parents=True, exist_ok=True)
    also.write_text(json.dumps(tel.summary(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(tel.summary(), indent=2))
    print(f"wrote {path} and {also}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
