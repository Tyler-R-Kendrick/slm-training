#!/usr/bin/env python3
"""Benchmark accelerator + parallel-unmask throughput on TwoTower."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from slm_training.accel import detect_device, maybe_compile, sync_device
from slm_training.dsl.schema import load_jsonl
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _bench_train(model: TwoTowerModel, batch, steps: int, device: str) -> float:
    import torch

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-4)
    # warmup
    for _ in range(2):
        loss = model.training_loss(batch)
        loss.backward()
        opt.zero_grad(set_to_none=True)
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(batch)
        loss.backward()
        opt.step()
    sync_device(device)
    return (time.perf_counter() - t0) / steps


def _bench_generate(model: TwoTowerModel, prompts: list[str], rounds: int, device: str) -> float:
    # warmup
    model.generate_batch(prompts[:2])
    sync_device(device)
    t0 = time.perf_counter()
    n = 0
    for _ in range(rounds):
        outs = model.generate_batch(prompts)
        n += len(outs)
    sync_device(device)
    elapsed = time.perf_counter() - t0
    return elapsed / max(1, n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("outputs/train_data/v1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gen-rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    accel = detect_device(args.device)
    device = accel.device if args.device in {"auto", "best"} else args.device
    records = load_jsonl(args.train_dir / "records.jsonl")[:64]
    prompts = [r.prompt for r in records[:16]]

    report: dict = {
        "accel": {
            "device": device,
            "backend": accel.backend,
            "num_threads": accel.num_threads,
            "note": accel.note,
        },
        "variants": [],
    }

    for compile_on, unmask in (
        (False, "topk"),
        (False, "adaptive"),
        (True, "adaptive"),
    ):
        cfg = TwoTowerConfig(
            context_backend="scratch",
            d_model=128,
            n_heads=4,
            context_layers=2,
            denoiser_layers=4,
            grammar_ltr_primary=True,
            grammar_constrained=True,
            structural_bias=2.5,
            parallel_unmask=unmask,
            use_compile=compile_on,
            gen_steps=8,
            seed=0,
        )
        model = TwoTowerModel.from_records(records, config=cfg, device=device)
        if compile_on:
            model.denoiser = maybe_compile(model.denoiser, enabled=True, mode="default")
        batch = records[: args.batch_size]
        train_s = _bench_train(model, batch, args.steps, device)
        gen_s = _bench_generate(model, prompts, args.gen_rounds, device)
        report["variants"].append(
            {
                "compile": compile_on,
                "parallel_unmask": unmask,
                "train_sec_per_step": round(train_s, 4),
                "generate_sec_per_prompt": round(gen_s, 4),
                "train_steps_per_sec": round(1.0 / train_s, 3),
                "generate_prompts_per_sec": round(1.0 / gen_s, 3),
            }
        )

    out = Path("outputs/runs/accel_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
