#!/usr/bin/env python3
"""Benchmark accelerator + train-speed microbenches on TwoTower."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from slm_training.levers import DEFAULT_TRAIN_DATA_DIR

from slm_training.runtime.accel import detect_device, maybe_compile, sync_device
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


def _micro_variant(
    *,
    records,
    device: str,
    batch_size: int,
    steps: int,
    label: str,
    context_backend: str,
    cache_context: bool,
    fuse_ltr_loss: bool,
    compile_on: bool = False,
    local_files_only: bool = False,
) -> dict:
    cfg = TwoTowerConfig(
        context_backend=context_backend,
        d_model=128,
        n_heads=4,
        context_layers=2,
        denoiser_layers=4,
        grammar_ltr_primary=True,
        grammar_constrained=True,
        structural_bias=2.5,
        parallel_unmask="adaptive",
        use_compile=compile_on,
        cache_context=cache_context,
        fuse_ltr_loss=fuse_ltr_loss,
        gen_steps=8,
        freeze_context=True,
        local_files_only=local_files_only,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device=device)
    if compile_on:
        model.denoiser = maybe_compile(model.denoiser, enabled=True, mode="default")
    batch = records[:batch_size]
    train_s = _bench_train(model, batch, steps, device)
    return {
        "label": label,
        "context_backend": context_backend,
        "cache_context": cache_context,
        "fuse_ltr_loss": fuse_ltr_loss,
        "compile": compile_on,
        "train_sec_per_step": round(train_s, 4),
        "train_steps_per_sec": round(1.0 / train_s, 3) if train_s > 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DATA_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gen-rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--microbench",
        action="store_true",
        help="Run cache/fuse train microbenches and write train-microbench.json.",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HF context microbench (offline / no transformers cache).",
    )
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

    if args.microbench:
        micros: list[dict] = []
        micros.append(
            _micro_variant(
                records=records,
                device=device,
                batch_size=args.batch_size,
                steps=args.steps,
                label="scratch_baseline_no_fuse_no_cache",
                context_backend="scratch",
                cache_context=False,
                fuse_ltr_loss=False,
            )
        )
        micros.append(
            _micro_variant(
                records=records,
                device=device,
                batch_size=args.batch_size,
                steps=args.steps,
                label="scratch_fuse_cache",
                context_backend="scratch",
                cache_context=True,
                fuse_ltr_loss=True,
            )
        )
        micros.append(
            _micro_variant(
                records=records,
                device=device,
                batch_size=args.batch_size,
                steps=args.steps,
                label="scratch_fuse_cache_compile",
                context_backend="scratch",
                cache_context=True,
                fuse_ltr_loss=True,
                compile_on=True,
            )
        )
        if not args.skip_hf:
            try:
                micros.append(
                    _micro_variant(
                        records=records,
                        device=device,
                        batch_size=min(4, args.batch_size),
                        steps=max(5, args.steps // 2),
                        label="hf_no_cache_fuse",
                        context_backend="hf",
                        cache_context=False,
                        fuse_ltr_loss=True,
                        local_files_only=True,
                    )
                )
                micros.append(
                    _micro_variant(
                        records=records,
                        device=device,
                        batch_size=min(4, args.batch_size),
                        steps=max(5, args.steps // 2),
                        label="hf_cache_fuse",
                        context_backend="hf",
                        cache_context=True,
                        fuse_ltr_loss=True,
                        local_files_only=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                micros.append({"label": "hf_skipped", "error": str(exc)})

        # Rank by steps/sec among successful scratch rows.
        ranked = sorted(
            [m for m in micros if m.get("train_steps_per_sec")],
            key=lambda m: float(m["train_steps_per_sec"]),
            reverse=True,
        )
        micro_report = {
            "accel": report["accel"],
            "microbenches": micros,
            "winner": ranked[0] if ranked else None,
            "notes": (
                "Scratch microbench understates HF production cost; "
                "HF cache row is the primary train-speed win for frozen backbone."
            ),
        }
        micro_out = Path("docs/design/train-microbench.json")
        micro_out.parent.mkdir(parents=True, exist_ok=True)
        micro_out.write_text(json.dumps(micro_report, indent=2) + "\n", encoding="utf-8")
        also = Path("outputs/runs/train_microbench.json")
        also.write_text(json.dumps(micro_report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(micro_report, indent=2))
        print(f"wrote {micro_out} and {also}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
