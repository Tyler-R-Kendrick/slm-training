#!/usr/bin/env python3
"""Self-distillation stage: select traces → SFT from mid-trained anchor (P2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sel = sub.add_parser("select", help="Stratified coverage selection from a trace store.")
    p_sel.add_argument("--traces", type=Path, required=True)
    p_sel.add_argument("--out", type=Path, required=True)
    p_sel.add_argument("--budget", type=int, default=500)
    p_sel.add_argument("--corpus", default="self_distilled_success")
    p_sel.add_argument("--seed", type=int, default=0)

    p_train = sub.add_parser("train", help="SFT from selected traces + optional anchor mix.")
    p_train.add_argument("--checkpoint", type=Path, required=True, help="Mid-trained anchor.")
    p_train.add_argument("--traces", type=Path, required=True)
    p_train.add_argument("--out-dir", type=Path, default=Path("outputs/runs/self_distill"))
    p_train.add_argument("--anchor-train-dir", type=Path, default=None)
    p_train.add_argument("--budget", type=int, default=500)
    p_train.add_argument("--steps", type=int, default=50)
    p_train.add_argument("--lambda-traj", type=float, default=1.0)
    p_train.add_argument("--lambda-anchor", type=float, default=0.3)
    p_train.add_argument("--dropout", type=float, default=0.0)
    p_train.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)

    if args.cmd == "select":
        from slm_training.harnesses.distill.select import SelectConfig, select_traces
        from slm_training.harnesses.distill.trace_store import TraceStore

        store = TraceStore(args.traces)
        selected = select_traces(
            store.iter_traces(),
            config=SelectConfig(
                budget=args.budget, corpus=args.corpus, seed=args.seed
            ),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for trace in selected:
                handle.write(json.dumps(trace) + "\n")
        print(json.dumps({"selected": len(selected), "out": str(args.out)}, indent=2))
        return 0

    from slm_training.harnesses.distill.sft import train_self_distill_from_paths

    summary = train_self_distill_from_paths(
        args.checkpoint,
        args.traces,
        out_dir=args.out_dir,
        anchor_train_dir=args.anchor_train_dir,
        steps=args.steps,
        device=args.device,
        budget=args.budget,
        lambda_traj=args.lambda_traj,
        lambda_anchor=args.lambda_anchor,
        dropout=args.dropout,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
