#!/usr/bin/env python3
"""Collect verified decode trajectories into an append-only trace store.

Runs MaskGIT generation over a suite (production inputs only — no gold in the
model's context), records every intermediate canvas / commit / remask event,
verifies the final program, and appends labeled traces. This store is the
substrate for offline self-distillation SFT and trajectory-aligned RL (E64).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-dir", type=Path, default=None, help="Versioned test artifacts dir."
    )
    parser.add_argument("--suite", default="held_out")
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Explicit records.jsonl (alternative to --test-dir/--suite).",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/traces/latest"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Rollouts per prompt (diverse trajectories need sample decode).",
    )
    parser.add_argument(
        "--no-canvases",
        action="store_true",
        help="Skip per-step canvas snapshots (events only; smaller traces).",
    )
    parser.add_argument(
        "--record-support",
        action="store_true",
        help="Persist grammar allowed_id_set on each commit (E64 support match).",
    )
    args = parser.parse_args(argv)

    import torch

    from slm_training.distill.trace_store import (
        DecodeTraceRecorder,
        TraceStore,
        checkpoint_sha,
        decode_config_hash,
    )
    from slm_training.dsl.schema import load_jsonl
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.preference import (
        composite_reward,
        grammar_score,
        layout_metrics,
        placeholder_score,
    )

    if args.records is not None:
        records = load_jsonl(args.records)
    elif args.test_dir is not None:
        from slm_training.harnesses.model_build.data import load_suite_records

        records = load_suite_records(args.test_dir, args.suite)
    else:
        parser.error("provide --records or --test-dir")
        return 2
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]

    torch.manual_seed(int(args.seed))
    model = TwoTowerModel.from_checkpoint(args.checkpoint, device=args.device)
    # Trajectories come from the MaskGIT diffusion path (E64 target), not the
    # LTR-primary shortcut.
    model.config.grammar_ltr_primary = False

    policy_sha = checkpoint_sha(args.checkpoint)
    decode_hash = decode_config_hash(model.config)
    store = TraceStore(args.out)

    appended = 0
    accepted = 0
    for record in records:
        for sample_index in range(max(1, int(args.samples_per_prompt))):
            recorder = DecodeTraceRecorder(
                record_canvases=not bool(args.no_canvases),
                record_support=bool(args.record_support),
            )
            model.trace_recorder = recorder
            try:
                text = model.generate(record.prompt, gold=None, design_md=None)
            finally:
                model.trace_recorder = None

            g = grammar_score(text)
            reward = {
                "grammar": g,
                "placeholder": placeholder_score(text, record),
                "layout": layout_metrics(text),
                "composite": composite_reward(text, gold=record, design_md=None),
            }
            labels = {
                "accepted": g > 0.0,
                "exact_gold": text.strip() == record.openui.strip(),
            }
            trace = recorder.finalize(
                final_text=text,
                reward=reward,
                labels=labels,
                record_id=record.id,
                prompt=record.prompt,
                sample_index=sample_index,
                source_suite=args.suite if args.test_dir else None,
                policy_checkpoint_sha=policy_sha,
                policy_checkpoint=str(args.checkpoint),
                decode_config_hash=decode_hash,
                tokenizer_version=getattr(model.tokenizer, "version", None),
                seed=int(args.seed),
            )
            store.append(trace)
            appended += 1
            accepted += int(labels["accepted"])

    print(
        json.dumps(
            {
                "out": str(args.out),
                "traces": appended,
                "accepted": accepted,
                "accept_rate": round(accepted / appended, 4) if appended else None,
                "policy_checkpoint_sha": policy_sha,
                "decode_config_hash": decode_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
