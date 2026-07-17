#!/usr/bin/env python3
"""Build preference pairs and/or run DPO-style preference training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.harnesses.preference import (
    collect_pairs_with_generator,
    write_pairs,
)
from slm_training.harnesses.preference.train import train_preference_from_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-pairs", help="Build preference pairs from train records")
    build.add_argument("--train-records", type=Path, required=True)
    build.add_argument("--out", type=Path, default=Path("outputs/data/preference/pairs.jsonl"))
    build.add_argument(
        "--soft-corrupt",
        action="store_true",
        default=True,
        help="Synthesize valid-but-worse rejects (default on; prefer over BrokenText).",
    )
    build.add_argument(
        "--no-soft-corrupt",
        action="store_true",
        help="Disable soft-corrupt rejects.",
    )
    build.add_argument(
        "--corrupt",
        action="store_true",
        help="Also synthesize BrokenText-style rejects (discouraged).",
    )
    build.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        help="Generate candidates from a TwoTower checkpoint (model samples).",
    )
    build.add_argument("--limit", type=int, default=None, help="Optional record cap.")
    build.add_argument("--device", default="cpu")
    build.add_argument(
        "--samples-per-prompt",
        type=int,
        default=2,
        help="When using --from-checkpoint, generate this many samples per prompt.",
    )
    build.add_argument(
        "--allow-invalid-rejects",
        action="store_true",
        help="Allow grammar-invalid rejects when ranking pairs (default: prefer valid).",
    )
    build.add_argument(
        "--no-gold",
        action="store_true",
        help=(
            "Policy-only self-distillation corpus: never inject the gold target "
            "and never synthesize gold-derived rejects. Requires "
            "--from-checkpoint. Pairs are tagged pair_corpus=self_distilled."
        ),
    )

    train = sub.add_parser("train", help="Run preference training from a checkpoint")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--pairs", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, default=Path("outputs/runs/preference"))
    train.add_argument("--steps", type=int, default=50)
    train.add_argument("--device", default="cpu")

    events = sub.add_parser(
        "build-local-events", help="Mine exact local decision events from traces"
    )
    events.add_argument("--traces", type=Path, required=True)
    events.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/data/preference/local_decisions.jsonl"),
    )
    events.add_argument("--manifest-out", type=Path, default=None)
    events.add_argument("--dataset-id", default=None)
    events.add_argument("--source-record-manifest", type=Path, default=None)
    events.add_argument(
        "--evidence-kind",
        choices=("all", "constraint_shadow", "counterfactual"),
        default="all",
        help="Filter mined events; semantic training must use counterfactual.",
    )

    local = sub.add_parser("train-local", help="Train on exact local decision events")
    local.add_argument("--checkpoint", type=Path, required=True)
    local.add_argument("--events", type=Path, required=True)
    local.add_argument("--out-dir", type=Path, default=Path("outputs/runs/local_preference"))
    local.add_argument(
        "--objective",
        choices=("ce_margin", "unlikelihood", "ftpo_single", "ftpo_set"),
        required=True,
    )
    local.add_argument("--reference-checkpoint", type=Path, default=None)
    local.add_argument("--steps", type=int, default=50)
    local.add_argument("--lr", type=float, default=5e-5)
    local.add_argument("--epsilon", type=float, default=2.0)
    local.add_argument("--tau", type=float, default=1.0)
    local.add_argument("--non-target-tether", type=float, default=0.0)
    local.add_argument("--target-tether", type=float, default=0.0)
    local.add_argument("--target-grace", type=float, default=1.0)
    local.add_argument("--balanced", action="store_true")
    local.add_argument("--seed", type=int, default=0)
    local.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)

    if args.cmd == "build-local-events":
        from slm_training.harnesses.preference.local_decisions import (
            decision_event_manifest,
            events_from_trace,
            load_trace_rows,
            write_decision_events,
            write_decision_event_manifest,
        )

        traces = load_trace_rows(args.traces)
        mined = [
            event
            for trace in traces
            for event in events_from_trace(trace)
            if args.evidence_kind == "all"
            or event.evidence_kind == args.evidence_kind
        ]
        count = write_decision_events(args.out, mined)
        if args.manifest_out is not None:
            if not args.dataset_id:
                parser.error("--manifest-out requires --dataset-id")
            source_fingerprint = None
            if args.source_record_manifest is not None:
                source_fingerprint = json.loads(
                    args.source_record_manifest.read_text(encoding="utf-8")
                ).get("content_fingerprint")
            manifest = decision_event_manifest(
                mined,
                dataset_id=args.dataset_id,
                records_path=args.out.name,
                source_trace_ids=(
                    str(trace.get("trace_id"))
                    for trace in traces
                    if trace.get("trace_id")
                ),
                source_record_fingerprint=source_fingerprint,
            )
            write_decision_event_manifest(args.manifest_out, manifest)
        print(json.dumps({"events": count, "out": str(args.out)}, indent=2))
        return 0

    if args.cmd == "train-local":
        from slm_training.harnesses.preference.local_train import train_local_from_paths

        summary = train_local_from_paths(
            args.checkpoint,
            args.events,
            out_dir=args.out_dir,
            objective=args.objective,
            reference_checkpoint=args.reference_checkpoint,
            steps=args.steps,
            device=args.device,
            lr=args.lr,
            epsilon=args.epsilon,
            tau=args.tau,
            non_target_tether=args.non_target_tether,
            target_tether=args.target_tether,
            target_grace=args.target_grace,
            balanced=args.balanced,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "build-pairs":
        records = load_jsonl(args.train_records)
        if args.limit is not None:
            records = records[: max(0, int(args.limit))]

        from slm_training.harnesses.quality import soft_corrupt_openui

        include_gold = not bool(args.no_gold)
        if not include_gold and args.from_checkpoint is None:
            parser.error("--no-gold requires --from-checkpoint (policy-only pairs)")
        use_soft = (
            bool(args.soft_corrupt)
            and not bool(args.no_soft_corrupt)
            and include_gold  # soft-corrupt rejects are gold-derived
        )
        prefer_valid = not bool(args.allow_invalid_rejects)

        if args.from_checkpoint is not None:
            from slm_training.models.twotower import TwoTowerModel

            model = TwoTowerModel.from_checkpoint(
                args.from_checkpoint, device=args.device
            )
            model.config.grammar_ltr_primary = True
            model.config.design_md_in_context = False
            n_samp = max(1, int(args.samples_per_prompt))

            def gen(record):
                cands = [record.openui] if include_gold else []
                for _ in range(n_samp):
                    cands.append(model.generate(record.prompt, gold=None))
                if use_soft:
                    cands.append(soft_corrupt_openui(record.openui))
                if args.corrupt and include_gold:
                    bad = record.openui.replace("TextContent", "BrokenText", 1)
                    if bad == record.openui:
                        bad = "root = Broken()"
                    cands.append(bad)
                return cands
        else:

            def gen(record):
                cands = [record.openui]
                if use_soft:
                    cands.append(soft_corrupt_openui(record.openui))
                if args.corrupt or not use_soft:
                    bad = record.openui.replace("TextContent", "BrokenText", 1)
                    if bad == record.openui:
                        bad = "root = Broken()"
                    cands.append(bad)
                return cands

        pairs = collect_pairs_with_generator(
            records,
            gen,
            prefer_valid_rejects=prefer_valid,
            structure_only=True,
            include_gold=include_gold,
            generator_checkpoint=(
                str(args.from_checkpoint) if args.from_checkpoint else None
            ),
        )
        n = write_pairs(args.out, pairs)
        print(
            json.dumps(
                {
                    "pairs": n,
                    "out": str(args.out),
                    "pair_corpus": (
                        "gold_correction" if include_gold else "self_distilled"
                    ),
                    "soft_corrupt": use_soft,
                    "prefer_valid_rejects": prefer_valid,
                    "structure_only": True,
                },
                indent=2,
            )
        )
        return 0

    summary = train_preference_from_paths(
        args.checkpoint,
        args.pairs,
        out_dir=args.out_dir,
        steps=args.steps,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
