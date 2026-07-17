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
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--run-id", default="trajectory-latest")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-kind",
        default=None,
        help="Collect only records with this target kind (for example, document).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decode-policy",
        choices=("maskgit", "strict_compiler_tree"),
        default="maskgit",
        help="Decode policy to trace; strict compiler-tree matches V9/V10 controls.",
    )
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
    parser.add_argument(
        "--counterfactual-semantic",
        action="store_true",
        help="Replay and independently judge grammar-legal exact-state alternatives.",
    )
    parser.add_argument(
        "--counterfactual-states-per-record",
        type=int,
        default=4,
        help=(
            "Maximum judge probes per record, stratified across compiler-derived "
            "decision kinds and relative trajectory depth."
        ),
    )
    parser.add_argument("--counterfactual-candidates", type=int, default=4)
    args = parser.parse_args(argv)

    if args.counterfactual_semantic and (
        args.decode_policy != "strict_compiler_tree" or not args.record_support
    ):
        parser.error(
            "--counterfactual-semantic requires --decode-policy "
            "strict_compiler_tree and --record-support"
        )

    import torch

    from slm_training.harnesses.distill.trace_store import (
        DecodeTraceRecorder,
        TraceStore,
        checkpoint_sha,
        decode_config_hash,
    )
    from slm_training.dsl.schema import load_jsonl
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.harnesses.preference import (
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
    if args.output_kind is not None:
        records = [
            record for record in records if record.target_kind == args.output_kind
        ]
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]
    if args.decode_policy == "strict_compiler_tree" and any(
        record.target_kind != "document" for record in records
    ):
        parser.error(
            "strict_compiler_tree supports document records only; "
            "pass --output-kind document"
        )

    torch.manual_seed(int(args.seed))
    model = TwoTowerModel.from_checkpoint(args.checkpoint, device=args.device)
    if args.decode_policy == "strict_compiler_tree":
        from slm_training.harnesses.model_build.eval_policy import (
            apply_strict_compiler_tree_policy,
        )

        apply_strict_compiler_tree_policy(model.config)
    else:
        # E64 trajectories come from MaskGIT, not the LTR-primary shortcut.
        model.config.grammar_ltr_primary = False

    policy_sha = checkpoint_sha(args.checkpoint)
    decode_hash = decode_config_hash(model.config)
    from slm_training.runtime.telemetry import run_trace

    trace_context = run_trace(args.run_id, "trajectory.collect")
    trace_context.__enter__()
    output = args.out or trace_context.domain_path("decode")
    store = TraceStore(
        output,
        run_id=args.run_id,
        trace_id=trace_context.trace_id,
        span_id=trace_context.span_id,
    )

    appended = 0
    accepted = 0
    counterfactual = {
        "states": 0,
        "candidates": 0,
        "judge_passed": 0,
        "verified": 0,
        "events": 0,
    }
    for record in records:
        for sample_index in range(max(1, int(args.samples_per_prompt))):
            recorder = DecodeTraceRecorder(
                record_canvases=not bool(args.no_canvases),
                record_support=bool(args.record_support),
            )
            model.trace_recorder = recorder
            request = None
            if args.decode_policy == "strict_compiler_tree":
                from slm_training.data.contract import GenerationRequest
                from slm_training.harnesses.quality import compact_schema_snippet
                from slm_training.models.template_fill import ensure_prompt_inventory

                request = GenerationRequest.from_record(
                    record,
                    schema=compact_schema_snippet(budget=600),
                    include_design_md=False,
                )
                visible_prompt = ensure_prompt_inventory(
                    request.prompt, list(request.slot_contract)
                )
                context_text = model._context_prompts(
                    [visible_prompt],
                    golds=[None],
                    design_mds=[request.design_md],
                    slot_contracts=[list(request.slot_contract)],
                    schemas=[request.schema],
                    output_kinds=[request.output_kind],
                    output_categories=[request.output_category],
                )[0]
            else:
                context_text = model._context_prompts(
                    [record.prompt], golds=[None], design_mds=[None]
                )[0]
            try:
                text = (
                    model.generate_batch_requests([request])[0]
                    if request is not None
                    else model.generate(record.prompt, gold=None, design_md=None)
                )
            finally:
                model.trace_recorder = None

            if args.counterfactual_semantic:
                from slm_training.harnesses.preference.counterfactuals import (
                    mine_semantic_counterfactuals,
                )

                mined = mine_semantic_counterfactuals(
                    model,
                    recorder,
                    record,
                    context_text,
                    max_states=int(args.counterfactual_states_per_record),
                    max_candidates=int(args.counterfactual_candidates),
                    seed=int(args.seed),
                )
                for key, value in mined.items():
                    counterfactual[key] += value

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
                tokenizer_sha=model.artifact_identity()["tokenizer_sha"],
                context_text=context_text,
                seed=int(args.seed),
            )
            store.append(trace)
            appended += 1
            accepted += int(labels["accepted"])

    print(
        json.dumps(
            {
                "out": str(output),
                "trace_id": trace_context.trace_id,
                "traces": appended,
                "accepted": accepted,
                "accept_rate": round(accepted / appended, 4) if appended else None,
                "policy_checkpoint_sha": policy_sha,
                "decode_config_hash": decode_hash,
                "decode_policy": args.decode_policy,
                "counterfactual": counterfactual,
            },
            indent=2,
        )
    )
    trace_context.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
