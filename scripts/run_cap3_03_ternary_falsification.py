#!/usr/bin/env python3
"""Run the CAP3-03 equal-storage ternary falsification matrix.

Example::

    python -m scripts.run_cap3_03_ternary_falsification \
        --checkpoint toy --synthetic-traces 256 --samples 64 \
        --formats ternary,learned4zero --out-dir outputs/runs/cap3-03-toy

CPU/toy runs are wiring evidence only; they do not claim ship-grade retention
or physical speedup.  Ship-grade claims need GPU + local E224+ checkpoints and
a full ``--ship-gates`` evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from slm_training.harnesses.distill.grammar_trace import GrammarTraceRecorder
from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.harnesses.experiments.cap3_03_ternary_falsification import (
    FORMAT_FACTORIES,
    MatrixReport,
    run_matrix,
)
from slm_training.harnesses.quantization.calibration import (
    PRIMARY_STRATEGIES,
    build_calibration_corpus,
    load_grammar_decision_traces,
)
from slm_training.models.local_action_head import LocalFlatHead
from slm_training.models.quantization.formats import QuantFormat


DEFAULT_FORMATS = (
    "ternary",
    "learned4zero",
    "symmetric4",
    "binary_plus_mask",
    "int4",
)


class ToyLocalModel(nn.Module):
    """Minimal model owning a local flat head for fixture wiring."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.local_head = LocalFlatHead(hidden_dim)


def _load_model(checkpoint: str | None, hidden_dim: int) -> nn.Module:
    if checkpoint is None or checkpoint == "toy":
        return ToyLocalModel(hidden_dim)
    path = Path(checkpoint)
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)
    try:
        from slm_training.models.twotower import TwoTowerModel

        return TwoTowerModel.from_pretrained(checkpoint)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not load checkpoint {checkpoint!r}: {exc}") from exc


def _format_from_id(format_id: str, group_size: int) -> QuantFormat:
    factory = FORMAT_FACTORIES.get(format_id)
    if factory is None:
        raise ValueError(f"Unknown format: {format_id}")
    return factory(group_size=group_size)


def _synthetic_trace_records(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Generate deterministic grammar_decision records for fixture dry-runs."""
    rng = random.Random(seed)
    actions = ["component:root", "component:arg0", "bind:root", "literal:text", "ref:global"]
    recorder = GrammarTraceRecorder(
        run_id="synthetic",
        checkpoint_id="toy",
        dataset_id="toy",
        example_id="ex",
        seed=seed,
        capture_logits=True,
    )
    for i in range(n):
        legal = rng.sample(actions, k=rng.randint(1, len(actions)))
        selected = rng.choice(legal)
        logits = [rng.gauss(0, 1) for _ in legal]
        idx = legal.index(selected)
        logits[idx] += 1.0
        recorder.record(
            state_fingerprint=f"state-{i % 8}",
            state_signature_version="1",
            legal_action_ids=legal,
            selected_action_id=selected,
            compiler_coverage="complete",
            logits_or_energies=logits,
            scope_signature=f"scope-{i % 3}",
            template_signature=f"tpl-{i % 2}",
            sensitivity={"grad_norm": rng.random()},
            verification_outcome="counterexample" if i % 7 == 0 else None,
        )
    return recorder.finalize()


def _write_trace_store(trace_dir: Path, records: list[dict[str, Any]]) -> None:
    store = TraceStore(trace_dir, run_id="synthetic")
    for record in records:
        record["kind"] = "grammar_decision"
        store.append(record)


def _format_bytes_table(report: MatrixReport) -> list[str]:
    lines = [
        "## Physical storage",
        "",
        "| arm_id | format_id | physical_weight_bytes | total_bytes |",
        "| ------ | --------- | --------------------- | ----------- |",
    ]
    for r in report.arms:
        lines.append(
            f"| {r.arm_id} | {r.format_id} | {r.physical_weight_bytes} | {r.total_bytes} |"
        )
    return lines


def _format_results_table(report: MatrixReport) -> list[str]:
    lines = [
        "## Results",
        "",
        "| arm_id | top1_acc | teacher_top1_acc | flip_rate | KL | margin | mean_regret | cvar90 | zero_rate | support_rate | entropy_bits | status |",
        "| ------ | -------- | ---------------- | --------- | -- | ------ | ----------- | ------ | --------- | ------------ | ------------ | ------ |",
    ]
    for r in report.arms:
        notes = " ".join(r.notes)
        status = f"{r.status} {notes}".strip()
        lines.append(
            f"| {r.arm_id} | {r.top1_accuracy:.4f} | {r.teacher_top1_accuracy:.4f} | "
            f"{r.action_flip_rate:.4f} | {r.kl_to_teacher:.4f} | {r.margin_preservation:.4f} | "
            f"{r.mean_regret:.4f} | {r.cvar90_regret:.4f} | {r.zero_rate:.4f} | "
            f"{r.support_rate:.4f} | {r.symbol_entropy_bits:.4f} | {status} |"
        )
    return lines


def _markdown_report(report: MatrixReport) -> str:
    lines = [
        "# CAP3-03 equal-storage ternary falsification matrix",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Version:* `{report.version}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*Checkpoint:* {report.checkpoint_id}  ",
        f"*Formats:* {list(report.formats)}  ",
        f"*Group size:* {report.group_size}  ",
        f"*Seeds:* {list(report.seeds)}  ",
        f"*Samples:* {report.sample_count}  ",
        f"*Sampling strategy:* {report.sampling_strategy}",
        "",
        "## Honest caveat",
        "",
        "This is a CPU/toy fixture run.  It verifies the CAP3-03 matrix can be",
        "instantiated, every low-bit arm produces a matched-conditions ledger, and",
        "metrics are collected; it does not train a production model or make a",
        "ship-quality claim.  Ship-grade falsification needs GPU + local E224+",
        "checkpoints + full ``--ship-gates``.",
        "",
    ]
    lines.extend(_format_results_table(report))
    lines.append("")
    lines.extend(_format_bytes_table(report))
    lines.append("")
    lines.append("## Matched-conditions parity")
    lines.append("")
    failed = [r for r in report.arms if r.status != "ok"]
    if failed:
        lines.append(f"- **FAIL** {len(failed)} arm(s) failed matched-conditions or evaluation:")
        for r in failed:
            lines.append(f"  - `{r.arm_id}`: {'; '.join(r.notes)}")
    else:
        lines.append("- **PASS** every arm satisfied the matched-conditions parity check.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="toy")
    parser.add_argument("--trace-dir", type=Path, default=Path("outputs/traces/cap3_03"))
    parser.add_argument("--synthetic-traces", type=int, default=None)
    parser.add_argument(
        "--formats",
        type=str,
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated low-bit format ids to compare.",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid_coverage_margin",
        choices=sorted(PRIMARY_STRATEGIES),
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds.")
    parser.add_argument("--qat-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/cap3_03_ternary_falsification"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    if args.dry_run:
        print("CAP3-03 dry-run:")
        print(f"  checkpoint={args.checkpoint}")
        print(f"  formats={formats}")
        print(f"  group_size={args.group_size}")
        print(f"  seeds={seeds}")
        print(f"  samples={args.samples}")
        print(f"  qat_steps={args.qat_steps}")
        return 0

    model = _load_model(args.checkpoint, args.hidden_dim)
    device = torch.device("cpu")
    model.to(device)

    trace_dir = args.trace_dir
    if args.synthetic_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)
        records = _synthetic_trace_records(args.synthetic_traces, seed=seeds[0])
        _write_trace_store(trace_dir, records)

    if not trace_dir.exists():
        raise FileNotFoundError(f"trace directory not found: {trace_dir}")

    samples, coverage, violations = load_grammar_decision_traces(
        trace_dir,
        checkpoint_id=args.checkpoint,
    )
    if violations:
        print(f"warning: {len(violations)} trace violation(s)")

    manifest, selected = build_calibration_corpus(
        samples,
        args.strategy,
        args.samples,
        seed=seeds[0],
        checkpoint_id=args.checkpoint,
        teacher_id=args.checkpoint,
    )

    report = run_matrix(
        model,
        manifest,
        selected,
        formats=formats,
        group_size=args.group_size,
        seeds=seeds,
        qat_steps=args.qat_steps,
        qat_lr=args.lr,
        hidden_dim=args.hidden_dim,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cap3_03_ternary_falsification_{report.run_id}.json"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"cap3_03_ternary_falsification_{report.run_id}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    # Also emit a small metrics envelope for downstream tooling.
    metrics = {
        "run_id": report.run_id,
        "checkpoint_id": report.checkpoint_id,
        "formats": list(report.formats),
        "group_size": report.group_size,
        "seeds": list(report.seeds),
        "sample_count": report.sample_count,
        "sampling_strategy": report.sampling_strategy,
        "arm_count": len(report.arms),
        "ok_count": sum(1 for r in report.arms if r.status == "ok"),
        "error_count": sum(1 for r in report.arms if r.status != "ok"),
        "coverage": coverage,
        "trace_violations": violations,
        "caveat": (
            "CPU/toy fixture wiring evidence; ship-grade falsification requires "
            "GPU + local E224+ checkpoints + full --ship-gates."
        ),
    }
    metrics_path = out_dir / f"cap3_03_ternary_falsification_{report.run_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    failed = any(r.status != "ok" for r in report.arms)
    print(f"Wrote {json_path}, {md_path}, and {metrics_path}")
    print(f"OK arms: {metrics['ok_count']}/{len(report.arms)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
