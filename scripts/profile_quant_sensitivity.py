#!/usr/bin/env python3
"""Profile grammar-conditioned quantization sensitivity (CAP3-04).

Example::

    python -m scripts.profile_quant_sensitivity \
        --checkpoint toy --synthetic-traces 256 --samples 64 \
        --formats ternary,learned4zero,int4,int8 \
        --out-dir outputs/runs/cap3-04-sensitivity

CPU/toy runs are wiring evidence only; ship-grade profiling needs GPU + local
E224+ checkpoints + full --ship-gates.
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
from slm_training.harnesses.quantization.calibration import (
    PRIMARY_STRATEGIES,
    build_calibration_corpus,
    load_grammar_decision_traces,
)
from slm_training.harnesses.quantization.sensitivity import (
    SensitivityReport,
    default_grouping_policy,
    profile_group_sensitivity,
)
from slm_training.models.local_action_head import LocalFlatHead


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


def _markdown_report(report: SensitivityReport) -> str:
    lines = [
        "# CAP3-04 grammar-conditioned quantization sensitivity profile",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Version:* `{report.version}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*Checkpoint:* {report.checkpoint_id}  ",
        f"*Grouping policy:* {report.grouping_policy_version}  ",
        f"*Formats:* {list(report.formats)}  ",
        f"*Samples:* {report.sample_count}",
        "",
        "## Honest caveat",
        "",
        "This is a CPU/toy fixture run. It verifies the sensitivity profiler can",
        "group parameters, quantize one group at a time, measure local-head",
        "perturbation, and restore the baseline. It is not a ship-grade",
        "profiler.",
        "",
        "## Gradient proxies (diagnostic)",
        "",
    ]
    lines.append("| group_id | gradient_proxy |")
    lines.append("| -------- | -------------- |")
    for gid, val in sorted(report.gradient_proxies.items()):
        lines.append(f"| {gid} | {val:.6f} |")
    lines.append("")
    lines.append("## Direct-perturbation points")
    lines.append("")
    lines.append(
        "| group_id | format_id | bytes | flip_rate | KL | margin | mean_regret | cvar90 | status |"
    )
    lines.append(
        "| -------- | --------- | ----- | --------- | -- | ------ | ----------- | ------ | ------ |"
    )
    for p in report.points:
        lines.append(
            f"| {p.group_id} | {p.format_id} | {p.total_bytes} | "
            f"{p.action_flip_rate:.4f} | {p.kl_to_teacher:.4f} | {p.margin_preservation:.4f} | "
            f"{p.mean_regret:.4f} | {p.cvar90_regret:.4f} | {p.status} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="toy")
    parser.add_argument("--trace-dir", type=Path, default=Path("outputs/traces/cap3_04"))
    parser.add_argument("--synthetic-traces", type=int, default=None)
    parser.add_argument(
        "--formats",
        type=str,
        default="ternary,learned4zero,int4,int8",
        help="Comma-separated format ids to profile.",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid_coverage_margin",
        choices=sorted(PRIMARY_STRATEGIES),
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds (only first used).")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/cap3_04_sensitivity"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    if args.dry_run:
        print("CAP3-04 dry-run:")
        print(f"  checkpoint={args.checkpoint}")
        print(f"  formats={formats}")
        print(f"  group_size={args.group_size}")
        print(f"  samples={args.samples}")
        print(f"  grouping_policy={default_grouping_policy().version}")
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

    report = profile_group_sensitivity(
        model,
        manifest,
        selected,
        default_grouping_policy(),
        formats=formats,
        group_size=args.group_size,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"sensitivity_report_{report.run_id}.json"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"sensitivity_report_{report.run_id}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    metrics = {
        "run_id": report.run_id,
        "checkpoint_id": report.checkpoint_id,
        "formats": list(report.formats),
        "group_count": len({p.group_id for p in report.points}),
        "point_count": len(report.points),
        "ok_count": sum(1 for p in report.points if p.status == "ok"),
        "coverage": coverage,
        "trace_violations": violations,
        "caveat": (
            "CPU/toy fixture wiring evidence; ship-grade profiling requires "
            "GPU + local E224+ checkpoints + full --ship-gates."
        ),
    }
    metrics_path = out_dir / f"sensitivity_report_{report.run_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote {json_path}, {md_path}, and {metrics_path}")
    print(f"OK points: {metrics['ok_count']}/{len(report.points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
