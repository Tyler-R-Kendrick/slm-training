#!/usr/bin/env python3
"""Calibration + low-bit adaptation CLI for the local scorer (CAP3-02).

Builds a grammar-stratified calibration corpus from CAP1-02 traces and runs
PTQ scale calibration, short QAT reconstruction, or distillation-objective
adaptation on a ``LocalActionHead``.  CPU/toy runs are wiring evidence only;
they do not claim ship-grade retention or speedup.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.harnesses.distill.grammar_trace import GrammarTraceRecorder
from slm_training.harnesses.quantization.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    PRIMARY_STRATEGIES,
    _fake_quantize_with_ste,
    build_calibration_corpus,
    calibrate_scales_ptq,
    load_grammar_decision_traces,
    mixed_task_distillation_objective,
    qat_reconstruct_local_scorer,
)
from slm_training.models.local_action_head import LocalActionHead, LocalFlatHead, StateContext
from slm_training.models.quantization import (
    binary_format,
    binary_plus_mask_format,
    build_model_ledger,
    int4_format,
    int8_format,
    learned_four_level_zero_format,
    symmetric_four_level_format,
    ternary_format,
)
from slm_training.models.quantization.convert import QuantizationPolicy, convert_twotower
from slm_training.models.quantization.formats import QuantFormat


FORMAT_FACTORIES: dict[str, Any] = {
    "int8": int8_format,
    "int4": int4_format,
    "binary": binary_format,
    "ternary": ternary_format,
    "symmetric_four_level": symmetric_four_level_format,
    "learned4zero": learned_four_level_zero_format,
    "learned_four_level_zero": learned_four_level_zero_format,
    "binary_plus_mask": binary_plus_mask_format,
}


class ToyLocalModel(nn.Module):
    """Minimal model owning a local action head for dry-run wiring."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.local_head = LocalFlatHead(hidden_dim)


def _find_target_heads(
    model: nn.Module,
    target: str,
) -> list[tuple[str, LocalActionHead]]:
    """Return (path, head) pairs matching the requested target."""
    if target == "local_action_head":
        return [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, LocalActionHead)
        ]
    # exact path match
    for name, module in model.named_modules():
        if name == target and isinstance(module, LocalActionHead):
            return [(name, module)]
    return []


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


def _format_from_args(args: argparse.Namespace) -> QuantFormat:
    factory = FORMAT_FACTORIES.get(args.format)
    if factory is None:
        raise ValueError(f"Unknown format: {args.format}")
    return factory(group_size=args.group_size)


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
        logits[idx] += 1.0  # bias selected action
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
    from slm_training.harnesses.distill.trace_store import TraceStore

    store = TraceStore(trace_dir, run_id="synthetic")
    for record in records:
        record["kind"] = "grammar_decision"
        store.append(record)


def _sample_to_batch(
    sample: Any,
    head: LocalFlatHead,
    hidden_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Build a synthetic (hidden, legal_actions, teacher_logits) batch from a sample."""
    legal = list(sample.legal_action_ids)
    if not legal:
        legal = ["__noop__"]
    hidden = torch.randn(1, hidden_dim, device=device)
    teacher_logits = torch.zeros(1, len(legal), device=device)
    if sample.selected_action_id is not None and sample.selected_action_id in legal:
        idx = legal.index(sample.selected_action_id)
        teacher_logits[0, idx] = sample.top1_margin if sample.top1_margin is not None else 1.0
    # Warm embeddings so they exist before QAT.
    with torch.no_grad():
        head.score(hidden, StateContext(state_family_id="synthetic"), legal)
    return hidden, legal, teacher_logits


def _run_ptq(
    head: LocalFlatHead,
    fmt: QuantFormat,
    samples: list[Any],
) -> dict[str, Any]:
    """PTQ: calibrate scales for every action embedding touched by samples."""
    mses: list[float] = []
    actions = {a for s in samples for a in s.legal_action_ids}
    for action in actions:
        if action not in head.action_embeddings:
            continue
        param = head.action_embeddings[action]
        q, scale, _ = calibrate_scales_ptq(param, fmt)
        mse = float((q - param).pow(2).mean().item())
        mses.append(mse)
    return {
        "mode": "ptq",
        "quantized_actions": len(mses),
        "mean_embedding_mse": sum(mses) / len(mses) if mses else None,
        "scale_shape": tuple(scale.shape) if "scale" in locals() and scale is not None else (),
    }


def _run_distill(
    head: LocalFlatHead,
    fmt: QuantFormat,
    batches: list[tuple[torch.Tensor, list[str], torch.Tensor]],
    steps: int,
    lr: float,
    task_weight: float,
    distill_weight: float,
) -> dict[str, Any]:
    """Short distillation loop using mixed task + KL objective."""
    parameters = list(head.action_embeddings.values())
    if not parameters:
        return {"mode": "distill", "status": "no_parameters"}
    optimizer = torch.optim.SGD(parameters, lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        step_loss = 0.0
        count = 0
        for hidden, legal_actions, teacher_logits in batches:
            # Build student logits with quantized embeddings.
            embeddings = []
            for action in legal_actions:
                param = head.action_embeddings[action]
                q = _fake_quantize_with_ste(param, fmt)
                embeddings.append(q)
            stacked = torch.stack(embeddings, dim=0)
            student_logits = hidden @ stacked.T
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            target = teacher_probs.argmax(dim=-1)
            loss = mixed_task_distillation_objective(
                student_logits,
                teacher_probs,
                target,
                task_weight=task_weight,
                distill_weight=distill_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss += float(loss.item())
            count += 1
        losses.append(step_loss / max(count, 1))
    return {
        "mode": "distill",
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "loss_history": losses,
    }


def _build_ledger(model: nn.Module, fmt: QuantFormat) -> dict[str, Any]:
    """Return a physical-cost ledger for the model under the chosen format."""
    policy = QuantizationPolicy(default_format=fmt)
    try:
        converted, _ = convert_twotower(model, policy, fail_on_tied=False, in_place=False)
    except Exception:  # noqa: BLE001
        converted = model
    ledger = build_model_ledger(converted, {}, default_format=fmt)
    return ledger.as_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--trace-dir", type=Path, default=Path("outputs/traces/calibration"))
    parser.add_argument("--target", type=str, default="local_action_head")
    parser.add_argument("--format", type=str, default="ternary")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid_coverage_margin",
        choices=sorted(PRIMARY_STRATEGIES),
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--qat-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--calibration-mode",
        type=str,
        default="qat",
        choices=["ptq", "qat", "distill", "mixed"],
    )
    parser.add_argument("--task-weight", type=float, default=0.5)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/calibration"))
    parser.add_argument("--docs-out", type=Path, default=None)
    parser.add_argument("--synthetic-traces", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    fmt = _format_from_args(args)
    model = _load_model(args.checkpoint, args.hidden_dim)
    device = torch.device("cpu")
    model.to(device)

    trace_dir = args.trace_dir
    if args.synthetic_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)
        records = _synthetic_trace_records(args.synthetic_traces, seed=args.seed)
        _write_trace_store(trace_dir, records)

    if not trace_dir.exists():
        raise FileNotFoundError(f"trace directory not found: {trace_dir}")

    targets = _find_target_heads(model, args.target)
    if not targets:
        print(f"warning: no target heads matched {args.target!r}")

    manifest, selected = build_calibration_corpus(
        load_grammar_decision_traces(trace_dir, checkpoint_id=args.checkpoint or None)[0],
        args.strategy,
        args.samples,
        seed=args.seed,
        checkpoint_id=args.checkpoint or "toy",
        teacher_id=args.checkpoint or "toy",
    )

    adaptation_results: list[dict[str, Any]] = []
    for path, head in targets:
        if isinstance(head, LocalFlatHead):
            batches = [
                _sample_to_batch(s, head, args.hidden_dim, device)
                for s in selected
            ]
            if args.calibration_mode == "ptq":
                result = _run_ptq(head, fmt, selected)
            elif args.calibration_mode in {"qat", "mixed"}:
                result = qat_reconstruct_local_scorer(
                    head,
                    fmt,
                    batches,
                    steps=args.qat_steps,
                    lr=args.lr,
                )
            else:  # distill
                result = _run_distill(
                    head,
                    fmt,
                    batches,
                    steps=args.qat_steps,
                    lr=args.lr,
                    task_weight=args.task_weight,
                    distill_weight=args.distill_weight,
                )
            result["target_path"] = path
            adaptation_results.append(result)

    ledger = _build_ledger(model, fmt)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "selected_samples.jsonl").write_text(
        "\n".join(json.dumps(s.to_dict()) for s in selected) + "\n",
        encoding="utf-8",
    )
    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )

    metrics = {
        "run_id": out_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "format": args.format,
        "group_size": args.group_size,
        "strategy": args.strategy,
        "samples": args.samples,
        "calibration_mode": args.calibration_mode,
        "manifest_schema": CALIBRATION_SCHEMA_VERSION,
        "adaptation": adaptation_results,
        "evidence": {
            "selected_count": len(selected),
            "unique_selected_states": len({s.state_fingerprint for s in selected}),
            "coverage": manifest.coverage_fields,
        },
        "caveat": (
            "Reference fake-quantization calibration on CPU/toy data; "
            "ship-grade run requires GPU + local E224+ checkpoints + full --ship-gates."
        ),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    docs_out = args.docs_out
    if docs_out:
        docs_out.parent.mkdir(parents=True, exist_ok=True)
        docs_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {docs_out}")

    print(json.dumps(metrics, indent=2))
    print(f"wrote outputs under {out_dir}")
    if args.dry_run:
        print("dry-run: no checkpoint or model card updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
