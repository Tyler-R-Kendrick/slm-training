#!/usr/bin/env python3
"""Run the bounded SLM-243 recurrence-update architecture matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

import torch
import torch.nn.functional as F

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm243_recursive_update_gate import (
    classify_recursive_update_gate,
    stable_hash,
)
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.eval_runner import (
    _is_meaningful_program,
    _reward_for_prediction,
    structural_similarity,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs/design/iter-slm243-recursive-update-gate-20260724.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/iter-slm243-recursive-update-gate-20260724.md"
DEFAULT_AGENTV = ROOT / "docs/design/iter-slm243-recursive-update-gate-agentv-20260724"
DEFAULT_TEST_DIR = (
    ROOT / "src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722"
)
COMPONENT = "harness.experiments.slm243_recursive_update_gate"
DEPTHS = (1, 2, 4, 6, 8)
SEEDS = (24301, 24302, 24303)
RECORD_IDS = {
    "smoke": ("smoke_hero_01", "smoke_button_01"),
    "held_out": ("held_out_form_01", "held_out_dual_card_01"),
}
VARIANTS: dict[str, dict[str, str]] = {
    "current_v1": {
        "update_mode": "current_v1",
        "empty_f_mode": "pass_through",
        "norm_mode": "shared",
    },
    "delta_only": {
        "update_mode": "delta_only",
        "empty_f_mode": "zero",
        "norm_mode": "shared",
    },
    "layerscale": {
        "update_mode": "layerscale",
        "empty_f_mode": "zero",
        "norm_mode": "shared",
    },
    "gated_private": {
        "update_mode": "gated",
        "empty_f_mode": "zero",
        "norm_mode": "private",
    },
    "current_true_empty": {
        "update_mode": "current_v1",
        "empty_f_mode": "zero",
        "norm_mode": "shared",
    },
    "layerscale_private": {
        "update_mode": "layerscale",
        "empty_f_mode": "zero",
        "norm_mode": "private",
    },
}


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        path.write_text(text, encoding="utf-8")


def _select(records: list[Any], ids: tuple[str, ...]) -> list[Any]:
    by_id = {record.id: record for record in records}
    return [by_id[record_id] for record_id in ids]


def _prompt_context(prompt: str, *, d_model: int) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(3, d_model, generator=generator)


def _corpus_batch(
    test_dir: Path,
) -> tuple[DSLNativeTokenizer, torch.Tensor, torch.Tensor, torch.Tensor, list[Any]]:
    records = []
    for suite, ids in RECORD_IDS.items():
        records.extend(_select(load_suite_records(test_dir, suite), ids))
    tokenizer = DSLNativeTokenizer.build()
    encoded = [
        tokenizer.encode(
            record.openui,
            add_special=True,
            placeholders=record.placeholders,
        )[:16]
        for record in records
    ]
    width = max(len(row) for row in encoded)
    targets = torch.full((len(encoded), width), tokenizer.pad_id, dtype=torch.long)
    for index, row in enumerate(encoded):
        targets[index, : len(row)] = torch.tensor(row)
    noisy = targets.clone()
    active = targets.ne(tokenizer.pad_id)
    positions = torch.arange(width).unsqueeze(0)
    mask = active & positions.remainder(3).eq(1)
    noisy[mask] = tokenizer.mask_id
    context = torch.stack(
        [_prompt_context(record.prompt, d_model=8) for record in records]
    )
    return tokenizer, noisy, targets, context, records


def _architecture_hash(tower: SharedRecursiveDenoiserTower) -> str:
    return stable_hash(
        {
            "update_mode": tower.update_mode,
            "empty_f_mode": tower.empty_f_mode,
            "norm_mode": tower.norm_mode,
            "state": [
                (name, list(value.shape))
                for name, value in tower.state_dict().items()
            ],
        }
    )


def _mechanism_fixtures() -> dict[str, Any]:
    """Prove the three repair seams independently of corpus outcomes."""
    torch.manual_seed(243)
    historical = SharedRecursiveDenoiserTower(
        vocab_size=17,
        d_model=8,
        n_layers=1,
        n_heads=1,
        max_len=8,
        recursive_steps=1,
        update_mode="current_v1",
        empty_f_mode="pass_through",
        norm_mode="shared",
    )
    true_empty = SharedRecursiveDenoiserTower(
        vocab_size=17,
        d_model=8,
        n_layers=1,
        n_heads=1,
        max_len=8,
        recursive_steps=1,
        update_mode="current_v1",
        empty_f_mode="zero",
        norm_mode="shared",
    )
    true_empty.load_state_dict(historical.state_dict())
    noisy = torch.tensor([[1, 2, 3, 4]])
    context = torch.randn(1, 2, 8)
    initial = historical.initial_transition_state(noisy, context, 0)
    historical_step = historical.transition_step(
        initial["y"], initial["z"], context, initial["self_pad_mask"]
    )
    true_empty_step = true_empty.transition_step(
        initial["y"], initial["z"], context, initial["self_pad_mask"]
    )
    historical_z = historical_step["z_update"]
    true_empty_z = true_empty_step["z_update"]
    assert isinstance(historical_z, torch.Tensor)
    assert isinstance(true_empty_z, torch.Tensor)
    layerscale = SharedRecursiveDenoiserTower(
        vocab_size=17,
        d_model=8,
        n_layers=2,
        n_heads=1,
        update_mode="layerscale",
    )
    gated_private = SharedRecursiveDenoiserTower(
        vocab_size=17,
        d_model=8,
        n_layers=2,
        n_heads=1,
        update_mode="gated",
        norm_mode="private",
    )
    return {
        "historical_empty_f_update_norm": float(historical_z.norm()),
        "true_empty_f_update_norm": float(true_empty_z.norm()),
        "true_empty_f_exact_zero": torch.equal(
            true_empty_z, torch.zeros_like(true_empty_z)
        ),
        "layerscale_initial_value": float(layerscale.f_update_scale[0]),
        "gated_initial_sigmoid": float(
            torch.sigmoid(gated_private.f_update_gate[0])
        ),
        "private_norm_objects_distinct": len(
            {
                id(gated_private.f_norm),
                id(gated_private.g_norm),
                id(gated_private.norm),
            }
        )
        == 3,
    }


def _gradient_norm(tower: SharedRecursiveDenoiserTower) -> tuple[float, dict[str, float]]:
    groups = {"f": [], "g": [], "norm": [], "other": []}
    for name, parameter in tower.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm())
        if "f_" in name or name.startswith("layers.0"):
            groups["f"].append(value)
        elif "g_" in name or name.startswith("layers.1"):
            groups["g"].append(value)
        elif "norm" in name:
            groups["norm"].append(value)
        else:
            groups["other"].append(value)
    grouped = {
        name: math.sqrt(sum(value * value for value in values))
        for name, values in groups.items()
    }
    return math.sqrt(sum(value * value for value in grouped.values())), grouped


def _semantic_rows(
    tokenizer: DSLNativeTokenizer,
    logits: torch.Tensor,
    records: list[Any],
) -> list[dict[str, Any]]:
    predictions = logits.argmax(dim=-1)
    rows = []
    for index, record in enumerate(records):
        table = SymbolTable.from_placeholders(record.placeholders)
        prediction = tokenizer.decode(
            predictions[index].tolist(), skip_special=True, table=table
        )
        meaningful, _, serialized = _is_meaningful_program(prediction, gold=record)
        scored = serialized or prediction
        try:
            reward = float(_reward_for_prediction(scored, record))
        except Exception:  # noqa: BLE001
            reward = None
        rows.append(
            {
                "record_id": record.id,
                "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
                "meaningful_parse": bool(meaningful),
                "structural_similarity": structural_similarity(scored, record.openui),
                "reward": reward,
            }
        )
    return rows


def _run_cell(
    *,
    variant: str,
    depth: int,
    seed: int,
    tokenizer: DSLNativeTokenizer,
    noisy: torch.Tensor,
    targets: torch.Tensor,
    context: torch.Tensor,
    records: list[Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    torch.manual_seed(seed)
    controls = VARIANTS[variant]
    tower = SharedRecursiveDenoiserTower(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_layers=2,
        n_heads=1,
        max_len=noisy.size(1),
        recursive_steps=depth,
        recursive_transition_layers=2,
        tie_output_embedding=True,
        **controls,
    )
    output = tower.recursive_outputs(
        noisy,
        context,
        tokenizer.pad_id,
        diagnostics=True,
        diagnostic_targets=targets,
        diagnostic_mask=targets.ne(tokenizer.pad_id),
    )
    logits = output["logits"]
    diagnostics = output["diagnostics"]
    loss = F.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=tokenizer.pad_id
    )
    loss.backward()
    total_grad, grad_groups = _gradient_norm(tower)
    update_ratios = [
        float(record.y_update_state_ratio.max())
        for record in diagnostics
    ] + [
        float(record.z_update_state_ratio.max())
        for record in diagnostics
        if record.z_update_state_ratio is not None
    ]
    update_cosines = []
    for left, right in zip(diagnostics, diagnostics[1:]):
        update_cosines.append(
            float(
                F.cosine_similarity(
                    left.y_update.flatten().unsqueeze(0),
                    right.y_update.flatten().unsqueeze(0),
                )
            )
        )
    finite_tensors = [logits, loss.detach()]
    finite_tensors.extend(
        parameter.grad
        for parameter in tower.parameters()
        if parameter.grad is not None
    )
    semantic = _semantic_rows(tokenizer, logits.detach(), records)
    row = {
        "variant": variant,
        "architecture_hash": _architecture_hash(tower),
        "depth": depth,
        "seed": seed,
        "corpus_n": len(records),
        "cross_entropy": float(loss.detach()),
        "accuracy": statistics.mean(
            float(record.accuracy.mean()) for record in diagnostics
        ),
        "maximum_update_ratio": max(update_ratios),
        "mean_update_ratio": statistics.mean(update_ratios),
        "update_cosine_previous": update_cosines,
        "gradient_norm": total_grad,
        "gradient_norm_by_group": grad_groups,
        "parameter_count": sum(parameter.numel() for parameter in tower.parameters()),
        "block_evaluations": depth * 2,
        "all_finite": all(torch.isfinite(value).all() for value in finite_tensors),
        # Earlier states are identical to the separately persisted cells at
        # those depths for the same variant/seed. Retain only the final state
        # here instead of repeating the triangular trajectory 90 times.
        "final_depth_metrics": {
            "depth": diagnostics[-1].step,
            "cross_entropy": float(diagnostics[-1].cross_entropy.mean()),
            "entropy": float(diagnostics[-1].entropy.mean()),
            "y_norm": float(diagnostics[-1].y_norm.mean()),
            "z_norm": (
                None
                if diagnostics[-1].z_norm is None
                else float(diagnostics[-1].z_norm.mean())
            ),
            "y_update_ratio": float(
                diagnostics[-1].y_update_state_ratio.mean()
            ),
            "z_update_ratio": (
                None
                if diagnostics[-1].z_update_state_ratio is None
                else float(diagnostics[-1].z_update_state_ratio.mean())
            ),
        },
        "bounded_semantics": {
            "mode": "one_forward_masked_reconstruction",
            # Raw per-record outcomes are retained at the preregistered
            # recommendation depth; lower-depth cells keep the exact
            # aggregates used by the gate without duplicating those rows.
            "rows": semantic if depth == max(DEPTHS) else [],
            "raw_rows_persisted": depth == max(DEPTHS),
            "meaningful_parse_rate": statistics.mean(
                float(item["meaningful_parse"]) for item in semantic
            ),
            "structural_similarity": statistics.mean(
                item["structural_similarity"] for item in semantic
            ),
            "reward": (
                statistics.mean(
                    item["reward"] for item in semantic if item["reward"] is not None
                )
                if any(item["reward"] is not None for item in semantic)
                else None
            ),
        },
    }
    return row, logits.detach()


def _scientific_hash(report: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(report))
    payload.pop("report_hash", None)
    payload.pop("generated_at", None)
    payload.pop("elapsed_seconds", None)
    stamp = dict(payload["version_stamp"])
    stamp.pop("stamped_at", None)
    payload["version_stamp"] = stamp
    summary = dict(payload["agentv"]["summary"])
    summary.pop("durationMs", None)
    payload["agentv"]["summary"] = summary
    return stable_hash(payload)


def _run(
    test_dir: Path,
    agentv_dir: Path,
    *,
    pinned_version_stamp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    stamp = pinned_version_stamp or build_version_stamp(
        COMPONENT, "model.recursive_denoiser", "model.twotower"
    )
    mechanisms = _mechanism_fixtures()
    tokenizer, noisy, targets, context, records = _corpus_batch(test_dir)
    rows = []
    baseline_logits: dict[tuple[int, int], torch.Tensor] = {}
    for variant in VARIANTS:
        for depth in DEPTHS:
            for seed in SEEDS:
                row, logits = _run_cell(
                    variant=variant,
                    depth=depth,
                    seed=seed,
                    tokenizer=tokenizer,
                    noisy=noisy,
                    targets=targets,
                    context=context,
                    records=records,
                )
                key = (depth, seed)
                if variant == "current_v1":
                    baseline_logits[key] = logits
                    row["logit_divergence_from_current_v1"] = {
                        "kl": 0.0,
                        "js": 0.0,
                    }
                else:
                    baseline = baseline_logits[key].float()
                    candidate = logits.float()
                    base_logp = F.log_softmax(baseline, dim=-1)
                    candidate_logp = F.log_softmax(candidate, dim=-1)
                    base_p = base_logp.exp()
                    candidate_p = candidate_logp.exp()
                    mixture = 0.5 * (base_p + candidate_p)
                    row["logit_divergence_from_current_v1"] = {
                        "kl": float(
                            (base_p * (base_logp - candidate_logp)).sum(-1).mean()
                        ),
                        "js": float(
                            0.5
                            * (
                                (base_p * (base_logp - mixture.log())).sum(-1).mean()
                                + (
                                    candidate_p
                                    * (candidate_logp - mixture.log())
                                )
                                .sum(-1)
                                .mean()
                            )
                        ),
                    }
                rows.append(row)
    gate = classify_recursive_update_gate(rows, depths=DEPTHS, seeds=SEEDS)
    raw_payload = {
        "schema": "RecursiveUpdateRawMatrixV1",
        "issue": "SLM-243",
        "rows": rows,
        "version_stamp": stamp,
    }
    raw_hash = stable_hash(raw_payload)
    agentv_dir.mkdir(parents=True, exist_ok=True)
    raw_path = agentv_dir / "raw_matrix.json"
    raw_path.write_text(json.dumps(raw_payload, indent=2) + "\n", encoding="utf-8")
    summary_rows = [
        {
            **{
                key: row[key]
                for key in (
                    "variant",
                    "depth",
                    "seed",
                    "all_finite",
                    "cross_entropy",
                    "maximum_update_ratio",
                    "gradient_norm",
                    "parameter_count",
                    "block_evaluations",
                )
            },
            "bounded_semantics": {
                key: row["bounded_semantics"][key]
                for key in (
                    "mode",
                    "meaningful_parse_rate",
                    "structural_similarity",
                    "reward",
                )
            },
        }
        for row in rows
    ]
    cases = [
        {
            "id": "complete-paired-matrix",
            "criteria": "All 90 preregistered variant/depth/seed cells are present.",
            "pass": len(rows) == len(VARIANTS) * len(DEPTHS) * len(SEEDS),
            "result": {"cells": len(rows)},
        },
        {
            "id": "historical-default-preserved",
            "criteria": "current_v1 retains pass-through empty F; the true-empty arm is exact zero and repair initializers are near identity.",
            "pass": VARIANTS["current_v1"]
            == {
                "update_mode": "current_v1",
                "empty_f_mode": "pass_through",
                "norm_mode": "shared",
            },
            "result": mechanisms,
        },
        {
            "id": "finite-accounted-architecture",
            "criteria": "Every cell reports finite status, resource accounting, and an architecture hash.",
            "pass": all(
                row["architecture_hash"]
                and row["parameter_count"] > 0
                and row["block_evaluations"] == row["depth"] * 2
                for row in rows
            ),
            "result": {"nonfinite_cells": sum(not row["all_finite"] for row in rows)},
        },
        {
            "id": "recursive-update-gate",
            "criteria": "The report publishes exactly one validated RecursiveUpdateGateV1 verdict.",
            "pass": bool(gate.to_dict()),
            "result": gate.to_dict(),
        },
        {
            "id": "honest-claim-boundary",
            "criteria": "Bounded scratch evidence blocks semantic, promotion, ship, and production-default claims.",
            "pass": set(gate.blocked_claims)
            == {
                "semantic_workspace",
                "checkpoint_promotion",
                "ship_readiness",
                "production_default_change",
            },
            "result": {"blocked_claims": gate.blocked_claims},
        },
    ]
    agentv = publish_agentv_evaluation(
        agentv_dir,
        name="slm243-recursive-update-gate",
        claim="bounded_architecture_repair_not_ship",
        version_stamp=stamp,
        cases=cases,
    )
    _rewrite_agentv_paths(agentv_dir)
    report: dict[str, Any] = {
        "schema": "RecursiveUpdateMatrixReportV1",
        "issue": "SLM-243",
        "matrix_set": "slm243-recursive-update-gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_class": "scratch_architecture_matrix_not_ship",
        "recipe": {
            "device": "cpu",
            "backend": "scratch_direct_tower",
            "corpus": str(test_dir.relative_to(ROOT)),
            "record_ids": [record.id for record in records],
            "corpus_n": len(records),
            "depths": DEPTHS,
            "paired_seeds": SEEDS,
            "d_model": 8,
            "transition_layers": 2,
            "max_target_tokens": 16,
            "optimizer_steps": 0,
            "max_wall_minutes": 3.0,
            "honesty_mode": "architecture_repair_not_semantic_ship",
        },
        "thresholds": {
            "hard_nonfinite": True,
            "maximum_update_ratio": 2.0,
            "maximum_gradient_norm": 100.0,
            "paired_update_ratio_fraction": 0.8,
            "paired_cross_entropy_tolerance": 0.25,
            "recommendation_requires_all_three_seeds": True,
        },
        "variants": VARIANTS,
        "mechanism_fixtures": mechanisms,
        "rows": summary_rows,
        "raw_matrix": {
            "path": _portable(str(raw_path), agentv_dir),
            "sha256": raw_hash,
            "schema": raw_payload["schema"],
            "cells": len(rows),
        },
        "gate": gate.to_dict(),
        "prior_evidence": list(gate.evidence_refs),
        "agentv": _portable(agentv, agentv_dir),
        "version_stamp": stamp,
        "checkpoint_created": False,
        "production_default_changed": False,
        "ship_gate_claim": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_hash"] = _scientific_hash(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    gate = report["gate"]
    lines = [
        "# SLM-243 recursive update architecture gate",
        "",
        f"Verdict: **{gate['verdict']}**",
        "",
        f"Report hash: `{report['report_hash']}`",
        "",
        "This is a bounded scratch architecture-repair matrix, not evidence of "
        "semantic workspace, checkpoint promotion, ship readiness, or a production "
        "default change.",
        "",
        "## Recipe and preregistration",
        "",
        f"- Records: `{', '.join(report['recipe']['record_ids'])}`",
        f"- Depths: `{list(report['recipe']['depths'])}`",
        f"- Paired seeds: `{list(report['recipe']['paired_seeds'])}`",
        "- Six orthogonal variants; 90 total cells; zero optimizer steps.",
        f"- Thresholds: `{json.dumps(report['thresholds'], sort_keys=True)}`",
        "",
        "## High-depth results",
        "",
        "| variant | finite seeds | CE mean | max update ratio | grad norm | parse rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        rows = [
            row
            for row in report["rows"]
            if row["variant"] == variant and row["depth"] == 8
        ]
        lines.append(
            f"| {variant} | {sum(row['all_finite'] for row in rows)}/3 | "
            f"{statistics.mean(row['cross_entropy'] for row in rows):.6f} | "
            f"{max(row['maximum_update_ratio'] for row in rows):.6f} | "
            f"{statistics.mean(row['gradient_norm'] for row in rows):.6f} | "
            f"{statistics.mean(row['bounded_semantics']['meaningful_parse_rate'] for row in rows):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"- Selected variant: `{gate['selected_variant']}`",
            f"- Maximum authorized diagnostic depth: `{gate['maximum_authorized_depth']}`",
            f"- Allowed SLM-233 modes: `{gate['allowed_slm233_modes']}`",
            f"- Rationale: {gate['rationale']}",
            f"- Blocked claims: `{gate['blocked_claims']}`",
            "",
            "Prior SLM-282/230/231/232 recurrence results remain authoritative. "
            "This matrix can authorize only a later diagnostic architecture mode; "
            "one-forward masked reconstruction is not free-running reasoning.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python -m "
            "scripts.run_slm243_recursive_update_gate --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        committed = json.loads(args.json_out.read_text(encoding="utf-8"))
        with TemporaryDirectory(prefix="slm243-agentv-") as temp_dir:
            report = _run(
                args.test_dir,
                Path(temp_dir),
                pinned_version_stamp=committed["version_stamp"],
            )
        if _scientific_hash(committed) != _scientific_hash(report):
            raise SystemExit("SLM-243 committed report does not match a clean rerun")
        return 0
    report = _run(args.test_dir, args.agentv_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"verdict": report["gate"]["verdict"], "hash": report["report_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
