"""Evaluation runner for ModelPlugin implementations."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import load_suite_records, load_train_records
from slm_training.harnesses.model_build.factory import build_model

_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")


def _placeholder_fidelity(pred: str, gold: ExampleRecord) -> float:
    pred_set = set(extract_placeholders(pred))
    gold_set = set(gold.placeholders) or set(extract_placeholders(gold.openui))
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    return len(pred_set & gold_set) / len(gold_set)


def _tree_match(pred: str, gold_openui: str) -> float:
    if pred.strip() == gold_openui.strip():
        return 1.0
    try:
        pred_p = validate(pred)
        gold_p = validate(gold_openui)
        if pred_p.serialized and gold_p.serialized:
            return (
                1.0
                if pred_p.serialized.strip() == gold_p.serialized.strip()
                else 0.0
            )
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def _component_multiset(source: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in _COMPONENT_RE.findall(source):
        counts[name] = counts.get(name, 0) + 1
    return counts


def structural_similarity(pred: str, gold_openui: str) -> float:
    """Jaccard-like similarity over component multisets + depth proxy."""
    pred_c = _component_multiset(pred)
    gold_c = _component_multiset(gold_openui)
    keys = set(pred_c) | set(gold_c)
    if not keys:
        return 0.0
    inter = sum(min(pred_c.get(k, 0), gold_c.get(k, 0)) for k in keys)
    union = sum(max(pred_c.get(k, 0), gold_c.get(k, 0)) for k in keys)
    jaccard = inter / union if union else 0.0
    depth_p = pred.count("[") + pred.count("(")
    depth_g = gold_openui.count("[") + gold_openui.count("(")
    depth_sim = 1.0 - min(1.0, abs(depth_p - depth_g) / max(1, depth_g))
    return round(0.7 * jaccard + 0.3 * depth_sim, 4)


def _design_lint_for_record(record: ExampleRecord) -> float | None:
    if not record.design_md:
        return None
    try:
        from slm_training.design_md import bridge_available, lint

        if not bridge_available():
            return None
        return float(lint(record.design_md).get("score") or 0.0)
    except Exception:  # noqa: BLE001
        return None


def evaluate(
    config: ModelBuildConfig,
    model=None,
    checkpoint: Path | None = None,
) -> dict:
    if config.test_dir is None:
        raise ValueError("test_dir is required for evaluation")

    records = load_suite_records(config.test_dir, config.suite)
    ckpt = checkpoint or (config.checkpoint_dir / "last.pt")

    if model is not None:
        plugin = model
        if ckpt.exists() and hasattr(plugin, "load"):
            try:
                plugin.load(ckpt)
            except Exception:  # noqa: BLE001 — model may already be loaded
                pass
    else:
        train_records = []
        if config.train_dir.exists():
            try:
                train_records = load_train_records(config.train_dir)
            except FileNotFoundError:
                train_records = []
        plugin = build_model(
            config,
            train_records or records,
            checkpoint=ckpt if ckpt.exists() else None,
        )

    n = len(records)
    parse_ok = 0
    fidelity_sum = 0.0
    exact_sum = 0.0
    struct_sum = 0.0
    design_scores: list[float] = []
    latencies: list[float] = []
    details: list[dict] = []

    for record in records:
        t0 = time.perf_counter()
        # Pass gold for DESIGN.md context when the model supports it.
        try:
            pred = plugin.generate(record.prompt, gold=record)
        except TypeError:
            pred = plugin.generate(record.prompt, gold=None)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ok = False
        error = None
        try:
            validate(pred)
            ok = True
            parse_ok += 1
        except ParseError as exc:
            error = str(exc)
        fid = _placeholder_fidelity(pred, record)
        exact = _tree_match(pred, record.openui)
        struct = structural_similarity(pred, record.openui)
        dscore = _design_lint_for_record(record)
        fidelity_sum += fid
        exact_sum += exact
        struct_sum += struct
        if dscore is not None:
            design_scores.append(dscore)
        details.append(
            {
                "id": record.id,
                "parse_ok": ok,
                "error": error,
                "placeholder_fidelity": fid,
                "exact_match": exact,
                "structural_similarity": struct,
                "design_lint_score": dscore,
                "latency_ms": round(latencies[-1], 2),
                "prediction": pred[:500],
            }
        )

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else None
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else None

    metrics = {
        "suite": config.suite,
        "n": n,
        "parse_rate": (parse_ok / n) if n else 0.0,
        "placeholder_fidelity": (fidelity_sum / n) if n else 0.0,
        "exact_match": (exact_sum / n) if n else 0.0,
        "structural_similarity": (struct_sum / n) if n else 0.0,
        "design_lint_score": (
            sum(design_scores) / len(design_scores) if design_scores else None
        ),
        "latency_ms_p50": round(p50, 2) if p50 is not None else None,
        "latency_ms_p95": round(p95, 2) if p95 is not None else None,
        "checkpoint": str(ckpt),
        "model": config.model_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "eval.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metrics["output"] = str(out_path)
    return metrics
