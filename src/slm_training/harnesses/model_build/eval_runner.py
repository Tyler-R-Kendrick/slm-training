"""Evaluation runner for ModelPlugin implementations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import load_suite_records, load_train_records
from slm_training.harnesses.model_build.factory import build_model


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
        # Build from train records for vocab when needed, then load weights
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
    details: list[dict] = []

    for record in records:
        # No gold oracle — prompt only
        pred = plugin.generate(record.prompt, gold=None)
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
        fidelity_sum += fid
        exact_sum += exact
        details.append(
            {
                "id": record.id,
                "parse_ok": ok,
                "error": error,
                "placeholder_fidelity": fid,
                "exact_match": exact,
                "prediction": pred[:500],
            }
        )

    metrics = {
        "suite": config.suite,
        "n": n,
        "parse_rate": (parse_ok / n) if n else 0.0,
        "placeholder_fidelity": (fidelity_sum / n) if n else 0.0,
        "exact_match": (exact_sum / n) if n else 0.0,
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
