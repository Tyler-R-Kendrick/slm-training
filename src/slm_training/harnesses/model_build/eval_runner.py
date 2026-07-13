"""Evaluation runner for ModelPlugin implementations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.plugin import ModelPlugin, StubModel


def _placeholder_fidelity(pred: str, gold: ExampleRecord) -> float:
    pred_set = set(extract_placeholders(pred))
    gold_set = set(gold.placeholders) or set(extract_placeholders(gold.openui))
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    return len(pred_set & gold_set) / len(gold_set)


def _tree_match(pred: str, gold_openui: str) -> float:
    """Canonical OpenUI match via official serializer when possible."""
    if pred.strip() == gold_openui.strip():
        return 1.0
    try:
        from slm_training.dsl import validate

        pred_p = validate(pred)
        gold_p = validate(gold_openui)
        if pred_p.serialized and gold_p.serialized:
            return 1.0 if pred_p.serialized.strip() == gold_p.serialized.strip() else 0.0
    except Exception:  # noqa: BLE001 — fall back to exact string miss
        return 0.0
    return 0.0


def evaluate(
    config: ModelBuildConfig,
    model: ModelPlugin | None = None,
    checkpoint: Path | None = None,
) -> dict:
    if config.test_dir is None:
        raise ValueError("test_dir is required for evaluation")

    records = load_suite_records(config.test_dir, config.suite)
    plugin: ModelPlugin = model or StubModel(
        noise_rate=config.noise_rate,
        seed=config.seed,
    )
    ckpt = checkpoint or (config.checkpoint_dir / "last.pt")
    if ckpt.exists():
        plugin.load(ckpt)

    n = len(records)
    parse_ok = 0
    fidelity_sum = 0.0
    exact_sum = 0.0
    details: list[dict] = []

    for record in records:
        pred = plugin.generate(record.prompt, gold=record)
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
            }
        )

    metrics = {
        "suite": config.suite,
        "n": n,
        "parse_rate": (parse_ok / n) if n else 0.0,
        "placeholder_fidelity": (fidelity_sum / n) if n else 0.0,
        "exact_match": (exact_sum / n) if n else 0.0,
        "checkpoint": str(ckpt),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "eval.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metrics["output"] = str(out_path)
    return metrics
