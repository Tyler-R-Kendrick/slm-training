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
    """Exact placeholder overlap with gold (strict)."""
    pred_set = set(extract_placeholders(pred))
    gold_set = set(gold.placeholders) or set(extract_placeholders(gold.openui))
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    return len(pred_set & gold_set) / len(gold_set)


def _normalize_placeholder(token: str) -> str:
    """Drop leading namespace segment so :smoke.hero.title ~= :hero.title."""
    body = token[1:] if token.startswith(":") else token
    parts = body.split(".")
    if len(parts) >= 3:
        return ".".join(parts[1:])
    return body


def _placeholder_validity(pred: str, gold: ExampleRecord) -> float:
    """
    Soft placeholder quality for diagnostics only (not a ship gate alone).
    Prefer placeholder_fidelity for readiness claims.
    """
    pred_set = set(extract_placeholders(pred))
    gold_set = set(gold.placeholders) or set(extract_placeholders(gold.openui))
    if not gold_set:
        return 1.0 if not pred_set else 0.5
    if not pred_set:
        return 0.0
    well_formed = sum(1 for p in pred_set if p.startswith(":") and "." in p) / len(pred_set)
    pred_n = {_normalize_placeholder(p) for p in pred_set}
    gold_n = {_normalize_placeholder(p) for p in gold_set}
    overlap = len(pred_n & gold_n) / len(gold_n) if gold_n else 0.0
    return round(0.4 * well_formed + 0.6 * overlap, 4)


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


def component_type_recall(pred: str, gold_openui: str) -> float:
    """Recall of non-Stack gold component types present in the prediction."""
    gold_types = {k for k in _component_multiset(gold_openui) if k != "Stack"}
    if not gold_types:
        return 1.0
    pred_types = {k for k in _component_multiset(pred) if k != "Stack"}
    return len(pred_types & gold_types) / len(gold_types)


def _gold_design_lint_score(record: ExampleRecord) -> float | None:
    """
    Lint the *gold* DESIGN.md context quality — not the prediction.
    Kept for dataset diagnostics; must not be treated as model skill.
    """
    if not record.design_md:
        return None
    try:
        from slm_training.design_md import bridge_available, lint

        if not bridge_available():
            return None
        return float(lint(record.design_md).get("score") or 0.0)
    except Exception:  # noqa: BLE001
        return None


def _reward_for_prediction(pred: str, record: ExampleRecord) -> float:
    """
    Composite preference reward on the generated layout.

    Intentionally does **not** pass gold DESIGN.md — linting gold context was
    inflating reward when design_md_in_context was false.
    """
    try:
        from slm_training.preference import composite_reward

        return float(composite_reward(pred, gold=record, design_md=None))
    except Exception:  # noqa: BLE001
        return 0.0


def _is_meaningful_program(
    pred: str,
    *,
    gold: ExampleRecord | None = None,
    min_component_recall: float = 0.5,
) -> tuple[bool, str | None, str | None]:
    """
    Validate and reject trivial / off-task programs.
    Empty Stack/Card, no content components, no placeholders, and (when gold
    is provided) low component-type recall vs the gold layout.
    """
    try:
        program = validate(pred)
    except ParseError as exc:
        return False, str(exc), None
    serialized = (program.serialized or pred).strip()
    compact = serialized.replace(" ", "")
    if "Stack([])" in compact or "Stack([]," in compact:
        return False, "empty_root_stack", serialized
    if "Card([])" in compact:
        return False, "empty_card", serialized
    comps = _component_multiset(serialized)
    non_stack = {k: v for k, v in comps.items() if k != "Stack"}
    if not non_stack:
        return False, "no_content_components", serialized
    if not extract_placeholders(serialized):
        return False, "no_placeholders", serialized
    if gold is not None and min_component_recall > 0:
        recall = component_type_recall(serialized, gold.openui)
        if recall < min_component_recall:
            return False, f"low_component_recall:{recall:.2f}", serialized
    return True, None, serialized


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
    validity_sum = 0.0
    exact_sum = 0.0
    struct_sum = 0.0
    reward_sum = 0.0
    recall_sum = 0.0
    gold_design_scores: list[float] = []
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
        ok, error, serialized = _is_meaningful_program(pred, gold=record)
        scored_pred = serialized or pred
        if ok:
            parse_ok += 1
        fid = _placeholder_fidelity(scored_pred, record)
        ph_valid = _placeholder_validity(scored_pred, record)
        exact = _tree_match(scored_pred, record.openui)
        struct = structural_similarity(scored_pred, record.openui)
        recall = component_type_recall(scored_pred, record.openui)
        reward = _reward_for_prediction(scored_pred, record)
        gold_dscore = _gold_design_lint_score(record)
        fidelity_sum += fid
        validity_sum += ph_valid
        exact_sum += exact
        struct_sum += struct
        recall_sum += recall
        reward_sum += reward
        if gold_dscore is not None:
            gold_design_scores.append(gold_dscore)
        details.append(
            {
                "id": record.id,
                "parse_ok": ok,
                "error": error,
                "placeholder_fidelity": fid,
                "placeholder_validity": ph_valid,
                "exact_match": exact,
                "structural_similarity": struct,
                "component_type_recall": recall,
                "reward_score": reward,
                "gold_design_lint_score": gold_dscore,
                # Back-compat alias — gold context only, not model skill.
                "design_lint_score": gold_dscore,
                "latency_ms": round(latencies[-1], 2),
                "prediction": pred[:500],
                "serialized": (serialized or "")[:500] if serialized else None,
            }
        )

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else None
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else None
    gold_design_mean = (
        sum(gold_design_scores) / len(gold_design_scores) if gold_design_scores else None
    )

    metrics = {
        "suite": config.suite,
        "n": n,
        "parse_rate": (parse_ok / n) if n else 0.0,
        "placeholder_fidelity": (fidelity_sum / n) if n else 0.0,
        "placeholder_validity": (validity_sum / n) if n else 0.0,
        "exact_match": (exact_sum / n) if n else 0.0,
        "structural_similarity": (struct_sum / n) if n else 0.0,
        "component_type_recall": (recall_sum / n) if n else 0.0,
        "reward_score": (reward_sum / n) if n else 0.0,
        "gold_design_lint_score": gold_design_mean,
        # Alias kept for older dashboards; do not gate ship on this.
        "design_lint_score": gold_design_mean,
        "latency_ms_p50": round(p50, 2) if p50 is not None else None,
        "latency_ms_p95": round(p95, 2) if p95 is not None else None,
        "checkpoint": str(ckpt),
        "model": config.model_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    suite_path = run_dir / f"eval_{config.suite}.json"
    payload = json.dumps(metrics, indent=2) + "\n"
    suite_path.write_text(payload, encoding="utf-8")
    if config.suite == "smoke":
        (run_dir / "eval.json").write_text(payload, encoding="utf-8")
    metrics["output"] = str(suite_path)
    return metrics


def evaluate_suites(
    config: ModelBuildConfig,
    suites: list[str],
    *,
    checkpoint: Path | None = None,
    write_gates: bool = False,
) -> dict[str, dict]:
    """Run eval across multiple suites; write scoreboard.json (and optional gates)."""
    from dataclasses import replace

    from slm_training.harnesses.model_build.ship_gates import write_ship_gates

    board: dict[str, dict] = {}
    for suite in suites:
        suite_config = replace(config, suite=suite)
        metrics = evaluate(suite_config, checkpoint=checkpoint)
        board[suite] = {k: v for k, v in metrics.items() if k != "details"}
    scoreboard = {
        "run_id": config.run_id,
        "checkpoint": str(
            checkpoint or (config.checkpoint_dir / "last.pt")
        ),
        "suites": board,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "scoreboard.json"
    path.write_text(json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8")
    scoreboard["output"] = str(path)
    if write_gates:
        gates = write_ship_gates(run_dir, board)
        scoreboard["gates"] = {k: gates[k] for k in ("pass", "failures", "output")}
    return scoreboard
