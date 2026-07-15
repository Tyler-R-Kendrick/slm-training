"""Evaluation runner for ModelPlugin implementations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.structure import strip_style_literals
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import (
    load_suite_records,
    load_train_records,
)
from slm_training.harnesses.model_build.factory import build_model
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES

_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")


def _nearest_rank(sorted_values: list[float], fraction: float) -> float | None:
    """Return a monotonic nearest-rank percentile for small samples."""
    if not sorted_values:
        return None
    index = max(
        0, min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1)
    )
    return sorted_values[index]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _placeholder_fidelity_normalized(pred: str, gold: ExampleRecord) -> float:
    """Namespace-stripped placeholder overlap (diagnostic / ablation metric)."""
    pred_set = set(extract_placeholders(pred))
    gold_set = set(gold.placeholders) or set(extract_placeholders(gold.openui))
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    pred_n = {_normalize_placeholder(p) for p in pred_set}
    gold_n = {_normalize_placeholder(p) for p in gold_set}
    return len(pred_n & gold_n) / len(gold_n)


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
    well_formed = sum(1 for p in pred_set if p.startswith(":") and "." in p) / len(
        pred_set
    )
    pred_n = {_normalize_placeholder(p) for p in pred_set}
    gold_n = {_normalize_placeholder(p) for p in gold_set}
    overlap = len(pred_n & gold_n) / len(gold_n) if gold_n else 0.0
    return round(0.4 * well_formed + 0.6 * overlap, 4)


def _tree_match(pred: str, gold_openui: str) -> float:
    """Exact match on structure-normalized programs (style args ignored)."""
    pred_s = strip_style_literals(pred).strip()
    gold_s = strip_style_literals(gold_openui).strip()
    if pred_s == gold_s:
        return 1.0
    try:
        pred_p = validate(pred_s)
        gold_p = validate(gold_s)
        if pred_p.serialized and gold_p.serialized:
            ps = strip_style_literals(pred_p.serialized).strip()
            gs = strip_style_literals(gold_p.serialized).strip()
            return 1.0 if ps == gs else 0.0
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def _component_multiset(source: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in _COMPONENT_RE.findall(strip_style_literals(source)):
        counts[name] = counts.get(name, 0) + 1
    return counts


def structural_similarity(pred: str, gold_openui: str) -> float:
    """Jaccard-like similarity over component multisets + depth (style-agnostic)."""
    pred_s = strip_style_literals(pred)
    gold_s = strip_style_literals(gold_openui)
    pred_c = _component_multiset(pred_s)
    gold_c = _component_multiset(gold_s)
    keys = set(pred_c) | set(gold_c)
    if not keys:
        return 0.0
    inter = sum(min(pred_c.get(k, 0), gold_c.get(k, 0)) for k in keys)
    union = sum(max(pred_c.get(k, 0), gold_c.get(k, 0)) for k in keys)
    jaccard = inter / union if union else 0.0
    depth_p = pred_s.count("[") + pred_s.count("(")
    depth_g = gold_s.count("[") + gold_s.count("(")
    depth_sim = 1.0 - min(1.0, abs(depth_p - depth_g) / max(1, depth_g))
    return round(0.7 * jaccard + 0.3 * depth_sim, 4)


def _raw_syntax_valid(pred: str) -> bool:
    """True when ``validate()`` accepts the prediction (syntax only)."""
    try:
        validate(pred)
        return True
    except ParseError:
        return False


def _contract_precision(pred: str, record: ExampleRecord) -> float:
    """Fraction of predicted placeholders that appear in the record contract."""
    pred_set = set(extract_placeholders(pred))
    gold_set = set(record.placeholders or ())
    if not pred_set:
        return 1.0 if not gold_set else 0.0
    return len(pred_set & gold_set) / len(pred_set)


def _contract_recall(pred: str, record: ExampleRecord) -> float:
    """Fraction of record contract placeholders present in the prediction."""
    pred_set = set(extract_placeholders(pred))
    gold_set = set(record.placeholders or ())
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    return len(pred_set & gold_set) / len(gold_set)


def tree_edit_similarity(pred: str, gold_openui: str) -> float:
    """Structural similarity proxy until a dedicated tree-edit metric lands."""
    return structural_similarity(pred, gold_openui)


def component_type_recall(pred: str, gold_openui: str) -> float:
    """Recall of non-Stack gold component types present in the prediction."""
    gold_types = {k for k in _component_multiset(gold_openui) if k != "Stack"}
    if not gold_types:
        return 1.0
    pred_types = {k for k in _component_multiset(pred) if k != "Stack"}
    return len(pred_types & gold_types) / len(gold_types)


def _gold_design_lint_score(record: ExampleRecord) -> float | None:
    """
    Gold DESIGN.md context quality — diagnostic only, never model skill.

    Prefer the score already attached at corpus build time (meta.design_lint)
    so eval does not spawn a Node lint per record (~75ms each).
    Style tokens in DESIGN.md must not affect ship gates or reward_score.
    """
    meta = (record.meta or {}).get("design_lint") or {}
    if meta.get("score") is not None:
        try:
            return float(meta["score"])
        except (TypeError, ValueError):
            pass
    if not record.design_md:
        return None
    try:
        from slm_training.dsl.design_md import bridge_available, lint

        if not bridge_available():
            return None
        return float(lint(record.design_md).get("score") or 0.0)
    except Exception:  # noqa: BLE001
        return None


def _reward_for_prediction(pred: str, record: ExampleRecord) -> float:
    """
    Structure-only composite reward on the generated layout.

    Never passes gold DESIGN.md — style/color lint must not affect eval or
    ship ``reward_score`` gates.
    """
    try:
        from slm_training.harnesses.preference import composite_reward

        return float(
            composite_reward(
                strip_style_literals(pred),
                gold=record,
                design_md=None,
            )
        )
    except Exception:  # noqa: BLE001
        return 0.0


def _decode_canvas_cap(plugin: object) -> int | None:
    """Best-effort LTR canvas cap from a loaded model plugin."""
    cfg = getattr(plugin, "config", None)
    if cfg is None:
        return None
    cap = int(getattr(cfg, "grammar_ltr_max_tokens", 0) or 0)
    return cap if cap > 0 else None


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
    *,
    publish_agentv: bool = True,
) -> dict:
    if config.test_dir is None:
        raise ValueError("test_dir is required for evaluation")

    records = load_suite_records(config.test_dir, config.suite)
    suite_limit = getattr(config, "eval_limit", None)
    if suite_limit is None and config.suite == "rico_held":
        suite_limit = getattr(config, "rico_eval_limit", None)
    if suite_limit is not None:
        records = records[: max(0, int(suite_limit))]
    ckpt = checkpoint or (config.checkpoint_dir / "last.pt")

    if model is not None:
        if checkpoint is not None:
            raise ValueError(
                "provide either a preloaded model or a checkpoint, not both"
            )
        plugin = model
        loaded_checkpoint: Path | None = None
        checkpoint_sha256: str | None = None
    else:
        if not ckpt.exists():
            raise FileNotFoundError(f"evaluation checkpoint not found: {ckpt}")
        train_records = []
        if config.train_dir.exists():
            try:
                train_records = load_train_records(config.train_dir)
            except FileNotFoundError:
                train_records = []
        plugin = build_model(
            config,
            train_records or records,
            checkpoint=ckpt,
        )
        loaded_checkpoint = ckpt
        checkpoint_sha256 = _sha256_file(ckpt)

    # V7 decode telemetry: reset per-suite so forwards/hit-rate are suite-local.
    spec_stats = getattr(plugin, "speculative_stats", None)
    if spec_stats is not None and hasattr(spec_stats, "reset"):
        spec_stats.reset()

    n = len(records)
    parse_ok = 0
    raw_syntax_ok = 0
    fidelity_sum = 0.0
    fidelity_norm_sum = 0.0
    validity_sum = 0.0
    exact_sum = 0.0
    struct_sum = 0.0
    tree_edit_sum = 0.0
    reward_sum = 0.0
    recall_sum = 0.0
    contract_precision_sum = 0.0
    contract_recall_sum = 0.0
    gold_design_scores: list[float] = []
    latencies: list[float] = []
    details: list[dict] = []
    task_cases: list[dict] = []
    topology_evidence: list[dict[str, Any]] = []
    topology_target_evidence: list[dict[str, Any]] = []
    failure_breakdown: dict[str, int] = {}
    canvas_cap = _decode_canvas_cap(plugin)
    score_topology_targets = getattr(plugin, "score_topology_targets", None)
    if callable(score_topology_targets):
        topology_target_evidence = list(score_topology_targets(records))

    batch_size = 1
    generate_batch_requests = getattr(plugin, "generate_batch_requests", None)
    generate_batch = getattr(plugin, "generate_batch", None)
    if callable(generate_batch_requests) or callable(generate_batch):
        batch_size = max(
            1,
            int(
                getattr(getattr(plugin, "config", None), "generate_batch_size", 8) or 8
            ),
        )

    def _eval_schema() -> str | None:
        if not getattr(config, "schema_in_context", False):
            return None
        from slm_training.harnesses.quality import compact_schema_snippet

        budget = min(600, int(getattr(config, "design_md_budget", 1800) or 1800))
        return compact_schema_snippet(budget=budget)

    def _requests_for(chunk: list[ExampleRecord]) -> list[GenerationRequest]:
        schema = _eval_schema()
        return [GenerationRequest.from_record(r, schema=schema) for r in chunk]

    def _generate_chunk_unbounded(
        chunk: list[ExampleRecord],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Generate without passing gold ExampleRecord to the model."""
        if callable(generate_batch_requests):
            predictions = generate_batch_requests(_requests_for(chunk))
            consume = getattr(plugin, "consume_generation_evidence", None)
            evidence = consume() if callable(consume) else []
            return predictions, list(evidence)
        prompts = [r.prompt for r in chunk]
        if callable(generate_batch):
            try:
                return generate_batch(prompts), []
            except TypeError:
                try:
                    return generate_batch(prompts, golds=None), []
                except TypeError:
                    pass
        out: list[str] = []
        for prompt in prompts:
            try:
                out.append(plugin.generate(prompt))
            except TypeError:
                out.append(plugin.generate(prompt, gold=None))
        consume = getattr(plugin, "consume_generation_evidence", None)
        evidence = consume() if callable(consume) else []
        return out, list(evidence)

    decode_timeout_count = 0

    def _generate_chunk(
        chunk: list[ExampleRecord],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Generate a chunk, converting an explicit diagnostic timeout to failures."""
        nonlocal decode_timeout_count
        seconds = float(getattr(config, "decode_timeout_seconds", 0) or 0)
        if seconds <= 0 or not hasattr(signal, "setitimer"):
            return _generate_chunk_unbounded(chunk)

        def _alarm(_signum: int, _frame: object) -> None:
            raise TimeoutError(f"decode exceeded {seconds:g}s")

        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return _generate_chunk_unbounded(chunk)
        except TimeoutError:
            decode_timeout_count += len(chunk)
            return ["" for _ in chunk], []
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def _score_one(
        record: ExampleRecord,
        pred: str,
        latency_ms: float,
        prediction_evidence: dict[str, Any] | None = None,
    ) -> None:
        nonlocal parse_ok, raw_syntax_ok, fidelity_sum, fidelity_norm_sum, validity_sum
        nonlocal exact_sum, struct_sum, tree_edit_sum, reward_sum, recall_sum
        nonlocal contract_precision_sum, contract_recall_sum
        ok, error, serialized = _is_meaningful_program(pred, gold=record)
        scored_pred = serialized or pred
        if not ok:
            from slm_training.harnesses.model_build.decode_feasibility import (
                classify_parse_failure,
            )

            bucket = classify_parse_failure(
                pred,
                error=error,
                gold=record,
                canvas_cap=canvas_cap,
            )
            failure_breakdown[bucket] = failure_breakdown.get(bucket, 0) + 1
        if ok:
            parse_ok += 1
        if _raw_syntax_valid(scored_pred):
            raw_syntax_ok += 1
        fid = _placeholder_fidelity(scored_pred, record)
        fid_norm = _placeholder_fidelity_normalized(scored_pred, record)
        ph_valid = _placeholder_validity(scored_pred, record)
        exact = _tree_match(scored_pred, record.openui)
        struct = structural_similarity(scored_pred, record.openui)
        tree_edit = tree_edit_similarity(scored_pred, record.openui)
        recall = component_type_recall(scored_pred, record.openui)
        contract_prec = _contract_precision(scored_pred, record)
        contract_rec = _contract_recall(scored_pred, record)
        reward = _reward_for_prediction(scored_pred, record)
        evidence = dict(prediction_evidence or {})
        if len(topology_target_evidence) > len(topology_evidence):
            evidence.update(topology_target_evidence[len(topology_evidence)])
        codec = getattr(plugin, "codec", None)
        if codec is not None:
            from slm_training.models.grammar_diffusion import (
                production_sequence_accuracy,
                topology_arity_accuracy,
            )

            evidence["production_accuracy"] = production_sequence_accuracy(
                codec, scored_pred, record.openui
            )
            evidence["arity_accuracy"] = topology_arity_accuracy(
                codec, scored_pred, record.openui
            )
        topology_evidence.append(evidence)
        gold_dscore = _gold_design_lint_score(record)
        fidelity_sum += fid
        fidelity_norm_sum += fid_norm
        validity_sum += ph_valid
        exact_sum += exact
        struct_sum += struct
        tree_edit_sum += tree_edit
        recall_sum += recall
        contract_precision_sum += contract_prec
        contract_recall_sum += contract_rec
        reward_sum += reward
        if gold_dscore is not None:
            gold_design_scores.append(gold_dscore)
        details.append(
            {
                "id": record.id,
                "parse_ok": ok,
                "raw_syntax_valid": _raw_syntax_valid(scored_pred),
                "error": error,
                "placeholder_fidelity": fid,
                "placeholder_fidelity_normalized": fid_norm,
                "placeholder_validity": ph_valid,
                "contract_precision": contract_prec,
                "contract_recall": contract_rec,
                "exact_match": exact,
                "structural_similarity": struct,
                "tree_edit_similarity": tree_edit,
                "component_type_recall": recall,
                "reward_score": reward,
                "gold_design_lint_score": gold_dscore,
                "design_lint_score": gold_dscore,
                "latency_ms": round(latency_ms, 2),
                "prediction": pred[:500],
                "serialized": (serialized or "")[:500] if serialized else None,
                "topology_evidence": evidence or None,
            }
        )
        task_cases.append(
            {
                "id": record.id,
                "task": str((record.meta or {}).get("task") or "unknown"),
                "gold": record.openui,
                "prediction": scored_pred,
                "abstraction_level": (record.meta or {}).get("abstraction_level"),
                "prediction_evidence": evidence,
            }
        )

    if batch_size > 1 and (
        callable(generate_batch_requests) or callable(generate_batch)
    ):
        for start in range(0, n, batch_size):
            chunk = records[start : start + batch_size]
            t0 = time.perf_counter()
            preds, evidence_rows = _generate_chunk(chunk)
            elapsed = (time.perf_counter() - t0) * 1000.0
            per = elapsed / max(1, len(chunk))
            for index, (record, pred) in enumerate(zip(chunk, preds)):
                latencies.append(per)
                evidence = evidence_rows[index] if index < len(evidence_rows) else None
                _score_one(record, pred, per, evidence)
    else:
        for record in records:
            t0 = time.perf_counter()
            predictions, evidence_rows = _generate_chunk([record])
            pred = predictions[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)
            _score_one(
                record,
                pred,
                latencies[-1],
                evidence_rows[0] if evidence_rows else None,
            )

    lat_sorted = sorted(latencies)

    p50 = _nearest_rank(lat_sorted, 0.50)
    p95 = _nearest_rank(lat_sorted, 0.95)
    gold_design_mean = (
        sum(gold_design_scores) / len(gold_design_scores)
        if gold_design_scores
        else None
    )

    metrics = {
        "suite": config.suite,
        "n": n,
        "eval_limit": suite_limit,
        "diagnostic_subset": suite_limit is not None,
        "parse_rate": (parse_ok / n) if n else 0.0,
        "raw_syntax_validity": (raw_syntax_ok / n) if n else 0.0,
        "contract_precision": (contract_precision_sum / n) if n else 0.0,
        "contract_recall": (contract_recall_sum / n) if n else 0.0,
        "residual_mask_rate": 0.0,
        "oov_rate": 0.0,
        "fallback_count": 0,
        "placeholder_fidelity": (fidelity_sum / n) if n else 0.0,
        "placeholder_fidelity_normalized": (fidelity_norm_sum / n) if n else 0.0,
        "placeholder_validity": (validity_sum / n) if n else 0.0,
        "exact_match": (exact_sum / n) if n else 0.0,
        "structural_similarity": (struct_sum / n) if n else 0.0,
        "tree_edit_similarity": (tree_edit_sum / n) if n else 0.0,
        "component_type_recall": (recall_sum / n) if n else 0.0,
        "reward_score": (reward_sum / n) if n else 0.0,
        "gold_design_lint_score": gold_design_mean,
        # Alias kept for older dashboards; do not gate ship on this.
        "design_lint_score": gold_design_mean,
        "latency_ms_p50": round(p50, 2) if p50 is not None else None,
        "latency_ms_p95": round(p95, 2) if p95 is not None else None,
        "checkpoint": str(loaded_checkpoint) if loaded_checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_source": ("checkpoint" if loaded_checkpoint else "preloaded_model"),
        "model": config.model_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "failure_breakdown": failure_breakdown,
        "decode_timeout_count": decode_timeout_count,
        "decode_canvas_cap": canvas_cap,
        "details": details,
    }
    from slm_training.evals.task_scoreboard import build_task_scoreboard

    metrics["task_scoreboard"] = build_task_scoreboard(task_cases)
    scored_details = metrics["task_scoreboard"].get("details") or []

    def _available_mean(name: str) -> float | None:
        values = [
            float(metric["value"])
            for row in scored_details
            if (metric := (row.get("metrics") or {}).get(name))
            and metric.get("value") is not None
        ]
        return sum(values) / len(values) if values else None

    metrics["ast_node_f1"] = _available_mean("ast_node_f1")
    metrics["ast_edge_f1"] = _available_mean("ast_edge_f1")
    if topology_evidence and all(
        all(
            key in row
            for key in (
                "action_macro_f1",
                "production_accuracy",
                "arity_accuracy",
                "critic_ece",
                "efficiency_score",
            )
        )
        for row in topology_evidence
    ):

        def mean(key: str) -> float:
            return sum(float(row[key]) for row in topology_evidence) / len(
                topology_evidence
            )

        quality = (
            2.0 * float(metrics["parse_rate"])
            + 2.0 * float(metrics["placeholder_fidelity"])
            + float(metrics["structural_similarity"])
            + 0.5 * float(metrics["reward_score"])
        ) / 5.5
        ast_node = metrics["ast_node_f1"]
        ast_edge = metrics["ast_edge_f1"]
        if ast_node is not None and ast_edge is not None:
            topology = (
                float(ast_node)
                + float(ast_edge)
                + float(metrics["tree_edit_similarity"])
            ) / 3.0
            trace = (
                mean("action_macro_f1")
                + mean("production_accuracy")
                + mean("arity_accuracy")
                + (1.0 - mean("critic_ece"))
            ) / 4.0
            efficiency = mean("efficiency_score")
            metrics.update(
                {
                    "topology_quality_score": quality,
                    "topology_structure_score": topology,
                    "topology_trace_score": trace,
                    "topology_efficiency_score": efficiency,
                    "topology_composite": (
                        0.45 * quality
                        + 0.25 * topology
                        + 0.20 * trace
                        + 0.10 * efficiency
                    ),
                    "topology_telemetry": {
                        key: mean(key)
                        for key in topology_evidence[0]
                        if isinstance(topology_evidence[0].get(key), (int, float))
                        and all(
                            isinstance(row.get(key), (int, float))
                            for row in topology_evidence
                        )
                    },
                }
            )
    # V7: speculative-denoising decode telemetry (MaskGIT path only).
    if (
        spec_stats is not None
        and hasattr(spec_stats, "as_dict")
        and getattr(spec_stats, "generates", 0)
    ):
        metrics["speculative_stats"] = spec_stats.as_dict()

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    suite_path = run_dir / f"eval_{config.suite}.json"
    metrics["output"] = str(suite_path)
    if publish_agentv:
        from slm_training.evals.agentv import publish_model_evaluation

        metrics["agentv"] = publish_model_evaluation(run_dir, {config.suite: metrics})
    payload = json.dumps(metrics, indent=2) + "\n"
    suite_path.write_text(payload, encoding="utf-8")
    if config.suite == "smoke":
        (run_dir / "eval.json").write_text(payload, encoding="utf-8")
    return metrics


def evaluate_suites(
    config: ModelBuildConfig,
    suites: list[str],
    *,
    checkpoint: Path | None = None,
    model=None,
    write_gates: bool = False,
) -> dict[str, dict]:
    """Run eval across multiple suites; write scoreboard.json (and optional gates)."""
    from dataclasses import replace

    from slm_training.harnesses.model_build.ship_gates import write_ship_gates

    board: dict[str, dict] = {}
    for suite in suites:
        suite_config = replace(config, suite=suite)
        metrics = evaluate(
            suite_config,
            model=model,
            checkpoint=checkpoint,
            publish_agentv=False,
        )
        board[suite] = {k: v for k, v in metrics.items() if k != "details"}
    scoreboard = {
        "run_id": config.run_id,
        "checkpoint": (
            None
            if model is not None
            else str(checkpoint or (config.checkpoint_dir / "last.pt"))
        ),
        "checkpoint_source": "preloaded_model" if model is not None else "checkpoint",
        "checkpoint_sha256": next(iter(board.values()), {}).get("checkpoint_sha256"),
        "suites": board,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "scoreboard.json"
    scoreboard["output"] = str(path)
    if write_gates:
        gates = write_ship_gates(run_dir, board)
        scoreboard["gates"] = {k: gates[k] for k in ("pass", "failures", "output")}
    from slm_training.evals.agentv import publish_model_evaluation

    scoreboard["agentv"] = publish_model_evaluation(
        run_dir,
        board,
        include_missing_suites=set(suites) == set(DEFAULT_SHIP_GATES),
    )
    path.write_text(json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8")
    return scoreboard
