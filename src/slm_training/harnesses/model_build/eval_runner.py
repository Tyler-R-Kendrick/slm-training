"""Evaluation runner for ModelPlugin implementations."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import signal
import sys
import time
from dataclasses import fields
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from slm_training.data.contract import RuntimeSymbol
from slm_training.data.structure import strip_style_literals
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.eval_cache import (
    EvalCache,
    EvalCacheMode,
    suite_result_key,
)
from slm_training.harnesses.eval.harness_replay import (
    UNKNOWN_NOT_CAPTURED,
    HarnessProvenanceV1,
    harness_provenance_id,
    prediction_lineage,
)
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import (
    load_suite_records,
    load_train_records,
)
from slm_training.harnesses.model_build.decode_outcome import outcome_counts
from slm_training.harnesses.model_build.factory import build_model
from slm_training.harnesses.model_build.full_state import _git_dirty, _git_sha
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_MIN_SUITE_N,
    DEFAULT_SHIP_GATES,
)
from slm_training.models.decode_stats import (
    DecodeStats,
    aggregate_stats,
    check_decode_deadline,
    clear_decode_deadline,
    collect_decode_stats,
    set_decode_deadline,
)
from slm_training.versioning import build_version_stamp, component_version

_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")
_LANGSMITH_METRIC_KEYS = (
    "n",
    "completed_document_n",
    "incomplete_document_n",
    "parse_rate",
    "placeholder_fidelity",
    "structural_similarity",
    "ast_beq_rate",
    "canonical_beq_rate",
    "reward_score",
    "decode_timeout_count",
    "decode_timeout_rate",
)


def _persist_decode_progress(
    config: ModelBuildConfig,
    *,
    status: str,
    processed_record_n: int,
    active_chunk: list[ExampleRecord],
    stats_rows: list[DecodeStats],
    version_stamp: dict[str, Any],
) -> Path:
    """Atomically persist non-scoreable decode work before a supervisor stop."""
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "decode_progress.json"
    tmp = run_dir / "decode_progress.json.tmp"
    payload = {
        "schema_version": "DecodeProgressV1",
        "run_id": config.run_id,
        "suite": config.suite,
        "status": status,
        "measurement_complete": False,
        "scoreable": False,
        "processed_record_n": int(processed_record_n),
        "active_record_ids": [record.id for record in active_chunk],
        "decode_stats": aggregate_stats(stats_rows),
        "version_stamp": version_stamp,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _evaluation_version_components(config: ModelBuildConfig) -> tuple[str, ...]:
    components = (
        "config.levers",
        "harness.model_build.eval",
        "harness.experiment_feature_flags",
        "evals.meaningful_program",
        "evals.power_protocol",
        "evals.scoring",
    )
    return components + (("model.twotower",) if config.model_name == "twotower" else ())


def _record_langsmith_evaluation(
    config, *, suites: dict[str, dict], scoreboard: dict
) -> None:
    """Publish only aggregate evaluation data to the active summary trace."""
    from slm_training.runtime.telemetry import current_trace

    trace = current_trace()
    if trace is None:
        return
    summary = {
        suite: {key: metrics[key] for key in _LANGSMITH_METRIC_KEYS if key in metrics}
        for suite, metrics in suites.items()
    }
    trace.record_summary(
        "evaluation.summary",
        inputs={"run_id": config.run_id, "suites": sorted(suites)},
        outputs={
            "suites": summary,
            "gates": scoreboard.get("gates"),
            "agentv": scoreboard.get("agentv"),
        },
        metadata={
            key: scoreboard.get(key)
            for key in (
                "run_class",
                "checkpoint_sha256",
                "eval_data_manifest_sha",
                "code_git_sha",
                "version_stamp",
            )
            if scoreboard.get(key) is not None
        },
    )


def _annotate_decode_trace_records(
    stats: object,
    records: list[ExampleRecord],
) -> None:
    """Attach stable eval identities before per-call decode stats are aggregated."""
    for trace in getattr(stats, "constrained_selection_traces", ()):
        row = trace.get("row")
        if isinstance(row, int) and 0 <= row < len(records):
            trace["record_id"] = records[row].id
        elif len(records) == 1:
            # Single-request paths have no batch row.  Preserve only the stable
            # evaluation identity; feature projection below still excludes every
            # final outcome and post-decode field.
            trace.setdefault("record_id", records[0].id)


_TEMPORAL_DECODE_TRACE_FIELDS = (
    "position",
    "legal_candidates",
    "forced",
    "phase",
    "decision_source",
    "choice_changed",
)


def _temporal_decode_evidence(
    stats: object | None, record_id: str
) -> list[dict[str, object]]:
    """Return the prefix-time, model-available trace projection for one record.

    This is deliberately not a ``DecodeStats.as_dict()`` snapshot: aggregate
    counters and terminal fields can describe work that happens after a given
    prefix.  Final parse/semantic/error/timeout/fallback outcomes remain labels
    in the surrounding eval detail, never features.
    """
    if stats is None:
        return []
    evidence: list[dict[str, object]] = []
    for trace in getattr(stats, "constrained_selection_traces", ()):
        if trace.get("record_id") != record_id:
            continue
        item = {
            key: trace[key] for key in _TEMPORAL_DECODE_TRACE_FIELDS if key in trace
        }
        if "position" in item:
            evidence.append(item)
    return evidence


@lru_cache(maxsize=1024)
def _placeholders_of(source: str) -> frozenset[str]:
    """Placeholder set for a source; several per-record metrics share it."""
    return frozenset(extract_placeholders(source))


def _nearest_rank(sorted_values: list[float], fraction: float) -> float | None:
    """Return a monotonic nearest-rank percentile for small samples."""
    if not sorted_values:
        return None
    index = max(
        0, min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1)
    )
    return sorted_values[index]


def _aggregate_scope_contract_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    scoped = [row for row in rows if "scope_kind" in row]
    if not scoped:
        return None

    mean_keys = (
        "scope_gate_accuracy",
        "scope_summary_definitions_mae",
        "scope_summary_uses_mae",
        "scope_summary_slots_mae",
        "scope_summary_realized_size_mae",
        "failure_cone_predicted_size",
        "failure_cone_target_size",
    )

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"sample_count": len(group)}
        for key in mean_keys:
            values = [
                float(row[key])
                for row in group
                if isinstance(row.get(key), (int, float))
            ]
            if values:
                summary[f"{key}_mean" if key.endswith("_size") else key] = sum(
                    values
                ) / len(values)
        tp = sum(int(row.get("failure_cone_tp", 0)) for row in group)
        fp = sum(int(row.get("failure_cone_fp", 0)) for row in group)
        fn = sum(int(row.get("failure_cone_fn", 0)) for row in group)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        summary.update(
            {
                "failure_cone_precision": precision,
                "failure_cone_recall": recall,
                "failure_cone_f1": (
                    2.0 * precision * recall / (precision + recall)
                    if precision is not None
                    and recall is not None
                    and precision + recall
                    else None
                ),
            }
        )
        return summary

    result = summarize(scoped)
    for field, output_key in (
        ("scope_kind", "by_scope_kind"),
        ("scope_family", "by_scope_family"),
    ):
        labels = sorted({str(row[field]) for row in scoped})
        result[output_key] = {
            label: summarize([row for row in scoped if str(row[field]) == label])
            for label in labels
        }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _placeholder_fidelity_normalized(pred: str, gold: ExampleRecord) -> float | None:
    """
    Namespace-stripped placeholder overlap (diagnostic / ablation metric).

    ``None`` when gold has no placeholders and the prediction adds none: 0/0 is
    undefined evidence, never a vacuous 1.0.
    """
    pred_set = _placeholders_of(pred)
    gold_set = set(gold.placeholders) or _placeholders_of(gold.openui)
    if not gold_set:
        return None if not pred_set else 0.0
    pred_n = {_normalize_placeholder(p) for p in pred_set}
    gold_n = {_normalize_placeholder(p) for p in gold_set}
    return len(pred_n & gold_n) / len(gold_n)


def _placeholder_fidelity(pred: str, gold: ExampleRecord) -> float | None:
    """
    Exact placeholder overlap with gold (strict).

    ``None`` when gold has no placeholders and the prediction adds none: 0/0 is
    undefined evidence, never a vacuous 1.0.
    """
    pred_set = _placeholders_of(pred)
    gold_set = set(gold.placeholders) or _placeholders_of(gold.openui)
    if not gold_set:
        return None if not pred_set else 0.0
    return len(pred_set & gold_set) / len(gold_set)


def _normalize_placeholder(token: str) -> str:
    """Drop leading namespace segment so :smoke.hero.title ~= :hero.title."""
    body = token.removeprefix(":")
    parts = body.split(".")
    if len(parts) >= 3:
        return ".".join(parts[1:])
    return body


def _placeholder_validity(pred: str, gold: ExampleRecord) -> float | None:
    """
    Soft placeholder quality for diagnostics only (not a ship gate alone).
    Prefer placeholder_fidelity for readiness claims.
    ``None`` when neither side has placeholders (undefined, not perfect).
    """
    pred_set = _placeholders_of(pred)
    gold_set = set(gold.placeholders) or _placeholders_of(gold.openui)
    if not gold_set:
        return None if not pred_set else 0.5
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
    """
    Exact match on structure-normalized programs (style args ignored).

    An unparseable *prediction* is a real mismatch (0.0). A gold-side parse
    failure raises — that is harness/data breakage, not model quality, and the
    caller records it as an error instead of a fabricated 0.0 score.
    """
    pred_s = strip_style_literals(pred).strip()
    gold_s = strip_style_literals(gold_openui).strip()
    if pred_s == gold_s:
        return 1.0
    try:
        pred_p = validate(pred_s)
    except ParseError:
        return 0.0
    gold_p = validate(gold_s)
    if pred_p.serialized and gold_p.serialized:
        ps = strip_style_literals(pred_p.serialized).strip()
        gs = strip_style_literals(gold_p.serialized).strip()
        return 1.0 if ps == gs else 0.0
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


def _contract_precision(pred: str, record: ExampleRecord) -> float | None:
    """
    Fraction of predicted placeholders that appear in the record contract.

    ``None`` when the prediction has no placeholders and the contract is empty:
    0/0 is undefined evidence, never a vacuous 1.0.
    """
    pred_set = _placeholders_of(pred)
    gold_set = set(record.placeholders or ())
    if not pred_set:
        return None if not gold_set else 0.0
    return len(pred_set & gold_set) / len(pred_set)


def _contract_recall(pred: str, record: ExampleRecord) -> float | None:
    """
    Fraction of record contract placeholders present in the prediction.

    ``None`` when the contract is empty and the prediction adds nothing: 0/0 is
    undefined evidence, never a vacuous 1.0.
    """
    pred_set = _placeholders_of(pred)
    gold_set = set(record.placeholders or ())
    if not gold_set:
        return None if not pred_set else 0.0
    return len(pred_set & gold_set) / len(gold_set)


def _binder_reference_f1(precision: float | None, recall: float | None) -> float | None:
    """Harmonic mean for the existing binder/reference contract evidence."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def tree_edit_similarity(pred: str, gold_openui: str) -> float:
    """Structural similarity proxy until a dedicated tree-edit metric lands."""
    return structural_similarity(pred, gold_openui)


def component_type_recall(pred: str, gold_openui: str) -> float | None:
    """
    Recall of non-Stack gold component types present in the prediction.

    ``None`` when gold has no non-Stack components — recall over an empty type
    set is undefined evidence, never a vacuous 1.0.
    """
    gold_types = {k for k in _component_multiset(gold_openui) if k != "Stack"}
    if not gold_types:
        return None
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
    ship ``reward_score`` gates. ``composite_reward`` scores unparseable input
    as 0.0 itself, so an exception here is harness breakage: it propagates and
    the caller records an error instead of laundering it into a 0.0 score.
    """
    from slm_training.harnesses.preference import composite_reward

    return float(
        composite_reward(
            strip_style_literals(pred),
            gold=record,
            design_md=None,
        )
    )


def _decode_canvas_cap(plugin: object) -> int | None:
    """Best-effort LTR canvas cap from a loaded model plugin."""
    cfg = getattr(plugin, "config", None)
    if cfg is None:
        return None
    cap = int(getattr(cfg, "grammar_ltr_max_tokens", 0) or 0)
    return cap if cap > 0 else None


def _effective_evaluation_policy(
    config: ModelBuildConfig, plugin: object
) -> dict[str, object]:
    """Report the loaded model's effective settings after runtime overrides."""
    model_config = getattr(plugin, "config", None)

    def value(name: str) -> object:
        effective = getattr(model_config, name, None)
        return effective if effective is not None else getattr(config, name, None)

    def optional_bool(name: str) -> bool | None:
        effective = value(name)
        return None if effective is None else bool(effective)

    def json_value(item: object) -> object:
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (list, tuple, set, frozenset)):
            return [json_value(value) for value in item]
        if isinstance(item, dict):
            return {str(key): json_value(value) for key, value in item.items()}
        return str(item)

    if model_config is None:
        model_snapshot: dict[str, object] = {}
    else:
        try:
            names = [field.name for field in fields(model_config)]
        except TypeError:
            names = sorted(vars(model_config))
        model_snapshot = {
            name: json_value(getattr(model_config, name)) for name in names
        }

    return {
        "evaluation_policy": str(
            getattr(config, "evaluation_policy", "checkpoint_declared")
        ),
        "context_backend": value("context_backend"),
        "local_files_only": bool(value("local_files_only")),
        "grammar_constrained": optional_bool("grammar_constrained"),
        "grammar_ltr_primary": optional_bool("grammar_ltr_primary"),
        "grammar_ltr_repair": optional_bool("grammar_ltr_repair"),
        "compiler_decode_mode": (
            None
            if value("compiler_decode_mode") is None
            else str(value("compiler_decode_mode"))
        ),
        "schema_in_context": bool(value("schema_in_context")),
        "slot_contract_in_context": bool(value("slot_contract_in_context")),
        "semantic_role_contract_in_context": bool(
            value("semantic_role_contract_in_context")
        ),
        "slot_contract_constrained_decode": bool(
            value("slot_contract_constrained_decode")
        ),
        **{
            field.name: float(value(field.name) or 0.0)
            for field in fields(ModelBuildConfig)
            if field.name.endswith("_decode_weight")
        },
        "semantic_role_schema_candidates": bool(
            value("semantic_role_schema_candidates")
        ),
        "honest_slot_contract": bool(value("honest_slot_contract")),
        "grammar_skip_exact_stream_probe": optional_bool(
            "grammar_skip_exact_stream_probe"
        ),
        "grammar_verify_chosen_only": optional_bool("grammar_verify_chosen_only"),
        "grammar_top_k": (
            None if value("grammar_top_k") is None else int(value("grammar_top_k"))
        ),
        "generate_max_attempts": (
            None
            if value("generate_max_attempts") is None
            else int(value("generate_max_attempts"))
        ),
        "decode_timeout_seconds": value("decode_timeout_seconds"),
        "evaluation_wall_seconds": value("evaluation_wall_seconds"),
        "allow_unconstrained_fallback": bool(value("allow_unconstrained_fallback")),
        "gen_steps": int(value("gen_steps") or 0),
        "grammar_ltr_max_tokens": int(value("grammar_ltr_max_tokens") or 0),
        # Complete post-load/post-override model state. The summary fields above
        # stay convenient for dashboards, while this snapshot makes behavior
        # reconstructible without guessing which non-default field mattered.
        "effective_model_config": model_snapshot,
    }


def _first_empty_children_component(node: Any) -> str | None:
    """
    Recursively find an element whose ``children`` array is genuinely empty.

    Walks the real parsed AST (``Program.root``, an ``ElementNode`` tree)
    rather than the serialized text. A component's ``children`` array is a
    *separate* key from its other (positional) properties in this AST, so a
    component that had non-content required slots stuffed into its remaining
    arguments (e.g. serialized as ``Card([], ":a", ":b")``) is still caught
    here even though a literal ``"Card([])"`` substring match on the text
    would miss it -- the ``variant``/``direction`` stuffing lives in
    ``props``, ``children`` is still ``[]`` regardless of what else was
    stuffed into sibling props. Returns the component's ``typeName`` (e.g.
    ``"Card"``, ``"Modal"``, ``"Carousel"``, ``"Stack"``) on the first empty
    match found, else ``None``. Any component type the parser itself marks
    with a ``children`` prop is covered -- no hardcoded type allowlist.
    """
    if isinstance(node, dict):
        if node.get("type") == "element":
            props = node.get("props")
            if isinstance(props, dict):
                children = props.get("children")
                if isinstance(children, list):
                    if not children:
                        type_name = node.get("typeName")
                        return type_name if isinstance(type_name, str) else "unknown"
                    for child in children:
                        found = _first_empty_children_component(child)
                        if found:
                            return found
                for key, value in props.items():
                    if key == "children":
                        continue
                    found = _first_empty_children_component(value)
                    if found:
                        return found
            return None
        for value in node.values():
            found = _first_empty_children_component(value)
            if found:
                return found
        return None
    if isinstance(node, list):
        for item in node:
            found = _first_empty_children_component(item)
            if found:
                return found
    return None


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
    from slm_training.dsl.language_contract import output_contract_violations

    if output_contract_violations(serialized):
        return False, "free_form_output_string", serialized
    compact = serialized.replace(" ", "")
    if "Stack([])" in compact or "Stack([]," in compact:
        return False, "empty_root_stack", serialized
    if "Card([])" in compact:
        return False, "empty_card", serialized
    # AST-structural check (E631): catches an empty ``children`` array that
    # the literal substring checks above miss -- e.g. a Card whose non-content
    # properties absorbed extra values (``Card([], ":a", ":b")``), or any
    # other children-bearing component (Modal, Carousel, ...) the literal
    # checks never covered. Purely additive: every case the literal checks
    # already caught is unaffected (same branch, same reason string, still
    # runs first); this only adds new rejections, never removes one.
    empty_type = _first_empty_children_component(program.root)
    if empty_type == "Stack":
        return False, "empty_root_stack", serialized
    if empty_type == "Card":
        return False, "empty_card", serialized
    if empty_type is not None:
        return False, f"empty_children:{empty_type}", serialized
    comps = _component_multiset(serialized)
    non_stack = {k: v for k, v in comps.items() if k != "Stack"}
    if not non_stack:
        return False, "no_content_components", serialized
    if not extract_placeholders(serialized):
        return False, "no_placeholders", serialized
    if gold is not None and min_component_recall > 0:
        recall = component_type_recall(serialized, gold.openui)
        # None = recall undefined (gold has only Stacks); nothing to reject on.
        if recall is not None and recall < min_component_recall:
            return False, f"low_component_recall:{recall:.2f}", serialized
    return True, None, serialized


# Public version lock: historical scoreboards and ship thresholds remain v1.
meaningful_program_v1 = _is_meaningful_program

#: Schema marker for the typed v1 reason-code report (SLM-288).
MEANINGFUL_V1_REASON_SCHEMA = "meaningful_program_v1_reasons/v1"

#: Typed clause codes in evaluation order. ``component_recall_unobservable``
#: is an UNKNOWN/unobservable annotation, never a failure.
MEANINGFUL_V1_CLAUSE_CODES: tuple[str, ...] = (
    "parse_failed",
    "free_form_output_string",
    "empty_root_stack",
    "empty_card",
    "empty_children",
    "no_content_components",
    "no_placeholders",
    "low_component_recall",
    "component_recall_unobservable",
)


def meaningful_program_v1_report(
    pred: str,
    *,
    gold: ExampleRecord | None = None,
    min_component_recall: float = 0.5,
) -> dict[str, Any]:
    """Typed, ordered reason-code view of :func:`meaningful_program_v1` (SLM-288).

    Every clause is evaluated (no short-circuit) so downstream analysis sees
    the full failure profile; ``verdict`` and the first failure's legacy
    reason string remain byte-identical to v1. UNKNOWN/unobservable contract
    cases (gold missing, or recall undefined because gold has only Stacks)
    are reported explicitly as ``component_recall_unobservable`` and are never
    counted as semantic failures.
    """
    checks: list[dict[str, Any]] = []

    def _add(code: str, status: str, detail: str | None = None) -> None:
        checks.append({"code": code, "status": status, "detail": detail})

    serialized: str | None = None
    program: Any = None
    try:
        program = validate(pred)
    except ParseError as exc:
        _add("parse_failed", "fail", str(exc))
        fails = [c for c in checks if c["status"] == "fail"]
        return {
            "schema": MEANINGFUL_V1_REASON_SCHEMA,
            "verdict": False,
            "reason_codes": [c["code"] for c in checks if c["status"] != "pass"],
            "checks": checks,
            "legacy_reason": fails[0]["detail"],
            "serialized": None,
            "component_recall": None,
            "min_component_recall": min_component_recall,
        }
    serialized = (program.serialized or pred).strip()
    from slm_training.dsl.language_contract import output_contract_violations

    if output_contract_violations(serialized):
        _add("free_form_output_string", "fail")
    compact = serialized.replace(" ", "")
    empty_literal_stack = "Stack([])" in compact or "Stack([]," in compact
    empty_literal_card = "Card([])" in compact
    empty_type = _first_empty_children_component(program.root)
    if empty_literal_stack or empty_type == "Stack":
        _add("empty_root_stack", "fail")
    if empty_literal_card or empty_type == "Card":
        _add("empty_card", "fail")
    if empty_type is not None and empty_type not in {"Stack", "Card"}:
        _add("empty_children", "fail", empty_type)
    comps = _component_multiset(serialized)
    non_stack = {k: v for k, v in comps.items() if k != "Stack"}
    if not non_stack:
        _add("no_content_components", "fail")
    if not extract_placeholders(serialized):
        _add("no_placeholders", "fail")
    recall: float | None = None
    if gold is not None and min_component_recall > 0:
        recall = component_type_recall(serialized, gold.openui)
        if recall is None:
            _add(
                "component_recall_unobservable",
                "unknown",
                "recall undefined: gold has only Stack components",
            )
        elif recall < min_component_recall:
            _add("low_component_recall", "fail", f"{recall:.2f}")
    else:
        _add(
            "component_recall_unobservable",
            "unknown",
            "no gold reference or recall floor disabled",
        )
    fails = [c for c in checks if c["status"] == "fail"]
    first = fails[0] if fails else None
    legacy_reason: str | None = None
    if first is not None:
        if first["code"] == "low_component_recall":
            legacy_reason = f"low_component_recall:{first['detail']}"
        elif first["code"] == "empty_children":
            legacy_reason = f"empty_children:{first['detail']}"
        else:
            legacy_reason = first["detail"] or first["code"]
    return {
        "schema": MEANINGFUL_V1_REASON_SCHEMA,
        "verdict": not fails,
        "reason_codes": [c["code"] for c in checks if c["status"] != "pass"],
        "checks": checks,
        "legacy_reason": legacy_reason,
        "serialized": serialized,
        "component_recall": recall,
        "min_component_recall": min_component_recall,
    }


def _eval_data_sha(directory: Path) -> str | None:
    """Content fingerprint of an eval dataset dir (manifest or records hash)."""
    from slm_training.harnesses.model_build.full_state import data_manifest_sha

    try:
        return data_manifest_sha(directory)
    except Exception:  # noqa: BLE001 - identity stamping must never break evals
        return None


def _seed_eval_rng(seed: int) -> None:
    """Lock Python / NumPy / Torch RNG for greedy constrained evals."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001,S110 - numpy optional in some envs
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001,S110 - torch optional at import time
        pass


def _clear_repeating_decode_alarm(previous: Any) -> None:
    """Cancel SIGALRM without letting a pending interval tick escape cleanup."""
    if not (hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")):
        return
    old_mask = None
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if callable(pthread_sigmask):
        old_mask = pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        # Ignoring SIGALRM while blocked discards any already-pending interval
        # tick before the caller's previous disposition is restored.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL, 0)
        if previous is not None:
            signal.signal(signal.SIGALRM, previous)
    finally:
        if callable(pthread_sigmask) and old_mask is not None:
            pthread_sigmask(signal.SIG_SETMASK, old_mask)


_EVALUATION_FINALIZATION_RESERVE_SECONDS = 2.0


def _effective_record_decode_timeout(
    requested_seconds: float,
    *,
    evaluation_deadline: float | None,
    remaining_record_n: int,
    chunk_record_n: int,
    now: float | None = None,
) -> float:
    """Allocate a fair, bounded wall-time budget for one decode chunk."""

    requested_chunk_seconds = requested_seconds * max(1, int(chunk_record_n))
    if evaluation_deadline is None:
        return requested_chunk_seconds
    current = time.monotonic() if now is None else float(now)
    usable = max(
        0.05,
        float(evaluation_deadline) - current - _EVALUATION_FINALIZATION_RESERVE_SECONDS,
    )
    fair_share = usable * max(1, int(chunk_record_n)) / max(1, int(remaining_record_n))
    if requested_seconds > 0:
        fair_share = min(requested_chunk_seconds, fair_share)
    return max(0.05, fair_share)


def evaluate(
    config: ModelBuildConfig,
    model=None,
    checkpoint: Path | None = None,
    *,
    model_checkpoint_sha256: str | None = None,
    model_checkpoint_path: Path | None = None,
    publish_agentv: bool = True,
    cache: EvalCache | None = None,
    generation_overrides: dict[str, Any] | None = None,
    evaluation_deadline: float | None = None,
    evaluation_remaining_records: list[int] | None = None,
) -> dict:
    from slm_training.harnesses.model_build.feature_flags import resolve, save_snapshot

    generation_overrides = dict(generation_overrides or {})
    _seed_eval_rng(int(getattr(config, "seed", 0) or 0))
    if config.test_dir is None:
        raise ValueError("test_dir is required for evaluation")

    records = load_suite_records(config.test_dir, config.suite)
    suite_offset = int(config.eval_offset)
    if suite_offset < 0:
        raise ValueError("eval_offset must be non-negative")
    suite_limit = getattr(config, "eval_limit", None)
    if suite_limit is None and config.suite == "rico_held":
        suite_limit = getattr(config, "rico_eval_limit", None)
    records = records[
        suite_offset : (
            suite_offset + max(0, int(suite_limit)) if suite_limit is not None else None
        )
    ]
    if evaluation_deadline is None and config.evaluation_wall_seconds:
        evaluation_deadline = time.monotonic() + float(config.evaluation_wall_seconds)
    if evaluation_remaining_records is None and evaluation_deadline is not None:
        evaluation_remaining_records = [len(records)]
    ckpt = checkpoint or (config.checkpoint_dir / "last.pt")

    if model is not None:
        if checkpoint is not None:
            raise ValueError(
                "provide either a preloaded model or a checkpoint, not both"
            )
        if (model_checkpoint_sha256 is None) != (model_checkpoint_path is None):
            raise ValueError(
                "preloaded model checkpoint identity requires both "
                "model_checkpoint_sha256 and model_checkpoint_path"
            )
        if model_checkpoint_path is not None and not model_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"model checkpoint identity path not found: {model_checkpoint_path}"
            )
        if (
            model_checkpoint_path is not None
            and _sha256_file(model_checkpoint_path) != model_checkpoint_sha256
        ):
            raise ValueError(
                "model_checkpoint_sha256 does not match model_checkpoint_path"
            )
        plugin = model
        # A caller may bind an immutable preloaded model to the exact
        # checkpoint that produced it.  This is the only safe way to enable
        # suite-cache reuse for a preloaded model; live training models omit
        # the digest and continue to fail closed below.
        loaded_checkpoint = model_checkpoint_path
        checkpoint_sha256 = model_checkpoint_sha256
    else:
        if model_checkpoint_sha256 is not None or model_checkpoint_path is not None:
            raise ValueError(
                "model checkpoint identity requires a preloaded model"
            )
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

    plugin_config = getattr(plugin, "config", None)
    allowed_overrides = config.runtime_override_fields
    if plugin_config is not None and allowed_overrides is not None:
        for item in fields(config):
            if item.name not in allowed_overrides and hasattr(plugin_config, item.name):
                setattr(config, item.name, getattr(plugin_config, item.name))
    config, flag_snapshot = resolve(config, phase="evaluation")

    # V7 decode telemetry: reset per-suite so forwards/hit-rate are suite-local.
    spec_stats = getattr(plugin, "speculative_stats", None)
    if spec_stats is not None and hasattr(spec_stats, "reset"):
        spec_stats.reset()

    n = len(records)
    document_n = sum(record.target_kind == "document" for record in records)
    parse_ok = 0
    syntax_parse_ok = 0
    raw_syntax_ok = 0
    # Per-metric defined values only: undefined (None) results are excluded so
    # aggregates can never fabricate a vacuous 0.0/1.0 out of unmeasured data.
    fidelity_vals: list[float] = []
    fidelity_norm_vals: list[float] = []
    validity_vals: list[float] = []
    exact_vals: list[float] = []
    ast_beq_vals: list[float] = []
    canonical_beq_vals: list[float] = []
    certificate_equiv_vals: list[float] = []
    struct_vals: list[float] = []
    tree_edit_vals: list[float] = []
    reward_vals: list[float] = []
    recall_vals: list[float] = []
    contract_precision_vals: list[float] = []
    contract_recall_vals: list[float] = []
    binder_reference_f1_vals: list[float] = []
    match_error_count = 0
    reward_error_count = 0
    empty_prediction_count = 0
    gold_design_scores: list[float] = []
    latencies: list[float] = []
    details: list[dict] = []
    semantic_meaning_reports_v2: list[Any] = []
    task_cases: list[dict] = []
    topology_evidence: list[dict[str, Any]] = []
    topology_target_evidence: list[dict[str, Any]] = []
    failure_breakdown: dict[str, int] = {}
    decode_stats_rows: list[object] = []
    canvas_cap = _decode_canvas_cap(plugin)
    score_topology_targets = getattr(plugin, "score_topology_targets", None)
    if callable(score_topology_targets):
        topology_target_evidence = list(score_topology_targets(records))

    # SDE3-01: optional suite-level content-addressed cache.  Key is built from
    # every dependency that can change the suite result.
    eval_data_manifest_sha = _eval_data_sha(Path(config.test_dir))
    eval_suite_manifest_sha = _eval_data_sha(
        Path(config.test_dir) / "suites" / config.suite
    )
    evaluation_policy = _effective_evaluation_policy(config, plugin)
    harness_provenance = HarnessProvenanceV1(
        source_eval_sha256=eval_suite_manifest_sha or UNKNOWN_NOT_CAPTURED,
        evaluation_policy=evaluation_policy,
        timeout_seconds=(
            float(config.decode_timeout_seconds)
            if config.decode_timeout_seconds is not None
            else None
        ),
        canvas_cap=canvas_cap,
        parser_fallback=(
            "allow_unconstrained_fallback"
            if evaluation_policy["allow_unconstrained_fallback"]
            else "forbidden"
        ),
        repair_policy=(
            str(evaluation_policy["grammar_ltr_repair"])
            if evaluation_policy["grammar_ltr_repair"] is not None
            else UNKNOWN_NOT_CAPTURED
        ),
        runtime=(
            f"python/{platform.python_version()} "
            f"{platform.system().lower()}/{platform.machine().lower()}"
        ),
        verifier="meaningful_program/v1+binding_aware_meaningful_v2",
        target_length=canvas_cap,
    )
    harness_provenance_ref = harness_provenance_id(harness_provenance)
    progress_version_stamp = build_version_stamp(
        *_evaluation_version_components(config)
    )
    cache_key = None
    cache_dependencies: dict[str, Any] = {}
    cache_bypass_reason: str | None = None
    # A preloaded model is mutable (the training loop evaluates the live
    # weights before saving the next checkpoint).  Without an explicit
    # checkpoint digest, a suite cache key would be shared by unrelated model
    # states and could replay stale quality as if the current arm had learned.
    # Fail closed: recompute rather than infer identity from architecture-only
    # metadata.  Checkpoint-backed evaluation retains the normal cache path.
    if (
        model is not None
        and checkpoint_sha256 is None
        and cache is not None
        and cache.config.mode is not EvalCacheMode.OFF
    ):
        cache_bypass_reason = "preloaded_model_without_checkpoint_identity"
    elif cache is not None and cache.config.mode is not EvalCacheMode.OFF:
        try:
            component_versions = {
                cid: component_version(cid)
                for cid in _evaluation_version_components(config)
            }
        except Exception:  # noqa: BLE001 - missing version identity forbids reuse
            cache_bypass_reason = "component_version_unavailable"
        else:
            cache_dependencies = {
                "checkpoint_sha256": checkpoint_sha256,
                "eval_data_manifest_sha": eval_data_manifest_sha,
                "eval_suite_manifest_sha": eval_suite_manifest_sha,
                "suite_limit": suite_limit,
                "suite_offset": suite_offset,
                "evaluation_policy": evaluation_policy,
                "generation_overrides": generation_overrides,
                "component_versions": component_versions,
            }
            cache_key = suite_result_key(
                suite=config.suite,
                checkpoint_sha256=checkpoint_sha256,
                eval_data_manifest_sha=eval_data_manifest_sha,
                eval_suite_manifest_sha=eval_suite_manifest_sha,
                eval_limit=suite_limit,
                evaluation_policy=evaluation_policy,
                component_versions=component_versions,
                extra={
                    "eval_offset": suite_offset,
                    "generation_overrides": generation_overrides,
                },
            )
            if cache.config.mode in (EvalCacheMode.READ, EvalCacheMode.READ_WRITE):
                cached_metrics = cache.get(cache_key)
                if cached_metrics is not None:
                    # Replay: keep predictions/metrics byte-identical, but update
                    # the output path to the current run directory.
                    run_dir = config.run_dir
                    run_dir.mkdir(parents=True, exist_ok=True)
                    suite_path = run_dir / f"eval_{config.suite}.json"
                    cached_metrics = dict(cached_metrics)
                    cached_metrics["output"] = str(suite_path)
                    cached_metrics["cache_replay"] = True
                    suite_path.write_text(
                        json.dumps(cached_metrics, indent=2) + "\n", encoding="utf-8"
                    )
                    if config.suite == "smoke":
                        (run_dir / "eval.json").write_text(
                            json.dumps(cached_metrics, indent=2) + "\n", encoding="utf-8"
                        )
                    if evaluation_remaining_records is not None:
                        evaluation_remaining_records[0] = max(
                            0, evaluation_remaining_records[0] - len(records)
                        )
                    (run_dir / "decode_progress.json").unlink(missing_ok=True)
                    return cached_metrics

    batch_size = 1
    generate_batch_requests = getattr(plugin, "generate_batch_requests", None)
    generate_batch = getattr(plugin, "generate_batch", None)
    generate_with_stats = getattr(plugin, "generate_with_stats", None)
    prepare_generation = getattr(plugin, "prepare_generation", None)
    decode_initialization_ms = 0.0
    if callable(prepare_generation):
        # Process-scoped parser/schema/static-artifact initialization is not a
        # document decode.  Run it once, outside every per-chunk deadline, and
        # disclose its wall cost separately so the first record is not the
        # accidental owner of cold harness setup while later records are warm.
        initialization_started = time.perf_counter()
        prepare_generation()
        decode_initialization_ms = (
            time.perf_counter() - initialization_started
        ) * 1000.0
    if callable(generate_batch_requests) or callable(generate_batch):
        batch_size = max(
            1,
            int(
                getattr(getattr(plugin, "config", None), "generate_batch_size", 8) or 8
            ),
        )
    # Prefer the production batch API when a plugin exposes both interfaces.
    # collect_decode_stats() retains row-tagged evidence for the batch path;
    # merely exposing the legacy single-record stats method must not disable
    # I4 row compaction and force sequential decode.

    def _eval_schema() -> str | None:
        if not getattr(config, "schema_in_context", False):
            return None
        from slm_training.harnesses.quality import compact_schema_snippet

        budget = min(600, int(getattr(config, "design_md_budget", 1800) or 1800))
        return compact_schema_snippet(budget=budget)

    def _request_for(record: ExampleRecord) -> GenerationRequest:
        schema = _eval_schema()
        request = GenerationRequest.from_record(record, schema=schema)
        # Template markers are opaque codec surfaces (TEMPLATE_MARKERS_ARE_OPAQUE /
        # RuntimeSymbol law). Never derive semantic_role from placeholder text.
        # Typed authority must come from caller-declared metadata elsewhere.
        data = request.to_dict()
        data["runtime_symbols"] = [
            RuntimeSymbol(
                surface=slot,
                role="external_entity",
            ).to_dict()
            for slot in request.slot_contract
        ]
        return GenerationRequest.from_dict(data)

    def _effective_request_for(record: ExampleRecord) -> GenerationRequest:
        request = _request_for(record)
        data = request.to_dict()
        if not getattr(config, "design_md_in_context", False):
            data.pop("design_md", None)
        if not getattr(config, "slot_contract_in_context", False):
            data["slot_contract"] = []
        return GenerationRequest.from_dict(data)

    def _requests_for(chunk: list[ExampleRecord]) -> list[GenerationRequest]:
        return [_request_for(record) for record in chunk]

    def _generate_chunk_unbounded(
        chunk: list[ExampleRecord],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Generate without passing gold ExampleRecord to the model."""
        if callable(generate_batch_requests):
            with collect_decode_stats() as stats:
                requests = _requests_for(chunk)
                try:
                    predictions = generate_batch_requests(
                        requests, max_len=canvas_cap, **generation_overrides
                    )
                except TypeError:
                    predictions = generate_batch_requests(
                        requests, **generation_overrides
                    )
            _annotate_decode_trace_records(stats, chunk)
            decode_stats_rows.append(stats)
            consume = getattr(plugin, "consume_generation_evidence", None)
            evidence = consume() if callable(consume) else []
            return predictions, list(evidence)
        if callable(generate_with_stats) and len(chunk) == 1:
            try:
                text, stats = generate_with_stats(
                    chunk[0].prompt, max_len=canvas_cap, **generation_overrides
                )
            except TypeError:
                text, stats = generate_with_stats(
                    chunk[0].prompt, **generation_overrides
                )
            _annotate_decode_trace_records(stats, chunk)
            decode_stats_rows.append(stats)
            return [text], []
        prompts = [r.prompt for r in chunk]
        if callable(generate_batch):
            try:
                return generate_batch(
                    prompts, max_len=canvas_cap, **generation_overrides
                ), []
            except TypeError:
                try:
                    return generate_batch(
                        prompts, golds=None, **generation_overrides
                    ), []
                except TypeError:
                    pass
        out: list[str] = []
        for prompt in prompts:
            try:
                out.append(
                    plugin.generate(prompt, max_len=canvas_cap, **generation_overrides)
                )
            except TypeError:
                out.append(plugin.generate(prompt, gold=None, **generation_overrides))
        consume = getattr(plugin, "consume_generation_evidence", None)
        evidence = consume() if callable(consume) else []
        return out, list(evidence)

    decode_timeout_count = 0
    # Document rows that hit the budget wall. Quality rates use only completed
    # documents so a timeout is incompleteness, never a false parse/quality 0.
    decode_timeout_document_n = 0
    # SLM-303: per-chunk decode-outcome evidence (parallel to the chunk loop);
    # each entry carries the timeout fact plus the chunk's DecodeStats row (or
    # None when the plugin exposes no stats) so every scored record can be
    # classified into the decode-outcome taxonomy.
    chunk_decode_meta: list[dict[str, Any]] = []
    decode_chunk_sizes: list[int] = []
    processed_record_n = 0
    effective_decode_timeouts: list[float] = []

    def _generate_chunk(
        chunk: list[ExampleRecord],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Generate a chunk, converting an explicit diagnostic timeout to failures.

        Hard wall: cooperative monotonic deadline (checked in LTR/MaskGIT loops)
        plus a SIGALRM backup. The alarm uses a short interval re-fire so a
        one-shot timeout swallowed by bare ``except Exception`` cannot leave
        decode running for minutes past ``decode_timeout_seconds``.
        """
        nonlocal decode_timeout_count, processed_record_n
        decode_chunk_sizes.append(len(chunk))
        requested_seconds = float(getattr(config, "decode_timeout_seconds", 0) or 0)
        remaining_record_n = (
            int(evaluation_remaining_records[0])
            if evaluation_remaining_records is not None
            else len(chunk)
        )
        seconds = _effective_record_decode_timeout(
            requested_seconds,
            evaluation_deadline=evaluation_deadline,
            remaining_record_n=remaining_record_n,
            chunk_record_n=len(chunk),
        )
        if evaluation_deadline is not None:
            effective_decode_timeouts.append(seconds)
            if evaluation_remaining_records is not None:
                evaluation_remaining_records[0] = max(
                    0, evaluation_remaining_records[0] - len(chunk)
                )

        def _run(timed_out: bool) -> tuple[list[str], list[dict[str, Any]]]:
            nonlocal processed_record_n
            stats_before = len(decode_stats_rows)
            # Re-check at chunk entry so deadline is live even without LTR hooks.
            if not timed_out:
                check_decode_deadline()
            result = _generate_chunk_unbounded(chunk)
            stats = (
                decode_stats_rows[-1] if len(decode_stats_rows) > stats_before else None
            )
            chunk_decode_meta.append(
                {
                    "timed_out": timed_out,
                    "stats": stats,
                    "effective_timeout_seconds": seconds,
                }
            )
            processed_record_n += len(chunk)
            return result

        def _persist_interrupted(exc: KeyboardInterrupt) -> None:
            try:
                stats = getattr(exc, "decode_stats", None)
                if isinstance(stats, DecodeStats):
                    _annotate_decode_trace_records(stats, chunk)
                partial_rows = [
                    row for row in decode_stats_rows if isinstance(row, DecodeStats)
                ]
                if isinstance(stats, DecodeStats) and all(
                    stats is not row for row in partial_rows
                ):
                    partial_rows.append(stats)
                _persist_decode_progress(
                    config,
                    status="interrupted",
                    processed_record_n=processed_record_n,
                    active_chunk=chunk,
                    stats_rows=partial_rows,
                    version_stamp=progress_version_stamp,
                )
            except Exception as progress_error:  # noqa: BLE001
                print(
                    f"decode progress persistence failed: {progress_error}",
                    file=sys.stderr,
                )

        if seconds <= 0:
            try:
                return _run(timed_out=False)
            except KeyboardInterrupt as exc:
                _persist_interrupted(exc)
                raise

        set_decode_deadline(seconds)
        use_alarm = hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")
        previous = None
        if use_alarm:

            def _alarm(_signum: int, _frame: object) -> None:
                raise TimeoutError(f"decode exceeded {seconds:g}s")

            previous = signal.signal(signal.SIGALRM, _alarm)
            # First fire at budget; then every 0.5s until cleared. Re-firing
            # prevents a swallowed one-shot from disabling the wall, while
            # leaving the outer timeout handler enough time to disarm before
            # another tick can interrupt its bookkeeping.
            signal.setitimer(signal.ITIMER_REAL, seconds, 0.5)
        try:
            return _run(timed_out=False)
        except TimeoutError as exc:
            if use_alarm:
                _clear_repeating_decode_alarm(previous)
                use_alarm = False
            stats = getattr(exc, "decode_stats", None)
            if stats is not None:
                _annotate_decode_trace_records(stats, chunk)
                decode_stats_rows.append(stats)
            chunk_decode_meta.append(
                {
                    "timed_out": True,
                    "stats": stats,
                    "effective_timeout_seconds": seconds,
                }
            )
            decode_timeout_count += len(chunk)
            processed_record_n += len(chunk)
            return ["" for _ in chunk], []
        except KeyboardInterrupt as exc:
            if use_alarm:
                _clear_repeating_decode_alarm(previous)
                use_alarm = False
            _persist_interrupted(exc)
            raise
        finally:
            if use_alarm:
                _clear_repeating_decode_alarm(previous)
            clear_decode_deadline()

    def _decode_outcome_fields(
        pred: str,
        *,
        parse_ok: bool | None,
        error: str | None,
        decode_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """SLM-303 additive per-record decode-outcome classification.

        Computed from the same evidence the runner already has (chunk timeout
        fact, chunk DecodeStats fallback counters, parse verdict); never
        changes any existing field.
        """
        from slm_training.harnesses.model_build.decode_outcome import (
            MODEL_VALID,
            classify_decode_outcome,
            fallback_counter_total,
        )

        meta = decode_meta or {}
        stats = meta.get("stats")
        fallbacks = fallback_counter_total(stats)
        timed_out = bool(meta.get("timed_out"))
        outcome = classify_decode_outcome(
            parse_ok=parse_ok,
            error=error,
            fallback_counters=fallbacks,
            timed_out=timed_out,
            abstained=not pred.strip(),
            harness_exception=False,
        )
        if timed_out:
            stop_reason = "decode_timeout"
        else:
            stop_reason = ""
            for name in (
                "compiler_lattice_termination_reason",
                "solver_terminal_status",
            ):
                value = str(getattr(stats, name, "") or "") if stats is not None else ""
                if value:
                    stop_reason = value
                    break
            if not stop_reason:
                stop_reason = "empty_prediction" if not pred.strip() else "completed"
        detail = None
        if outcome == MODEL_VALID and parse_ok is None:
            detail = "parse_not_evaluated"
        return {
            "decode_outcome": outcome,
            "stop_reason": stop_reason,
            "fallback_used": fallbacks > 0,
            "decode_outcome_detail": detail,
        }

    def _score_one(
        record: ExampleRecord,
        pred: str,
        latency_ms: float,
        prediction_evidence: dict[str, Any] | None = None,
        decode_meta: dict[str, Any] | None = None,
    ) -> None:
        nonlocal parse_ok, syntax_parse_ok, raw_syntax_ok
        nonlocal match_error_count, reward_error_count, empty_prediction_count
        nonlocal decode_timeout_document_n
        timed_out = bool((decode_meta or {}).get("timed_out"))
        evidence = dict(prediction_evidence or {})
        if len(topology_target_evidence) > len(topology_evidence):
            evidence.update(topology_target_evidence[len(topology_evidence)])

        # Runtime timeout is incompleteness: record the budget fact, leave
        # quality fields unmeasured (None), and never count the empty stand-in
        # prediction as abstention or parse/fidelity/structure/reward failure.
        # Topology slot stays for index alignment but is marked incomplete so
        # every quality aggregate skips it.
        if timed_out:
            if record.target_kind == "document":
                decode_timeout_document_n += 1
            lineage = prediction_lineage(pred)
            incomplete_evidence = {"incomplete": True, "decode_timeout": True}
            topology_evidence.append(incomplete_evidence)
            details.append(
                {
                    "id": record.id,
                    "target_kind": record.target_kind,
                    "incomplete": True,
                    "parse_ok": None,
                    "meaningful_program_v1": None,
                    "binding_aware_meaningful_v2": None,
                    "syntax_parse_valid": None,
                    "raw_syntax_valid": None,
                    "error": "decode_timeout",
                    "placeholder_fidelity": None,
                    "placeholder_fidelity_normalized": None,
                    "placeholder_validity": None,
                    "contract_precision": None,
                    "contract_recall": None,
                    "binder_reference_f1": None,
                    "exact_match": None,
                    "ast_beq": None,
                    "canonical_beq": None,
                    "certificate_equivalent": None,
                    "structural_similarity": None,
                    "tree_edit_similarity": None,
                    "component_type_recall": None,
                    "reward_score": None,
                    "gold_design_lint_score": None,
                    "design_lint_score": None,
                    "latency_ms": round(latency_ms, 2),
                    "prediction": pred,
                    "prediction_sha256": lineage["raw_prediction_sha256"],
                    **lineage,
                    "harness_provenance_id": harness_provenance_ref,
                    "generation_request": _effective_request_for(record).to_dict(),
                    "source_record_sha256": hashlib.sha256(
                        json.dumps(
                            record.to_dict(), sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                    # Do not attach precomputed target topology scores: those
                    # would launder gold-side evidence into a timed-out row.
                    "topology_evidence": None,
                    "temporal_decode_evidence": _temporal_decode_evidence(
                        (decode_meta or {}).get("stats"), record.id
                    ),
                    **_decode_outcome_fields(
                        pred,
                        parse_ok=None,
                        error="decode_timeout",
                        decode_meta=decode_meta,
                    ),
                }
            )
            return

        if not pred.strip():
            empty_prediction_count += 1
        if record.target_kind != "document":
            from slm_training.evals.task_scoreboard import score_output_targets

            target_score = score_output_targets(pred, record.output_targets)
            lineage = prediction_lineage(pred)
            topology_evidence.append(evidence)
            details.append(
                {
                    "id": record.id,
                    "target_kind": record.target_kind,
                    "target_score": target_score,
                    "latency_ms": round(latency_ms, 2),
                    "prediction": pred,
                    # Keep the legacy digest while recording explicit replay
                    # lineage. There is no captured intermediate decoder output.
                    "prediction_sha256": lineage["raw_prediction_sha256"],
                    **lineage,
                    "harness_provenance_id": harness_provenance_ref,
                    "generation_request": _effective_request_for(record).to_dict(),
                    "source_record_sha256": hashlib.sha256(
                        json.dumps(
                            record.to_dict(), sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                    "topology_evidence": evidence or None,
                    "temporal_decode_evidence": _temporal_decode_evidence(
                        (decode_meta or {}).get("stats"), record.id
                    ),
                    **_decode_outcome_fields(
                        pred, parse_ok=None, error=None, decode_meta=decode_meta
                    ),
                }
            )
            task_cases.append(
                {
                    "id": record.id,
                    "task": str((record.meta or {}).get("task") or "unknown"),
                    "gold": record.openui,
                    "prediction": pred,
                    "target_kind": record.target_kind,
                    "target_category": record.target_category,
                    "accepted_outputs": [
                        target.__dict__ for target in record.accepted_outputs
                    ],
                    "prediction_evidence": evidence,
                }
            )
            return
        ok, error, serialized = _is_meaningful_program(pred, gold=record)
        from slm_training.evals.meaningful_program import binding_aware_meaningful_v2

        semantic_report_v2 = binding_aware_meaningful_v2(
            pred, record=record, request=_effective_request_for(record)
        )
        semantic_meaning_reports_v2.append(semantic_report_v2)
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
        syntax_ok = _raw_syntax_valid(scored_pred)
        if syntax_ok:
            syntax_parse_ok += 1
            raw_syntax_ok += 1
        fid = _placeholder_fidelity(scored_pred, record)
        fid_norm = _placeholder_fidelity_normalized(scored_pred, record)
        ph_valid = _placeholder_validity(scored_pred, record)
        exact: float | None
        try:
            exact = _tree_match(scored_pred, record.openui)
        except Exception:  # noqa: BLE001 — gold-side/harness failure, not model quality
            match_error_count += 1
            exact = None
        struct = structural_similarity(scored_pred, record.openui)
        # tree_edit_similarity is currently an alias of structural_similarity;
        # reuse the value instead of recomputing the full metric.
        tree_edit = struct
        from slm_training.evals.semantic_fidelity import (
            canonical_beq as _canonical_beq,
        )
        from slm_training.evals.semantic_fidelity import (
            certificate_equivalent as _certificate_equivalent,
        )

        # BEq analogues: Boolean equality predicates (not soft similarity).
        # exact_match already tracks structure-normalized equality; ast_beq
        # reuses that bit for the ship-gate field (avoids a second validate).
        ast_beq_bit: float | None
        if exact is None:
            ast_beq_bit = None
        else:
            ast_beq_bit = 1.0 if exact >= 1.0 else 0.0
        can_beq_bit = 1.0 if _canonical_beq(scored_pred, record.openui) else 0.0
        # Optional certificate pair from record meta / prediction evidence.
        pred_cert = (evidence or {}).get("support_certificate") or (evidence or {}).get(
            "formal_object"
        )
        gold_cert = (record.meta or {}).get("support_certificate") or (
            record.meta or {}
        ).get("formal_object")
        cert_bit: float | None
        if pred_cert is not None or gold_cert is not None:
            cert_bit = 1.0 if _certificate_equivalent(pred_cert, gold_cert) else 0.0
        else:
            cert_bit = None
        recall = component_type_recall(scored_pred, record.openui)
        contract_prec = _contract_precision(scored_pred, record)
        contract_rec = _contract_recall(scored_pred, record)
        binder_reference_f1 = _binder_reference_f1(contract_prec, contract_rec)
        reward: float | None
        try:
            reward = _reward_for_prediction(scored_pred, record)
        except Exception:  # noqa: BLE001 — reward harness failure, not model quality
            reward_error_count += 1
            reward = None
        codec = getattr(plugin, "codec", None)
        if codec is not None:
            try:
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
            except ImportError:
                # Torch-free hosts cannot import grammar_diffusion (module-level
                # torch). Fall back to a pure string-ratio stand-in so topology
                # composite evidence stays defined for scoreboard wiring tests.
                from difflib import SequenceMatcher

                ratio = SequenceMatcher(
                    a=scored_pred, b=record.openui, autojunk=False
                ).ratio()
                evidence["production_accuracy"] = ratio
                evidence["arity_accuracy"] = ratio
        topology_evidence.append(evidence)
        gold_dscore = _gold_design_lint_score(record)
        for defined_values, value in (
            (fidelity_vals, fid),
            (fidelity_norm_vals, fid_norm),
            (validity_vals, ph_valid),
            (exact_vals, exact),
            (ast_beq_vals, ast_beq_bit),
            (canonical_beq_vals, can_beq_bit),
            (certificate_equiv_vals, cert_bit),
            (struct_vals, struct),
            (tree_edit_vals, tree_edit),
            (recall_vals, recall),
            (contract_precision_vals, contract_prec),
            (contract_recall_vals, contract_rec),
            (binder_reference_f1_vals, binder_reference_f1),
            (reward_vals, reward),
        ):
            if value is not None:
                defined_values.append(float(value))
        if gold_dscore is not None:
            gold_design_scores.append(gold_dscore)
        lineage = prediction_lineage(pred)
        details.append(
            {
                "id": record.id,
                "parse_ok": ok,
                "meaningful_program_v1": ok,
                "binding_aware_meaningful_v2": semantic_report_v2.verdict,
                "semantic_meaning_report_v2": semantic_report_v2.to_dict(),
                "syntax_parse_valid": syntax_ok,
                "raw_syntax_valid": _raw_syntax_valid(scored_pred),
                "error": error,
                "placeholder_fidelity": fid,
                "placeholder_fidelity_normalized": fid_norm,
                "placeholder_validity": ph_valid,
                "contract_precision": contract_prec,
                "contract_recall": contract_rec,
                "binder_reference_f1": binder_reference_f1,
                "exact_match": exact,
                "ast_beq": None if ast_beq_bit is None else bool(ast_beq_bit >= 1.0),
                "canonical_beq": bool(can_beq_bit >= 1.0),
                "certificate_equivalent": (
                    None if cert_bit is None else bool(cert_bit >= 1.0)
                ),
                "structural_similarity": struct,
                "tree_edit_similarity": tree_edit,
                "component_type_recall": recall,
                "reward_score": reward,
                "gold_design_lint_score": gold_dscore,
                "design_lint_score": gold_dscore,
                "latency_ms": round(latency_ms, 2),
                # Full text + digest make every new metric report replayable.
                "prediction": pred,
                "prediction_sha256": lineage["raw_prediction_sha256"],
                **lineage,
                "harness_provenance_id": harness_provenance_ref,
                "generation_request": _effective_request_for(record).to_dict(),
                "source_record_sha256": hashlib.sha256(
                    json.dumps(
                        record.to_dict(), sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "semantic_factor": str(
                    (record.meta or {}).get("semantic_factor")
                    or record.target_category
                    or record.target_kind
                ),
                "complexity": {
                    "program_chars": len(record.openui),
                    "statement_count": record.openui.count("\n") + 1,
                    "bucket": (
                        "small"
                        if len(record.openui) < 160
                        else "medium"
                        if len(record.openui) < 400
                        else "large"
                    ),
                },
                "serialized": serialized,
                "topology_evidence": evidence or None,
                "temporal_decode_evidence": _temporal_decode_evidence(
                    (decode_meta or {}).get("stats"), record.id
                ),
                **_decode_outcome_fields(
                    pred, parse_ok=ok, error=error, decode_meta=decode_meta
                ),
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
                "target_kind": record.target_kind,
                "target_category": record.target_category,
                "accepted_outputs": [
                    target.__dict__ for target in record.accepted_outputs
                ],
            }
        )

    incomplete_latencies: list[float] = []
    if batch_size > 1 and (
        callable(generate_batch_requests) or callable(generate_batch)
    ):
        for start in range(0, n, batch_size):
            chunk = records[start : start + batch_size]
            t0 = time.perf_counter()
            preds, evidence_rows = _generate_chunk(chunk)
            chunk_meta = chunk_decode_meta[-1] if chunk_decode_meta else None
            timed_out = bool((chunk_meta or {}).get("timed_out"))
            elapsed = (time.perf_counter() - t0) * 1000.0
            per = elapsed / max(1, len(chunk))
            for index, (record, pred) in enumerate(zip(chunk, preds)):
                if timed_out:
                    incomplete_latencies.append(per)
                else:
                    latencies.append(per)
                evidence = evidence_rows[index] if index < len(evidence_rows) else None
                _score_one(record, pred, per, evidence, chunk_meta)
    else:
        for record in records:
            t0 = time.perf_counter()
            predictions, evidence_rows = _generate_chunk([record])
            chunk_meta = chunk_decode_meta[-1] if chunk_decode_meta else None
            timed_out = bool((chunk_meta or {}).get("timed_out"))
            pred = predictions[0]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if timed_out:
                incomplete_latencies.append(elapsed_ms)
            else:
                latencies.append(elapsed_ms)
            _score_one(
                record,
                pred,
                elapsed_ms,
                evidence_rows[0] if evidence_rows else None,
                chunk_meta,
            )

    # Primary latency percentiles are completed attempts only so a budget wall
    # does not look like "slow but finishing" quality. Timeout walls stay under
    # incomplete_* / decode_timeout_* runtime fields.
    lat_sorted = sorted(latencies)
    all_lat_sorted = sorted(latencies + incomplete_latencies)

    p50 = _nearest_rank(lat_sorted, 0.50)
    p95 = _nearest_rank(lat_sorted, 0.95)
    p50_all = _nearest_rank(all_lat_sorted, 0.50)
    p95_all = _nearest_rank(all_lat_sorted, 0.95)
    gold_design_mean = (
        sum(gold_design_scores) / len(gold_design_scores)
        if gold_design_scores
        else None
    )

    def _mean_or_none(defined_values: list[float]) -> float | None:
        """Mean over defined values; None (never a fabricated 0/1) when empty."""
        return sum(defined_values) / len(defined_values) if defined_values else None

    # Real fallback telemetry from the decode path; None (gate fails as
    # unmeasured) when the plugin exposes no decode stats — never hardcoded 0.
    if decode_stats_rows:
        fallback_count = sum(
            int(getattr(row, name, 0) or 0)
            for row in decode_stats_rows
            for name in (
                "unconstrained_retries",
                "compiler_fallbacks",
                "seeded_fallbacks",
                "template_fallback_count",
                "certified_fallbacks",
            )
        )
    else:
        fallback_count = None

    from slm_training.evals.power_protocol import binomial_rate_evidence
    from slm_training.evals.record_schema import RUN_CLASSES, SCHEMA_VERSION

    def _rate_evidence(successes: int, total: int) -> dict[str, Any]:
        evidence_class = (
            "unmeasured"
            if total == 0
            else (
                "diagnostic_subset"
                if suite_limit is not None or suite_offset > 0
                else (
                    "meets_default_suite_n"
                    if total >= DEFAULT_MIN_SUITE_N
                    else "fixture_under_minimum_n"
                )
            )
        )
        return binomial_rate_evidence(
            successes,
            total,
            seed_count=1,
            evidence_class=evidence_class,
        )

    # Quality denominators exclude runtime timeouts (incomplete), not full
    # document_n. All-timeout suites report rates as null, never false 0.0.
    completed_document_n = document_n - decode_timeout_document_n
    syntax_rate_evidence = _rate_evidence(syntax_parse_ok, completed_document_n)
    meaningful_rate_evidence = _rate_evidence(parse_ok, completed_document_n)
    timeout_rate_evidence = _rate_evidence(decode_timeout_document_n, document_n)
    unmeasured_rate_evidence = {
        "schema": "binomial_rate_evidence/v1",
        "numerator": None,
        "denominator": 0,
        "seed_count": 1,
        "interval": {
            "method": "wilson_score",
            "n": 0,
            "estimate": None,
            "low": None,
            "high": None,
            "confidence_level": 0.95,
        },
        "evidence_class": "unmeasured",
    }

    run_class = (
        config.run_class if config.run_class in RUN_CLASSES else "scratch_matrix"
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_class": run_class,
        "suite": config.suite,
        "n": n,
        "document_n": document_n,
        "completed_document_n": completed_document_n,
        "incomplete_document_n": decode_timeout_document_n,
        "fragment_n": n - document_n,
        "eval_limit": suite_limit,
        "eval_offset": suite_offset,
        "seed": int(getattr(config, "seed", 0) or 0),
        "diagnostic_subset": suite_limit is not None or suite_offset > 0,
        "evaluation_wall_seconds": config.evaluation_wall_seconds,
        "effective_decode_timeout_seconds_min": (
            min(effective_decode_timeouts) if effective_decode_timeouts else None
        ),
        "effective_decode_timeout_seconds_max": (
            max(effective_decode_timeouts) if effective_decode_timeouts else None
        ),
        "decode_batch_size_configured": batch_size,
        "decode_batch_size_max": max(decode_chunk_sizes) if decode_chunk_sizes else 0,
        "decode_chunk_n": len(decode_chunk_sizes),
        # Persist the effective decode policy beside every scoreboard.  This
        # is essential for comparing historical runs: checkpoint defaults and
        # CLI diagnostic overrides can materially change quality and timeout
        # metrics even when the checkpoint hash is identical.
        "evaluation_policy": evaluation_policy,
        "harness_provenance": harness_provenance.to_dict(),
        "harness_provenance_id": harness_provenance_ref,
        # Quality rates are over completed (non-timeout) documents only.
        # None (JSON null) when nothing completed — never a fabricated 0.0.
        "parse_rate": (
            (syntax_parse_ok / completed_document_n) if completed_document_n else None
        ),
        "meaningful_program_rate": (
            (parse_ok / completed_document_n) if completed_document_n else None
        ),
        "syntax_parse_rate": (
            (syntax_parse_ok / completed_document_n) if completed_document_n else None
        ),
        "raw_syntax_validity": (
            (raw_syntax_ok / completed_document_n) if completed_document_n else None
        ),
        "parse_rate_ci95": [
            round(float(bound), 4)
            for bound in (
                syntax_rate_evidence["interval"]["low"],
                syntax_rate_evidence["interval"]["high"],
            )
            if bound is not None
        ]
        or None,
        "meaningful_program_rate_ci95": [
            round(float(bound), 4)
            for bound in (
                meaningful_rate_evidence["interval"]["low"],
                meaningful_rate_evidence["interval"]["high"],
            )
            if bound is not None
        ]
        or None,
        "rate_evidence": {
            "parse_rate": syntax_rate_evidence,
            "syntax_parse_rate": syntax_rate_evidence,
            "raw_syntax_validity": _rate_evidence(raw_syntax_ok, completed_document_n),
            "meaningful_program_rate": meaningful_rate_evidence,
            "exact_match": _rate_evidence(int(sum(exact_vals)), len(exact_vals)),
            "decode_timeout_rate": timeout_rate_evidence,
            "residual_mask_rate": unmeasured_rate_evidence,
            "oov_rate": unmeasured_rate_evidence,
        },
        "contract_precision": _mean_or_none(contract_precision_vals),
        "contract_recall": _mean_or_none(contract_recall_vals),
        "binder_reference_f1": _mean_or_none(binder_reference_f1_vals),
        # Not computed by any current decode path; None (not a fake 0.0) until
        # a plugin actually measures them.
        "residual_mask_rate": None,
        "oov_rate": None,
        "fallback_count": fallback_count,
        "placeholder_fidelity": _mean_or_none(fidelity_vals),
        "placeholder_fidelity_normalized": _mean_or_none(fidelity_norm_vals),
        "placeholder_validity": _mean_or_none(validity_vals),
        "exact_match": _mean_or_none(exact_vals),
        # Semantic-fidelity BEq rates (first-class ship gates — not syntax alone).
        "ast_beq_rate": _mean_or_none(ast_beq_vals),
        "canonical_beq_rate": _mean_or_none(canonical_beq_vals),
        "certificate_equivalence_rate": _mean_or_none(certificate_equiv_vals),
        "certificates_compared": len(certificate_equiv_vals),
        "structural_similarity": _mean_or_none(struct_vals),
        "tree_edit_similarity": _mean_or_none(tree_edit_vals),
        "component_type_recall": _mean_or_none(recall_vals),
        "reward_score": _mean_or_none(reward_vals),
        # How many document records actually defined each mean above — the
        # denominator disclosure that separates "measured 0" from "unmeasured".
        "metric_defined_n": {
            "contract_precision": len(contract_precision_vals),
            "contract_recall": len(contract_recall_vals),
            "binder_reference_f1": len(binder_reference_f1_vals),
            "placeholder_fidelity": len(fidelity_vals),
            "placeholder_fidelity_normalized": len(fidelity_norm_vals),
            "placeholder_validity": len(validity_vals),
            "exact_match": len(exact_vals),
            "ast_beq_rate": len(ast_beq_vals),
            "canonical_beq_rate": len(canonical_beq_vals),
            "certificate_equivalence_rate": len(certificate_equiv_vals),
            "structural_similarity": len(struct_vals),
            "tree_edit_similarity": len(tree_edit_vals),
            "component_type_recall": len(recall_vals),
            "reward_score": len(reward_vals),
        },
        "match_error_count": match_error_count,
        "reward_error_count": reward_error_count,
        "empty_prediction_count": empty_prediction_count,
        "gold_design_lint_score": gold_design_mean,
        # Alias kept for older dashboards; do not gate ship on this.
        "design_lint_score": gold_design_mean,
        # Completed-only (quality/perf signal). Timeout walls live under
        # latency_ms_*_including_incomplete and decode_timeout_*.
        "latency_ms_p50": round(p50, 2) if p50 is not None else None,
        "latency_ms_p95": round(p95, 2) if p95 is not None else None,
        "latency_ms_p50_including_incomplete": (
            round(p50_all, 2) if p50_all is not None else None
        ),
        "latency_ms_p95_including_incomplete": (
            round(p95_all, 2) if p95_all is not None else None
        ),
        "completed_latency_n": len(latencies),
        "incomplete_latency_n": len(incomplete_latencies),
        "checkpoint": str(loaded_checkpoint) if loaded_checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_source": ("checkpoint" if loaded_checkpoint else "preloaded_model"),
        # Pin the exact eval data alongside the model identity so every
        # reported number is reproducible (run + checkpoint + dataset).
        "test_dir": str(config.test_dir),
        "eval_data_manifest_sha": _eval_data_sha(Path(config.test_dir)),
        "eval_suite_manifest_sha": _eval_data_sha(
            Path(config.test_dir) / "suites" / config.suite
        ),
        "model": config.model_name,
        "code_git_sha": _git_sha(),
        "code_dirty": _git_dirty(),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "failure_breakdown": failure_breakdown,
        # Timeout is a runtime/completeness signal, not a quality rate. Ship
        # gates already fail decode_timeout_count under runtime_failures.
        "decode_timeout_count": decode_timeout_count,
        "decode_timeout_document_count": decode_timeout_document_n,
        "decode_timeout_rate": (
            (decode_timeout_document_n / document_n) if document_n else None
        ),
        "decode_initialization_ms": round(decode_initialization_ms, 3),
        "decode_canvas_cap": canvas_cap,
        # SLM-303: per-taxonomy decode-outcome counts over details[] rows
        # (additive; every taxonomy key always present).
        "decode_outcome_counts": outcome_counts(
            [str(row.get("decode_outcome")) for row in details]
        ),
        "details": details,
        "generation_evidence_schemas": sorted(
            {
                str(row["schema"])
                for row in topology_evidence
                if isinstance(row, dict) and row.get("schema")
            }
        ),
    }
    from slm_training.evals.meaningful_program import aggregate_meaning_reports_v2

    meaning_v2 = aggregate_meaning_reports_v2(semantic_meaning_reports_v2)
    strict_positive_n = sum(report.verdict for report in semantic_meaning_reports_v2)
    covered_reports_v2 = [
        report for report in semantic_meaning_reports_v2 if report.coverage_known
    ]
    covered_positive_n = sum(report.verdict for report in covered_reports_v2)
    covered_n = len(covered_reports_v2)
    metrics.update(
        {
            "meaningful_program_v1_rate": metrics["meaningful_program_rate"],
            "binding_aware_meaningful_v2_rate_strict": (
                meaning_v2["strict_rate"] if semantic_meaning_reports_v2 else None
            ),
            "binding_aware_meaningful_v2_rate_coverage_conditioned": (
                meaning_v2["coverage_conditioned_rate"] if covered_n else None
            ),
            "binding_aware_meaningful_v2_coverage": (
                meaning_v2["coverage"] if semantic_meaning_reports_v2 else None
            ),
            "meaningful_metric_primary": "meaningful_program_v1",
            "meaningful_metric_versions": {
                "meaningful_program_v1": "1.0.0",
                "binding_aware_meaningful_v2": meaning_v2,
            },
        }
    )
    metrics["rate_evidence"].update(
        {
            "meaningful_program_v1_rate": _rate_evidence(
                parse_ok, completed_document_n
            ),
            "binding_aware_meaningful_v2_rate_strict": _rate_evidence(
                strict_positive_n, len(semantic_meaning_reports_v2)
            ),
            "binding_aware_meaningful_v2_rate_coverage_conditioned": _rate_evidence(
                covered_positive_n, covered_n
            ),
            "binding_aware_meaningful_v2_coverage": _rate_evidence(
                covered_n, len(semantic_meaning_reports_v2)
            ),
        }
    )
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
    # Topology / scope aggregates exclude incomplete (timeout) rows so pre-scored
    # target evidence never dilutes completed quality.
    completed_topology_evidence = [
        row
        for row in topology_evidence
        if not (isinstance(row, dict) and row.get("incomplete"))
    ]
    scope_contract_metrics = _aggregate_scope_contract_metrics(
        completed_topology_evidence
    )
    if scope_contract_metrics is not None:
        metrics["scope_contract_metrics"] = scope_contract_metrics
    if completed_topology_evidence and all(
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
        for row in completed_topology_evidence
    ):

        def mean(key: str) -> float:
            return sum(float(row[key]) for row in completed_topology_evidence) / len(
                completed_topology_evidence
            )

        quality_inputs = (
            metrics["meaningful_program_rate"],
            metrics["placeholder_fidelity"],
            metrics["structural_similarity"],
            metrics["reward_score"],
            metrics["tree_edit_similarity"],
        )
        quality = (
            None
            if any(value is None for value in quality_inputs)
            else (
                2.0 * float(metrics["meaningful_program_rate"])
                + 2.0 * float(metrics["placeholder_fidelity"])
                + float(metrics["structural_similarity"])
                + 0.5 * float(metrics["reward_score"])
            )
            / 5.5
        )
        ast_node = metrics["ast_node_f1"]
        ast_edge = metrics["ast_edge_f1"]
        if quality is not None and ast_node is not None and ast_edge is not None:
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
                        for key in completed_topology_evidence[0]
                        if isinstance(
                            completed_topology_evidence[0].get(key), (int, float)
                        )
                        and all(
                            isinstance(row.get(key), (int, float))
                            for row in completed_topology_evidence
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
    if decode_stats_rows:
        from slm_training.models.decode_stats import aggregate_stats

        metrics["decode_stats"] = aggregate_stats(decode_stats_rows)
        retries = sum(
            int(getattr(row, "unconstrained_retries", 0)) for row in decode_stats_rows
        )
        metrics["constrained_fallback_rate"] = retries / len(decode_stats_rows)
        metrics["rate_evidence"]["constrained_fallback_rate"] = {
            "schema": "rate_evidence/v1",
            "numerator": retries,
            "denominator": len(decode_stats_rows),
            "seed_count": 1,
            "interval": None,
            "evidence_class": "decode_batch_telemetry_non_binomial",
        }

    # Metrics the active decode policy enforces by construction: consumers must
    # not read them as learned model skill (e.g. constrained decode guarantees
    # syntax; slot-contract injection supplies the contract placeholders).
    eval_policy = metrics["evaluation_policy"]
    decoder_guaranteed: list[str] = []
    if (
        eval_policy.get("grammar_constrained")
        or eval_policy.get("grammar_ltr_primary")
        or eval_policy.get("compiler_decode_mode") not in (None, "off")
    ):
        decoder_guaranteed += ["parse_rate", "syntax_parse_rate", "raw_syntax_validity"]
    if eval_policy.get("slot_contract_constrained_decode"):
        decoder_guaranteed += [
            "contract_precision",
            "contract_recall",
            "placeholder_fidelity",
            "placeholder_fidelity_normalized",
            "placeholder_validity",
        ]
    metrics["decoder_guaranteed"] = decoder_guaranteed
    if cache_bypass_reason is not None:
        metrics["cache_bypass_reason"] = cache_bypass_reason

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_snapshot(run_dir, flag_snapshot)
    suite_path = run_dir / f"eval_{config.suite}.json"
    metrics["version_stamp"] = progress_version_stamp
    metrics["output"] = str(suite_path)
    if publish_agentv:
        if config.suite in DEFAULT_SHIP_GATES:
            from slm_training.evals.agentv import publish_model_evaluation

            # Single-suite runs publish only the suite that actually ran —
            # never four missing_suite auto-failures dressed up as 5/5 failed.
            metrics["agentv"] = publish_model_evaluation(
                run_dir,
                {config.suite: metrics},
                include_missing_suites=False,
            )
            metrics["agentv"]["suites_run"] = [config.suite]
        else:
            metrics["agentv"] = {
                "skipped": f"suite {config.suite!r} is not in the ship-gate policy"
            }
    # SDE3-01: persist the full suite result for exact replay when enabled.
    if (
        cache is not None
        and cache_key is not None
        and cache.config.mode
        in (
            EvalCacheMode.READ_WRITE,
            EvalCacheMode.REFRESH,
        )
    ):
        try:
            cache.put(cache_key, metrics, dependencies=cache_dependencies)
        except Exception:  # noqa: BLE001,S110 - cache write must never break eval
            pass

    payload = json.dumps(metrics, indent=2) + "\n"
    suite_path.write_text(payload, encoding="utf-8")
    if config.suite == "smoke":
        (run_dir / "eval.json").write_text(payload, encoding="utf-8")
    (run_dir / "decode_progress.json").unlink(missing_ok=True)
    if publish_agentv:
        _record_langsmith_evaluation(
            config,
            suites={config.suite: metrics},
            scoreboard={
                "run_class": config.run_class,
                "checkpoint_sha256": metrics.get("checkpoint_sha256"),
                "eval_data_manifest_sha": metrics.get("eval_data_manifest_sha"),
                "code_git_sha": metrics.get("code_git_sha"),
                "version_stamp": metrics["version_stamp"],
                "agentv": metrics.get("agentv"),
            },
        )
    return metrics


def evaluate_grammar_leakage_audit(
    config: ModelBuildConfig,
    model=None,
    checkpoint: Path | None = None,
    *,
    publish_agentv: bool = True,
    variant_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compare deterministic constrained decoders on one checkpoint.

    Raw logits remain available as constraint-shadow telemetry, but are never
    emitted as an unconstrained program.
    """
    from contextlib import contextmanager
    from dataclasses import replace

    all_variants = {
        "constrained_native": {
            "grammar_constrained": True,
            "grammar_ltr_repair": True,
            "grammar_uniform_at_unforced": False,
            "compiler_decode_mode": "off",
        },
        "constrained_compiler": {
            "grammar_constrained": True,
            "grammar_ltr_repair": True,
            "grammar_uniform_at_unforced": False,
            "compiler_decode_mode": "tree",
        },
    }
    selected_names = variant_names or tuple(all_variants)
    if "constrained_native" not in selected_names or not set(selected_names).issubset(
        all_variants
    ):
        raise ValueError(
            "grammar leakage audit variants must include constrained_native"
        )
    variants = {name: all_variants[name] for name in selected_names}

    @contextmanager
    def _temporary_plugin_config(overrides: dict[str, Any]):
        plugin_config = getattr(model, "config", None)
        saved = {
            key: getattr(plugin_config, key)
            for key in overrides
            if plugin_config is not None and hasattr(plugin_config, key)
        }
        try:
            for key, value in overrides.items():
                if plugin_config is not None and hasattr(plugin_config, key):
                    setattr(plugin_config, key, value)
            yield
        finally:
            for key, value in saved.items():
                setattr(plugin_config, key, value)

    results: dict[str, dict[str, Any]] = {}
    for name, overrides in variants.items():
        variant_config = replace(
            config,
            run_id=f"{config.run_id}/grammar-leakage-{name}",
            **overrides,
        )
        with _temporary_plugin_config(overrides):
            results[name] = evaluate(
                variant_config,
                model=model,
                checkpoint=checkpoint,
                publish_agentv=publish_agentv,
                cache=None,
                generation_overrides={
                    "grammar_constrained": overrides["grammar_constrained"]
                },
            )

    metric_names = (
        "meaningful_program_rate",
        "parse_rate",
        "placeholder_fidelity",
        "contract_precision",
        "contract_recall",
        "binder_reference_f1",
        "structural_similarity",
    )
    baseline = results["constrained_native"]
    deltas = {
        name: {
            metric: (
                None
                if metrics.get(metric) is None or baseline.get(metric) is None
                else float(metrics[metric]) - float(baseline[metric])
            )
            for metric in metric_names
        }
        for name, metrics in results.items()
        if name != "constrained_native"
    }

    def _strata(metrics: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
            "semantic_factor": {},
            "complexity": {},
        }
        for detail in metrics.get("details") or ():
            grouped["semantic_factor"].setdefault(
                str(detail.get("semantic_factor") or "unknown"), []
            ).append(detail)
            grouped["complexity"].setdefault(
                str((detail.get("complexity") or {}).get("bucket") or "unknown"), []
            ).append(detail)

        def _completed_rate(rows: list[dict[str, Any]], key: str) -> float | None:
            completed = [
                row
                for row in rows
                if not row.get("incomplete") and row.get(key) is not None
            ]
            if not completed:
                return None
            return sum(bool(row.get(key)) for row in completed) / len(completed)

        return {
            axis: {
                label: {
                    "n": len(rows),
                    "completed_n": sum(1 for row in rows if not row.get("incomplete")),
                    "incomplete_n": sum(1 for row in rows if row.get("incomplete")),
                    "meaningful_program_rate": _completed_rate(
                        rows, "binding_aware_meaningful_v2"
                    ),
                    "parse_rate": _completed_rate(rows, "parse_ok"),
                }
                for label, rows in labels.items()
            }
            for axis, labels in grouped.items()
        }

    payload: dict[str, Any] = {
        "schema_version": "grammar-leakage-audit/v1",
        "run_id": config.run_id,
        "suite": config.suite,
        "variants": results,
        "baseline_deltas": deltas,
        "strata": {name: _strata(metrics) for name, metrics in results.items()},
        "claim_scope": (
            "evaluation-only constrained decoder comparison; raw logits are "
            "diagnostic shadows and never emitted"
        ),
    }
    from slm_training.versioning import build_version_stamp

    payload["version_stamp"] = build_version_stamp(
        *_evaluation_version_components(config)
    )
    output = config.run_dir / "grammar_leakage_audit.json"
    payload["output"] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def evaluate_suites(
    config: ModelBuildConfig,
    suites: list[str],
    *,
    checkpoint: Path | None = None,
    model=None,
    write_gates: bool = False,
    cache: EvalCache | None = None,
    suite_reachability: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Run eval across multiple suites; write scoreboard.json (and optional gates)."""
    from dataclasses import replace

    from slm_training.harnesses.model_build.ship_gates import write_ship_gates

    # Loading a checkpoint-backed model once is materially cheaper than
    # rebuilding it for every suite.  Bind the shared instance to the exact
    # checkpoint digest so suite-result caching remains content-addressed and
    # live mutable training models still use the fail-closed no-identity path.
    shared_model = model
    shared_checkpoint_sha256: str | None = None
    shared_checkpoint_path: Path | None = None
    if shared_model is None:
        shared_checkpoint_path = checkpoint or (config.checkpoint_dir / "last.pt")
        if not shared_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"evaluation checkpoint not found: {shared_checkpoint_path}"
            )
        shared_checkpoint_sha256 = _sha256_file(shared_checkpoint_path)
        train_records = []
        if config.train_dir.exists():
            try:
                train_records = load_train_records(config.train_dir)
            except FileNotFoundError:
                train_records = []
        shared_model = build_model(
            config,
            train_records or load_suite_records(config.test_dir, suites[0]),
            checkpoint=shared_checkpoint_path,
        )

    board: dict[str, dict] = {}
    evaluation_deadline = (
        time.monotonic() + float(config.evaluation_wall_seconds)
        if config.evaluation_wall_seconds
        else None
    )
    remaining_records = None
    if evaluation_deadline is not None:
        def selected_record_n(suite: str) -> int:
            records = load_suite_records(config.test_dir, suite)
            limit = config.eval_limit
            if limit is None and suite == "rico_held":
                limit = config.rico_eval_limit
            offset = max(0, int(config.eval_offset))
            stop = offset + max(0, int(limit)) if limit is not None else None
            return len(records[offset:stop])

        remaining_records = [
            sum(selected_record_n(suite) for suite in suites)
        ]
    for suite in suites:
        suite_config = replace(config, suite=suite)
        metrics = evaluate(
            suite_config,
            model=shared_model,
            model_checkpoint_sha256=shared_checkpoint_sha256,
            model_checkpoint_path=shared_checkpoint_path,
            publish_agentv=False,
            cache=cache,
            evaluation_deadline=evaluation_deadline,
            evaluation_remaining_records=remaining_records,
        )
        board[suite] = {k: v for k, v in metrics.items() if k != "details"}
    from slm_training.evals.record_schema import RUN_CLASSES, SCHEMA_VERSION
    from slm_training.versioning import build_version_stamp

    scoreboard = {
        "schema_version": SCHEMA_VERSION,
        "run_class": (
            config.run_class if config.run_class in RUN_CLASSES else "scratch_matrix"
        ),
        "run_id": config.run_id,
        "checkpoint": (
            None
            if shared_checkpoint_path is None
            else str(shared_checkpoint_path)
        ),
        "checkpoint_source": (
            "preloaded_model" if shared_checkpoint_path is None else "checkpoint"
        ),
        "checkpoint_sha256": next(iter(board.values()), {}).get("checkpoint_sha256"),
        "test_dir": str(config.test_dir),
        "eval_data_manifest_sha": next(iter(board.values()), {}).get(
            "eval_data_manifest_sha"
        ),
        "code_git_sha": next(iter(board.values()), {}).get("code_git_sha"),
        "code_dirty": next(iter(board.values()), {}).get("code_dirty"),
        "suites": board,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "version_stamp": build_version_stamp(*_evaluation_version_components(config)),
    }
    # Ceiling + length-budget diagnostics ride with every board so a zero
    # scoreboard is attributable: harness/data breakage (ceiling < 1, budget
    # overflow) vs genuine model failure. Diagnostic breakage is recorded, not
    # allowed to sink the eval itself.
    diagnostics: dict[str, Any] = {}
    try:
        from slm_training.harnesses.model_build.diagnostic import ceiling_report

        diagnostics["ceiling"] = {
            suite: {key: value for key, value in report.items() if key != "failures"}
            for suite, report in ceiling_report(
                config.test_dir, suites=tuple(suites)
            ).items()
        }
    except Exception as exc:  # noqa: BLE001
        diagnostics["ceiling_error"] = str(exc)
    try:
        from slm_training.harnesses.model_build.diagnostic import length_budget_report

        ltr_cap = int(getattr(config, "grammar_ltr_max_tokens", 0) or 0)
        budget = length_budget_report(
            train_dir=config.train_dir,
            test_dir=config.test_dir,
            suites=tuple(suites),
            **({"grammar_ltr_max_tokens": ltr_cap} if ltr_cap > 0 else {}),
        )
        diagnostics["length_budget"] = {
            "ok": bool(budget.get("ok")),
            "effective_budget": budget.get("effective_budget"),
            "failures": budget.get("failures"),
        }
    except Exception as exc:  # noqa: BLE001
        diagnostics["length_budget_error"] = str(exc)
    scoreboard["diagnostics"] = diagnostics

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    from slm_training.harnesses.model_build.feature_flags import load_snapshot

    scoreboard["feature_flags"] = load_snapshot(run_dir)
    if suite_reachability is not None:
        scoreboard["suite_reachability"] = dict(suite_reachability)
    path = run_dir / "scoreboard.json"
    scoreboard["output"] = str(path)
    gate_suites = sorted(suite for suite in suites if suite in DEFAULT_SHIP_GATES)
    if gate_suites:
        from slm_training.evals.agentv import publish_model_evaluation

        scoreboard["evals"] = publish_model_evaluation(
            run_dir,
            board,
            include_missing_suites=set(suites) == set(DEFAULT_SHIP_GATES),
            suite_reachability=suite_reachability,
        )
        scoreboard["evals"]["suites_run"] = gate_suites
    else:
        scoreboard["evals"] = {"skipped": "no ship-gate policy suites evaluated"}
    if write_gates:
        gates = write_ship_gates(
            run_dir,
            board,
            suite_reachability=suite_reachability,
            evals_result=scoreboard["evals"],
        )
        scoreboard["gates"] = {
            key: gates[key]
            for key in (
                "authority",
                "pass",
                "failures",
                "evidence_volume_failures",
                "measurement_integrity_failures",
                "quality_threshold_failures",
                "runtime_failures",
                "output",
            )
        }
    path.write_text(json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8")
    _record_langsmith_evaluation(config, suites=board, scoreboard=scoreboard)
    return scoreboard
