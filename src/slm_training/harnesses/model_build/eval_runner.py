"""Evaluation runner for ModelPlugin implementations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import signal
import time
from datetime import datetime, timezone
from functools import lru_cache
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
from slm_training.harnesses.model_build.full_state import _git_dirty, _git_sha
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.evals.eval_cache import (
    EvalCache,
    EvalCacheMode,
    suite_result_key,
)
from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.versioning import component_version

_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")


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
            values = [float(row[key]) for row in group if isinstance(row.get(key), (int, float))]
            if values:
                summary[f"{key}_mean" if key.endswith("_size") else key] = sum(values) / len(values)
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
                    if precision is not None and recall is not None and precision + recall
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
    body = token[1:] if token.startswith(":") else token
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

    return {
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
        "semantic_role_decode_weight": float(
            value("semantic_role_decode_weight") or 0.0
        ),
        "semantic_role_schema_candidates": bool(
            value("semantic_role_schema_candidates")
        ),
        "slot_coverage_close_decode_weight": float(
            value("slot_coverage_close_decode_weight") or 0.0
        ),
        "schema_value_decode_weight": float(
            value("schema_value_decode_weight") or 0.0
        ),
        "schema_enum_close_decode_weight": float(
            value("schema_enum_close_decode_weight") or 0.0
        ),
        "schema_opaque_decode_weight": float(
            value("schema_opaque_decode_weight") or 0.0
        ),
        "schema_opaque_close_decode_weight": float(
            value("schema_opaque_close_decode_weight") or 0.0
        ),
        "schema_role_slot_decode_weight": float(
            value("schema_role_slot_decode_weight") or 0.0
        ),
        "semantic_plan_decode_weight": float(
            value("semantic_plan_decode_weight") or 0.0
        ),
        "semantic_plan_margin_decode_weight": float(
            value("semantic_plan_margin_decode_weight") or 0.0
        ),
        "semantic_plan_seed_decode_weight": float(
            value("semantic_plan_seed_decode_weight") or 0.0
        ),
        "semantic_plan_inline_decode_weight": float(
            value("semantic_plan_inline_decode_weight") or 0.0
        ),
        "semantic_plan_binding_decode_weight": float(
            value("semantic_plan_binding_decode_weight") or 0.0
        ),
        "semantic_plan_root_decode_weight": float(
            value("semantic_plan_root_decode_weight") or 0.0
        ),
        "semantic_plan_root_margin_decode_weight": float(
            value("semantic_plan_root_margin_decode_weight") or 0.0
        ),
        "visible_reference_decode_weight": float(
            value("visible_reference_decode_weight") or 0.0
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
        "allow_unconstrained_fallback": bool(value("allow_unconstrained_fallback")),
        "gen_steps": int(value("gen_steps") or 0),
        "grammar_ltr_max_tokens": int(value("grammar_ltr_max_tokens") or 0),
    }


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
        # None = recall undefined (gold has only Stacks); nothing to reject on.
        if recall is not None and recall < min_component_recall:
            return False, f"low_component_recall:{recall:.2f}", serialized
    return True, None, serialized


# Public version lock: historical scoreboards and ship thresholds remain v1.
meaningful_program_v1 = _is_meaningful_program


def _eval_data_sha(directory: Path) -> str | None:
    """Content fingerprint of an eval dataset dir (manifest or records hash)."""
    from slm_training.harnesses.model_build.full_state import data_manifest_sha

    try:
        return data_manifest_sha(directory)
    except Exception:  # noqa: BLE001 - identity stamping must never break evals
        return None


def evaluate(
    config: ModelBuildConfig,
    model=None,
    checkpoint: Path | None = None,
    *,
    publish_agentv: bool = True,
    cache: EvalCache | None = None,
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
    struct_vals: list[float] = []
    tree_edit_vals: list[float] = []
    reward_vals: list[float] = []
    recall_vals: list[float] = []
    contract_precision_vals: list[float] = []
    contract_recall_vals: list[float] = []
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
    cache_key = None
    cache_dependencies: dict[str, Any] = {}
    if cache is not None and cache.config.mode is not EvalCacheMode.OFF:
        try:
            component_versions = {
                cid: component_version(cid)
                for cid in (
                    "harness.model_build.eval",
                    "evals.meaningful_program",
                    "evals.scoring",
                )
            }
        except Exception:  # noqa: BLE001 - degrade gracefully if registry unavailable
            component_versions = {}
        cache_dependencies = {
            "checkpoint_sha256": checkpoint_sha256,
            "eval_data_manifest_sha": eval_data_manifest_sha,
            "eval_suite_manifest_sha": eval_suite_manifest_sha,
            "suite_limit": suite_limit,
            "evaluation_policy": evaluation_policy,
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
                return cached_metrics

    batch_size = 1
    generate_batch_requests = getattr(plugin, "generate_batch_requests", None)
    generate_batch = getattr(plugin, "generate_batch", None)
    generate_with_stats = getattr(plugin, "generate_with_stats", None)
    if callable(generate_batch_requests) or callable(generate_batch):
        batch_size = max(
            1,
            int(
                getattr(getattr(plugin, "config", None), "generate_batch_size", 8) or 8
            ),
        )
    if callable(generate_with_stats):
        batch_size = 1

    def _eval_schema() -> str | None:
        if not getattr(config, "schema_in_context", False):
            return None
        from slm_training.harnesses.quality import compact_schema_snippet

        budget = min(600, int(getattr(config, "design_md_budget", 1800) or 1800))
        return compact_schema_snippet(budget=budget)

    def _request_for(record: ExampleRecord) -> GenerationRequest:
        schema = _eval_schema()
        return GenerationRequest.from_record(record, schema=schema)

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
                        requests, max_len=canvas_cap
                    )
                except TypeError:
                    predictions = generate_batch_requests(requests)
            decode_stats_rows.append(stats)
            consume = getattr(plugin, "consume_generation_evidence", None)
            evidence = consume() if callable(consume) else []
            return predictions, list(evidence)
        if callable(generate_with_stats) and len(chunk) == 1:
            try:
                text, stats = generate_with_stats(
                    chunk[0].prompt, max_len=canvas_cap
                )
            except TypeError:
                text, stats = generate_with_stats(chunk[0].prompt)
            decode_stats_rows.append(stats)
            return [text], []
        prompts = [r.prompt for r in chunk]
        if callable(generate_batch):
            try:
                return generate_batch(prompts, max_len=canvas_cap), []
            except TypeError:
                try:
                    return generate_batch(prompts, golds=None), []
                except TypeError:
                    pass
        out: list[str] = []
        for prompt in prompts:
            try:
                out.append(plugin.generate(prompt, max_len=canvas_cap))
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
        except TimeoutError as exc:
            stats = getattr(exc, "decode_stats", None)
            if stats is not None:
                decode_stats_rows.append(stats)
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
        nonlocal parse_ok, syntax_parse_ok, raw_syntax_ok
        nonlocal match_error_count, reward_error_count, empty_prediction_count
        if not pred.strip():
            empty_prediction_count += 1
        evidence = dict(prediction_evidence or {})
        if len(topology_target_evidence) > len(topology_evidence):
            evidence.update(topology_target_evidence[len(topology_evidence)])
        if record.target_kind != "document":
            from slm_training.evals.task_scoreboard import score_output_targets

            target_score = score_output_targets(pred, record.output_targets)
            topology_evidence.append(evidence)
            details.append(
                {
                    "id": record.id,
                    "target_kind": record.target_kind,
                    "target_score": target_score,
                    "latency_ms": round(latency_ms, 2),
                    "prediction": pred[:500],
                    "topology_evidence": evidence or None,
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
        recall = component_type_recall(scored_pred, record.openui)
        contract_prec = _contract_precision(scored_pred, record)
        contract_rec = _contract_recall(scored_pred, record)
        reward: float | None
        try:
            reward = _reward_for_prediction(scored_pred, record)
        except Exception:  # noqa: BLE001 — reward harness failure, not model quality
            reward_error_count += 1
            reward = None
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
        for defined_values, value in (
            (fidelity_vals, fid),
            (fidelity_norm_vals, fid_norm),
            (validity_vals, ph_valid),
            (exact_vals, exact),
            (struct_vals, struct),
            (tree_edit_vals, tree_edit),
            (recall_vals, recall),
            (contract_precision_vals, contract_prec),
            (contract_recall_vals, contract_rec),
            (reward_vals, reward),
        ):
            if value is not None:
                defined_values.append(float(value))
        if gold_dscore is not None:
            gold_design_scores.append(gold_dscore)
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
                "exact_match": exact,
                "structural_similarity": struct,
                "tree_edit_similarity": tree_edit,
                "component_type_recall": recall,
                "reward_score": reward,
                "gold_design_lint_score": gold_dscore,
                "design_lint_score": gold_dscore,
                "latency_ms": round(latency_ms, 2),
                # Full text + digest make every new metric report replayable.
                "prediction": pred,
                "prediction_sha256": hashlib.sha256(pred.encode("utf-8")).hexdigest(),
                "generation_request": _effective_request_for(record).to_dict(),
                "source_record_sha256": hashlib.sha256(
                    json.dumps(
                        record.to_dict(), sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "serialized": serialized,
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
                "target_kind": record.target_kind,
                "target_category": record.target_category,
                "accepted_outputs": [
                    target.__dict__ for target in record.accepted_outputs
                ],
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
            )
        )
    else:
        fallback_count = None

    from slm_training.evals.record_schema import RUN_CLASSES, SCHEMA_VERSION
    from slm_training.lineage.promotion import wilson_lower_bound

    def _wilson_ci95(successes: int, total: int) -> list[float] | None:
        """95% Wilson interval — makes tiny-n quantization visible (n=3 → ±0.5)."""
        if total <= 0:
            return None
        lower = wilson_lower_bound(successes, total)
        upper = 1.0 - wilson_lower_bound(total - successes, total)
        return [round(lower, 4), round(upper, 4)]

    run_class = config.run_class if config.run_class in RUN_CLASSES else "scratch_matrix"
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_class": run_class,
        "suite": config.suite,
        "n": n,
        "document_n": document_n,
        "fragment_n": n - document_n,
        "eval_limit": suite_limit,
        "diagnostic_subset": suite_limit is not None,
        # Persist the effective decode policy beside every scoreboard.  This
        # is essential for comparing historical runs: checkpoint defaults and
        # CLI diagnostic overrides can materially change quality and timeout
        # metrics even when the checkpoint hash is identical.
        "evaluation_policy": _effective_evaluation_policy(config, plugin),
        # Rates are None (JSON null) when no document records were measured —
        # "not measured" must never render as a fabricated 0.0.
        "parse_rate": (syntax_parse_ok / document_n) if document_n else None,
        "meaningful_program_rate": (parse_ok / document_n) if document_n else None,
        "syntax_parse_rate": (
            (syntax_parse_ok / document_n) if document_n else None
        ),
        "raw_syntax_validity": (raw_syntax_ok / document_n) if document_n else None,
        "parse_rate_ci95": _wilson_ci95(syntax_parse_ok, document_n),
        "meaningful_program_rate_ci95": _wilson_ci95(parse_ok, document_n),
        "contract_precision": _mean_or_none(contract_precision_vals),
        "contract_recall": _mean_or_none(contract_recall_vals),
        # Not computed by any current decode path; None (not a fake 0.0) until
        # a plugin actually measures them.
        "residual_mask_rate": None,
        "oov_rate": None,
        "fallback_count": fallback_count,
        "placeholder_fidelity": _mean_or_none(fidelity_vals),
        "placeholder_fidelity_normalized": _mean_or_none(fidelity_norm_vals),
        "placeholder_validity": _mean_or_none(validity_vals),
        "exact_match": _mean_or_none(exact_vals),
        "structural_similarity": _mean_or_none(struct_vals),
        "tree_edit_similarity": _mean_or_none(tree_edit_vals),
        "component_type_recall": _mean_or_none(recall_vals),
        "reward_score": _mean_or_none(reward_vals),
        # How many document records actually defined each mean above — the
        # denominator disclosure that separates "measured 0" from "unmeasured".
        "metric_defined_n": {
            "contract_precision": len(contract_precision_vals),
            "contract_recall": len(contract_recall_vals),
            "placeholder_fidelity": len(fidelity_vals),
            "placeholder_fidelity_normalized": len(fidelity_norm_vals),
            "placeholder_validity": len(validity_vals),
            "exact_match": len(exact_vals),
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
        "latency_ms_p50": round(p50, 2) if p50 is not None else None,
        "latency_ms_p95": round(p95, 2) if p95 is not None else None,
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
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "failure_breakdown": failure_breakdown,
        "decode_timeout_count": decode_timeout_count,
        "decode_canvas_cap": canvas_cap,
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
    metrics.update(
        {
            "meaningful_program_v1_rate": metrics["meaningful_program_rate"],
            "binding_aware_meaningful_v2_rate_strict": meaning_v2["strict_rate"],
            "binding_aware_meaningful_v2_rate_coverage_conditioned": meaning_v2[
                "coverage_conditioned_rate"
            ],
            "binding_aware_meaningful_v2_coverage": meaning_v2["coverage"],
            "meaningful_metric_primary": "meaningful_program_v1",
            "meaningful_metric_versions": {
                "meaningful_program_v1": "1.0.0",
                "binding_aware_meaningful_v2": meaning_v2,
            },
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
    scope_contract_metrics = _aggregate_scope_contract_metrics(topology_evidence)
    if scope_contract_metrics is not None:
        metrics["scope_contract_metrics"] = scope_contract_metrics
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
    if decode_stats_rows:
        from slm_training.models.decode_stats import aggregate_stats
        metrics["decode_stats"] = aggregate_stats(decode_stats_rows)
        retries = sum(int(getattr(row, "unconstrained_retries", 0)) for row in decode_stats_rows)
        metrics["constrained_fallback_rate"] = retries / len(decode_stats_rows)

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

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    suite_path = run_dir / f"eval_{config.suite}.json"
    from slm_training.versioning import build_version_stamp

    version_components = [
        "harness.model_build.eval",
        "evals.meaningful_program",
        "evals.scoring",
    ]
    if config.model_name == "twotower":
        version_components.append("model.twotower")
    metrics["version_stamp"] = build_version_stamp(*version_components)
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
    if cache is not None and cache_key is not None and cache.config.mode in (
        EvalCacheMode.READ_WRITE,
        EvalCacheMode.REFRESH,
    ):
        try:
            cache.put(cache_key, metrics, dependencies=cache_dependencies)
        except Exception:  # noqa: BLE001 - cache write must never break eval
            pass

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
    cache: EvalCache | None = None,
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
            cache=cache,
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
            if model is not None
            else str(checkpoint or (config.checkpoint_dir / "last.pt"))
        ),
        "checkpoint_source": "preloaded_model" if model is not None else "checkpoint",
        "checkpoint_sha256": next(iter(board.values()), {}).get("checkpoint_sha256"),
        "test_dir": str(config.test_dir),
        "eval_data_manifest_sha": next(iter(board.values()), {}).get(
            "eval_data_manifest_sha"
        ),
        "code_git_sha": next(iter(board.values()), {}).get("code_git_sha"),
        "code_dirty": next(iter(board.values()), {}).get("code_dirty"),
        "suites": board,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "version_stamp": build_version_stamp(
            *(
                [
                    "harness.model_build.eval",
                    "evals.meaningful_program",
                    "evals.scoring",
                ]
                + (["model.twotower"] if config.model_name == "twotower" else [])
            )
        ),
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
    path = run_dir / "scoreboard.json"
    scoreboard["output"] = str(path)
    if write_gates:
        gates = write_ship_gates(run_dir, board)
        scoreboard["gates"] = {k: gates[k] for k in ("pass", "failures", "output")}
    gate_suites = sorted(suite for suite in suites if suite in DEFAULT_SHIP_GATES)
    if gate_suites:
        from slm_training.evals.agentv import publish_model_evaluation

        scoreboard["agentv"] = publish_model_evaluation(
            run_dir,
            board,
            include_missing_suites=set(suites) == set(DEFAULT_SHIP_GATES),
        )
        scoreboard["agentv"]["suites_run"] = gate_suites
    else:
        scoreboard["agentv"] = {"skipped": "no ship-gate policy suites evaluated"}
    path.write_text(json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8")
    return scoreboard
