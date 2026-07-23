"""Canonical experiment-record schema and read-side normalizer.

``docs/design`` accumulated a dozen-plus metric dialects over time: canonical
long keys, a ``syntax_parse_rate`` variant, short perf keys (``parse`` /
``meaningful`` / ``fidelity`` / ``structure``), ``honest_evaluation.suites``
nesting, and many single-suite blocks (``eval`` / ``observed`` / ``metrics`` /
top-level). Committed records are immutable evidence, so readers normalize
through this module; writers emit only the canonical shape going forward
(``canonical_envelope``).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

SCHEMA_VERSION = 1

# fixture_demo: committed-fixture wiring runs; scratch_matrix: local/CI matrix
# budgets; ship_eval: full honest ship-gate evaluations.
RUN_CLASSES = ("fixture_demo", "scratch_matrix", "ship_eval")

KNOWN_SUITES = ("smoke", "held_out", "adversarial", "ood", "rico_held")

CANONICAL_SUITE_METRIC_KEYS = (
    "n",
    "parse_rate",
    "meaningful_program_rate",
    "syntax_parse_rate",
    "raw_syntax_validity",
    "structural_similarity",
    "component_type_recall",
    "placeholder_fidelity",
    "placeholder_validity",
    "contract_precision",
    "contract_recall",
    "reward_score",
    "exact_match",
    "tree_edit_similarity",
    "latency_ms_p50",
    "latency_ms_p95",
    "fallback_count",
)

SHORT_KEY_ALIASES = {
    "parse": "parse_rate",
    "meaningful": "meaningful_program_rate",
    "fidelity": "placeholder_fidelity",
    "structure": "structural_similarity",
    "structural": "structural_similarity",
    "component_recall": "component_type_recall",
    "reward": "reward_score",
}

# A dict is treated as a metric block when it carries at least two of these.
_METRIC_HINTS = frozenset(
    key for key in CANONICAL_SUITE_METRIC_KEYS if key not in {"n", "fallback_count"}
) | frozenset(SHORT_KEY_ALIASES)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_MAX_DEPTH = 3


def _json_pointer(path: tuple[str, ...]) -> str:
    return "".join(
        f"/{part.replace('~', '~0').replace('/', '~1')}" for part in path
    )


def normalize_suite_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Re-key one suite's metrics onto the canonical vocabulary.

    Unknown keys pass through untouched (telemetry stays visible). The
    legacy ``parse_rate`` → ``meaningful_program_rate`` substitution applies
    only to pre-split boards — records that carry neither
    ``meaningful_program_rate`` nor ``syntax_parse_rate`` — and is tagged via
    ``meaningful_source`` so consumers can badge it instead of presenting
    decoder-guaranteed syntax as meaningful quality.
    """
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        out.setdefault(SHORT_KEY_ALIASES.get(key, key), value)
    explicit_v1 = out.get("meaningful_program_v1_rate")
    if explicit_v1 is None:
        explicit_v1 = out.get("meaningful_v1")
    if (
        out.get("meaningful_program_rate") is None
        and isinstance(explicit_v1, (int, float))
    ):
        out["meaningful_program_rate"] = explicit_v1
    if (
        out.get("meaningful_program_rate") is None
        and out.get("syntax_parse_rate") is None
        and isinstance(out.get("parse_rate"), (int, float))
    ):
        out["meaningful_program_rate"] = out["parse_rate"]
        out["meaningful_source"] = "parse_rate_legacy"
    return out


def _is_metric_block(value: Any, min_hints: int = 2) -> bool:
    if not isinstance(value, dict):
        return False
    hints = sum(1 for key in value if key in _METRIC_HINTS)
    return hints >= min_hints


def _board_members(board: Mapping[str, Any]) -> dict[str, dict]:
    """Suite members of a candidate board.

    A known suite name vouches for a single-metric row; unknown keys need two
    metric hints so arbitrary telemetry dicts don't masquerade as suites.
    """
    return {
        str(key): member
        for key, member in board.items()
        if _is_metric_block(member, 1 if str(key) in KNOWN_SUITES else 2)
    }


def _is_suites_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return bool(_board_members(value))


def _walk(payload: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict]]:
    found: list[tuple[tuple[str, ...], dict]] = []
    if not isinstance(payload, dict) or len(path) > _MAX_DEPTH:
        return found
    found.append((path, payload))
    for key, value in payload.items():
        found.extend(_walk(value, (*path, str(key))))
    return found


def synthesize_run_id(payload: Mapping[str, Any], *, stem: str) -> str:
    for candidate in (
        payload.get("run_id"),
        (payload.get("train_result") or {}).get("run_id")
        if isinstance(payload.get("train_result"), dict)
        else None,
        (payload.get("train") or {}).get("run_id")
        if isinstance(payload.get("train"), dict)
        else None,
        (payload.get("evaluation") or {}).get("run_id")
        if isinstance(payload.get("evaluation"), dict)
        else None,
    ):
        if isinstance(candidate, str) and _RUN_ID_RE.fullmatch(candidate):
            return candidate
    fallback = re.sub(r"[^A-Za-z0-9._-]", "-", stem).strip("-") or "record"
    return fallback[:128]


def normalize_experiment_record(
    payload: Any, *, stem: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize one committed record to (record, None) or (None, reason).

    ``record["suites"]`` maps suite name → canonical-key metrics.
    ``record["board_context"]`` is the dict that contained the chosen board so
    callers can read sibling evidence (gates, agentv). Rejections are typed:
    ``unreadable`` (not a JSON object) or ``no_metric_blocks`` (a design doc /
    benchmark / report without an eval scoreboard — not an experiment record).
    """
    if not isinstance(payload, dict):
        return None, "unreadable"
    nodes = _walk(payload)

    # Preferred shape: an explicit suites dict (deepest evidence wins ties by
    # suite count; document order breaks remaining ties deterministically).
    boards = [
        (path, node["suites"])
        for path, node in nodes
        if _is_suites_dict(node.get("suites"))
    ]
    if boards:
        path, board = max(
            enumerate(boards),
            key=lambda item: (len(_board_members(item[1][1])), -item[0]),
        )[1]
        context = next(node for node_path, node in nodes if node_path == path)
        suites = {
            name: normalize_suite_metrics(member)
            for name, member in _board_members(board).items()
        }
        return (
            {
                "run_id": synthesize_run_id(payload, stem=stem),
                "suites": suites,
                "board_context": context,
                "source_schema": "suites@" + ("/".join(path) or "<root>"),
                "source_pointer": _json_pointer((*path, "suites")),
                "boards_found": len(boards),
            },
            None,
        )

    # Records that declare the canonical writer contract must not fall back to
    # mining arbitrary nested metric blocks. The first v1 records, however,
    # persisted their single-suite evidence explicitly in ``result`` before
    # canonical writers introduced root ``suites``. Keep that named historical
    # contract visible; a malformed canonical record is still safer to reject
    # than to publish an arbitrary diagnostic replay as its primary suite.
    legacy_result = payload.get("result")
    if payload.get("schema_version") == SCHEMA_VERSION and payload.get(
        "run_class"
    ) in RUN_CLASSES and not _is_metric_block(legacy_result):
        return None, "canonical_missing_suites"

    # Single-suite blocks (eval / observed / metrics / top-level …): pick the
    # richest block, shallowest first on ties.
    blocks = [
        (path, node)
        for path, node in nodes
        if _is_metric_block(node) and "suites" not in node
    ]
    if not blocks:
        return None, "no_metric_blocks"
    path, block = max(
        blocks,
        key=lambda item: (
            sum(1 for key in item[1] if key in _METRIC_HINTS),
            -len(item[0]),
        ),
    )
    suite = block.get("suite")
    if not isinstance(suite, str):
        suite = next((part for part in reversed(path) if part in KNOWN_SUITES), None)
    if not isinstance(suite, str):
        top_suite = payload.get("suite")
        suite = top_suite if isinstance(top_suite, str) else "unspecified"
    metrics = normalize_suite_metrics(block)
    if metrics.get("n") is None and isinstance(payload.get("n"), (int, float)):
        metrics["n"] = payload["n"]
    parent_path = path[:-1] if path else ()
    context = next(node for node_path, node in nodes if node_path == parent_path)
    return (
        {
            "run_id": synthesize_run_id(payload, stem=stem),
            "suites": {suite: metrics},
            "board_context": context,
            "source_schema": "block@" + ("/".join(path) or "<root>"),
            "source_pointer": _json_pointer(path),
            "boards_found": 0,
        },
        None,
    )


def canonical_envelope(
    *,
    run_id: str,
    suites: Mapping[str, Mapping[str, Any]],
    run_class: str = "scratch_matrix",
    **extra: Any,
) -> dict[str, Any]:
    """Canonical writer shape for new docs/design experiment records."""
    if run_class not in RUN_CLASSES:
        raise ValueError(f"run_class must be one of {RUN_CLASSES}, got {run_class!r}")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_class": run_class,
        "suites": {name: dict(metrics) for name, metrics in suites.items()},
        **extra,
    }
