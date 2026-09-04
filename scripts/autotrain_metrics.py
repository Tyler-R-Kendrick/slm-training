"""Reading metrics out of a run's eval payloads.

One responsibility: turning raw eval JSON into the numbers the loop reasons
about -- primary-metric leaves, per-suite metrics, paired observations, NLL
records. Comparison and adequacy live in ``autotrain_measurement``.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autotrain_io import (
    read_json,
)


def metric_from_eval(path: Path, key: str) -> float | None:
    data = read_json(path)
    if key in data and isinstance(data[key], (int, float)):
        return float(data[key])
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(key), (int, float)):
        return float(metrics[key])
    for suite_name in ("smoke", "held_out"):
        suite = data.get(suite_name)
        if isinstance(suite, dict) and isinstance(suite.get(key), (int, float)):
            return float(suite[key])
    return None


METRIC_LEAVES = (
    "latency_ms_p50",
    "parse_rate",
    "meaningful_program_rate",
    "structural_similarity",
    "binder_reference_f1",
    "eval_nll",
)


def run_metrics(
    camp_dir: Path,
    run_id: str,
    *,
    prefer_held_out: bool = False,
) -> dict[str, float | None]:
    """Load smoke (+ held_out when present) metrics for Phase A classification.

    Screening primaries use smoke leaves. Promotion primaries use
    ``held_out.*`` (policy ``held_out.structural_similarity``). When
    ``prefer_held_out`` is true and held-out eval exists, leaf keys are filled
    from held_out so climb_policy leaf lookup and tradeoff paths see the same
    suite as the dotted primary.
    """
    run_dir = camp_dir / "runs" / run_id
    smoke = run_dir / "eval_smoke.json"
    if not smoke.exists():
        smoke = run_dir / "eval.json"
    held = run_dir / "eval_held_out.json"
    out: dict[str, float | None] = {leaf: None for leaf in METRIC_LEAVES}

    if smoke.is_file():
        for leaf in METRIC_LEAVES:
            val = metric_from_eval(smoke, leaf)
            if val is not None:
                out[leaf] = val
                out[f"smoke.{leaf}"] = val

    if held.is_file():
        for leaf in METRIC_LEAVES:
            val = metric_from_eval(held, leaf)
            if val is not None:
                out[f"held_out.{leaf}"] = val
                if prefer_held_out:
                    out[leaf] = val
    scoreboard = run_dir / "scoreboard.json"
    if scoreboard.is_file():
        suites = read_json(scoreboard).get("suites")
        smoke_sb = suites.get("smoke") if isinstance(suites, dict) else None
        if isinstance(smoke_sb, dict) and isinstance(
            smoke_sb.get("eval_nll"), (int, float)
        ):
            nll = float(smoke_sb["eval_nll"])
            out["eval_nll"] = nll
            out["smoke.eval_nll"] = nll
    return out


EVAL_NLL_RECORDS_SCHEMA = "eval_nll_records/v1"

EVAL_NLL_RECORDS_NAME = "eval_nll_records.json"


def read_eval_nll_records(run_dir: Path) -> tuple[dict[str, float], str | None]:
    """``({record_id: nll}, definition_hash)`` from ``eval_nll_records.json``."""

    data = read_json(Path(run_dir) / EVAL_NLL_RECORDS_NAME)
    if data.get("schema") != EVAL_NLL_RECORDS_SCHEMA:
        return {}, None
    raw = data.get("records")
    if not isinstance(raw, dict):
        return {}, None
    records: dict[str, float] = {}
    for record_id, value in raw.items():
        number = finite_metric(value)
        if number is not None:
            records[str(record_id)] = number
    digest = data.get("definition_hash")
    return records, (str(digest) if digest else None)


def find_nested_key(payload: Any, key: str, *, depth: int = 6) -> Any:
    """First value of ``key`` in a nested JSON payload (depth-limited)."""

    if depth < 0:
        return None
    if isinstance(payload, dict):
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
        for value in payload.values():
            found = find_nested_key(value, key, depth=depth - 1)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_nested_key(value, key, depth=depth - 1)
            if found is not None:
                return found
    return None


def finite_metric(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def metric_leaf(row: Mapping[str, Any], name: str) -> float | None:
    raw = row.get(name)
    if raw is None:
        raw = row.get(f"smoke.{name}")
    return finite_metric(raw)


def rate_to_pm(value: object) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return int(max(0, min(1000, round(float(value) * 1000.0))))


def run_suite_metrics(camp_dir: Path, run_id: str) -> dict[str, float | None]:
    """Load parse / structure / latency from smoke or held_out eval JSON."""
    run_dir = camp_dir / "runs" / run_id
    out: dict[str, float | None] = {
        "latency_ms_p50": None,
        "parse_rate": None,
        "structural_similarity": None,
        "meaningful_program_rate": None,
    }
    for name in ("eval_held_out.json", "eval_smoke.json", "eval.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        data = read_json(path)
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else data
        if not isinstance(metrics, dict):
            continue
        for key in out:
            if out[key] is None and isinstance(metrics.get(key), (int, float)):
                out[key] = float(metrics[key])
        # Nested suite blocks
        for suite_key in ("held_out", "smoke"):
            suite = data.get(suite_key)
            if isinstance(suite, dict):
                for key in out:
                    if out[key] is None and isinstance(suite.get(key), (int, float)):
                        out[key] = float(suite[key])
    # Fallback to smoke helpers used by Phase A
    base = run_metrics(camp_dir, run_id)
    for key, val in base.items():
        if out.get(key) is None and val is not None:
            out[key] = val
    return out


def raw_metric_observations(
    camp_dir: Path, run_id: str
) -> tuple[dict[str, list[int]] | None, Path | None]:
    """Read per-example promotion observations; aggregates are not samples."""

    run_dir = camp_dir / "runs" / run_id
    for name in ("eval_held_out.json", "eval_smoke.json", "eval.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        details = read_json(path).get("details")
        if not isinstance(details, list):
            continue
        structural: list[int] = []
        parse: list[int] = []
        for row in details:
            if not isinstance(row, dict) or row.get("incomplete") is True:
                continue
            score = rate_to_pm(row.get("structural_similarity"))
            if score is not None:
                structural.append(score)
            parse_ok = row.get("parse_ok")
            if isinstance(parse_ok, bool):
                parse.append(1000 if parse_ok else 0)
        if structural and parse:
            return {
                "held_out_structural_similarity_pm": structural,
                "parse_rate_pm": parse,
            }, path
    return None, None


def run_has_usable_metrics(camp_dir: Path, run_id: str) -> bool:
    """True when suite metrics include parse or structure for cert export."""
    if not run_id:
        return False
    metrics = run_suite_metrics(camp_dir, run_id)
    return (
        metrics.get("parse_rate") is not None
        or metrics.get("structural_similarity") is not None
        or metrics.get("meaningful_program_rate") is not None
    )


def primary_harness_family(camp_dir: Path) -> str:
    for path in sorted((camp_dir / "artifacts" / "outcomes").glob("*.json")):
        payload = read_json(path)
        for signal in payload.get("harness_signals") or []:
            if signal.get("reproduced_on_frozen_input"):
                return str(signal.get("family") or "model_build")
    return "model_build"


def effective_primary_metric(
    *,
    role: str,
    policy_metric: str,
    requested_metric: str,
    replay_metric: str | None = None,
) -> str:
    effective = replay_metric or policy_metric
    requested_parts = requested_metric.rsplit(".", maxsplit=1)
    effective_parts = effective.rsplit(".", maxsplit=1)
    requested_scope = requested_parts[0] if len(requested_parts) == 2 else ""
    effective_scope = effective_parts[0] if len(effective_parts) == 2 else ""
    if (
        replay_metric is None
        and role == "screening"
        and requested_metric
        and requested_parts[-1] == effective_parts[-1]
        and requested_scope in {"", effective_scope}
    ):
        return requested_metric
    return effective
