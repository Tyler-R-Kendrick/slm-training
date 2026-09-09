"""Selection, persistence and completeness contracts for the canonical evaluator.

These are operational evidence guards, not new quality metrics or ship gates.
Legacy evidence remains readable but cannot acquire missing resume identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any

PARTIAL_SCOREBOARD_SCHEMA = "EvalPartialScoreboardV1"


def partial_policy_digest(policy: dict[str, Any]) -> str:
    """Invocation wall controls scheduling, not reuse of committed row evidence.

    The explicit per-record timeout and every decode/scoring field stay bound.
    Timed-out rows retain their recorded incomplete outcome; this does not retry
    them until lucky or turn an interruption into a quality observation.
    """
    semantic = {key: value for key, value in policy.items()
                if key != "evaluation_wall_seconds"}
    payload = {"schema": "partial_evaluation_policy/v1", "policy": semantic}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def exclusive_suite_writer(function):
    """Local Linux writer exclusion; the controller separately owns fencing."""
    @wraps(function)
    def wrapped(config, *args, **kwargs):
        import fcntl

        if not config.suite or Path(config.suite).name != config.suite:
            raise ValueError("invalid evaluation suite namespace")
        config.run_dir.mkdir(parents=True, exist_ok=True)
        with (config.run_dir / f".eval_{config.suite}.lock").open("a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("evaluation suite writer already active") from exc
            return function(config, *args, **kwargs)
    return wrapped


def validate_selection(records, *, limit: int | None = None) -> None:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("eval_limit must be a positive integer")
    if not records:
        raise ValueError("evaluation selection is empty")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation selection contains duplicate record IDs")


def partial_scoreboard_path(run_dir: Path, suite: str) -> Path:
    return Path(run_dir) / f"eval_{suite}.partial.json"


def load_partial_scoreboard(run_dir: Path | None, suite: str) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = partial_scoreboard_path(run_dir, suite)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != PARTIAL_SCOREBOARD_SCHEMA
            or not isinstance(payload.get("records"), dict)):
        return None
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Unique staging file, fsync then replace; no shared fixed .tmp name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def resume_rejection(prior: dict[str, Any], identity: dict[str, Any]) -> str | None:
    for key in ("checkpoint_sha256", "eval_suite_manifest_sha"):
        if identity.get(key) is None or prior.get(key) is None:
            return f"{key}_unavailable"
    for key, value in identity.items():
        if prior.get(key) != value:
            return f"{key}_mismatch"
    rows = prior.get("records", {})
    if not isinstance(rows, dict) or set(rows) - set(identity["record_ids"]):
        return "stored_record_ids_mismatch"
    for record_id, row in rows.items():
        if not valid_stored_row(record_id, row):
            return "stored_record_invalid"
    return None


def valid_stored_row(record_id: str, row: Any) -> bool:
    if not isinstance(row, dict) or row.get("id") != record_id:
        return False
    if not isinstance(row.get("prediction"), str) or type(row.get("timed_out")) is not bool:
        return False
    for key in ("decode_ms", "amortized_ms"):
        value = row.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return False
    return True


def preserve_rejected_partial(run_dir: Path, source: Path, reason: str | None) -> str | None:
    if reason is None or not source.is_file():
        return None
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    destination = run_dir / "rejected_partials" / f"{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("xb") as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
    return str(destination.relative_to(run_dir))


def selection_identity(records) -> str:
    payload = [record.to_dict() for record in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def decode_seed(seed: int, suite: str, records) -> int:
    payload = json.dumps([seed, suite, [record.id for record in records]])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def evaluator_identity(version_stamp: dict[str, Any]) -> str:
    # Versions bind governed metric dependencies; bytes additionally detect an
    # unpublished local scorer edit before the component registry is bumped.
    root = Path(__file__).parents[2]
    paths = (Path(__file__), Path(__file__).with_name("eval_runner.py"),
             root / "evals/meaningful_program.py", root / "data/verify/stack.py",
             root / "data/quality.py", root / "evals/measurement_identity.py")
    payload = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in paths}
    payload["component_versions"] = version_stamp.get("components", {})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def suite_result_cacheable(metrics: dict[str, Any]) -> bool:
    if not isinstance(metrics, dict):
        return False
    counts = ("n", "document_n", "completed_document_n", "incomplete_document_n",
              "decode_timeout_count")
    if any(type(metrics.get(key)) is not int or metrics[key] < 0 for key in counts):
        return False
    if not _cache_numbers_valid(metrics):
        return False
    selected = metrics.get("selected_record_ids")
    if (not isinstance(selected, list) or len(selected) != metrics["n"]
            or any(not isinstance(value, str) or not value for value in selected)
            or len(set(selected)) != len(selected)):
        return False
    fragments = metrics.get("fragment_n", 0)
    pending = metrics.get("pending_document_n", 0)
    return (type(fragments) is int and fragments >= 0
            and type(pending) is int and pending == 0
            and metrics["document_n"] > 0
            and metrics["document_n"] + fragments == metrics["n"]
            and metrics["completed_document_n"] == metrics["document_n"]
            and metrics["incomplete_document_n"] == 0
            and metrics["decode_timeout_count"] == 0
            and metrics.get("measurement_complete") is True)


def _cache_numbers_valid(metrics: dict[str, Any]) -> bool:
    """Validate measurements, not gate thresholds; null optional metrics stay null."""
    try:
        json.dumps(metrics, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return all(isinstance(key, str) and _cache_number_valid(key, value, metrics)
               for key, value in metrics.items())


def _cache_number_valid(key: str, value: Any, metrics: dict[str, Any]) -> bool:
    bounded = {
        "raw_syntax_validity", "contract_precision", "contract_recall",
        "binder_reference_f1", "placeholder_fidelity", "placeholder_fidelity_normalized",
        "placeholder_validity", "exact_match", "structural_similarity",
        "tree_edit_similarity", "component_type_recall", "reward_score",
        "ast_node_f1", "ast_edge_f1", "binding_aware_meaningful_v2_coverage",
        "topology_quality_score", "topology_structure_score", "topology_trace_score",
        "topology_efficiency_score", "topology_composite",
    }
    if key == "metric_defined_n":
        return _defined_counts_valid(value, metrics)
    if key.endswith(("_n", "_count")) or key == "certificates_compared":
        return type(value) is int and value >= 0
    if key in bounded or key.endswith("_rate") or "_rate_" in key:
        return _cache_rate_valid(key, value)
    if "_ms" in key or key == "decode_records_per_second":
        return value is None or (type(value) in (int, float) and value >= 0)
    return True


def _cache_rate_valid(key: str, value: Any) -> bool:
    if value is None:
        return key not in {"parse_rate", "syntax_parse_rate", "meaningful_program_rate", "raw_syntax_validity"}
    if key.endswith("_ci95"):
        return (isinstance(value, list) and len(value) == 2
                and all(type(bound) in (int, float) for bound in value)
                and 0 <= value[0] <= value[1] <= 1)
    # Existing retries-per-batch telemetry is not a binomial rate.
    upper = math.inf if key == "constrained_fallback_rate" else 1.0
    return type(value) in (int, float) and 0 <= value <= upper


def _defined_counts_valid(counts: Any, metrics: dict[str, Any]) -> bool:
    if not isinstance(counts, dict):
        return False
    return all(
        type(count) is int and 0 <= count <= metrics.get("document_n", 0)
        and ((metrics.get(key) is not None) == (count > 0))
        for key, count in counts.items()
    )


def measurement_states(metrics: dict[str, Any], *, pending_n: int) -> dict[str, Any]:
    """Operational exhaustion and usable quality are deliberately distinct."""
    operational = metrics["n"] > 0 and pending_n == 0
    decoded = operational and metrics["decode_timeout_count"] == 0
    return {
        "operational_complete": operational,
        "measurement_complete": decoded,
        "decoded_probe_complete": decoded,
        "confirmation_complete": False,
        "ship_eligible": False,
        "publication_complete": False,
    }


def publish_suite_metrics(run_dir: Path, suite: str, metrics: dict[str, Any]) -> None:
    from slm_training.evals.agentv import publish_model_evaluation
    from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES

    if not metrics["measurement_complete"]:
        metrics["agentv"] = {"skipped": "measurement incomplete"}
    elif suite in DEFAULT_SHIP_GATES:
        metrics["agentv"] = publish_model_evaluation(
            run_dir, {suite: metrics}, include_missing_suites=False,
        )
        metrics["agentv"]["suites_run"] = [suite]
        metrics["publication_complete"] = True
    else:
        metrics["agentv"] = {"skipped": f"suite {suite!r} is not in the ship-gate policy"}


def replay_cached_suite(config, metrics, *, record_n, publish_agentv=False,
                        evaluation_remaining_records=None):
    """Reuse measurements, never another run's external publication receipt."""
    run_dir = config.run_dir
    suite_path = run_dir / f"eval_{config.suite}.json"
    replay = {key: value for key, value in metrics.items() if key != "agentv"}
    replay.update(output=str(suite_path), cache_replay=True, publication_complete=False)
    # A failed SDK call must retain measured evidence with publication still false.
    write_json_atomic(suite_path, replay)
    if config.suite == "smoke":
        write_json_atomic(run_dir / "eval.json", replay)
    if publish_agentv:
        publish_suite_metrics(run_dir, config.suite, replay)
        write_json_atomic(suite_path, replay)
    if config.suite == "smoke":
        write_json_atomic(run_dir / "eval.json", replay)
    if evaluation_remaining_records is not None:
        evaluation_remaining_records[0] = max(0, evaluation_remaining_records[0] - record_n)
    (run_dir / "decode_progress.json").unlink(missing_ok=True)
    return replay
