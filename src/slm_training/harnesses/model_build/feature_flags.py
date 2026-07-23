"""OpenFeature-backed, persisted experiment-lever snapshots.

The model-build config is the one canonical lever surface.  Every field is a
feature flag unless it identifies a run, a filesystem location, or output
bookkeeping.  This keeps new behaviour knobs visible without a parallel UI
registry that can drift from the executable configuration.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, get_origin, get_type_hints

from openfeature import api
from openfeature.provider.in_memory_provider import InMemoryFlag, InMemoryProvider

from slm_training.harnesses.model_build.config import ModelBuildConfig

SNAPSHOT_SCHEMA = "experiment_feature_flags/v1"
_LOCK = threading.Lock()

# These describe execution identity or artifact plumbing, not a model/training/
# decode behaviour. All other config fields intentionally become flags by
# default; a future exceptional field must be added here with a reason.
_NON_LEVER_FIELDS = frozenset(
    {
        "train_dir",
        "test_dir",
        "suite",
        "run_class",
        "run_root",
        "run_id",
        "runtime_override_fields",
        "max_wall_minutes",
        "device",
        "local_files_only",
        "resume_from",
        "initialize_from",
        "replay_train_dir",
        "mixture_manifest",
        "targeted_margin_manifest",
        "action_alias_manifest",
        "checkpoint_bucket",
        "full_state_checkpoint",
        "register_promoted",
        "campaign_manifest",
        "campaign_result",
        "campaign_store_root",
        "campaign_artifact_root",
        "emit_record_nll",
        "telemetry",
    }
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _type_name(annotation: Any, default: Any) -> str:
    origin = get_origin(annotation)
    if origin is Literal:
        return "string"
    if origin in (tuple, list, dict):
        return "object"
    if annotation is bool or isinstance(default, bool):
        return "boolean"
    if annotation in (int, float) or (
        isinstance(default, (int, float)) and not isinstance(default, bool)
    ):
        return "number"
    return "string"


@dataclass(frozen=True)
class FeatureFlag:
    key: str
    config_field: str
    type: Literal["boolean", "number", "string", "object"]
    default: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "type": self.type,
            "default": _json_value(self.default),
        }


def feature_flags() -> tuple[FeatureFlag, ...]:
    """Return the generated catalog in stable field-name order."""
    hints = get_type_hints(ModelBuildConfig)
    rows: list[FeatureFlag] = []
    for item in fields(ModelBuildConfig):
        if item.name in _NON_LEVER_FIELDS:
            continue
        if item.default is MISSING and item.default_factory is MISSING:
            raise ValueError(f"unclassified config field {item.name!r}")
        default = (
            item.default
            if item.default is not MISSING
            else item.default_factory()  # type: ignore[misc]
        )
        rows.append(
            FeatureFlag(
                key=f"slm.{item.name}",
                config_field=item.name,
                type=_type_name(hints.get(item.name), default),
                default=default,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.key))


def flag_key_for_config_field(field: str) -> str | None:
    """Return the canonical OpenFeature key for an internal config field."""
    return next((row.key for row in feature_flags() if row.config_field == field), None)


def registry_revision() -> str:
    payload = [row.to_dict() for row in feature_flags()]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def catalog() -> dict[str, Any]:
    rows = [flag.to_dict() for flag in feature_flags()]
    return {
        "schema": SNAPSHOT_SCHEMA,
        "revision": registry_revision(),
        "flags": rows,
        "count": len(rows),
        "boolean_count": sum(row["type"] == "boolean" for row in rows),
    }


def _coerce(value: Any, default: Any) -> Any:
    """Restore JSON/provider values to the config's declared runtime shape."""
    if isinstance(default, tuple):
        return tuple(value) if isinstance(value, list) else default
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _domain(config: ModelBuildConfig, phase: str, values: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"slm.{config.run_id}.{phase}.{digest}"


def resolve(config: ModelBuildConfig, *, phase: str) -> tuple[ModelBuildConfig, dict[str, Any]]:
    """Evaluate the complete config through OpenFeature and return its snapshot."""
    catalog_rows = feature_flags()
    raw = {row.key: _json_value(getattr(config, row.config_field)) for row in catalog_rows}
    domain = _domain(config, phase, raw)
    provider = InMemoryProvider(
        {
            # Object values let one standards-compliant OpenFeature evaluation
            # preserve bools, numbers, strings, tuples, and optional values.
            key: InMemoryFlag("on", {"on": {"value": value}})
            for key, value in raw.items()
        }
    )
    # OpenFeature providers are globally registered, but domains isolate every
    # run/phase snapshot. Lock registration for concurrent matrix workers.
    with _LOCK:
        api.set_provider_and_wait(provider, domain)
    client = api.get_client(domain)
    resolved: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for lever in catalog_rows:
        default = raw[lever.key]
        envelope = client.get_object_value(lever.key, {"value": default})
        value = envelope.get("value", default) if isinstance(envelope, dict) else default
        coerced = _coerce(value, lever.default)
        resolved[lever.config_field] = coerced
        rows.append(
            {
                **lever.to_dict(),
                "value": _json_value(coerced),
                "state": "enabled" if lever.type == "boolean" and coerced else "disabled"
                if lever.type == "boolean"
                else "overridden"
                if _json_value(coerced) != _json_value(lever.default)
                else "default",
                "recorded": True,
            }
        )
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "registry_revision": registry_revision(),
        "phase": phase,
        "provider": "openfeature.in_memory",
        "domain": domain,
        "flags": rows,
    }
    return replace(config, **resolved), snapshot


def save_snapshot(run_dir: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Atomically merge one phase snapshot into the run's durable flag file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "feature_flags.json"
    with _LOCK:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {"schema": SNAPSHOT_SCHEMA, "snapshots": {}}
        snapshots = payload.setdefault("snapshots", {})
        snapshots[str(snapshot["phase"])] = snapshot
        payload["registry_revision"] = snapshot["registry_revision"]
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=run_dir, delete=False
        ) as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary = Path(handle.name)
        temporary.replace(path)
    return payload


def load_snapshot(run_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((run_dir / "feature_flags.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None
