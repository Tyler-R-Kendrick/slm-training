"""Schema-derived action-description catalog for semantic production tokens.

The catalog is tokenizer-agnostic: it keys descriptions by production-token
strings such as ``+Card``, ``-``, ``*Run``, and ``r=``.  Callers that want to
initialize model embeddings map their own token strings to these keys.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from slm_training.dsl.lang_core import library_schema
from slm_training.dsl.production_codec import (
    ACTION_STMT,
    CLOSE,
    MUTATION_STMT,
    QUERY_STMT,
    ROOT_STMT,
    STATE_STMT,
    _prop_order,
)
from slm_training.models.choice_tokenizer import _BUILTIN_NAMES

__all__ = [
    "ActionDescription",
    "ActionDescriptionCatalog",
    "FixtureDescriptionEncoder",
    "masked_mean_pool",
    "coverage_report",
    "compute_nearest_neighbor_metrics",
    "centroid_distance",
]

_RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
_EXPANDED_JSON_PATH = _RESOURCE_DIR / "action_descriptions_expanded.json"


_SIBLING_FAMILY: dict[str, str] = {
    "Card": "container",
    "Stack": "container",
    "Tabs": "nav",
    "Accordion": "nav",
    "Steps": "nav",
    "Button": "action",
    "Buttons": "action",
    "Input": "input",
    "Select": "input",
    "Slider": "input",
    "TextArea": "input",
    "DatePicker": "input",
    "CheckBoxGroup": "input",
    "CheckBoxItem": "input",
    "RadioGroup": "input",
    "RadioItem": "input",
    "SwitchGroup": "input",
    "SwitchItem": "input",
    "TextContent": "content",
    "MarkDownRenderer": "content",
    "Label": "content",
    "Tag": "content",
    "CodeBlock": "content",
    "Image": "content",
    "ImageBlock": "content",
    "ImageGallery": "content",
    "Separator": "content",
    "Form": "form",
    "FormControl": "form",
    "Modal": "overlay",
    "Callout": "overlay",
    "Table": "data",
    "Col": "data",
    "BarChart": "data",
    "LineChart": "data",
    "AreaChart": "data",
    "RadarChart": "data",
    "PieChart": "data",
    "RadialChart": "data",
    "ScatterChart": "data",
    "HorizontalBarChart": "data",
    "SingleStackedBarChart": "data",
}


def _type_label(prop_schema: dict[str, Any]) -> str:
    """Return a compact, deterministic type label for a property schema."""
    if not prop_schema:
        return "any"
    if "$ref" in prop_schema:
        ref = str(prop_schema["$ref"])
        if ref.startswith("#/$defs/"):
            return f"ref:{ref.rsplit('/', 1)[-1]}"
        return "ref"
    if "anyOf" in prop_schema:
        labels = sorted({_type_label(dict(o)) for o in prop_schema["anyOf"] if isinstance(o, dict)})
        return "|".join(labels) if labels else "any"
    expected = prop_schema.get("type")
    if isinstance(expected, list):
        return "|".join(sorted(str(t) for t in expected)) or "any"
    if expected == "array":
        items = dict(prop_schema.get("items") or {})
        return f"list<{_type_label(items)}>"
    if expected == "object":
        return "object"
    if expected in ("string", "number", "integer", "boolean"):
        if expected == "integer":
            return "number"
        return str(expected)
    if "enum" in prop_schema:
        return "enum"
    return "any"


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


@dataclass(frozen=True)
class ActionDescription:
    """Immutable description of one semantic production action."""

    action_key: str
    short_name: str
    signature: str
    description: str
    result_type: str | None
    argument_roles: tuple[str, ...]
    sibling_family: str | None
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionDescription":
        return cls(
            action_key=str(data["action_key"]),
            short_name=str(data["short_name"]),
            signature=str(data["signature"]),
            description=str(data["description"]),
            result_type=data.get("result_type"),
            argument_roles=tuple(data.get("argument_roles", [])),
            sibling_family=data.get("sibling_family"),
            provenance=str(data["provenance"]),
        )


@dataclass(frozen=True)
class ActionDescriptionCatalog:
    """Deterministic catalog of action descriptions built from the OpenUI schema."""

    entries: tuple[ActionDescription, ...] = field(default_factory=tuple)
    by_key: dict[str, ActionDescription] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if len(self.by_key) != len(self.entries):
            object.__setattr__(
                self,
                "by_key",
                {entry.action_key: entry for entry in self.entries},
            )

    @classmethod
    def build(cls, *, dsl: str | None = None) -> "ActionDescriptionCatalog":
        """Build a catalog from ``library_schema`` and ``_prop_order``."""
        schema = library_schema()
        definitions = dict(schema.get("$defs") or {})
        prop_order = _prop_order(dsl)
        entries: list[ActionDescription] = []

        component_names = sorted(set(prop_order) | set(definitions))
        for name in component_names:
            definition = dict(definitions.get(name) or {})
            properties = dict(definition.get("properties") or {})
            prop_names = list(prop_order.get(name) or [])

            role_parts: list[str] = []
            sig_parts: list[str] = []
            for prop_name in prop_names:
                prop_schema = dict(properties.get(prop_name) or {})
                label = _type_label(prop_schema)
                role_parts.append(f"{prop_name}:{label}")
                sig_parts.append(f"{prop_name}: {label}")

            signature = f"+{name}({', '.join(sig_parts)})" if sig_parts else f"+{name}()"
            description = _clean(definition.get("description")) or f"{name} UI component."
            if prop_names:
                description = f"{description} Args: {', '.join(role_parts)}."

            entries.append(
                ActionDescription(
                    action_key=f"+{name}",
                    short_name=name,
                    signature=signature,
                    description=description,
                    result_type="element",
                    argument_roles=tuple(role_parts),
                    sibling_family=_SIBLING_FAMILY.get(name),
                    provenance="schema",
                )
            )

        # Close token.
        entries.append(
            ActionDescription(
                action_key=CLOSE,
                short_name="close",
                signature="-",
                description="Close a component or builtin expression.",
                result_type=None,
                argument_roles=(),
                sibling_family=None,
                provenance="structural",
            )
        )

        # Builtin actions (choice-codec prefix style).
        for builtin in _BUILTIN_NAMES:
            entries.append(
                ActionDescription(
                    action_key=f"*{builtin}",
                    short_name=builtin,
                    signature=f"*{builtin}(args: any)",
                    description=f"Builtin aggregate/action {builtin}.",
                    result_type="any",
                    argument_roles=("args:any",),
                    sibling_family="builtin",
                    provenance="builtin",
                )
            )

        # Structural statement markers.
        for marker, short_name, desc in (
            (ROOT_STMT, "root_statement", "Root statement marker."),
            (STATE_STMT, "state_statement", "State declaration marker."),
            (QUERY_STMT, "query_statement", "Query statement marker."),
            (MUTATION_STMT, "mutation_statement", "Mutation statement marker."),
            (ACTION_STMT, "action_statement", "Action statement marker."),
        ):
            entries.append(
                ActionDescription(
                    action_key=marker,
                    short_name=short_name,
                    signature=marker,
                    description=desc,
                    result_type=None,
                    argument_roles=(),
                    sibling_family=None,
                    provenance="structural",
                )
            )

        return cls(entries=tuple(entries))

    def keys(self) -> tuple[str, ...]:
        return tuple(entry.action_key for entry in self.entries)

    def descriptions_for(self, source: str) -> dict[str, str]:
        """Return a mapping from action key to description string for ``source``."""
        if source == "none":
            return {}

        if source == "current_stub":
            return {entry.action_key: self._stub_for(entry) for entry in self.entries}

        if source == "schema_description":
            return {entry.action_key: entry.description for entry in self.entries}

        if source == "expanded_description":
            overrides = self._load_expanded_overrides()
            return {
                entry.action_key: overrides.get(entry.action_key, entry.description)
                for entry in self.entries
            }

        if source == "shuffled":
            base = {
                entry.action_key: entry.description for entry in self.entries
            }
            keys = sorted(base)
            values = [base[k] for k in keys]
            rng = random.Random(163)
            rng.shuffle(values)
            return dict(zip(keys, values))

        raise ValueError(f"unknown description source: {source!r}")

    def _stub_for(self, entry: ActionDescription) -> str:
        if entry.provenance == "schema":
            return f"{entry.short_name} UI component"
        if entry.provenance == "builtin":
            return f"{entry.short_name} builtin action"
        if entry.action_key == CLOSE:
            return "close token"
        return f"{entry.short_name} marker"

    @staticmethod
    def _load_expanded_overrides() -> dict[str, str]:
        if not _EXPANDED_JSON_PATH.exists():
            return {}
        try:
            data = json.loads(_EXPANDED_JSON_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if isinstance(data, dict):
            if data and all(isinstance(v, str) for v in data.values()):
                return {str(k): str(v) for k, v in data.items()}
            # Support {key: {"description": ...}} shape as well.
            return {
                str(k): str(v["description"])
                for k, v in data.items()
                if isinstance(v, dict) and "description" in v
            }
        return {}


class FixtureDescriptionEncoder:
    """Deterministic hash-based description encoder (CPU-only, no downloads)."""

    def __init__(self, d_model: int) -> None:
        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")

    def encode(self, text: str) -> Any:
        """Return a ``(d_model,)`` tensor for ``text``."""
        import torch

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = random.Random(seed)
        values = [rng.uniform(-1.0, 1.0) for _ in range(self.d_model)]
        vec = torch.tensor(values, dtype=torch.float32)
        # Normalize to unit variance so that cosine geometry is well-behaved.
        norm = vec.norm(dim=-1, keepdim=True)
        if norm.item() > 0:
            vec = vec / norm
        return vec


def masked_mean_pool(embeddings: Any, mask: Any) -> Any:
    """Mean-pool ``embeddings`` over positions where ``mask`` is True/1."""
    import torch

    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)
    while mask.dim() < embeddings.dim():
        mask = mask.unsqueeze(-1)
    masked = embeddings * mask
    denom = mask.sum(dim=-2, keepdim=True)
    out = masked.sum(dim=-2, keepdim=True)
    # Avoid divide-by-zero; return zeros for empty masks.
    out = torch.where(denom > 0, out / denom, torch.zeros_like(out))
    return out.squeeze(-2)


def coverage_report(descriptions: Mapping[str, str], catalog: ActionDescriptionCatalog) -> dict[str, Any]:
    """Return coverage fraction and missing keys relative to ``catalog``."""
    keys = set(catalog.keys())
    present = set(descriptions)
    missing = sorted(keys - present)
    return {
        "coverage_fraction": len(present) / max(1, len(keys)),
        "missing": missing,
    }


def compute_nearest_neighbor_metrics(vectors: Mapping[str, Any]) -> dict[str, Any]:
    """Return average nearest-neighbor cosine similarity and a neighbor map."""
    import torch
    import torch.nn.functional as F

    keys = sorted(vectors)
    if not keys:
        return {"mean_nearest_cosine": 0.0, "nearest_neighbor_map": {}}

    matrix = torch.stack([vectors[k] for k in keys])
    n = len(keys)
    sims = F.cosine_similarity(matrix.unsqueeze(1), matrix.unsqueeze(0), dim=-1)
    # Exclude self-similarity.
    sims.fill_diagonal_(-1.0)
    best_scores, best_indices = sims.max(dim=-1)
    mean_score = float(best_scores.mean().item())
    neighbor_map = {
        keys[i]: {"nearest": keys[int(best_indices[i].item())], "cosine": float(best_scores[i].item())}
        for i in range(n)
    }
    return {"mean_nearest_cosine": mean_score, "nearest_neighbor_map": neighbor_map}


def centroid_distance(
    vectors: Mapping[str, Any],
    set_a: set[str],
    set_b: set[str],
) -> float:
    """Euclidean distance between the mean vectors of ``set_a`` and ``set_b``."""
    import torch

    def _mean(keys: set[str]) -> Any | None:
        present = [vectors[k] for k in keys if k in vectors]
        if not present:
            return None
        return torch.stack(present).mean(dim=0)

    mean_a = _mean(set_a)
    mean_b = _mean(set_b)
    if mean_a is None or mean_b is None:
        return 0.0
    return float(torch.dist(mean_a, mean_b).item())
