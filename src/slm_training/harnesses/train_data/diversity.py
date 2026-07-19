"""Deterministic diversity fingerprints for synthetic training corpora.

Computes multi-resolution fingerprints for each record so that diversity
experiments can distinguish unique roots, sketches, topologies, type/action
multisets, prompt templates, and source lineages without re-running the parser.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from slm_training.data.leakage import fingerprint_openui_structure
from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.grammar.backends.ast_utils import component_multiset
from slm_training.dsl.parser import validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord

SCHEMA_VERSION = "diversity_fingerprints/v1"


@dataclass(frozen=True)
class DiversityFingerprints:
    """Multiple-resolution fingerprints for one synthetic record."""

    schema_version: str
    record_id: str
    canonical_root_id: str
    binding_aware_sketch: str
    topology_sketch: str
    type_action_multiset: str
    prompt_intent_fingerprint: str
    source_lineage_id: str
    exact_structure_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "canonical_root_id": self.canonical_root_id,
            "binding_aware_sketch": self.binding_aware_sketch,
            "topology_sketch": self.topology_sketch,
            "type_action_multiset": self.type_action_multiset,
            "prompt_intent_fingerprint": self.prompt_intent_fingerprint,
            "source_lineage_id": self.source_lineage_id,
            "exact_structure_fingerprint": self.exact_structure_fingerprint,
        }


def _hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_prompt_intent(prompt: str) -> str:
    """Strip surface slot namespaces while keeping structural intent words."""
    # Remove slot names but keep the sentence skeleton.
    text = re.sub(r":\S+", "", prompt)
    # Keep only alphabetic tokens, lowercase, sorted for intent fingerprinting.
    tokens = sorted({t.lower() for t in re.findall(r"[a-zA-Z]+", text) if len(t) > 2})
    return " ".join(tokens)


def _source_lineage_id(record: ExampleRecord) -> str:
    parent = record.meta.get("root_parent_id") or record.meta.get("parent_id")
    lineage = record.meta.get("transformation_lineage") or []
    synth = record.meta.get("synth") or record.source
    return _hash(
        {
            "parent_id": parent or record.id,
            "lineage": lineage,
            "synth": synth,
            "source": record.source,
        }
    )


def _topology_sketch(node: Any) -> dict[str, Any]:
    """Tree shape + list cardinalities without lexical leaves or binder names."""
    if node is None:
        return None  # type: ignore[return-value]
    if isinstance(node, dict):
        type_name = node.get("typeName")
        props: dict[str, Any] = {}
        children: list[Any] = []
        raw_props = node.get("props") or {}
        for key, value in raw_props.items():
            if key == "children" and isinstance(value, list):
                children = [_topology_sketch(c) for c in value]
            elif key in {"type", "typeName", "partial", "hasDynamicProps"}:
                continue
            elif isinstance(value, str) and not value.startswith(":"):
                # Non-placeholder string values are non-semantic surface literals.
                continue
            else:
                props[key] = _topology_sketch(value)
        if isinstance(node.get("children"), list):
            children = [_topology_sketch(c) for c in node["children"]]
        return {
            "t": type_name,
            "p": props or None,
            "c": children or None,
            "n": len(children) if children else 0,
        }
    if isinstance(node, list):
        return [_topology_sketch(c) for c in node]
    if isinstance(node, str):
        # Placeholders keep identity; other strings collapse to "_".
        return node if node.startswith(":") else "_"
    return node


def _binding_aware_sketch(node: Any) -> dict[str, Any]:
    """Component topology + role-bearing props + placeholders + refs."""
    if node is None:
        return None  # type: ignore[return-value]
    if isinstance(node, dict):
        type_name = node.get("typeName")
        props: dict[str, Any] = {}
        children: list[Any] = []
        raw_props = node.get("props") or {}
        for key, value in raw_props.items():
            if key == "children" and isinstance(value, list):
                children = [_binding_aware_sketch(c) for c in value]
            elif key in {"type", "typeName", "partial", "hasDynamicProps"}:
                continue
            else:
                props[key] = _binding_aware_sketch(value)
        if isinstance(node.get("children"), list):
            children = [_binding_aware_sketch(c) for c in node["children"]]
        placeholders = sorted(extract_placeholders(json.dumps(props)))
        return {
            "t": type_name,
            "r": props or None,
            "c": children or None,
            "s": placeholders or None,
        }
    if isinstance(node, list):
        return [_binding_aware_sketch(c) for c in node]
    return node


def _type_action_multiset(record: ExampleRecord, ast: Any) -> str:
    counts = component_multiset(ast)
    placeholders = sorted(set(extract_placeholders(record.openui)))
    return _hash({"components": counts, "placeholders": placeholders})


def fingerprint_record(record: ExampleRecord) -> DiversityFingerprints:
    """Return all diversity fingerprints for a single record."""
    program = validate(record.openui)
    root = program.root
    return DiversityFingerprints(
        schema_version=SCHEMA_VERSION,
        record_id=record.id,
        canonical_root_id=canonical_fingerprint(record.openui),
        binding_aware_sketch=_hash(_binding_aware_sketch(root)),
        topology_sketch=_hash(_topology_sketch(root)),
        type_action_multiset=_type_action_multiset(record, root),
        prompt_intent_fingerprint=_hash(_normalize_prompt_intent(record.prompt)),
        source_lineage_id=_source_lineage_id(record),
        exact_structure_fingerprint=fingerprint_openui_structure(record.openui),
    )


def summarize_fingerprints(
    fingerprints: list[DiversityFingerprints],
) -> dict[str, Any]:
    """Aggregate unique counts across fingerprint kinds."""
    counts: dict[str, int] = {}
    for key in (
        "canonical_root_id",
        "binding_aware_sketch",
        "topology_sketch",
        "type_action_multiset",
        "prompt_intent_fingerprint",
        "source_lineage_id",
        "exact_structure_fingerprint",
    ):
        unique = {getattr(fp, key) for fp in fingerprints}
        counts[key] = len(unique)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_records": len(fingerprints),
        "unique_counts": counts,
    }


__all__ = [
    "DiversityFingerprints",
    "SCHEMA_VERSION",
    "fingerprint_record",
    "summarize_fingerprints",
]
