"""Evidence-backed external-page projections into candidate OpenUI records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from slm_training.data.awwwards import AwwwardsConfig, fetch_url
from slm_training.data.governance import SourceProvenance, govern_record
from slm_training.data.leakage import find_leakage
from slm_training.data.progspec import ProgramSpec, emit_record
from slm_training.data.verify import RuntimeEvidence
from slm_training.dsl.schema import ExampleRecord

PROJECTION_STATUSES = frozenset({"exact", "approximated", "omitted", "unsupported"})
_TOKEN_RE = re.compile(r"[^a-z0-9_]+")
_FORM_ROLES = frozenset(
    {
        "button",
        "checkbox",
        "combobox",
        "input",
        "radio",
        "searchbox",
        "switch",
        "textbox",
    }
)


def _token(value: object, *, fallback: str) -> str:
    token = _TOKEN_RE.sub("_", str(value or "").lower()).strip("_")
    return token or fallback


@dataclass(frozen=True)
class BrowserCapture:
    """Inert browser evidence. Raw DOM is hashed, scanned, and never emitted."""

    source_url: str
    dom_snapshot: str | None = None
    accessibility_tree: tuple[Mapping[str, Any], ...] = ()
    computed_layout: tuple[Mapping[str, Any], ...] = ()
    screenshot_refs: tuple[str, ...] = ()
    interaction_trace: tuple[str, ...] = ()
    responsive_state: str = "default"

    def __post_init__(self) -> None:
        parsed = urlparse(self.source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        if not (
            self.accessibility_tree
            or self.computed_layout
            or self.screenshot_refs
            or self.interaction_trace
        ):
            raise ValueError("raw DOM alone is not projection ground truth")

    def evidence(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "dom_sha256": (
                hashlib.sha256(self.dom_snapshot.encode("utf-8")).hexdigest()
                if self.dom_snapshot is not None
                else None
            ),
            "screenshot_refs": sorted(self.screenshot_refs),
            "interaction_trace": list(self.interaction_trace),
            "responsive_state": self.responsive_state,
        }


@dataclass(frozen=True)
class UINode:
    node_id: str
    role: str
    accessible_name: str | None
    text: str | None
    bbox: tuple[float, float, float, float] | None
    visible: bool
    parent_id: str | None
    form_control: bool
    repeated_region: bool
    affordances: tuple[str, ...]
    responsive_state: str
    dom_path: str | None
    screenshot_ref: str | None
    dsl_node: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "accessible_name": self.accessible_name,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox else None,
            "visible": self.visible,
            "parent_id": self.parent_id,
            "form_control": self.form_control,
            "repeated_region": self.repeated_region,
            "affordances": list(self.affordances),
            "responsive_state": self.responsive_state,
            "dom_path": self.dom_path,
            "screenshot_ref": self.screenshot_ref,
            "dsl_node": self.dsl_node,
        }


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        value = [value.get(key) for key in ("x", "y", "width", "height")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(part) for part in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def normalize_capture(
    capture: BrowserCapture,
    *,
    namespace: str,
    candidate_links: Mapping[str, str] | None = None,
) -> tuple[UINode, ...]:
    """Normalize browser evidence while replacing every user-facing string."""
    candidate_links = candidate_links or {}
    layout = {
        str(row.get("node_id") or row.get("id")): row
        for row in capture.computed_layout
        if row.get("node_id") or row.get("id")
    }
    accessibility = {
        str(row.get("node_id") or row.get("id")): row
        for row in capture.accessibility_tree
        if row.get("node_id") or row.get("id")
    }
    ids = sorted(set(layout) | set(accessibility))
    prefix = _token(namespace, fallback="web")
    nodes: list[UINode] = []
    for index, raw_id in enumerate(ids):
        safe_id = _token(raw_id, fallback=f"node_{index}")
        a11y = accessibility.get(raw_id, {})
        box = layout.get(raw_id, {})
        role = _token(a11y.get("role") or box.get("role"), fallback="generic")
        name = f":{prefix}.{safe_id}.name" if a11y.get("name") else None
        text = f":{prefix}.{safe_id}.text" if a11y.get("text") else None
        parent = a11y.get("parent_id") or box.get("parent_id")
        affordances = tuple(
            sorted(str(value) for value in (a11y.get("affordances") or ()))
        )
        nodes.append(
            UINode(
                node_id=safe_id,
                role=role,
                accessible_name=name,
                text=text,
                bbox=_bbox(box.get("bbox") or a11y.get("bbox")),
                visible=bool(box.get("visible", a11y.get("visible", True))),
                parent_id=_token(parent, fallback="parent") if parent else None,
                form_control=bool(a11y.get("form_control", role in _FORM_ROLES)),
                repeated_region=bool(
                    a11y.get("repeated_region", box.get("repeated_region", False))
                ),
                affordances=affordances,
                responsive_state=str(
                    box.get("responsive_state") or capture.responsive_state
                ),
                dom_path=(str(a11y["dom_path"]) if a11y.get("dom_path") else None),
                screenshot_ref=(
                    str(box["screenshot_ref"]) if box.get("screenshot_ref") else None
                ),
                dsl_node=candidate_links.get(raw_id),
            )
        )
    return tuple(nodes)


@dataclass(frozen=True)
class ProjectionEvidence:
    projection_status: str
    element_statuses: tuple[tuple[str, str], ...]
    supported_element_ratio: float
    behavior_coverage: float
    review_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_status": self.projection_status,
            "element_statuses": dict(self.element_statuses),
            "supported_element_ratio": self.supported_element_ratio,
            "behavior_coverage": self.behavior_coverage,
            "review_required": self.review_required,
        }


def assess_projection(
    graph: tuple[UINode, ...],
    element_statuses: Mapping[str, str],
    *,
    matched_behaviors: int = 0,
    expected_behaviors: int = 0,
    human_reviewed: bool = False,
) -> ProjectionEvidence:
    """Compute stable projection metrics from explicit element correspondences."""
    invalid = sorted(set(element_statuses.values()) - PROJECTION_STATUSES)
    if invalid:
        raise ValueError(f"invalid projection status: {invalid[0]}")
    node_ids = {node.node_id for node in graph}
    unknown = sorted(set(element_statuses) - node_ids)
    if unknown:
        raise ValueError(f"projection status references unknown node: {unknown[0]}")
    if (
        matched_behaviors < 0
        or expected_behaviors < 0
        or matched_behaviors > expected_behaviors
    ):
        raise ValueError("behavior counts must satisfy 0 <= matched <= expected")
    statuses = tuple(
        (node.node_id, element_statuses.get(node.node_id, "omitted")) for node in graph
    )
    supported = sum(status in {"exact", "approximated"} for _, status in statuses)
    supported_ratio = supported / len(statuses) if statuses else 0.0
    behavior_coverage = (
        matched_behaviors / expected_behaviors if expected_behaviors else 1.0
    )
    values = {status for _, status in statuses}
    overall = (
        "unsupported"
        if not values or "unsupported" in values
        else "omitted"
        if "omitted" in values
        else "approximated"
        if "approximated" in values
        else "exact"
    )
    review_required = (
        not human_reviewed or overall != "exact" or behavior_coverage < 1.0
    )
    return ProjectionEvidence(
        projection_status=overall,
        element_statuses=statuses,
        supported_element_ratio=supported_ratio,
        behavior_coverage=behavior_coverage,
        review_required=review_required,
    )


def build_web_projection(
    *,
    projection_id: str,
    capture: BrowserCapture,
    candidate_openui: str,
    element_statuses: Mapping[str, str],
    runtime_evidence: RuntimeEvidence,
    provenance: SourceProvenance | None,
    candidate_links: Mapping[str, str] | None = None,
    matched_behaviors: int = 0,
    expected_behaviors: int = 0,
    human_reviewed: bool = False,
    eval_fingerprints: dict[str, set[str]] | None = None,
) -> ExampleRecord:
    """Create a governed, verified candidate label from independent page evidence."""
    graph = normalize_capture(
        capture,
        namespace=projection_id,
        candidate_links=candidate_links,
    )
    evidence = assess_projection(
        graph,
        element_statuses,
        matched_behaviors=matched_behaviors,
        expected_behaviors=expected_behaviors,
        human_reviewed=human_reviewed,
    )
    lineage = hashlib.sha256(capture.source_url.encode("utf-8")).hexdigest()[:16]
    spec = ProgramSpec.from_openui(
        id=projection_id,
        openui=candidate_openui,
        facts={"normalized_ui_graph": [node.to_dict() for node in graph]},
        program_family_id="web_projection",
        lineage_id=f"web_{lineage}",
        split_group_id=f"web_{lineage}",
        provenance={"source_url": capture.source_url},
    )
    binders = {
        line.partition("=")[0].strip()
        for line in spec.canonical_openui.splitlines()
        if "=" in line
    }
    unknown_links = sorted(
        str(dsl_node)
        for dsl_node in (candidate_links or {}).values()
        if str(dsl_node) not in binders
    )
    if unknown_links:
        raise ValueError(
            f"candidate link references unknown DSL node: {unknown_links[0]}"
        )
    meta: dict[str, Any] = {
        "source_kind": "web",
        "label_kind": "candidate",
        "raw_html_is_ground_truth": False,
        "capture_evidence": capture.evidence(),
        "normalized_ui_graph": [node.to_dict() for node in graph],
        "projection": evidence.to_dict(),
        "runtime_evidence": runtime_evidence.to_dict(),
        "require_runtime": True,
        "require_behavior": expected_behaviors > 0,
    }
    if human_reviewed:
        meta["human_audit_passed"] = True
    record = emit_record(
        spec,
        prompt="Recreate the captured web layout with the provided placeholder contract.",
        task="generation",
        source="web_projection",
        tier="Bronze",
        determinacy="evidence-backed-candidate",
        meta=meta,
    )
    record = govern_record(record, provenance, raw_content=capture.dom_snapshot)
    if eval_fingerprints:
        leakage = find_leakage(record, eval_fingerprints)
        if leakage:
            raise ValueError(f"web projection overlaps eval: {', '.join(leakage)}")
    return record


def load_manifest(
    path: Path | str = Path("src/slm_training/resources/deconstruct/manifest.jsonl"),
) -> list[dict[str, Any]]:
    """Load approved-source definitions in stable identifier order."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(rows, key=lambda row: str(row.get("id") or row.get("source_url")))


def fetch_approved_source(
    entry: Mapping[str, Any], *, config: AwwwardsConfig | None = None
) -> str:
    """Reuse the cache/rate/live gate; a manifest approval is also mandatory."""
    if entry.get("approved") is not True:
        raise ValueError("source is not approved for acquisition")
    return fetch_url(str(entry["source_url"]), config=config or AwwwardsConfig())


__all__ = [
    "BrowserCapture",
    "PROJECTION_STATUSES",
    "ProjectionEvidence",
    "UINode",
    "assess_projection",
    "build_web_projection",
    "fetch_approved_source",
    "load_manifest",
    "normalize_capture",
]
