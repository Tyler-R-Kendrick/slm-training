"""Governed external-page capture and weak OpenUI projection records."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from slm_training.bridge_utils import repo_root
from slm_training.data.governance import SourceProvenance, govern_record
from slm_training.data.leakage import (
    find_leakage,
    fingerprint_openui,
    fingerprint_openui_structure,
    fingerprint_pair,
    fingerprint_prompt,
)
from slm_training.data.progspec import ProgramSpec, emit_record
from slm_training.data.verify import RuntimeEvidence
from slm_training.dsl.schema import ExampleRecord

_SAFE_ID_RE = re.compile(r"[^a-z0-9_]+")


def _safe_id(value: str) -> str:
    return _SAFE_ID_RE.sub("_", value.lower()).strip("_")[:48] or "page"


@dataclass(frozen=True)
class ManifestEntry:
    """One page whose acquisition decision is explicit and reviewable."""

    id: str
    url: str
    approved: bool = False
    acquisition_date: str = ""
    terms_policy_id: str = ""
    legal_basis: str = ""
    license: str = ""
    attribution: str = ""
    asset_rights: Mapping[str, str] | None = None
    robots_policy: str = ""
    deletion_procedure: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestEntry:
        return cls(
            id=str(data.get("id") or ""),
            url=str(data.get("url") or ""),
            approved=bool(data.get("approved", False)),
            acquisition_date=str(data.get("acquisition_date") or ""),
            terms_policy_id=str(data.get("terms_policy_id") or ""),
            legal_basis=str(data.get("legal_basis") or ""),
            license=str(data.get("license") or ""),
            attribution=str(data.get("attribution") or ""),
            asset_rights=dict(data.get("asset_rights") or {}),
            robots_policy=str(data.get("robots_policy") or ""),
            deletion_procedure=str(data.get("deletion_procedure") or ""),
        )

    def provenance(self, content: str | bytes) -> SourceProvenance:
        return SourceProvenance.from_content(
            source_url=self.url,
            acquisition_date=self.acquisition_date,
            terms_policy_id=self.terms_policy_id,
            legal_basis=self.legal_basis,
            license=self.license,
            attribution=self.attribution,
            asset_rights=dict(self.asset_rights or {}),
            robots_policy=self.robots_policy,
            deletion_procedure=self.deletion_procedure,
            content=content,
            transformation_history=(
                "browser-capture",
                "dom-accessibility-layout-to-ui-graph",
                "placeholderize",
                "candidate-project-and-verify",
            ),
        )

    def acquisition_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.id.strip():
            errors.append("id")
        if not self.approved:
            errors.append("not_approved")
        errors.extend(self.provenance(b"").validation_errors())
        return tuple(dict.fromkeys(errors))


def load_manifest(path: Path | str) -> list[ManifestEntry]:
    """Load a deterministic JSONL acquisition manifest."""
    entries: list[ManifestEntry] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise TypeError("manifest row must be a JSON object")
                entry = ManifestEntry.from_dict(data)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            entries.append(entry)
    ids = [entry.id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("deconstruct manifest contains duplicate ids")
    return sorted(entries, key=lambda entry: entry.id)


@dataclass(frozen=True)
class DeconstructConfig:
    cache_dir: Path = Path("fixtures/deconstruct/cache")
    capture_script: Path = Path("tools/deconstruct/capture.mjs")
    rate_limit_s: float = 1.0
    timeout_s: float = 60.0
    user_agent: str = "slm-training-deconstruct/0.1 (+research; governed)"
    live: bool = False


@dataclass(frozen=True)
class BrowserCapture:
    source_url: str
    dom_snapshot: str
    accessibility_tree: Any
    elements: tuple[dict[str, Any], ...]
    viewport: Mapping[str, int]
    responsive_state: str
    screenshot_path: str | None = None
    interaction_trace: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BrowserCapture:
        return cls(
            source_url=str(data.get("source_url") or ""),
            dom_snapshot=str(data.get("dom_snapshot") or ""),
            accessibility_tree=data.get("accessibility_tree"),
            elements=tuple(dict(item) for item in data.get("elements") or ()),
            viewport={
                "width": int((data.get("viewport") or {}).get("width") or 0),
                "height": int((data.get("viewport") or {}).get("height") or 0),
            },
            responsive_state=str(data.get("responsive_state") or "unknown"),
            screenshot_path=(
                str(data["screenshot_path"]) if data.get("screenshot_path") else None
            ),
            interaction_trace=tuple(
                str(item) for item in data.get("interaction_trace") or ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "dom_snapshot": self.dom_snapshot,
            "accessibility_tree": self.accessibility_tree,
            "elements": list(self.elements),
            "viewport": dict(self.viewport),
            "responsive_state": self.responsive_state,
            "screenshot_path": self.screenshot_path,
            "interaction_trace": list(self.interaction_trace),
        }

    def content_blob(self) -> str:
        """Canonical raw evidence used only for scanning and content hashing."""
        return json.dumps(
            {
                "dom_snapshot": self.dom_snapshot,
                "accessibility_tree": self.accessibility_tree,
                "elements": list(self.elements),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def capture_page(
    entry: ManifestEntry,
    config: DeconstructConfig | None = None,
    *,
    root: Path | None = None,
) -> BrowserCapture:
    """Capture an approved page; cache reads work while live capture is disabled."""
    config = config or DeconstructConfig()
    errors = entry.acquisition_errors()
    if errors:
        raise PermissionError(f"acquisition blocked: {', '.join(errors)}")

    cache_key = hashlib.sha256(entry.url.encode("utf-8")).hexdigest()[:16]
    cache_path = config.cache_dir / f"{_safe_id(entry.id)}-{cache_key}.json"
    if cache_path.is_file():
        return BrowserCapture.from_dict(json.loads(cache_path.read_text("utf-8")))
    if not config.live:
        raise RuntimeError(f"live capture disabled and cache miss: {entry.url}")

    parsed = urlparse(entry.url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"unsupported capture URL: {entry.url}")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for Playwright capture")
    root = root or repo_root()
    script = config.capture_script
    if not script.is_absolute():
        script = root / script
    if not script.is_file():
        raise RuntimeError(f"Playwright capture script not found: {script}")

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = cache_path.with_suffix(".png")
    time.sleep(max(0.0, config.rate_limit_s))
    proc = subprocess.run(
        [node, str(script)],
        input=json.dumps(
            {
                "url": entry.url,
                "user_agent": config.user_agent,
                "screenshot_path": str(screenshot_path),
            }
        ),
        text=True,
        capture_output=True,
        cwd=root,
        timeout=config.timeout_s,
        check=False,
    )
    raw = proc.stdout.strip()
    if proc.returncode or not raw:
        detail = (proc.stderr or raw or f"exit {proc.returncode}").strip()
        raise RuntimeError(f"Playwright capture failed: {detail[:500]}")
    try:
        capture = BrowserCapture.from_dict(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Playwright capture returned invalid JSON: {raw[:500]}") from exc
    if capture.source_url != entry.url:
        raise RuntimeError("capture source_url does not match the approved manifest URL")
    cache_path.write_text(
        json.dumps(capture.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return capture


@dataclass(frozen=True)
class UIGraphNode:
    id: str
    role: str
    accessible_name: str | None
    text: str | None
    bbox: Mapping[str, float]
    visible: bool
    parent_id: str | None
    form: Mapping[str, Any]
    repeated_region: bool
    affordances: tuple[str, ...]
    responsive_state: str
    dom_path: str
    screenshot_ref: str | None
    dsl_node: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "accessible_name": self.accessible_name,
            "text": self.text,
            "bbox": dict(self.bbox),
            "visible": self.visible,
            "parent_id": self.parent_id,
            "form": dict(self.form),
            "repeated_region": self.repeated_region,
            "affordances": list(self.affordances),
            "responsive_state": self.responsive_state,
            "dom_path": self.dom_path,
            "screenshot_ref": self.screenshot_ref,
            "dsl_node": self.dsl_node,
        }


@dataclass(frozen=True)
class NormalizedUIGraph:
    source_url: str
    viewport: Mapping[str, int]
    nodes: tuple[UIGraphNode, ...]
    interaction_trace: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "viewport": dict(self.viewport),
            "nodes": [node.to_dict() for node in self.nodes],
            "interaction_trace": list(self.interaction_trace),
        }


def _placeholder(namespace: str, index: int, field: str, value: Any) -> str | None:
    if not str(value or "").strip():
        return None
    return f":{namespace}.node_{index}.{field}"


def normalize_capture(
    capture: BrowserCapture,
    *,
    namespace: str,
    candidate_links: Mapping[str, str] | None = None,
) -> NormalizedUIGraph:
    """Drop raw user text and retain only stable placeholder skeletons."""
    namespace = _safe_id(namespace)
    candidate_links = candidate_links or {}
    nodes: list[UIGraphNode] = []
    for index, raw in enumerate(capture.elements):
        bbox = raw.get("bbox") or {}
        form = dict(raw.get("form") or {})
        if form.get("value") not in (None, ""):
            form["value"] = f":{namespace}.node_{index}.value"
        nodes.append(
            UIGraphNode(
                id=str(raw.get("id") or f"node_{index}"),
                role=str(raw.get("role") or "generic"),
                accessible_name=_placeholder(
                    namespace, index, "name", raw.get("accessible_name")
                ),
                text=_placeholder(namespace, index, "text", raw.get("text")),
                bbox={
                    key: float(bbox.get(key) or 0.0)
                    for key in ("x", "y", "width", "height")
                },
                visible=bool(raw.get("visible", False)),
                parent_id=(str(raw["parent_id"]) if raw.get("parent_id") else None),
                form=form,
                repeated_region=bool(raw.get("repeated_region", False)),
                affordances=tuple(str(x) for x in raw.get("affordances") or ()),
                responsive_state=str(
                    raw.get("responsive_state") or capture.responsive_state
                ),
                dom_path=str(raw.get("dom_path") or ""),
                screenshot_ref=(
                    str(raw["screenshot_ref"])
                    if raw.get("screenshot_ref")
                    else None
                ),
                dsl_node=(
                    candidate_links.get(str(raw.get("id") or f"node_{index}"))
                    or (str(raw["dsl_node"]) if raw.get("dsl_node") else None)
                ),
            )
        )
    return NormalizedUIGraph(
        source_url=capture.source_url,
        viewport=dict(capture.viewport),
        nodes=tuple(nodes),
        interaction_trace=tuple(capture.interaction_trace),
    )


@dataclass(frozen=True)
class ProjectionEvidence:
    projection_status: str
    supported_element_ratio: float
    behavior_coverage: float
    appearance_similarity: float | None
    unsupported_elements: tuple[str, ...]
    review_required: bool
    confidence_tier: str

    @classmethod
    def from_counts(
        cls,
        *,
        matched_elements: int,
        total_source_elements: int,
        supported_behaviors: int = 0,
        total_behaviors: int = 0,
        appearance_similarity: float | None = None,
        unsupported_elements: Iterable[str] = (),
        human_reviewed: bool = False,
    ) -> ProjectionEvidence:
        counts = (
            matched_elements,
            total_source_elements,
            supported_behaviors,
            total_behaviors,
        )
        if any(value < 0 for value in counts):
            raise ValueError("projection counts must be non-negative")
        if matched_elements > total_source_elements:
            raise ValueError("matched_elements exceeds total_source_elements")
        if supported_behaviors > total_behaviors:
            raise ValueError("supported_behaviors exceeds total_behaviors")
        if appearance_similarity is not None and not 0 <= appearance_similarity <= 1:
            raise ValueError("appearance_similarity must be between 0 and 1")

        ratio = (
            matched_elements / total_source_elements if total_source_elements else 0.0
        )
        behavior = supported_behaviors / total_behaviors if total_behaviors else 1.0
        unsupported = tuple(sorted(set(str(x) for x in unsupported_elements)))
        appearance_ok = appearance_similarity is None or appearance_similarity >= 0.9
        if not total_source_elements or not matched_elements:
            status = "unsupported"
        elif ratio == 1.0 and behavior == 1.0 and appearance_ok and not unsupported:
            status = "exact"
        elif ratio >= 0.75 and behavior >= 0.5:
            status = "approximated"
        else:
            status = "omitted"
        return cls(
            projection_status=status,
            supported_element_ratio=round(ratio, 6),
            behavior_coverage=round(behavior, 6),
            appearance_similarity=appearance_similarity,
            unsupported_elements=unsupported,
            review_required=not human_reviewed,
            # External projections remain weak candidates even after review.
            confidence_tier="Bronze",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_status": self.projection_status,
            "supported_element_ratio": self.supported_element_ratio,
            "behavior_coverage": self.behavior_coverage,
            "appearance_similarity": self.appearance_similarity,
            "unsupported_elements": list(self.unsupported_elements),
            "review_required": self.review_required,
            "confidence_tier": self.confidence_tier,
        }

    @classmethod
    def compare(
        cls,
        source: NormalizedUIGraph,
        candidate: NormalizedUIGraph,
        *,
        appearance_similarity: float | None = None,
        human_reviewed: bool = False,
    ) -> ProjectionEvidence:
        """Compare rendered graphs by role and controlled interaction affordances."""
        source_roles: dict[str, int] = {}
        candidate_roles: dict[str, int] = {}
        for node in source.nodes:
            if node.visible:
                source_roles[node.role] = source_roles.get(node.role, 0) + 1
        for node in candidate.nodes:
            if node.visible:
                candidate_roles[node.role] = candidate_roles.get(node.role, 0) + 1
        matched = sum(
            min(count, candidate_roles.get(role, 0))
            for role, count in source_roles.items()
        )

        source_behaviors: dict[str, int] = {}
        candidate_behaviors: dict[str, int] = {}
        for graph, counts in (
            (source, source_behaviors),
            (candidate, candidate_behaviors),
        ):
            for node in graph.nodes:
                for affordance in node.affordances:
                    counts[affordance] = counts.get(affordance, 0) + 1
        supported_behaviors = sum(
            min(count, candidate_behaviors.get(name, 0))
            for name, count in source_behaviors.items()
        )
        unsupported = (
            f"{role}:{count - candidate_roles.get(role, 0)}"
            for role, count in source_roles.items()
            if count > candidate_roles.get(role, 0)
        )
        return cls.from_counts(
            matched_elements=matched,
            total_source_elements=sum(source_roles.values()),
            supported_behaviors=supported_behaviors,
            total_behaviors=sum(source_behaviors.values()),
            appearance_similarity=appearance_similarity,
            unsupported_elements=unsupported,
            human_reviewed=human_reviewed,
        )


def _eval_fingerprints(records: Iterable[ExampleRecord]) -> dict[str, set[str]]:
    fingerprints = {
        "ids": set(),
        "split_group_ids": set(),
        "prompts": set(),
        "openuis": set(),
        "structures": set(),
        "pairs": set(),
        "design_mds": set(),
    }
    for record in records:
        fingerprints["ids"].add(record.id)
        group = record.meta.get("split_group_id")
        if group:
            fingerprints["split_group_ids"].add(str(group))
        fingerprints["prompts"].add(fingerprint_prompt(record.prompt))
        fingerprints["openuis"].add(fingerprint_openui(record.openui))
        fingerprints["structures"].add(
            fingerprint_openui_structure(record.openui)
        )
        fingerprints["pairs"].add(fingerprint_pair(record.prompt, record.openui))
    return fingerprints


def assert_decontaminated(
    records: Iterable[ExampleRecord], eval_records: Iterable[ExampleRecord]
) -> None:
    """Reject exact, structural, pair, id, or split-group overlap with eval."""
    eval_fingerprints = _eval_fingerprints(eval_records)
    leaks: dict[str, list[str]] = {}
    for record in records:
        reasons = find_leakage(record, eval_fingerprints)
        if reasons:
            leaks[record.id] = reasons
    if leaks:
        raise ValueError(f"web projection leaks into eval: {leaks}")


def project_capture(
    *,
    entry: ManifestEntry,
    capture: BrowserCapture,
    candidate_openui: str,
    evidence: ProjectionEvidence,
    candidate_runtime: RuntimeEvidence,
    eval_records: Iterable[ExampleRecord],
    candidate_links: Mapping[str, str] | None = None,
    split: str = "train",
) -> ExampleRecord:
    """Emit a lineage-linked weak candidate, governed and checked against eval."""
    if capture.source_url != entry.url:
        raise ValueError("capture source_url does not match manifest entry")
    if capture.dom_snapshot and not (
        capture.elements
        or capture.accessibility_tree
        or capture.screenshot_path
        or capture.interaction_trace
    ):
        raise ValueError("raw DOM cannot be the sole projection evidence")
    graph = normalize_capture(
        capture,
        namespace=entry.id,
        candidate_links=candidate_links,
    )
    source_content = capture.content_blob()
    provenance = entry.provenance(source_content)
    digest = hashlib.sha256(f"{entry.id}\0{entry.url}".encode()).hexdigest()[:12]
    root_id = f"web_{_safe_id(entry.id)}_{digest}"
    spec = ProgramSpec.from_openui(
        id=root_id,
        openui=candidate_openui,
        facts={
            "normalized_ui_graph": graph.to_dict(),
            "projection_status": evidence.projection_status,
        },
        program_family_id=f"web_projection:{digest}",
        lineage_id=f"web_projection:{digest}",
        split_group_id=f"web_projection:{digest}",
        split=split,
        provenance={
            "source_url": entry.url,
            "content_hash": provenance.content_hash,
        },
    )
    graph_ids = {node.id for node in graph.nodes}
    unknown_source_nodes = sorted(set(candidate_links or {}) - graph_ids)
    if unknown_source_nodes:
        raise ValueError(
            f"candidate link references unknown capture node: {unknown_source_nodes[0]}"
        )
    binders = {
        line.partition("=")[0].strip()
        for line in spec.canonical_openui.splitlines()
        if "=" in line
    }
    unknown_dsl_nodes = sorted(
        node.dsl_node
        for node in graph.nodes
        if node.dsl_node and node.dsl_node not in binders
    )
    if unknown_dsl_nodes:
        raise ValueError(
            f"capture correspondence references unknown DSL node: {unknown_dsl_nodes[0]}"
        )
    namespace = _safe_id(entry.id)
    record = emit_record(
        spec,
        prompt=(
            f"Reconstruct :{namespace}.interface from normalized structural, "
            "accessibility, layout, visual, and interaction evidence. Preserve "
            "all user-facing content as placeholders."
        ),
        task="generation",
        source="web_projection",
        tier=evidence.confidence_tier,
        provenance=provenance.to_dict(),
        meta={
            "source_kind": "web",
            "source_family": "web_projection",
            "candidate_label": "weak_evidence_backed",
            "projection": evidence.to_dict(),
            "normalized_ui_graph": graph.to_dict(),
            "runtime_evidence": candidate_runtime.to_dict(),
            "require_runtime": True,
            "require_behavior": any(node.affordances for node in graph.nodes),
            "evidence_channels": [
                "dom",
                "accessibility",
                "computed_layout",
                "screenshot",
                "interaction_trace",
            ],
        },
    )
    record = govern_record(record, provenance, raw_content=source_content)
    eval_rows = list(eval_records)
    assert_decontaminated([record], eval_rows)
    record.meta["decontamination"] = {
        "eval_records_checked": len(eval_rows),
        "exact_structural_split_group_leakage": 0,
    }
    return record


__all__ = [
    "BrowserCapture",
    "DeconstructConfig",
    "ManifestEntry",
    "NormalizedUIGraph",
    "ProjectionEvidence",
    "UIGraphNode",
    "assert_decontaminated",
    "capture_page",
    "load_manifest",
    "normalize_capture",
    "project_capture",
]
