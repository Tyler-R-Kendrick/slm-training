"""Render-equivalence surrogates for OpenUI (SLM-172 / SDE2-05).

Three-tier diagnostic:

* Tier 0: canonical AST signature exact match + binding-graph equality.
* Tier 1: deterministic normalized render-tree subscores (component/role/
  topology/cardinality/binding/interaction overlap).
* Tier 2: optional Playwright/chromium pixel similarity; capability-gated to
  ``not_available`` when the renderer is unavailable.

The aggregate ``equivalent`` bool is fail-closed: parse/schema/compiler failure,
missing components, or binding/topology/interaction mismatches prevent
equivalence.  Tier-2 visual similarity never overrides structural mismatches.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Literal

from slm_training.data.render.capture import CaptureConfig, capture_program
from slm_training.dsl.canonicalize import canonical_equal, canonical_fingerprint
from slm_training.dsl.grammar.backends.ast_utils import component_multiset
from slm_training.dsl.parser import validate
from slm_training.dsl.production_codec import parse_statement_bindings
from slm_training.versioning import build_version_stamp

_INTERACTIVE_TYPES = frozenset(
    {
        "Button",
        "Input",
        "TextInput",
        "Select",
        "TextArea",
        "SwitchItem",
        "Slider",
        "Link",
        "Tabs",
        "TabItem",
    }
)

_PLACEHOLDER_RE = re.compile(r":[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class _Tree:
    label: str
    children: tuple["_Tree", ...] = ()

    @cached_property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)


def _semantic_tree(value: Any) -> _Tree | None:
    """Mirror of task_scoreboard._semantic_tree (kept local to avoid cycles)."""
    if not isinstance(value, dict):
        return None
    if value.get("type") == "element":
        children: list[_Tree] = []
        props = value.get("props")
        if isinstance(props, dict):
            stack = list(props.values())
            while stack:
                child = stack.pop(0)
                if isinstance(child, list):
                    stack[0:0] = child
                elif isinstance(child, dict):
                    node = _semantic_tree(child)
                    if node is not None:
                        children.append(node)
                    else:
                        stack[0:0] = list(child.values())
        # Some backends keep children at the top level.
        for child in value.get("children") or ():
            node = _semantic_tree(child)
            if node is not None:
                children.append(node)
        return _Tree(str(value.get("typeName") or "element"), tuple(children))
    for child in value.values():
        node = _semantic_tree(child)
        if node is not None:
            return node
    return None


@lru_cache(maxsize=4096)
def _tree_distance(left: _Tree, right: _Tree) -> int:
    substitution = int(left.label != right.label)
    a, b = left.children, right.children
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, child in enumerate(a, start=1):
        dp[i][0] = dp[i - 1][0] + child.size
    for j, child in enumerate(b, start=1):
        dp[0][j] = dp[0][j - 1] + child.size
    for i, child_a in enumerate(a, start=1):
        for j, child_b in enumerate(b, start=1):
            dp[i][j] = min(
                dp[i - 1][j] + child_a.size,
                dp[i][j - 1] + child_b.size,
                dp[i - 1][j - 1] + _tree_distance(child_a, child_b),
            )
    return substitution + dp[-1][-1]


def _multiset_f1(left: Counter[Any], right: Counter[Any]) -> float:
    total_left, total_right = sum(left.values()), sum(right.values())
    if not total_left and not total_right:
        return 1.0
    overlap = sum((left & right).values())
    precision = overlap / total_left if total_left else 0.0
    recall = overlap / total_right if total_right else 0.0
    return (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )


@dataclass(frozen=True)
class Tier0RenderEquivalenceReport:
    canonical_exact: bool
    canonical_fingerprint_pred: str | None
    canonical_fingerprint_gold: str | None
    binding_graph_equal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_exact": self.canonical_exact,
            "canonical_fingerprint_pred": self.canonical_fingerprint_pred,
            "canonical_fingerprint_gold": self.canonical_fingerprint_gold,
            "binding_graph_equal": self.binding_graph_equal,
        }


@dataclass(frozen=True)
class Tier1RenderEquivalenceReport:
    component_type_match: float
    role_match: float
    topology_match: float
    cardinality_match: float
    binding_graph_match: float
    interaction_dependency_match: float
    normalized_render_tree_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_type_match": self.component_type_match,
            "role_match": self.role_match,
            "topology_match": self.topology_match,
            "cardinality_match": self.cardinality_match,
            "binding_graph_match": self.binding_graph_match,
            "interaction_dependency_match": self.interaction_dependency_match,
            "normalized_render_tree_distance": self.normalized_render_tree_distance,
        }


@dataclass(frozen=True)
class Tier2RenderEquivalenceReport:
    visual_similarity: float | None
    status: Literal["not_available", "available"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_similarity": self.visual_similarity,
            "status": self.status,
        }


@dataclass(frozen=True)
class RenderEquivalenceReport:
    tier0: Tier0RenderEquivalenceReport
    tier1: Tier1RenderEquivalenceReport
    tier2: Tier2RenderEquivalenceReport
    equivalent: bool
    reason_codes: tuple[str, ...]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier0": self.tier0.to_dict(),
            "tier1": self.tier1.to_dict(),
            "tier2": self.tier2.to_dict(),
            "equivalent": self.equivalent,
            "reason_codes": list(self.reason_codes),
            "version_stamp": self.version_stamp,
        }


def _parse_root(source: str, dsl: str | None = None) -> Any:
    return validate(source, dsl=dsl).root


def _canonical_bindings(source: str, dsl: str | None = None) -> dict[str, Any]:
    from slm_training.dsl.canonicalize import canonicalize

    canonical = canonicalize(source, dsl=dsl, validate=False)
    return parse_statement_bindings(canonical, dsl=dsl, validate=False)


def _walk_elements(
    value: Any,
    parent_type: str | None = None,
    topology: Counter[tuple[str, str]] | None = None,
    roles: Counter[tuple[str, str]] | None = None,
) -> None:
    """Collect parent->child topology edges and component->placeholder roles."""
    if topology is None:
        topology = Counter()
    if roles is None:
        roles = Counter()
    if isinstance(value, dict) and value.get("type") == "element":
        type_name = str(value.get("typeName") or "element")
        if parent_type is not None:
            topology[(parent_type, type_name)] += 1
        props = value.get("props") or {}
        for prop_name, prop_val in props.items():
            if prop_name == "children" and isinstance(prop_val, list):
                for child in prop_val:
                    _walk_elements(child, type_name, topology, roles)
            elif isinstance(prop_val, dict) and prop_val.get("type") == "element":
                _walk_elements(prop_val, type_name, topology, roles)
            elif isinstance(prop_val, list):
                for item in prop_val:
                    _walk_elements(item, type_name, topology, roles)
            else:
                text = prop_val if isinstance(prop_val, str) else json.dumps(prop_val)
                for placeholder in _PLACEHOLDER_RE.findall(str(text)):
                    roles[(type_name, placeholder)] += 1
        for child in value.get("children") or ():
            _walk_elements(child, type_name, topology, roles)
    elif isinstance(value, list):
        for item in value:
            _walk_elements(item, parent_type, topology, roles)


def _statement_ref_edges(bindings: dict[str, Any]) -> Counter[tuple[str, str]]:
    edges: Counter[tuple[str, str]] = Counter()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "ref" and isinstance(node.get("name"), str):
                return
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    for owner, rhs in bindings.items():
        visit(rhs)

    def collect(node: Any, owner: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "ref" and isinstance(node.get("name"), str):
                edges[(owner, node["name"])] += 1
                return
            for value in node.values():
                collect(value, owner)
        elif isinstance(node, list):
            for item in node:
                collect(item, owner)

    for owner, rhs in bindings.items():
        collect(rhs, owner)
    return edges


def _statement_component_types(bindings: dict[str, Any]) -> dict[str, str]:
    types: dict[str, str] = {}

    def find(node: Any) -> str | None:
        if isinstance(node, dict):
            if node.get("type") == "element":
                return str(node.get("typeName") or "element")
            for value in node.values():
                found = find(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = find(item)
                if found:
                    return found
        return None

    for name, rhs in bindings.items():
        t = find(rhs)
        if t:
            types[name] = t
    return types


def _interaction_dependency_edges(
    bindings: dict[str, Any]
) -> Counter[tuple[str, str]]:
    ref_edges = _statement_ref_edges(bindings)
    stmt_types = _statement_component_types(bindings)
    edges: Counter[tuple[str, str]] = Counter()
    for owner, target in ref_edges:
        owner_type = stmt_types.get(owner, "")
        target_type = stmt_types.get(target, "")
        if owner_type in _INTERACTIVE_TYPES or target_type in _INTERACTIVE_TYPES:
            edges[(owner_type, target_type)] += 1
    return edges


def _render_tree_distance(pred_root: Any, gold_root: Any) -> float:
    pred_tree = _semantic_tree(pred_root)
    gold_tree = _semantic_tree(gold_root)
    if pred_tree is None or gold_tree is None:
        return 0.0
    distance = _tree_distance(pred_tree, gold_tree)
    denom = max(pred_tree.size, gold_tree.size, 1)
    return max(0.0, 1.0 - distance / denom)


def _visual_similarity(
    pred: str, gold: str, dsl: str | None = None
) -> tuple[float | None, Literal["not_available", "available"]]:
    """Capture and compare screenshots when Playwright/chromium/PIL/numpy exist."""
    try:
        from PIL import Image  # noqa: F401
        import numpy as np  # noqa: F401
        from playwright.sync_api import sync_playwright  # noqa: F401
        from slm_training.data.progspec.schema import ProgramSpec
    except Exception:
        return None, "not_available"

    try:
        pred_spec = ProgramSpec.from_openui(
            id="render_equiv_pred",
            openui=pred,
            facts={},
            program_family_id="render_equivalence",
            lineage_id="render_equivalence",
            split_group_id="render_equivalence",
            split="train",
        )
        gold_spec = ProgramSpec.from_openui(
            id="render_equiv_gold",
            openui=gold,
            facts={},
            program_family_id="render_equivalence",
            lineage_id="render_equivalence",
            split_group_id="render_equivalence",
            split="train",
        )
        config = CaptureConfig(
            viewports=((390, 844),),
            themes=("light",),
            render_states=("populated",),
            interaction_states=("idle",),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir) / "pred"
            gold_dir = Path(tmpdir) / "gold"
            pred_caps = capture_program(pred_spec, output_dir=pred_dir, config=config)
            gold_caps = capture_program(gold_spec, output_dir=gold_dir, config=config)
            pred_path = pred_dir / pred_caps[0].full_page_screenshot
            gold_path = gold_dir / gold_caps[0].full_page_screenshot
            pred_img = Image.open(pred_path).convert("RGB")
            gold_img = Image.open(gold_path).convert("RGB")
            gold_img = gold_img.resize(pred_img.size)
            arr_pred = np.array(pred_img, dtype=float)
            arr_gold = np.array(gold_img, dtype=float)
            mse = float(np.mean((arr_pred - arr_gold) ** 2))
            similarity = 1.0 - mse / (255.0**2)
            return max(0.0, min(1.0, similarity)), "available"
    except Exception:
        return None, "not_available"


def render_equivalence(
    pred: str,
    gold: str,
    *,
    request: Any = None,
    mode: str = "diagnostic",
    dsl: str | None = None,
) -> RenderEquivalenceReport:
    """Return a tiered render-equivalence report for ``(pred, gold)`` OpenUI.

    ``mode`` is accepted for API symmetry but only ``"diagnostic"`` behavior is
    implemented; ``"selector_feature"`` and ``"reward"`` are rejected because
    this module is wiring/fixture-only.
    """
    if mode not in {"diagnostic", "selector_feature"}:
        raise ValueError(
            f"render_equivalence mode {mode!r} is not authorized; "
            "use 'diagnostic' or 'selector_feature'."
        )
    del request  # reserved for future request-aware contracts

    version_stamp = build_version_stamp("evals.render_equivalence")

    pred_root: Any = None
    gold_root: Any = None
    parse_error: str | None = None
    try:
        pred_root = _parse_root(pred, dsl=dsl)
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc).splitlines()[0][:240]
    try:
        gold_root = _parse_root(gold, dsl=dsl)
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc).splitlines()[0][:240]

    # Tier 0: canonical signature + binding graph.
    canonical_exact = False
    pred_fp: str | None = None
    gold_fp: str | None = None
    binding_graph_equal = False
    if pred_root is not None and gold_root is not None:
        try:
            canonical_exact = canonical_equal(pred, gold, dsl=dsl)
            pred_fp = canonical_fingerprint(pred, dsl=dsl)
            gold_fp = canonical_fingerprint(gold, dsl=dsl)
        except Exception:  # noqa: BLE001
            pass
        try:
            pred_bindings = _canonical_bindings(pred, dsl=dsl)
            gold_bindings = _canonical_bindings(gold, dsl=dsl)
            binding_graph_equal = (
                _statement_ref_edges(pred_bindings)
                == _statement_ref_edges(gold_bindings)
            )
        except Exception:  # noqa: BLE001
            pass

    tier0 = Tier0RenderEquivalenceReport(
        canonical_exact=canonical_exact,
        canonical_fingerprint_pred=pred_fp,
        canonical_fingerprint_gold=gold_fp,
        binding_graph_equal=binding_graph_equal,
    )

    # Tier 1: normalized render-tree subscores.
    subscores = {
        "component_type_match": 0.0,
        "role_match": 0.0,
        "topology_match": 0.0,
        "cardinality_match": 0.0,
        "binding_graph_match": 0.0,
        "interaction_dependency_match": 0.0,
        "normalized_render_tree_distance": 0.0,
    }
    tier1_error: str | None = None
    if pred_root is not None and gold_root is not None:
        try:
            pred_bindings = _canonical_bindings(pred, dsl=dsl)
            gold_bindings = _canonical_bindings(gold, dsl=dsl)
            pred_topology: Counter[tuple[str, str]] = Counter()
            gold_topology: Counter[tuple[str, str]] = Counter()
            pred_roles: Counter[tuple[str, str]] = Counter()
            gold_roles: Counter[tuple[str, str]] = Counter()
            _walk_elements(pred_root, topology=pred_topology, roles=pred_roles)
            _walk_elements(gold_root, topology=gold_topology, roles=gold_roles)

            pred_components = Counter(component_multiset(pred_root))
            gold_components = Counter(component_multiset(gold_root))

            pred_binding_edges = _statement_ref_edges(pred_bindings)
            gold_binding_edges = _statement_ref_edges(gold_bindings)
            pred_interaction = _interaction_dependency_edges(pred_bindings)
            gold_interaction = _interaction_dependency_edges(gold_bindings)

            total_pred = sum(pred_components.values())
            total_gold = sum(gold_components.values())
            cardinality = (
                min(total_pred, total_gold) / max(total_pred, total_gold, 1)
            )

            subscores = {
                "component_type_match": _multiset_f1(pred_components, gold_components),
                "role_match": _multiset_f1(pred_roles, gold_roles),
                "topology_match": _multiset_f1(pred_topology, gold_topology),
                "cardinality_match": cardinality,
                "binding_graph_match": _multiset_f1(
                    pred_binding_edges, gold_binding_edges
                ),
                "interaction_dependency_match": _multiset_f1(
                    pred_interaction, gold_interaction
                ),
                "normalized_render_tree_distance": _render_tree_distance(
                    pred_root, gold_root
                ),
            }
        except Exception as exc:  # noqa: BLE001
            tier1_error = str(exc).splitlines()[0][:240]

    tier1 = Tier1RenderEquivalenceReport(**subscores)

    # Tier 2: optional visual diff.
    visual_similarity, visual_status = _visual_similarity(pred, gold, dsl=dsl)
    tier2 = Tier2RenderEquivalenceReport(
        visual_similarity=visual_similarity, status=visual_status
    )

    # Aggregate (fail-closed).
    reasons: list[str] = []
    if parse_error is not None:
        reasons.append("parser_failure")
    if not canonical_exact:
        reasons.append("canonical_mismatch")
    if not binding_graph_equal:
        reasons.append("binding_graph_mismatch")
    if subscores["component_type_match"] < 1.0 - 1e-9:
        reasons.append("component_type_mismatch")
    if subscores["role_match"] < 1.0 - 1e-9:
        reasons.append("role_mismatch")
    if subscores["topology_match"] < 1.0 - 1e-9:
        reasons.append("topology_mismatch")
    if subscores["cardinality_match"] < 1.0 - 1e-9:
        reasons.append("cardinality_mismatch")
    if subscores["binding_graph_match"] < 1.0 - 1e-9:
        reasons.append("binding_graph_submatch_mismatch")
    if subscores["interaction_dependency_match"] < 1.0 - 1e-9:
        reasons.append("interaction_dependency_mismatch")
    if subscores["normalized_render_tree_distance"] < 1.0 - 1e-9:
        reasons.append("render_tree_distance_mismatch")
    if tier1_error is not None:
        reasons.append("render_tree_analysis_unavailable")
    if visual_status == "not_available":
        reasons.append("visual_not_available")

    equivalent = (
        parse_error is None
        and canonical_exact
        and binding_graph_equal
        and all(
            subscores[name] >= 1.0 - 1e-9
            for name in (
                "component_type_match",
                "role_match",
                "topology_match",
                "cardinality_match",
                "binding_graph_match",
                "interaction_dependency_match",
                "normalized_render_tree_distance",
            )
        )
    )

    return RenderEquivalenceReport(
        tier0=tier0,
        tier1=tier1,
        tier2=tier2,
        equivalent=equivalent,
        reason_codes=tuple(dict.fromkeys(reasons)),
        version_stamp=version_stamp,
    )


__all__ = [
    "RenderEquivalenceReport",
    "Tier0RenderEquivalenceReport",
    "Tier1RenderEquivalenceReport",
    "Tier2RenderEquivalenceReport",
    "render_equivalence",
]
