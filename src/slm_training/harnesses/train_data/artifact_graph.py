"""Append-only dataset-sidecar derivation graph and leakage firewall."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

SCHEMA_VERSION = "artifact_graph_sidecar/v1"


class OverlapCode(str, Enum):
    FAMILY = "cross_split_family"
    PARENT = "cross_split_parent"
    EXACT = "cross_split_exact"
    ALPHA = "cross_split_alpha_equivalent"
    CANONICAL_AST = "cross_split_canonical_ast"
    NEAR_TEMPLATE = "cross_split_near_template"


@dataclass(frozen=True)
class ArtifactNodeV1:
    artifact_id: str
    artifact_type: str
    root_family_id: str
    split_group_id: str
    split: str
    parent_ids: tuple[str, ...]
    surface_sha256: str
    alpha_sha256: str
    canonical_ast_sha256: str
    near_template_sha256: str
    payload: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported graph schema {self.schema_version!r}")
        for name in (
            "artifact_id",
            "surface_sha256",
            "alpha_sha256",
            "canonical_ast_sha256",
            "near_template_sha256",
        ):
            _require_digest(getattr(self, name), name)
        if not self.artifact_type:
            raise ValueError("artifact_type is required")
        RootFamilySplitPolicyV1().require_inherited(
            root_family_id=self.root_family_id,
            split_group_id=self.split_group_id,
            split=self.split,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "root_family_id": self.root_family_id,
            "split_group_id": self.split_group_id,
            "split": self.split,
            "parent_ids": sorted(self.parent_ids),
            "surface_sha256": self.surface_sha256,
            "alpha_sha256": self.alpha_sha256,
            "canonical_ast_sha256": self.canonical_ast_sha256,
            "near_template_sha256": self.near_template_sha256,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class OverlapFindingV1:
    code: OverlapCode
    left_id: str
    right_id: str
    left_split: str
    right_split: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_split": self.left_split,
            "right_split": self.right_split,
            "evidence": self.evidence,
        }


class ArtifactGraphStore:
    def __init__(self, dataset_root: Path) -> None:
        self.root = dataset_root / "artifact_graph"
        self.records = self.root / "records"
        self.quarantine = self.root / "quarantine"

    def append(self, node: ArtifactNodeV1) -> Path:
        existing = self.load_nodes()
        parent_splits = tuple(
            existing[parent].split for parent in node.parent_ids if parent in existing
        )
        missing = sorted(set(node.parent_ids) - set(existing))
        if missing:
            return self.quarantine_node(
                node, ("unresolved_parent",), {"missing_parent_ids": missing}
            )
        RootFamilySplitPolicyV1().require_inherited(
            root_family_id=node.root_family_id,
            split_group_id=node.split_group_id,
            split=node.split,
            parent_splits=parent_splits,
        )
        findings = find_cross_split_overlaps((*existing.values(), node))
        blocking = tuple(
            finding
            for finding in findings
            if node.artifact_id
            in {
                finding.left_id,
                finding.right_id,
            }
        )
        if blocking:
            return self.quarantine_node(
                node,
                tuple(sorted({item.code.value for item in blocking})),
                {"findings": [item.to_dict() for item in blocking]},
            )
        return _write_idempotent(
            self.records / f"{node.artifact_id}.json", node.to_dict()
        )

    def quarantine_node(
        self,
        node: ArtifactNodeV1,
        reason_codes: tuple[str, ...],
        detail: dict[str, Any],
    ) -> Path:
        if not reason_codes:
            raise ValueError("quarantine requires reason codes")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact": node.to_dict(),
            "reason_codes": sorted(set(reason_codes)),
            "detail": detail,
        }
        return _write_idempotent(
            self.quarantine / f"{content_sha(payload)}.json", payload
        )

    def load_nodes(self) -> dict[str, ArtifactNodeV1]:
        nodes: dict[str, ArtifactNodeV1] = {}
        for path in sorted(self.records.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            value["parent_ids"] = tuple(value["parent_ids"])
            node = ArtifactNodeV1(**value)
            nodes[node.artifact_id] = node
        return nodes

    def ancestors(self, artifact_id: str) -> tuple[str, ...]:
        nodes = self.load_nodes()
        if artifact_id not in nodes:
            raise KeyError(f"unknown artifact {artifact_id}")
        seen: set[str] = set()
        frontier = list(nodes[artifact_id].parent_ids)
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(nodes[current].parent_ids)
        return tuple(sorted(seen))

    def explain_overlaps(self) -> tuple[OverlapFindingV1, ...]:
        findings = list(find_cross_split_overlaps(self.load_nodes().values()))
        for path in sorted(self.quarantine.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            for row in value.get("detail", {}).get("findings", ()):
                findings.append(
                    OverlapFindingV1(
                        code=OverlapCode(row["code"]),
                        left_id=str(row["left_id"]),
                        right_id=str(row["right_id"]),
                        left_split=str(row["left_split"]),
                        right_split=str(row["right_split"]),
                        evidence=str(row["evidence"]),
                    )
                )
        unique = {canonical_finding(item): item for item in findings}
        return tuple(unique[key] for key in sorted(unique))


def find_cross_split_overlaps(
    nodes: Iterable[ArtifactNodeV1],
) -> tuple[OverlapFindingV1, ...]:
    rows = tuple(sorted(nodes, key=lambda item: item.artifact_id))
    findings: list[OverlapFindingV1] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left.split == right.split:
                continue
            checks = (
                (OverlapCode.FAMILY, left.root_family_id, right.root_family_id),
                (OverlapCode.EXACT, left.surface_sha256, right.surface_sha256),
                (OverlapCode.ALPHA, left.alpha_sha256, right.alpha_sha256),
                (
                    OverlapCode.CANONICAL_AST,
                    left.canonical_ast_sha256,
                    right.canonical_ast_sha256,
                ),
                (
                    OverlapCode.NEAR_TEMPLATE,
                    left.near_template_sha256,
                    right.near_template_sha256,
                ),
            )
            for code, left_value, right_value in checks:
                if left_value and left_value == right_value:
                    findings.append(
                        OverlapFindingV1(
                            code=code,
                            left_id=left.artifact_id,
                            right_id=right.artifact_id,
                            left_split=left.split,
                            right_split=right.split,
                            evidence=left_value,
                        )
                    )
            shared_parents = sorted(set(left.parent_ids) & set(right.parent_ids))
            if shared_parents:
                findings.append(
                    OverlapFindingV1(
                        code=OverlapCode.PARENT,
                        left_id=left.artifact_id,
                        right_id=right.artifact_id,
                        left_split=left.split,
                        right_split=right.split,
                        evidence=",".join(shared_parents),
                    )
                )
    return tuple(findings)


def canonical_finding(finding: OverlapFindingV1) -> tuple[str, ...]:
    return (
        finding.code.value,
        finding.left_id,
        finding.right_id,
        finding.left_split,
        finding.right_split,
        finding.evidence,
    )


def _write_idempotent(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"content identity collision at {path}")
    return path


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be lowercase SHA-256")
