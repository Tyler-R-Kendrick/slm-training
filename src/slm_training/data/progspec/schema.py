"""Typed program roots with split-safe projection to training records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from slm_training.dsl.language_contract import contract_id as current_contract_id
from slm_training.dsl.opaque_regions import OpaqueRegion
from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ALLOWED_SPLITS, TASK_TOKENS, ExampleRecord

_CONTRACT_ID_RE = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True)
class ProgramSpec:
    """Canonical root object from which prompts, edits, and renders derive."""

    id: str
    ast: dict[str, Any]
    canonical_openui: str
    facts: dict[str, Any]
    contract_id: str
    program_family_id: str
    lineage_id: str
    split_group_id: str
    split: str = "train"
    derivative_refs: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    opaque_regions: tuple[OpaqueRegion, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "id",
            "canonical_openui",
            "program_family_id",
            "lineage_id",
            "split_group_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(
                f"invalid split {self.split!r}; expected one of {sorted(ALLOWED_SPLITS)}"
            )
        if not _CONTRACT_ID_RE.fullmatch(self.contract_id):
            raise ValueError("contract_id must be 16 lowercase hex characters")
        if not isinstance(self.ast, dict) or not isinstance(self.facts, dict):
            raise ValueError("ast and facts must be dictionaries")

    @classmethod
    def from_openui(
        cls,
        *,
        id: str,
        openui: str,
        facts: Mapping[str, Any],
        program_family_id: str,
        lineage_id: str,
        split_group_id: str,
        split: str = "train",
        derivative_refs: tuple[str, ...] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> ProgramSpec:
        program = validate(openui)
        return cls(
            id=id,
            ast=dict(program.root or {}),
            canonical_openui=program.serialized or openui.strip(),
            facts=dict(facts),
            contract_id=current_contract_id(),
            program_family_id=program_family_id,
            lineage_id=lineage_id,
            split_group_id=split_group_id,
            split=split,
            derivative_refs=tuple(derivative_refs),
            provenance=dict(provenance or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["derivative_refs"] = list(self.derivative_refs)
        data["opaque_regions"] = [region.to_dict() for region in self.opaque_regions]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProgramSpec:
        return cls(
            id=str(data["id"]),
            ast=dict(data["ast"]),
            canonical_openui=str(data["canonical_openui"]),
            facts=dict(data.get("facts") or {}),
            contract_id=str(data["contract_id"]),
            program_family_id=str(data["program_family_id"]),
            lineage_id=str(data["lineage_id"]),
            split_group_id=str(data["split_group_id"]),
            split=str(data.get("split") or "train"),
            derivative_refs=tuple(str(v) for v in data.get("derivative_refs") or ()),
            provenance=dict(data.get("provenance") or {}),
            opaque_regions=tuple(
                OpaqueRegion.from_dict(region)
                for region in data.get("opaque_regions", ())
            ),
        )


RecordVerifier = Callable[[ExampleRecord], object]


def emit_record(
    spec: ProgramSpec,
    *,
    prompt: str,
    task: str,
    openui: str | None = None,
    record_id: str | None = None,
    parent_id: str | None = None,
    source: str = "programspec_generated",
    abstraction_level: str | int | None = None,
    determinacy: str = "deterministic",
    tier: str = "Silver",
    provenance: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    verifier: RecordVerifier | None = None,
) -> ExampleRecord:
    """Project one derivative while preserving its root's split identity."""
    if task not in TASK_TOKENS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASK_TOKENS)}")
    active_contract_id = current_contract_id()
    if spec.contract_id != active_contract_id:
        raise ValueError(
            f"stale ProgramSpec contract {spec.contract_id}; "
            f"current contract is {active_contract_id}"
        )
    target = openui or spec.canonical_openui
    program = validate(target)
    canonical = program.serialized or target.strip()
    parent = parent_id or spec.id
    if record_id is None:
        payload = json.dumps(
            [spec.id, parent, task, prompt, canonical],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        record_id = f"{spec.id}_{task}_{digest}"

    record_meta = {
        **dict(meta or {}),
        "contract_id": spec.contract_id,
        "program_family_id": spec.program_family_id,
        "lineage_id": spec.lineage_id,
        "split_group_id": spec.split_group_id,
        "task": task,
        "determinacy": determinacy,
        "tier": tier,
        "provenance": {**spec.provenance, **dict(provenance or {})},
        "parent_id": parent,
    }
    if abstraction_level is not None:
        record_meta["abstraction_level"] = abstraction_level
    record = ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=canonical,
        placeholders=list(program.placeholders),
        split=spec.split,
        source=source,
        meta=record_meta,
    )
    if verifier is not None:
        verifier(record)
    return record
