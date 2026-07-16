"""Canonical example record schema shared by all harnesses."""

from __future__ import annotations

import os
import tempfile
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal


ALLOWED_SPLITS = frozenset(
    {"train", "held_out", "smoke", "adversarial", "ood", "rico_held"}
)
TASK_TOKENS = frozenset(
    {
        "generation",
        "completion",
        "inpaint",
        "repair",
        "patch",
        "edit",
        "state",
        "behavior",
        "noop",
        "adversarial",
    }
)

OutputKind = Literal["document", "statement", "expression", "lexical"]
OUTPUT_KINDS = frozenset({"document", "statement", "expression", "lexical"})


@dataclass(frozen=True)
class OutputTarget:
    """One accepted output surface for a record."""

    text: str
    kind: OutputKind = "document"
    category: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("output target text must be non-empty")
        if self.kind not in OUTPUT_KINDS:
            raise ValueError(f"invalid output target kind {self.kind!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> OutputTarget:
        if isinstance(data, str):
            return cls(text=data)
        return cls(
            text=str(data["text"]),
            kind=str(data.get("kind") or "document"),  # type: ignore[arg-type]
            category=(
                None if data.get("category") is None else str(data["category"])
            ),
        )


@dataclass
class ExampleRecord:
    id: str
    prompt: str
    openui: str
    placeholders: list[str] = field(default_factory=list)
    split: str = "train"
    source: str = "fixture"
    meta: dict[str, Any] = field(default_factory=dict)
    design_md: str | None = None
    target_kind: OutputKind = "document"
    target_category: str | None = None
    accepted_outputs: list[OutputTarget] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(
                f"invalid split {self.split!r}; expected one of {sorted(ALLOWED_SPLITS)}"
            )
        if not self.id:
            raise ValueError("id must be non-empty")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if not self.openui:
            raise ValueError("openui must be non-empty")
        if self.target_kind not in OUTPUT_KINDS:
            raise ValueError(f"invalid target kind {self.target_kind!r}")
        if self.design_md is not None and not isinstance(self.design_md, str):
            raise ValueError("design_md must be a string or None")
        self.accepted_outputs = [
            item if isinstance(item, OutputTarget) else OutputTarget.from_dict(item)
            for item in self.accepted_outputs
        ]
        _validate_meta(self.meta)

    @property
    def output_targets(self) -> tuple[OutputTarget, ...]:
        return (
            OutputTarget(self.openui, self.target_kind, self.target_category),
            *self.accepted_outputs,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("design_md") is None:
            data.pop("design_md", None)
        if data.get("target_kind") == "document":
            data.pop("target_kind", None)
        if data.get("target_category") is None:
            data.pop("target_category", None)
        if not data.get("accepted_outputs"):
            data.pop("accepted_outputs", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleRecord:
        design_md = data.get("design_md")
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            openui=str(data["openui"]),
            placeholders=list(data.get("placeholders") or []),
            split=str(data.get("split") or "train"),
            source=str(data.get("source") or "fixture"),
            meta=dict(data.get("meta") or {}),
            design_md=None if design_md is None else str(design_md),
            target_kind=str(data.get("target_kind") or "document"),  # type: ignore[arg-type]
            target_category=(
                None
                if data.get("target_category") is None
                else str(data["target_category"])
            ),
            accepted_outputs=[
                OutputTarget.from_dict(item)
                for item in data.get("accepted_outputs") or ()
            ],
        )


def load_jsonl(path: Path | str) -> list[ExampleRecord]:
    path = Path(path)
    records: list[ExampleRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ExampleRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path | str, records: Iterable[ExampleRecord]) -> int:
    """Atomically replace a JSONL file with canonical records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return count


def iter_jsonl(path: Path | str) -> Iterator[ExampleRecord]:
    yield from load_jsonl(path)


def _validate_meta(meta: dict[str, Any]) -> None:
    """Validate optional lineage conventions without changing the wire shape."""
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dictionary")
    task = meta.get("task")
    if task is not None and task not in TASK_TOKENS:
        raise ValueError(f"invalid meta.task {task!r}")
    split_group_id = meta.get("split_group_id")
    if split_group_id is not None and not str(split_group_id).strip():
        raise ValueError("meta.split_group_id must be non-empty")
    provenance = meta.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("meta.provenance must be a dictionary")
    for key in ("determinacy", "tier"):
        if key in meta and not isinstance(meta[key], str):
            raise ValueError(f"meta.{key} must be a string")
    abstraction_level = meta.get("abstraction_level")
    if abstraction_level is not None and not isinstance(
        abstraction_level, (str, int)
    ):
        raise ValueError("meta.abstraction_level must be a string or integer")
    for key in ("edit", "repair", "frontier"):
        if key in meta and not isinstance(meta[key], dict):
            raise ValueError(f"meta.{key} must be a dictionary")
