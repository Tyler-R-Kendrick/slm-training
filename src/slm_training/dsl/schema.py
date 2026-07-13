"""Canonical example record schema shared by all harnesses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


ALLOWED_SPLITS = frozenset(
    {"train", "held_out", "smoke", "adversarial", "ood", "rico_held"}
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
        if self.design_md is not None and not isinstance(self.design_md, str):
            raise ValueError("design_md must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("design_md") is None:
            data.pop("design_md", None)
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path | str) -> Iterator[ExampleRecord]:
    yield from load_jsonl(path)
