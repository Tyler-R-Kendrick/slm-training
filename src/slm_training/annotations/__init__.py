"""Human annotation records for playground thumbs + notes."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.preference import PreferencePair, load_pairs, write_pairs

Rating = Literal["up", "down"]

DEFAULT_FEEDBACK_PATH = Path("outputs/annotations/feedback.jsonl")
DEFAULT_HUMAN_TRAIN_PATH = Path("fixtures/annotations/human_train.jsonl")
DEFAULT_HUMAN_PAIRS_PATH = Path("outputs/preferences/human_pairs.jsonl")

_WRITE_LOCK = threading.Lock()


@dataclass
class AnnotationRecord:
    id: str
    ts: str
    prompt: str
    openui: str
    rating: Rating
    description: str | None = None
    design_md: str | None = None
    valid: bool | None = None
    checkpoint: str | None = None
    session_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("description") is None:
            data.pop("description", None)
        if data.get("design_md") is None:
            data.pop("design_md", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationRecord:
        rating = str(data.get("rating") or "").lower()
        if rating not in {"up", "down"}:
            raise ValueError(f"invalid rating {rating!r}")
        desc = data.get("description")
        design_md = data.get("design_md")
        return cls(
            id=str(data["id"]),
            ts=str(data.get("ts") or ""),
            prompt=str(data["prompt"]),
            openui=str(data["openui"]),
            rating=rating,  # type: ignore[arg-type]
            description=None if desc in (None, "") else str(desc),
            design_md=None if design_md in (None, "") else str(design_md),
            valid=data.get("valid"),
            checkpoint=None if data.get("checkpoint") is None else str(data["checkpoint"]),
            session_id=None if data.get("session_id") is None else str(data["session_id"]),
            meta=dict(data.get("meta") or {}),
        )


def new_annotation_id() -> str:
    return f"fb_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_annotation(path: Path | str, record: AnnotationRecord) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    return path


def load_annotations(path: Path | str) -> list[AnnotationRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[AnnotationRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(AnnotationRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return out


def recent_annotations(path: Path | str, limit: int = 20) -> list[AnnotationRecord]:
    rows = load_annotations(path)
    if limit <= 0:
        return rows
    return rows[-limit:]


def annotation_to_example(record: AnnotationRecord) -> ExampleRecord:
    placeholders = extract_placeholders(record.openui)
    return ExampleRecord(
        id=f"human_{record.id}",
        prompt=record.prompt.strip(),
        openui=record.openui.strip(),
        placeholders=placeholders,
        split="train",
        source="human",
        meta={
            "annotation_id": record.id,
            "rating": record.rating,
            "description": record.description,
            **dict(record.meta or {}),
        },
        design_md=record.design_md,
    )


def upsert_human_train_seed(
    record: AnnotationRecord,
    path: Path | str = DEFAULT_HUMAN_TRAIN_PATH,
) -> Path | None:
    """Promote a thumbs-up annotation into the human SFT seed file."""
    if record.rating != "up":
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = annotation_to_example(record)
    existing: list[ExampleRecord] = []
    if path.exists():
        existing = load_jsonl(path)
    # Replace same annotation id; otherwise append.
    kept = [r for r in existing if (r.meta or {}).get("annotation_id") != record.id]
    # Also skip exact openui+prompt duplicates.
    kept = [
        r
        for r in kept
        if not (r.prompt.strip() == example.prompt and r.openui.strip() == example.openui)
    ]
    kept.append(example)
    write_jsonl(path, kept)
    return path


def export_to_train_seeds(
    feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
    out_path: Path | str = DEFAULT_HUMAN_TRAIN_PATH,
) -> dict[str, Any]:
    rows = [r for r in load_annotations(feedback_path) if r.rating == "up"]
    # Keep latest per (prompt, openui).
    dedup: dict[tuple[str, str], AnnotationRecord] = {}
    for row in rows:
        dedup[(row.prompt.strip(), row.openui.strip())] = row
    examples = [annotation_to_example(r) for r in dedup.values()]
    examples.sort(key=lambda r: r.id)
    count = write_jsonl(out_path, examples)
    return {"count": count, "path": str(out_path)}


def export_to_preference_pairs(
    feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
    out_path: Path | str = DEFAULT_HUMAN_PAIRS_PATH,
) -> dict[str, Any]:
    rows = load_annotations(feedback_path)
    by_prompt: dict[str, list[AnnotationRecord]] = {}
    for row in rows:
        by_prompt.setdefault(row.prompt.strip(), []).append(row)

    pairs: list[PreferencePair] = []
    for prompt, group in by_prompt.items():
        ups = [r for r in group if r.rating == "up"]
        downs = [r for r in group if r.rating == "down"]
        if not ups or not downs:
            continue
        # Pair latest up with latest down that has different openui.
        chosen = ups[-1]
        rejected = next(
            (d for d in reversed(downs) if d.openui.strip() != chosen.openui.strip()),
            downs[-1],
        )
        if chosen.openui.strip() == rejected.openui.strip():
            continue
        pairs.append(
            PreferencePair(
                prompt=prompt,
                chosen=chosen.openui,
                rejected=rejected.openui,
                design_md=chosen.design_md or rejected.design_md,
                chosen_score=1.0,
                rejected_score=0.0,
                meta={
                    "source": "human",
                    "chosen_annotation_id": chosen.id,
                    "rejected_annotation_id": rejected.id,
                    "description": chosen.description or rejected.description,
                },
            )
        )
    count = write_pairs(out_path, pairs)
    return {"count": count, "path": str(out_path), "pairs": count}


def maybe_append_preference_pair(
    record: AnnotationRecord,
    feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
    pairs_path: Path | str = DEFAULT_HUMAN_PAIRS_PATH,
) -> PreferencePair | None:
    """If opposite rating exists for the same prompt, append one preference pair."""
    rows = load_annotations(feedback_path)
    opposite: Rating = "down" if record.rating == "up" else "up"
    match = next(
        (
            r
            for r in reversed(rows)
            if r.prompt.strip() == record.prompt.strip()
            and r.rating == opposite
            and r.openui.strip() != record.openui.strip()
        ),
        None,
    )
    if match is None:
        return None
    chosen = record if record.rating == "up" else match
    rejected = match if record.rating == "up" else record
    pair = PreferencePair(
        prompt=record.prompt.strip(),
        chosen=chosen.openui,
        rejected=rejected.openui,
        design_md=chosen.design_md or rejected.design_md,
        chosen_score=1.0,
        rejected_score=0.0,
        meta={
            "source": "human",
            "chosen_annotation_id": chosen.id,
            "rejected_annotation_id": rejected.id,
            "description": record.description or match.description,
        },
    )
    pairs_path = Path(pairs_path)
    existing = load_pairs(pairs_path) if pairs_path.exists() else []
    # Skip duplicate chosen/rejected for same prompt.
    for prev in existing:
        if (
            prev.prompt == pair.prompt
            and prev.chosen.strip() == pair.chosen.strip()
            and prev.rejected.strip() == pair.rejected.strip()
        ):
            return pair
    existing.append(pair)
    write_pairs(pairs_path, existing)
    return pair


def export_all(
    feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
    human_train_path: Path | str = DEFAULT_HUMAN_TRAIN_PATH,
    pairs_path: Path | str = DEFAULT_HUMAN_PAIRS_PATH,
) -> dict[str, Any]:
    return {
        "feedback_path": str(feedback_path),
        "feedback_count": len(load_annotations(feedback_path)),
        "train": export_to_train_seeds(feedback_path, human_train_path),
        "pairs": export_to_preference_pairs(feedback_path, pairs_path),
    }
