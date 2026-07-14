"""Human annotation records for playground thumbs + notes."""

from __future__ import annotations

from contextlib import contextmanager
from collections import deque
import os
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.harnesses.preference import PreferencePair, write_pairs

Rating = Literal["up", "down"]

DEFAULT_FEEDBACK_PATH = Path("outputs/annotations/feedback.jsonl")
DEFAULT_HUMAN_TRAIN_PATH = Path("fixtures/annotations/human_train.jsonl")
DEFAULT_HUMAN_PAIRS_PATH = Path("outputs/preferences/human_pairs.jsonl")
DEFAULT_BAD_OUTPUTS_PATH = Path("outputs/annotations/bad_outputs.jsonl")
DEFAULT_GENERATION_ATTEMPTS_PATH = Path(
    "outputs/annotations/generation_attempts.jsonl"
)

_WRITE_LOCK = threading.RLock()


@contextmanager
def _annotation_transaction(path: Path | str) -> Iterator[None]:
    lock_path = Path(str(Path(path)) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - WSL/Linux is the supported host
                fcntl = None  # type: ignore[assignment]
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    generation_id: str | None = None
    original_openui: str | None = None
    human_corrected: bool = False
    identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("description") is None:
            data.pop("description", None)
        if data.get("design_md") is None:
            data.pop("design_md", None)
        if data.get("generation_id") is None:
            data.pop("generation_id", None)
        if data.get("original_openui") is None:
            data.pop("original_openui", None)
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
            checkpoint=None
            if data.get("checkpoint") is None
            else str(data["checkpoint"]),
            session_id=None
            if data.get("session_id") is None
            else str(data["session_id"]),
            generation_id=None
            if data.get("generation_id") is None
            else str(data["generation_id"]),
            original_openui=None
            if data.get("original_openui") in (None, "")
            else str(data["original_openui"]),
            human_corrected=bool(data.get("human_corrected")),
            identities={
                str(role): dict(identity)
                for role, identity in dict(data.get("identities") or {}).items()
            },
            meta=dict(data.get("meta") or {}),
        )


def new_annotation_id() -> str:
    return f"fb_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class GenerationAttemptRecord:
    """One model attempt, retained as raw supervised or negative training data."""

    id: str
    ts: str
    prompt: str
    openui: str
    source: Literal["server", "browser"]
    attempt: int
    valid: bool
    error: str | None = None
    prior_failures: list[str] = field(default_factory=list)
    design_md: str | None = None
    checkpoint: str | None = None
    session_id: str | None = None
    identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("error", "design_md", "checkpoint", "session_id"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationAttemptRecord:
        source = str(data.get("source") or "")
        if source not in {"server", "browser"}:
            raise ValueError(f"invalid generation source {source!r}")
        return cls(
            id=str(data["id"]),
            ts=str(data.get("ts") or ""),
            prompt=str(data["prompt"]),
            openui=str(data.get("openui") or ""),
            source=source,  # type: ignore[arg-type]
            attempt=int(data["attempt"]),
            valid=bool(data.get("valid")),
            error=None if data.get("error") in (None, "") else str(data["error"]),
            prior_failures=[str(item) for item in data.get("prior_failures") or []],
            design_md=None
            if data.get("design_md") in (None, "")
            else str(data["design_md"]),
            checkpoint=None
            if data.get("checkpoint") is None
            else str(data["checkpoint"]),
            session_id=None
            if data.get("session_id") is None
            else str(data["session_id"]),
            identities={
                str(role): dict(identity)
                for role, identity in dict(data.get("identities") or {}).items()
            },
            meta=dict(data.get("meta") or {}),
        )


def new_generation_attempt_id(source: str, attempt: int) -> str:
    prefix = "srv" if source == "server" else "browser"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"gen_{prefix}_{attempt}_{stamp}_{uuid.uuid4().hex[:8]}"


def append_generation_attempt(
    path: Path | str, record: GenerationAttemptRecord
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with _annotation_transaction(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return path


def load_generation_attempts(
    path: Path | str,
) -> list[GenerationAttemptRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[GenerationAttemptRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                out.append(GenerationAttemptRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return out


@dataclass
class BadOutputRecord:
    """Invalid model output quarantined for negative training examples."""

    id: str
    ts: str
    prompt: str
    openui: str
    error: str
    checkpoint: str | None = None
    session_id: str | None = None
    attempt: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("session_id") is None:
            data.pop("session_id", None)
        if data.get("attempt") is None:
            data.pop("attempt", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BadOutputRecord:
        return cls(
            id=str(data["id"]),
            ts=str(data.get("ts") or ""),
            prompt=str(data["prompt"]),
            openui=str(data["openui"]),
            error=str(data.get("error") or ""),
            checkpoint=None if data.get("checkpoint") is None else str(data["checkpoint"]),
            session_id=None if data.get("session_id") is None else str(data["session_id"]),
            attempt=None if data.get("attempt") is None else int(data["attempt"]),
            meta=dict(data.get("meta") or {}),
        )


def new_bad_output_id() -> str:
    return f"bad_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"


def append_bad_output(path: Path | str, record: BadOutputRecord) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with _annotation_transaction(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return path


def load_bad_outputs(path: Path | str) -> list[BadOutputRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[BadOutputRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(BadOutputRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return out


def append_annotation(path: Path | str, record: AnnotationRecord) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    return path


def iter_annotations(path: Path | str) -> Iterator[AnnotationRecord]:
    """Stream annotation rows with line-aware validation."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield AnnotationRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def load_annotations(path: Path | str) -> list[AnnotationRecord]:
    return list(iter_annotations(path))


def recent_annotations(path: Path | str, limit: int = 20) -> list[AnnotationRecord]:
    if limit <= 0:
        return load_annotations(path)
    rows: deque[AnnotationRecord] = deque(maxlen=limit)
    rows.extend(iter_annotations(path))
    return list(rows)


def count_annotations(path: Path | str) -> int:
    """Count non-empty annotation rows without materializing the file."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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
            "generation_id": record.generation_id,
            "human_corrected": record.human_corrected,
            "original_openui": record.original_openui,
            "identities": record.identities,
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
        if not (
            r.prompt.strip() == example.prompt and r.openui.strip() == example.openui
        )
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
                    "chosen_identities": chosen.identities,
                    "rejected_identities": rejected.identities,
                    "chosen_human_corrected": chosen.human_corrected,
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
    """Append a preference pair using streaming scans and an atomic transaction."""
    opposite: Rating = "down" if record.rating == "up" else "up"
    match: AnnotationRecord | None = None
    for candidate in iter_annotations(feedback_path):
        if (
            candidate.prompt.strip() == record.prompt.strip()
            and candidate.rating == opposite
            and candidate.openui.strip() != record.openui.strip()
        ):
            match = candidate
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
            "chosen_identities": chosen.identities,
            "rejected_identities": rejected.identities,
            "chosen_human_corrected": chosen.human_corrected,
        },
    )
    pairs_path = Path(pairs_path)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    if pairs_path.exists():
        with pairs_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                previous = json.loads(line)
                if (
                    previous.get("prompt") == pair.prompt
                    and str(previous.get("chosen") or "").strip() == pair.chosen.strip()
                    and str(previous.get("rejected") or "").strip()
                    == pair.rejected.strip()
                ):
                    return pair
    with _WRITE_LOCK:
        with pairs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return pair


def persist_annotation(
    record: AnnotationRecord,
    *,
    feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
    human_train_path: Path | str = DEFAULT_HUMAN_TRAIN_PATH,
    pairs_path: Path | str = DEFAULT_HUMAN_PAIRS_PATH,
) -> tuple[Path, Path | None, PreferencePair | None]:
    """Persist feedback and derived training artifacts as one transaction."""
    with _annotation_transaction(feedback_path):
        feedback = append_annotation(feedback_path, record)
        human = upsert_human_train_seed(record, human_train_path)
        pair = maybe_append_preference_pair(
            record, feedback_path=feedback_path, pairs_path=pairs_path
        )
    return feedback, human, pair


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
