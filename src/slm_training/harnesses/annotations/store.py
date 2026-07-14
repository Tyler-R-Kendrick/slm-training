"""Pluggable durable storage for playground annotations."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.request import Request, urlopen

from slm_training.harnesses.annotations import (
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_GENERATION_ATTEMPTS_PATH,
    DEFAULT_HUMAN_PAIRS_PATH,
    DEFAULT_HUMAN_TRAIN_PATH,
    AnnotationRecord,
    GenerationAttemptRecord,
    append_generation_attempt,
    count_annotations,
    persist_annotation,
    recent_annotations,
)


class AnnotationStorageError(RuntimeError):
    """Raised when an annotation cannot be durably stored or retrieved."""


@dataclass(frozen=True)
class AnnotationPersistence:
    path: str
    backend: str
    human_train_path: str | None = None
    preference_pair: dict[str, Any] | None = None


class AnnotationStore(Protocol):
    backend: str

    def persist(self, record: AnnotationRecord) -> AnnotationPersistence: ...

    def persist_generation_attempt(
        self, record: GenerationAttemptRecord
    ) -> AnnotationPersistence: ...

    def recent(self, limit: int = 20) -> list[AnnotationRecord]: ...

    def count(self) -> int: ...


class FileAnnotationStore:
    """Local JSONL store used by the CLI and local playground."""

    backend = "filesystem"

    def __init__(
        self,
        feedback_path: Path = DEFAULT_FEEDBACK_PATH,
        human_train_path: Path = DEFAULT_HUMAN_TRAIN_PATH,
        pairs_path: Path = DEFAULT_HUMAN_PAIRS_PATH,
        generation_attempts_path: Path = DEFAULT_GENERATION_ATTEMPTS_PATH,
    ) -> None:
        self.feedback_path = Path(feedback_path)
        self.human_train_path = Path(human_train_path)
        self.pairs_path = Path(pairs_path)
        self.generation_attempts_path = Path(generation_attempts_path)

    def persist(self, record: AnnotationRecord) -> AnnotationPersistence:
        path, human_path, pair = persist_annotation(
            record,
            feedback_path=self.feedback_path,
            human_train_path=self.human_train_path,
            pairs_path=self.pairs_path,
        )
        return AnnotationPersistence(
            path=str(path),
            backend=self.backend,
            human_train_path=str(human_path) if human_path else None,
            preference_pair=pair.to_dict() if pair else None,
        )

    def recent(self, limit: int = 20) -> list[AnnotationRecord]:
        return recent_annotations(self.feedback_path, limit=limit)

    def persist_generation_attempt(
        self, record: GenerationAttemptRecord
    ) -> AnnotationPersistence:
        path = append_generation_attempt(self.generation_attempts_path, record)
        return AnnotationPersistence(path=str(path), backend=self.backend)

    def count(self) -> int:
        return count_annotations(self.feedback_path)


class VercelBlobAnnotationStore:
    """Private Vercel Blob store with one immutable object per annotation."""

    backend = "vercel-blob"

    def __init__(
        self,
        token: str,
        *,
        prefix: str = "annotations/v1/",
        generation_prefix: str = "generation-attempts/v1/",
        client: Any | None = None,
        list_fn: Callable[..., Any] | None = None,
        read_url: Callable[[str], bytes] | None = None,
    ) -> None:
        if not token.strip():
            raise AnnotationStorageError("BLOB_READ_WRITE_TOKEN is required")
        self.token = token
        self.prefix = prefix.rstrip("/") + "/"
        self.generation_prefix = generation_prefix.rstrip("/") + "/"
        if client is None or list_fn is None:
            try:
                from vercel.blob import BlobClient, list_objects
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise AnnotationStorageError("the Vercel Blob SDK is not installed") from exc
            client = client or BlobClient(token=self.token)
            list_fn = list_fn or list_objects
        self._client = client
        self._list = list_fn
        self._read_url = read_url or self._read_private_url

    def persist(self, record: AnnotationRecord) -> AnnotationPersistence:
        pathname = f"{self.prefix}{record.id}.json"
        payload = json.dumps(record.to_dict(), ensure_ascii=False).encode("utf-8")
        try:
            result = self._client.put(
                pathname,
                payload,
                access="private",
                content_type="application/json",
                add_random_suffix=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise AnnotationStorageError(f"Vercel Blob write failed: {exc}") from exc
        return AnnotationPersistence(
            path=str(getattr(result, "pathname", pathname)),
            backend=self.backend,
        )

    def persist_generation_attempt(
        self, record: GenerationAttemptRecord
    ) -> AnnotationPersistence:
        pathname = f"{self.generation_prefix}{record.id}.json"
        payload = json.dumps(record.to_dict(), ensure_ascii=False).encode("utf-8")
        try:
            result = self._client.put(
                pathname,
                payload,
                access="private",
                content_type="application/json",
                add_random_suffix=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise AnnotationStorageError(
                f"Vercel Blob generation-attempt write failed: {exc}"
            ) from exc
        return AnnotationPersistence(
            path=str(getattr(result, "pathname", pathname)),
            backend=self.backend,
        )

    def _list_all(self) -> list[Any]:
        blobs: list[Any] = []
        cursor: str | None = None
        try:
            while True:
                page = self._list(
                    prefix=self.prefix,
                    limit=1000,
                    cursor=cursor,
                    token=self.token,
                )
                blobs.extend(page.blobs)
                if not page.has_more:
                    break
                cursor = page.cursor
        except Exception as exc:  # noqa: BLE001
            raise AnnotationStorageError(f"Vercel Blob listing failed: {exc}") from exc
        return blobs

    def _read_private_url(self, url: str) -> bytes:
        request = Request(url, headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urlopen(request, timeout=15) as response:  # noqa: S310
                return response.read()
        except Exception as exc:  # noqa: BLE001
            raise AnnotationStorageError(f"Vercel Blob read failed: {exc}") from exc

    def recent(self, limit: int = 20) -> list[AnnotationRecord]:
        blobs = sorted(
            self._list_all(),
            key=lambda blob: (
                str(getattr(blob, "uploaded_at", "")),
                str(getattr(blob, "pathname", "")),
            ),
        )
        selected = blobs[-limit:] if limit > 0 else blobs
        try:
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(selected)))) as pool:
                payloads = list(pool.map(lambda blob: self._read_url(blob.url), selected))
            return [AnnotationRecord.from_dict(json.loads(payload)) for payload in payloads]
        except AnnotationStorageError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AnnotationStorageError(f"invalid annotation in Vercel Blob: {exc}") from exc

    def count(self) -> int:
        return len(self._list_all())


class UnavailableAnnotationStore:
    """Fail-closed store for deployments missing durable storage configuration."""

    backend = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def _raise(self) -> None:
        raise AnnotationStorageError(self.reason)

    def persist(self, record: AnnotationRecord) -> AnnotationPersistence:
        self._raise()
        raise AssertionError("unreachable")

    def persist_generation_attempt(
        self, record: GenerationAttemptRecord
    ) -> AnnotationPersistence:
        self._raise()
        raise AssertionError("unreachable")

    def recent(self, limit: int = 20) -> list[AnnotationRecord]:
        self._raise()
        raise AssertionError("unreachable")

    def count(self) -> int:
        self._raise()
        raise AssertionError("unreachable")
