"""Durable annotation storage and authorization tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException

from slm_training.harnesses.annotations import AnnotationRecord, GenerationAttemptRecord
from slm_training.harnesses.annotations.store import (
    AnnotationPersistence,
    AnnotationStorageError,
    VercelBlobAnnotationStore,
)
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.web.app import AnnotateRequest, create_app


def annotation(record_id: str = "fb_test") -> AnnotationRecord:
    return AnnotationRecord(
        id=record_id,
        ts="2026-07-13T12:00:00Z",
        prompt="Render a card",
        openui='root = Card({title: "Hello"})',
        rating="up",
        valid=True,
    )


def test_vercel_blob_store_writes_private_immutable_objects() -> None:
    puts: list[tuple[tuple, dict]] = []

    class Client:
        def put(self, *args, **kwargs):
            puts.append((args, kwargs))
            return SimpleNamespace(pathname=args[0])

    page = SimpleNamespace(blobs=[], has_more=False, cursor=None)
    store = VercelBlobAnnotationStore(
        "blob-secret",
        client=Client(),
        list_fn=lambda **kwargs: page,
        read_url=lambda url: b"{}",
    )
    saved = store.persist(annotation())

    assert saved.backend == "vercel-blob"
    assert saved.path == "annotations/v1/fb_test.json"
    args, kwargs = puts[0]
    assert args[0] == "annotations/v1/fb_test.json"
    assert json.loads(args[1])["id"] == "fb_test"
    assert kwargs == {
        "access": "private",
        "content_type": "application/json",
        "add_random_suffix": False,
    }


def test_vercel_blob_store_persists_generation_attempts_separately() -> None:
    puts: list[tuple[tuple, dict]] = []

    class Client:
        def put(self, *args, **kwargs):
            puts.append((args, kwargs))
            return SimpleNamespace(pathname=args[0])

    store = VercelBlobAnnotationStore(
        "blob-secret",
        client=Client(),
        list_fn=lambda **kwargs: SimpleNamespace(
            blobs=[], has_more=False, cursor=None
        ),
    )
    record = GenerationAttemptRecord(
        id="gen_browser_2_test",
        ts="2026-07-13T12:00:00Z",
        prompt="Render a card",
        openui="broken",
        source="browser",
        attempt=2,
        valid=False,
        error="parse failed",
        prior_failures=["server attempt 1 failed"],
    )

    saved = store.persist_generation_attempt(record)

    assert saved.path == "generation-attempts/v1/gen_browser_2_test.json"
    args, kwargs = puts[0]
    assert json.loads(args[1])["prior_failures"] == ["server attempt 1 failed"]
    assert kwargs["access"] == "private"


def test_vercel_blob_store_reads_recent_and_counts() -> None:
    older = annotation("fb_old")
    newer = annotation("fb_new")
    payloads = {
        "private://old": json.dumps(older.to_dict()).encode(),
        "private://new": json.dumps(newer.to_dict()).encode(),
    }
    page = SimpleNamespace(
        blobs=[
            SimpleNamespace(
                pathname="annotations/v1/fb_new.json",
                url="private://new",
                uploaded_at="2026-07-13T12:01:00Z",
            ),
            SimpleNamespace(
                pathname="annotations/v1/fb_old.json",
                url="private://old",
                uploaded_at="2026-07-13T12:00:00Z",
            ),
        ],
        has_more=False,
        cursor=None,
    )
    store = VercelBlobAnnotationStore(
        "blob-secret",
        client=SimpleNamespace(),
        list_fn=lambda **kwargs: page,
        read_url=lambda url: payloads[url],
    )

    assert [record.id for record in store.recent(limit=1)] == ["fb_new"]
    assert store.count() == 2


class RecordingStore:
    backend = "recording"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.last_record: AnnotationRecord | None = None

    def persist(self, record: AnnotationRecord) -> AnnotationPersistence:
        self.calls += 1
        self.last_record = record
        if self.fail:
            raise AnnotationStorageError("durable write failed")
        return AnnotationPersistence(path=record.id, backend=self.backend)

    def recent(self, limit: int = 20) -> list[AnnotationRecord]:
        return []

    def count(self) -> int:
        return self.calls


def payload() -> dict:
    return {
        "prompt": "Render a card",
        "openui": 'root = Card({title: "Hello"})',
        "rating": "up",
        "valid": True,
    }


def annotate_endpoint(app):
    return next(route.endpoint for route in app.routes if route.path == "/api/annotate")


def test_deployment_auth_fails_closed_before_writing() -> None:
    store = RecordingStore()
    app = create_app(
        checkpoint=PLAYGROUND_DEMO_CHECKPOINT,
        annotation_token_required=True,
        annotation_store=store,
    )
    with pytest.raises(HTTPException) as exc_info:
        annotate_endpoint(app)(AnnotateRequest(**payload()), None)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "annotation authorization is not configured"
    assert store.calls == 0


def test_only_valid_bearer_token_reaches_durable_store() -> None:
    store = RecordingStore()
    app = create_app(
        checkpoint=PLAYGROUND_DEMO_CHECKPOINT,
        annotation_token="authorized",
        annotation_token_required=True,
        annotation_store=store,
    )
    endpoint = annotate_endpoint(app)
    request = AnnotateRequest(**payload())
    with pytest.raises(HTTPException) as exc_info:
        endpoint(request, None)
    accepted = endpoint(request, "Bearer authorized")

    assert exc_info.value.status_code == 401
    assert store.calls == 1
    assert accepted["storage"] == "recording"


def test_durable_write_failure_is_reported_as_unavailable() -> None:
    store = RecordingStore(fail=True)
    app = create_app(
        checkpoint=PLAYGROUND_DEMO_CHECKPOINT,
        annotation_token="authorized",
        annotation_store=store,
    )
    with pytest.raises(HTTPException) as exc_info:
        annotate_endpoint(app)(AnnotateRequest(**payload()), "Bearer authorized")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "durable write failed"


def test_human_correction_persists_original_and_all_identities() -> None:
    store = RecordingStore()
    app = create_app(
        checkpoint=PLAYGROUND_DEMO_CHECKPOINT,
        annotation_token="authorized",
        annotation_store=store,
    )
    endpoint = annotate_endpoint(app)
    original = 'root = Card([title])\ntitle = TextContent(":hero.title")'
    corrected = (
        'root = Card([title, body])\n'
        'title = TextContent(":hero.title")\n'
        'body = TextContent(":hero.body")'
    )
    request = AnnotateRequest(
        prompt="Hero card with title and body",
        openui=corrected,
        original_openui=original,
        human_corrected=True,
        generation_id="gen_srv_1_identity",
        rating="up",
        valid=True,
        session_id="session-1",
        identities={
            "request_generator": {
                "kind": "system",
                "provider": "slm-training",
                "id": "prompt-bank-composer:v1",
                "model": "prompt-bank-composer",
            },
            "output_generator": {
                "kind": "model",
                "provider": "slm-training",
                "id": "twotower:last.pt",
                "model": "twotower",
            },
            "annotator": {
                "kind": "user",
                "provider": "playground-annotator",
                "id": "alice@example.com",
                "display_name": "Alice",
            },
        },
    )

    result = endpoint(request, "Bearer authorized")

    assert result["human_corrected"] is True
    assert store.last_record is not None
    assert store.last_record.original_openui == original
    assert store.last_record.human_corrected is True
    assert store.last_record.generation_id == "gen_srv_1_identity"
    assert store.last_record.identities["annotator"]["id"] == "alice@example.com"
    assert store.last_record.identities["correction_author"]["id"] == "alice@example.com"
    assert store.last_record.identities["output_generator"]["model"] == "twotower"
