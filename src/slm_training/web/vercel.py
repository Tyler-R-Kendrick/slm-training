"""Vercel FastAPI entrypoint for the annotation playground."""

from __future__ import annotations

import os
from pathlib import Path

import slm_training.web.service as playground_service
from slm_training.harnesses.annotations.store import (
    UnavailableAnnotationStore,
    VercelBlobAnnotationStore,
)
from slm_training.dsl.parser import (
    ParseError,
    stream_check as parser_stream_check,
    validate,
)
from slm_training.models.onnx_inference import OnnxTwoTowerModel
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.web.app import create_app


def _stream_check(source: str) -> dict:
    status = parser_stream_check(source)
    return {
        "ok": status.ok,
        "incomplete": status.incomplete,
        "has_root": status.has_root,
        "errors": list(status.error_codes),
        "unresolved": list(status.unresolved),
    }


playground_service.ParseError = ParseError
playground_service.stream_check = _stream_check
playground_service.validate = validate

runtime_root = Path("/tmp/slm-training")
blob_token = (os.getenv("BLOB_READ_WRITE_TOKEN") or "").strip()
if blob_token:
    annotation_store = VercelBlobAnnotationStore(blob_token)
else:
    annotation_store = UnavailableAnnotationStore(
        "durable annotation storage is not configured "
        "(missing BLOB_READ_WRITE_TOKEN)"
    )

app = create_app(
    checkpoint=PLAYGROUND_DEMO_CHECKPOINT,
    device="cpu",
    annotations_path=runtime_root / "annotations.jsonl",
    human_train_path=runtime_root / "human_train.jsonl",
    human_pairs_path=runtime_root / "human_pairs.jsonl",
    bad_outputs_path=runtime_root / "bad_outputs.jsonl",
    generation_attempts_path=runtime_root / "generation_attempts.jsonl",
    annotation_token=os.getenv("SLM_ANNOTATION_TOKEN"),
    annotation_token_required=True,
    annotation_store=annotation_store,
    model_factory=OnnxTwoTowerModel.from_checkpoint,
)
