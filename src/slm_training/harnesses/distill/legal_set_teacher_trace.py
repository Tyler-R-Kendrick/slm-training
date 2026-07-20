"""Teacher-trace manifest and fixture traces for legal-set distillation (SPV2-03).

These traces are synthetic evidence only: they carry deterministic teacher
logits/probs over compiler-legal action sets and the metadata needed to bind them
to a real scorer later. No external model is downloaded or run here.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - torch may be absent in minimal environments
    torch = None  # type: ignore[assignment]

from slm_training.harnesses.distill.legal_set_kl import (
    LegalSetDistillExample,
    _require_torch,
)

__all__ = [
    "LegalSetTeacherTrace",
    "TeacherTraceManifest",
    "build_teacher_trace_fixture",
    "load_teacher_trace_manifest",
    "load_teacher_traces",
    "trace_to_distill_examples",
    "write_teacher_trace_manifest",
    "write_teacher_traces",
]

_SCHEMA_VERSION = "spv2-03.v1"


@dataclass(frozen=True)
class TeacherTraceManifest:
    """Provenance envelope for a set of legal-set teacher traces."""

    manifest_id: str
    teacher_model_id: str
    teacher_revision: str
    prompt_template_hash: str
    pack_id: str
    compiler_version: str
    state_schema_version: str
    timestamp: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "teacher_model_id": self.teacher_model_id,
            "teacher_revision": self.teacher_revision,
            "prompt_template_hash": self.prompt_template_hash,
            "pack_id": self.pack_id,
            "compiler_version": self.compiler_version,
            "state_schema_version": self.state_schema_version,
            "timestamp": self.timestamp,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeacherTraceManifest":
        return cls(**payload)


@dataclass(frozen=True)
class LegalSetTeacherTrace:
    """One synthetic legal-set teacher scoring decision."""

    trace_id: str
    manifest_id: str
    state_id: str
    prompt_hash: str
    prefix_ids: tuple[int, ...]
    legal_action_ids: tuple[int, ...]
    teacher_logits: tuple[float, ...] | None
    teacher_probs: tuple[float, ...] | None
    accepted_action_ids: tuple[int, ...]
    source: str
    coverage: str
    approximate: bool
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "manifest_id": self.manifest_id,
            "state_id": self.state_id,
            "prompt_hash": self.prompt_hash,
            "prefix_ids": list(self.prefix_ids),
            "legal_action_ids": list(self.legal_action_ids),
            "teacher_logits": (
                list(self.teacher_logits) if self.teacher_logits is not None else None
            ),
            "teacher_probs": (
                list(self.teacher_probs) if self.teacher_probs is not None else None
            ),
            "accepted_action_ids": list(self.accepted_action_ids),
            "source": self.source,
            "coverage": self.coverage,
            "approximate": self.approximate,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LegalSetTeacherTrace":
        return cls(
            trace_id=payload["trace_id"],
            manifest_id=payload["manifest_id"],
            state_id=payload["state_id"],
            prompt_hash=payload["prompt_hash"],
            prefix_ids=tuple(payload["prefix_ids"]),
            legal_action_ids=tuple(payload["legal_action_ids"]),
            teacher_logits=(
                tuple(float(x) for x in payload["teacher_logits"])
                if payload.get("teacher_logits") is not None
                else None
            ),
            teacher_probs=(
                tuple(float(x) for x in payload["teacher_probs"])
                if payload.get("teacher_probs") is not None
                else None
            ),
            accepted_action_ids=tuple(payload.get("accepted_action_ids", [])),
            source=payload["source"],
            coverage=payload["coverage"],
            approximate=bool(payload["approximate"]),
            provenance=dict(payload.get("provenance", {})),
        )


def build_teacher_trace_fixture(
    n_states: int = 16,
    vocab_size: int = 32,
    seed: int = 0,
) -> tuple[TeacherTraceManifest, list[LegalSetTeacherTrace]]:
    """Return a deterministic fixture manifest and synthetic teacher traces."""
    rng = random.Random(seed)
    manifest_id = f"manifest-spv2-03-fixture-{seed}"
    manifest = TeacherTraceManifest(
        manifest_id=manifest_id,
        teacher_model_id="fixture/teacher",
        teacher_revision="fixture",
        prompt_template_hash=hashlib.sha256(b"fixture-prompt-template").hexdigest(),
        pack_id="pack-spv2-03-fixture",
        compiler_version="openui-fixture",
        state_schema_version=_SCHEMA_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        provenance={
            "source": "synthetic",
            "seed": seed,
            "n_states": n_states,
            "vocab_size": vocab_size,
        },
    )

    traces: list[LegalSetTeacherTrace] = []
    for i in range(n_states):
        prefix_len = rng.randint(1, 6)
        prefix_ids = tuple(rng.randrange(vocab_size) for _ in range(prefix_len))
        legal_size = rng.randint(1, min(8, vocab_size))
        legal_action_ids = tuple(sorted(rng.sample(range(vocab_size), legal_size)))

        raw_logits = tuple(float(rng.gauss(0.0, 1.0)) for _ in range(legal_size))
        teacher_logits: tuple[float, ...] | None = raw_logits
        teacher_probs: tuple[float, ...] | None = None

        # A quarter of the rows are stored as normalized probabilities instead
        # of logits to exercise the teacher_is_prob path.
        if rng.random() < 0.25:
            mx = max(raw_logits)
            exps = [math.exp(v - mx) for v in raw_logits]
            total = sum(exps)
            teacher_probs = tuple(e / total for e in exps)
            teacher_logits = None

        accepted_count = rng.randint(0, legal_size)
        accepted_action_ids = (
            tuple(rng.sample(legal_action_ids, accepted_count))
            if accepted_count
            else ()
        )

        traces.append(
            LegalSetTeacherTrace(
                trace_id=f"trace-{i:03d}",
                manifest_id=manifest_id,
                state_id=f"state-{i:03d}",
                prompt_hash=hashlib.sha256(f"prompt-{i}".encode()).hexdigest(),
                prefix_ids=prefix_ids,
                legal_action_ids=legal_action_ids,
                teacher_logits=teacher_logits,
                teacher_probs=teacher_probs,
                accepted_action_ids=accepted_action_ids,
                source="fixture",
                coverage="complete",
                approximate=(i % 4 == 0),
                provenance={"state_index": i, "seed": seed},
            )
        )

    return manifest, traces


def trace_to_distill_examples(
    traces: list[LegalSetTeacherTrace],
) -> list[LegalSetDistillExample]:
    """Convert teacher traces into distillation examples."""
    _require_torch()
    examples: list[LegalSetDistillExample] = []
    for trace in traces:
        if trace.teacher_logits is not None:
            teacher_logits = torch.tensor(
                trace.teacher_logits, dtype=torch.float32
            )
            teacher_probs = None
        elif trace.teacher_probs is not None:
            teacher_probs = torch.tensor(trace.teacher_probs, dtype=torch.float32)
            teacher_logits = None
        else:
            raise ValueError(
                f"trace {trace.trace_id!r} has neither teacher_logits nor teacher_probs"
            )
        examples.append(
            LegalSetDistillExample(
                state_id=trace.state_id,
                legal_action_ids=trace.legal_action_ids,
                teacher_logits=teacher_logits,
                teacher_probs=teacher_probs,
                accepted_action_ids=trace.accepted_action_ids,
                source=trace.source,
                coverage=trace.coverage,
            )
        )
    return examples


def write_teacher_trace_manifest(path: str | Path, manifest: TeacherTraceManifest) -> None:
    """Write a manifest to JSON."""
    Path(path).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_teacher_trace_manifest(path: str | Path) -> TeacherTraceManifest:
    """Load a manifest from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TeacherTraceManifest.from_dict(payload)


def write_teacher_traces(path: str | Path, traces: list[LegalSetTeacherTrace]) -> None:
    """Write traces as JSONL."""
    handle = Path(path).open("w", encoding="utf-8")
    for trace in traces:
        handle.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")
    handle.close()


def load_teacher_traces(path: str | Path) -> list[LegalSetTeacherTrace]:
    """Load traces from JSONL."""
    traces: list[LegalSetTeacherTrace] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        traces.append(LegalSetTeacherTrace.from_dict(json.loads(line)))
    return traces
