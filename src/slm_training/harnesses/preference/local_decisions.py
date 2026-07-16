"""Strict exact-state decision events mined from decode traces."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


Split = Literal["train", "held_out"]
EvidenceKind = Literal["constraint_shadow", "counterfactual"]


def _ids(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def split_for_group(group_id: str) -> Split:
    """Stable 80/20 split; every event from one prompt family stays together."""
    return "held_out" if int(_sha(group_id)[:8], 16) % 5 == 0 else "train"


@dataclass(frozen=True)
class DecisionEventV1:
    event_id: str
    group_id: str
    context_text: str
    canvas_ids: tuple[int, ...]
    position: int
    good_token_ids: tuple[int, ...]
    bad_token_ids: tuple[int, ...]
    legal_token_ids: tuple[int, ...]
    evidence_kind: EvidenceKind
    evidence_confidence: float
    decision_kind: str
    split: Split
    policy_checkpoint_sha: str
    tokenizer_sha: str
    decode_config_hash: str
    seed: int
    trajectory_id: str
    source_suite: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "canvas_ids", tuple(map(int, self.canvas_ids)))
        for field in ("good_token_ids", "bad_token_ids", "legal_token_ids"):
            object.__setattr__(self, field, _ids(getattr(self, field)))
        if self.version != 1:
            raise ValueError("unsupported decision event version")
        if not self.event_id or not self.group_id or not self.context_text:
            raise ValueError("event_id, group_id, and context_text are required")
        if not 0 <= self.position < len(self.canvas_ids):
            raise ValueError("position must address canvas_ids")
        if not self.good_token_ids or not self.bad_token_ids:
            raise ValueError("good_token_ids and bad_token_ids must be non-empty")
        if set(self.good_token_ids) & set(self.bad_token_ids):
            raise ValueError("good and bad token sets must be disjoint")
        if not set(self.good_token_ids).issubset(self.legal_token_ids):
            raise ValueError("good token ids must be verifier-legal")
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1]")
        if self.split != split_for_group(self.group_id):
            raise ValueError("split must be derived from group_id")
        for value in (
            self.policy_checkpoint_sha,
            self.tokenizer_sha,
            self.decode_config_hash,
            self.trajectory_id,
        ):
            if not value:
                raise ValueError("event identity fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionEventV1":
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown decision event fields: {sorted(unknown)}")
        return cls(**value)


def _event(
    trace: dict[str, Any],
    payload: dict[str, Any],
    *,
    evidence_kind: EvidenceKind,
    ordinal: str,
) -> DecisionEventV1:
    meta = trace.get("meta") or {}
    group_id = str(meta.get("record_id") or _sha(str(meta.get("prompt") or "")))
    canvas = tuple(map(int, payload["pre_canvas"]))
    position = int(payload["position"])
    good = _ids(payload["good_token_ids"])
    bad = _ids(payload["bad_token_ids"])
    legal = _ids(payload["legal_token_ids"])
    identity = {
        "trajectory_id": trace.get("trajectory_id"),
        "ordinal": ordinal,
        "position": position,
        "canvas": canvas,
        "good": good,
        "bad": bad,
        "evidence": evidence_kind,
    }
    return DecisionEventV1(
        event_id=_sha(identity),
        group_id=group_id,
        context_text=str(meta.get("context_text") or ""),
        canvas_ids=canvas,
        position=position,
        good_token_ids=good,
        bad_token_ids=bad,
        legal_token_ids=legal,
        evidence_kind=evidence_kind,
        evidence_confidence=float(payload.get("evidence_confidence", 1.0)),
        decision_kind=str(payload.get("decision_kind") or evidence_kind),
        split=split_for_group(group_id),
        policy_checkpoint_sha=str(meta.get("policy_checkpoint_sha") or ""),
        tokenizer_sha=str(meta.get("tokenizer_sha") or ""),
        decode_config_hash=str(meta.get("decode_config_hash") or ""),
        seed=int(meta.get("seed", 0)),
        trajectory_id=str(trace.get("trajectory_id") or ""),
        source_suite=meta.get("source_suite"),
    )


def events_from_trace(trace: dict[str, Any]) -> list[DecisionEventV1]:
    """Mine exact constraint shadows and explicitly verified counterfactuals."""
    out: list[DecisionEventV1] = []
    for step_index, step in enumerate(trace.get("steps") or []):
        for commit_index, commit in enumerate(step.get("commits") or []):
            allowed = _ids(commit.get("allowed_id_set") or ())
            raw = commit.get("raw_id")
            chosen = commit.get("id")
            canvas = commit.get("pre_canvas")
            if not allowed or raw is None or chosen is None or canvas is None:
                continue
            if int(raw) in allowed or int(chosen) not in allowed:
                continue
            out.append(
                _event(
                    trace,
                    {
                        "pre_canvas": canvas,
                        "position": int(commit["t"]),
                        "good_token_ids": [chosen],
                        "bad_token_ids": [raw],
                        "legal_token_ids": allowed,
                        "decision_kind": "constraint_shadow",
                    },
                    evidence_kind="constraint_shadow",
                    ordinal=f"{step_index}:{commit_index}",
                )
            )
    for index, item in enumerate(trace.get("events") or []):
        if item.get("kind") != "counterfactual_decision":
            continue
        if item.get("same_state_verified") is not True:
            raise ValueError("counterfactual decision lacks same-state verification")
        out.append(
            _event(
                trace,
                item,
                evidence_kind="counterfactual",
                ordinal=f"counterfactual:{index}",
            )
        )
    return out


def load_trace_rows(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "traces.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_decision_events(path: Path | str) -> list[DecisionEventV1]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [DecisionEventV1.from_dict(json.loads(line)) for line in handle if line.strip()]


def write_decision_events(path: Path | str, events: Iterable[DecisionEventV1]) -> int:
    path = Path(path)
    rows = sorted(events, key=lambda event: event.event_id)
    if len({event.event_id for event in rows}) != len(rows):
        raise ValueError("duplicate decision event ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        for event in rows:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return len(rows)
