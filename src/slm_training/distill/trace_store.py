"""Append-only decode trajectory store + MaskGIT trace recorder.

Random masking is not the same distribution as progressive MaskGIT decoding;
closing that gap (trajectory-aligned SFT / RL) requires the *actual*
intermediate canvases, commit decisions, remask events, and their
probabilities under the rollout policy. This module persists them.

Every trace carries the identity needed to make it reusable later: policy
checkpoint SHA, decode-config hash, tokenizer/grammar versions, seed, reward
vector (not just a scalar), and accept/reject labels. Rollouts from different
checkpoints must never be mixed unlabeled.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

TRACE_VERSION = 1

# Config keys that change decode behavior (used for the decode-config hash).
_DECODE_KEYS = (
    "gen_steps",
    "grammar_constrained",
    "grammar_dsl",
    "grammar_top_k",
    "structural_bias",
    "grammar_ltr_repair",
    "grammar_ltr_max_tokens",
    "grammar_ltr_stages",
    "grammar_ltr_primary",
    "grammar_finalize_validate",
    "grammar_prefer_structural",
    "grammar_trust_model",
    "grammar_sample_decode",
    "grammar_sample_temperature",
    "grammar_block_decode",
    "grammar_block_size",
    "grammar_fastpath",
    "grammar_fastpath_mode",
    "parallel_unmask",
    "remask_ratio",
    "remask_policy",
    "remask_span",
    "remask_to_mask",
    "remask_use_gate",
    "remask_use_entropy",
    "core_perturb_frac",
    "suffix_rollback_window",
    "slot_contract_constrained_decode",
    "template_fill_decode",
    "honest_slot_contract",
    "best_of_n",
    "output_tokenizer",
    "use_symbol_table",
)


def decode_config_hash(config: Any) -> str:
    """Stable hash over decode-relevant configuration."""
    if is_dataclass(config) and not isinstance(config, type):
        raw = asdict(config)
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raw = {k: getattr(config, k) for k in _DECODE_KEYS if hasattr(config, k)}
    payload = {k: raw.get(k) for k in _DECODE_KEYS if k in raw}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checkpoint_sha(path: Path | str) -> str | None:
    """Content hash of a policy checkpoint file."""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DecodeTraceRecorder:
    """Collects per-step decode events from TwoTower MaskGIT generation.

    Attach via ``model.trace_recorder = recorder`` before ``generate``; the
    decode path emits events with zero cost when no recorder is attached.
    Nested decode passes (e.g. the unconstrained retry) are recorded into the
    same trajectory with phase markers rather than resetting it.
    """

    def __init__(self, *, record_canvases: bool = True) -> None:
        self.record_canvases = bool(record_canvases)
        self.meta: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.final: dict[str, Any] | None = None
        self.nfe = 0
        self.commit_count = 0
        self.repair_commit_count = 0
        self.remask_count = 0
        self._depth = 0

    # ── model-side hooks ─────────────────────────────────────────────────

    def begin(self, **meta: Any) -> None:
        self._depth += 1
        if self._depth > 1:
            self.events.append({"kind": "nested_begin", "meta": meta})
            return
        self.meta.update(meta)

    def forward(self) -> None:
        """One denoiser forward evaluation (NFE accounting)."""
        self.nfe += 1

    def step(
        self,
        step: int,
        *,
        canvas: list[int] | None = None,
        unknown: list[bool] | None = None,
        commits: list[dict[str, Any]] | None = None,
        remasks: list[dict[str, Any]] | None = None,
    ) -> None:
        row: dict[str, Any] = {"step": step, "depth": self._depth}
        if self.record_canvases and canvas is not None:
            row["canvas"] = list(canvas)
            if unknown is not None:
                row["unknown_positions"] = [
                    i for i, u in enumerate(unknown) if u
                ]
        if commits:
            row["commits"] = commits
            self.commit_count += sum(
                1 for c in commits if c.get("phase") != "ltr_repair"
            )
            self.repair_commit_count += sum(
                1 for c in commits if c.get("phase") == "ltr_repair"
            )
        if remasks:
            row["remasks"] = remasks
            self.remask_count += sum(len(r.get("positions") or []) for r in remasks)
        self.steps.append(row)

    def event(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, "depth": self._depth, **payload})

    def end(self, *, canvas: list[int] | None = None, text: str | None = None) -> None:
        self._depth = max(0, self._depth - 1)
        if self._depth > 0:
            self.events.append({"kind": "nested_end"})
            return
        self.final = {
            "canvas": list(canvas) if canvas is not None else None,
            "text": text,
        }

    # ── consumer-side assembly ───────────────────────────────────────────

    def finalize(
        self,
        *,
        final_text: str | None = None,
        reward: dict[str, Any] | None = None,
        labels: dict[str, Any] | None = None,
        **meta: Any,
    ) -> dict[str, Any]:
        final = dict(self.final or {})
        if final_text is not None:
            final["text"] = final_text
        return {
            "version": TRACE_VERSION,
            "meta": {**self.meta, **meta},
            "steps": self.steps,
            "events": self.events,
            "final": final,
            "counters": {
                "nfe": self.nfe,
                "commits": self.commit_count,
                "repair_commits": self.repair_commit_count,
                "remasked_positions": self.remask_count,
                "decode_steps": len(self.steps),
            },
            "reward": reward or {},
            "labels": labels or {},
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }


class TraceStore:
    """Append-only JSONL trajectory store with a manifest.

    Existing traces are never rewritten or deleted through this API; new
    rollouts only append. Consumers filter by the identity fields in each
    trace (checkpoint SHA, decode-config hash, labels).
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.traces_path = self.root / "traces.jsonl"
        self.manifest_path = self.root / "manifest.json"

    def __len__(self) -> int:
        if not self.traces_path.exists():
            return 0
        with self.traces_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def append(self, trace: dict[str, Any]) -> str:
        index = len(self)
        digest = hashlib.sha256(
            json.dumps(trace, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        trace_id = f"{index:08d}-{digest[:12]}"
        row = {"trace_id": trace_id, **trace}
        with self.traces_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._update_manifest(count=index + 1)
        return trace_id

    def iter_traces(self) -> Iterator[dict[str, Any]]:
        if not self.traces_path.exists():
            return
        with self.traces_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def _update_manifest(self, *, count: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        manifest = {
            "kind": "trace_store",
            "version": TRACE_VERSION,
            "append_only": True,
            "count": count,
            "traces": str(self.traces_path.as_posix()),
            "updated_at": now,
        }
        if self.manifest_path.exists():
            try:
                prev = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                manifest["created_at"] = prev.get("created_at", now)
            except Exception:  # noqa: BLE001
                manifest["created_at"] = now
        else:
            manifest["created_at"] = now
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
