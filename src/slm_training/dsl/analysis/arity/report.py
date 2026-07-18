"""Versioned arity report serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from slm_training.dsl.analysis.arity.types import AnalysisBounds, StateSignature


@dataclass(frozen=True)
class ContinuationSummary:
    """One continuation state observed during exploration."""

    state_signature: StateSignature
    next_actions: tuple[str, ...]
    terminal: bool
    complete: bool


@dataclass(frozen=True)
class ArityReport:
    """Exact or estimated arity report under a declared frame."""

    frame_id: str
    bounds: AnalysisBounds
    exact: bool
    total_states: int
    minimized_states: int
    continuation_summaries: tuple[ContinuationSummary, ...]
    version: str
    digest: str

    REPORT_VERSION = "cap0-02-report-v1"

    @classmethod
    def from_summaries(
        cls,
        frame_id: str,
        bounds: AnalysisBounds,
        summaries: tuple[ContinuationSummary, ...],
    ) -> ArityReport:
        total_states = len(summaries)
        unique_sigs: set[StateSignature] = {s.state_signature for s in summaries}
        minimized_states = len(unique_sigs)
        digest = _digest(frame_id, bounds, summaries)
        return cls(
            frame_id=frame_id,
            bounds=bounds,
            exact=True,
            total_states=total_states,
            minimized_states=minimized_states,
            continuation_summaries=summaries,
            version=cls.REPORT_VERSION,
            digest=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for JSON)."""

        def _convert(value: object) -> object:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _convert(v) for k, v in value.items()}
            if isinstance(value, AnalysisBounds):
                return asdict(value)
            if isinstance(value, StateSignature):
                return {
                    "version": value.version,
                    "generation_order": value.generation_order,
                    "atoms": [repr(a) for a in value.atoms],
                    "atom_count": value.atom_count,
                    "fingerprint": value.fingerprint(),
                }
            if isinstance(value, ContinuationSummary):
                return {
                    "state_signature": _convert(value.state_signature),
                    "next_actions": list(value.next_actions),
                    "terminal": value.terminal,
                    "complete": value.complete,
                }
            return repr(value)

        base = asdict(self)
        return {k: _convert(v) for k, v in base.items()}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _digest(
    frame_id: str,
    bounds: AnalysisBounds,
    summaries: tuple[ContinuationSummary, ...],
) -> str:
    canonical = {
        "frame_id": frame_id,
        "bounds": asdict(bounds),
        "summaries": [
            {
                "state_signature": s.state_signature.fingerprint(),
                "next_actions": list(s.next_actions),
                "terminal": s.terminal,
                "complete": s.complete,
            }
            for s in sorted(summaries, key=lambda x: x.state_signature.fingerprint())
        ],
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload, usedforsecurity=False).hexdigest()[:32]
