"""Byte-preserving replay records for archived evaluation failures.

This owner deliberately separates an archived output from any repaired or
feasibility-only view: it never regenerates a model prediction.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = [
    "UNKNOWN_NOT_CAPTURED",
    "ArchivedFailureV1",
    "HarnessProvenanceV1",
    "collect_archived_failures",
    "harness_provenance_id",
    "prediction_lineage",
    "replay_failure",
    "select_suite_stratified_failures",
]

UNKNOWN_NOT_CAPTURED = "unknown_not_captured"
REPLAY_SUITES = ("smoke", "held_out", "adversarial", "ood", "rico_held")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HarnessProvenanceV1:
    """Settings that can affect a replay verdict, including unavailable data."""

    source_eval_sha256: str
    evaluation_policy: Mapping[str, Any]
    timeout_seconds: float | None
    canvas_cap: int | None
    parser_fallback: str
    repair_policy: str
    runtime: str
    verifier: str
    target_length: int | None = None
    browser: str = "not_applicable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchivedFailureV1:
    """One archived event with immutable raw bytes and explicit lineage gaps."""

    event_id: str
    suite: str
    record_id: str
    raw_prediction: str
    raw_prediction_sha256: str
    original_failure: str
    provenance: HarnessProvenanceV1
    constrained_id: str = "unknown_not_captured"
    repaired_id: str = "unknown_not_captured"

    def __post_init__(self) -> None:
        if _sha256(self.raw_prediction) != self.raw_prediction_sha256:
            raise ValueError("raw prediction digest mismatch")
        if not self.event_id or not self.suite or not self.record_id:
            raise ValueError("event, suite, and record IDs are required")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provenance"] = self.provenance.to_dict()
        return data


def harness_provenance_id(provenance: HarnessProvenanceV1) -> str:
    """Return the stable ID used by individual rows to reference suite policy."""
    payload = json.dumps(provenance.to_dict(), sort_keys=True, separators=(",", ":"))
    return f"sha256:{_sha256(payload)}"


def prediction_lineage(raw_prediction: str) -> dict[str, str]:
    """Describe only captured output bytes; never invent intermediate outputs."""
    digest = _sha256(raw_prediction)
    return {
        "raw_prediction_sha256": digest,
        "raw_prediction_id": f"sha256:{digest}",
        "constrained_id": UNKNOWN_NOT_CAPTURED,
        "repaired_id": UNKNOWN_NOT_CAPTURED,
    }


def replay_failure(case: ArchivedFailureV1) -> dict[str, Any]:
    """Classify replay-time sensitivity without changing archived bytes.

    Canvas and repair outcomes are feasibility metadata only unless separately
    captured constrained/repaired output IDs exist.  This avoids attributing a
    post-hoc transformation to the original model.
    """

    cap = case.provenance.canvas_cap
    raw = case.raw_prediction
    truncated = len(raw) == 500
    canvas_sensitive = cap is not None and len(raw) > cap
    classifications = ["stable_failure"]
    if truncated:
        classifications.append("truncation_sensitive")
    if canvas_sensitive:
        classifications.append("canvas_sensitive")
    if case.provenance.timeout_seconds is not None and not raw.strip():
        classifications.append("timeout_sensitive")
    lineage = prediction_lineage(raw)
    return {
        "case": case.to_dict(),
        **lineage,
        "harness_provenance_id": harness_provenance_id(case.provenance),
        "raw_prediction_preserved": lineage["raw_prediction_sha256"] == case.raw_prediction_sha256,
        "classifications": classifications,
        "actual_decode_replayed": False,
        "caveat": "Archived bytes were reclassified only; no model output was regenerated.",
        "version_stamp": build_version_stamp("harness.eval.replay"),
    }


def collect_archived_failures(
    archive_root: Path, limit: int | None = 100
) -> list[ArchivedFailureV1]:
    """Select failed archived detail rows without regenerating their output.

    Older envelopes remain usable for diagnostic replay, but absent provenance is
    represented explicitly rather than reconstructed from a current harness.
    """
    cases: list[ArchivedFailureV1] = []
    for path in sorted(archive_root.rglob("eval_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        details = payload.get("details")
        if not isinstance(details, list):
            continue
        policy = payload.get("evaluation_policy")
        if not isinstance(policy, dict):
            policy = {}
        provenance = HarnessProvenanceV1(
            source_eval_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            evaluation_policy=policy,
            timeout_seconds=payload.get("decode_timeout_seconds"),
            canvas_cap=payload.get("decode_canvas_cap"),
            parser_fallback=str(payload.get("parser_fallback", "unknown_not_captured")),
            repair_policy=str(policy.get("grammar_ltr_repair", "unknown_not_captured")),
            runtime=str(payload.get("runtime", "unknown_not_captured")),
            verifier=str(payload.get("verifier", "production_metric_replay")),
            target_length=(
                int(payload["target_length"])
                if isinstance(payload.get("target_length"), int)
                else (
                    int(policy["grammar_ltr_max_tokens"])
                    if isinstance(policy.get("grammar_ltr_max_tokens"), int)
                    else None
                )
            ),
        )
        suite = str(payload.get("suite") or path.stem.removeprefix("eval_"))
        for index, detail in enumerate(details):
            if not isinstance(detail, dict):
                continue
            prediction = detail.get("prediction")
            failed = detail.get("parse_ok") is False or detail.get("meaningful_program_v1") is False
            if not failed or not isinstance(prediction, str):
                continue
            expected = detail.get("prediction_sha256")
            actual = _sha256(prediction)
            if isinstance(expected, str) and expected != actual:
                continue
            cases.append(
                ArchivedFailureV1(
                    event_id=f"{path.relative_to(archive_root)}#{index}",
                    suite=suite,
                    record_id=str(detail.get("id") or f"unknown-{index}"),
                    raw_prediction=prediction,
                    raw_prediction_sha256=actual,
                    original_failure=str(detail.get("error") or "meaningful_program_failure"),
                    provenance=provenance,
                )
            )
            if limit is not None and len(cases) >= limit:
                return cases
    return cases


def _stratum_key(case: ArchivedFailureV1) -> tuple[str, str, str, str]:
    """Return archive-only diversity axes; never inspect or transform bytes."""
    family = case.event_id.split("/", 1)[0]
    length = len(case.raw_prediction)
    length_bucket = "short" if length < 128 else "medium" if length < 384 else "long"
    policy = case.provenance.evaluation_policy
    decode_path = str(
        policy.get("compiler_decode_mode")
        or policy.get("grammar_ltr_primary")
        or "unknown_not_captured"
    )
    return family, length_bucket, decode_path, case.original_failure


def select_suite_stratified_failures(
    cases: Sequence[ArchivedFailureV1],
    *,
    suite_quota: int = 20,
    suites: Sequence[str] = REPLAY_SUITES,
) -> list[ArchivedFailureV1]:
    """Select a deterministic, suite-balanced archive sample.

    Each suite round-robins source/failure/length/decode strata before taking a
    second row from any stratum.  This is selection-only: prediction bytes are
    neither modified nor used as model input.
    """
    if suite_quota < 1:
        raise ValueError("suite_quota must be positive")
    by_suite: dict[str, list[ArchivedFailureV1]] = defaultdict(list)
    for case in cases:
        by_suite[case.suite].append(case)
    missing = {
        suite: suite_quota - len(by_suite[suite])
        for suite in suites
        if len(by_suite[suite]) < suite_quota
    }
    if missing:
        details = ", ".join(f"{suite}: short {count}" for suite, count in missing.items())
        raise ValueError(f"insufficient archived failures for stratified replay ({details})")

    selected: list[ArchivedFailureV1] = []
    for suite in suites:
        strata: dict[tuple[str, str, str, str], list[ArchivedFailureV1]] = defaultdict(list)
        for case in by_suite[suite]:
            strata[_stratum_key(case)].append(case)
        queues = [sorted(group, key=lambda case: case.event_id) for _, group in sorted(strata.items())]
        chosen: list[ArchivedFailureV1] = []
        while len(chosen) < suite_quota:
            progressed = False
            for queue in queues:
                if queue and len(chosen) < suite_quota:
                    chosen.append(queue.pop(0))
                    progressed = True
            if not progressed:  # Guard against a future sampler regression.
                raise AssertionError(f"selection exhausted unexpectedly for {suite}")
        selected.extend(chosen)
    return selected
