"""LeverProof evidence export and fail-closed certificate verification.

The Lean project owns metric arithmetic and selection. This adapter owns the
explicitly trusted boundary around JSON, files, SHA-256, and process execution.
Promotion callers must replay a certificate with the kernel-backed checker; a
plausible-looking JSON document is never sufficient.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS

EVIDENCE_SCHEMA = "metric_evidence/v1"
CERTIFICATE_SCHEMA = "metric_certificate/v1"
CHECKER_ID = "leverproof-lean/v1"
DEFAULT_EVIDENCE_NAME = "metric-evidence.json"
DEFAULT_CERTIFICATE_NAME = "metric-certificate.json"


class VerifiedMetricError(ValueError):
    """The metric evidence or certificate cannot authorize promotion."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _campaign_sha256(path: Path | str) -> str:
    try:
        manifest = ExperimentCampaignV1.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise VerifiedMetricError(f"invalid campaign manifest {path}: {exc}") from exc
    return campaign_manifest_sha256(manifest)


def _sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise VerifiedMetricError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _natural_samples(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise VerifiedMetricError(f"{field} must be a non-empty list")
    samples: list[int] = []
    for index, sample in enumerate(value):
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            raise VerifiedMetricError(f"{field}[{index}] must be a natural number")
        samples.append(sample)
    return samples


def build_metric_evidence(
    *,
    run_id: str,
    evidence_bundle_path: Path | str,
    feature_flags_path: Path | str,
    campaign_manifest_path: Path | str | None,
    cold_requests: int,
    warm_requests: int,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build exact raw-sample evidence with content-addressed provenance."""
    if not run_id:
        raise VerifiedMetricError("run_id is required")
    if cold_requests < 0 or warm_requests < 0 or cold_requests + warm_requests == 0:
        raise VerifiedMetricError("workload must contain at least one request")
    if not candidates:
        raise VerifiedMetricError("at least one candidate is required")

    normalized: list[dict[str, Any]] = []
    hardware: str | None = None
    for index, candidate in enumerate(candidates):
        prefix = f"candidates[{index}]"
        candidate_id = str(candidate.get("id", ""))
        current_hardware = str(candidate.get("hardware", ""))
        if not candidate_id or not current_hardware:
            raise VerifiedMetricError(f"{prefix} requires id and hardware")
        if hardware is None:
            hardware = current_hardware
        elif current_hardware != hardware:
            raise VerifiedMetricError("all candidates must use the same hardware")
        row = {
            "id": candidate_id,
            "hardware": current_hardware,
            "lever_snapshot_sha256": _sha256(
                candidate.get("lever_snapshot_sha256"),
                field=f"{prefix}.lever_snapshot_sha256",
            ),
            "cold_ns": _natural_samples(
                candidate.get("cold_ns"), field=f"{prefix}.cold_ns"
            ),
            "warm_ns": _natural_samples(
                candidate.get("warm_ns"), field=f"{prefix}.warm_ns"
            ),
            "input_units": _natural_samples(
                candidate.get("input_units"), field=f"{prefix}.input_units"
            ),
            "passes": _natural_samples(
                candidate.get("passes"), field=f"{prefix}.passes"
            ),
            "energy_uj": _natural_samples(
                candidate.get("energy_uj"), field=f"{prefix}.energy_uj"
            ),
            "cost_micro_usd": _natural_samples(
                candidate.get("cost_micro_usd"),
                field=f"{prefix}.cost_micro_usd",
            ),
        }
        for field in ("successes", "quality_failures", "trainable_params"):
            value = candidate.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VerifiedMetricError(f"{prefix}.{field} must be natural")
            row[field] = value
        if row["trainable_params"] == 0:
            raise VerifiedMetricError(f"{prefix}.trainable_params must be positive")
        if row["successes"] > len(row["cold_ns"]) + len(row["warm_ns"]):
            raise VerifiedMetricError(f"{prefix}.successes exceeds measured requests")
        normalized.append(row)

    return {
        "schema": EVIDENCE_SCHEMA,
        "run_id": run_id,
        "source": {
            "evidence_sha256": sha256_file(evidence_bundle_path),
            "feature_flags_sha256": sha256_file(feature_flags_path),
            "campaign_manifest_sha256": (
                _campaign_sha256(campaign_manifest_path)
                if campaign_manifest_path is not None
                else None
            ),
        },
        "workload": {
            "cold_requests": cold_requests,
            "warm_requests": warm_requests,
        },
        "candidates": normalized,
    }


def write_metric_evidence(
    path: Path | str,
    **kwargs: Any,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_metric_evidence(**kwargs)
    destination.write_bytes(_canonical_json(payload) + b"\n")
    return destination


def _load_object(path: Path, *, schema: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerifiedMetricError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise VerifiedMetricError(f"{path} is not a {schema} document")
    return payload


def _resolve_checker(checker: Path | str | None) -> str:
    requested = str(checker or os.environ.get("LEVERPROOF_BIN") or "leverproof")
    resolved = shutil.which(requested)
    if resolved is None:
        raise VerifiedMetricError(
            "LeverProof checker is unavailable; set LEVERPROOF_BIN or "
            "pass leverproof_bin"
        )
    return resolved


def verify_metric_certificate(
    *,
    evidence_path: Path | str,
    certificate_path: Path | str,
    expected_campaign_manifest_sha256: str | None = None,
    expected_selected_candidate: str | None = None,
    checker: Path | str | None = None,
) -> dict[str, Any]:
    """Replay a whitelisted kernel-backed certificate and check bindings."""
    evidence_file = Path(evidence_path)
    certificate_file = Path(certificate_path)
    evidence = _load_object(evidence_file, schema=EVIDENCE_SCHEMA)
    certificate = _load_object(certificate_file, schema=CERTIFICATE_SCHEMA)
    if (
        certificate.get("checker") != CHECKER_ID
        or certificate.get("verified") is not True
    ):
        raise VerifiedMetricError("certificate is not a verified LeverProof result")
    if certificate.get("assurance") != "observed_raw_samples":
        raise VerifiedMetricError("certificate does not cover observed raw samples")

    source = evidence.get("source")
    if not isinstance(source, dict):
        raise VerifiedMetricError("evidence source provenance is missing")
    bindings = (
        ("run_id", evidence.get("run_id")),
        ("evidence_sha256", source.get("evidence_sha256")),
        ("feature_flags_sha256", source.get("feature_flags_sha256")),
        ("campaign_manifest_sha256", source.get("campaign_manifest_sha256")),
    )
    for field, expected in bindings:
        if certificate.get(field) != expected:
            raise VerifiedMetricError(f"certificate {field} does not match evidence")
    if expected_campaign_manifest_sha256 is not None and certificate.get(
        "campaign_manifest_sha256"
    ) != _sha256(
        expected_campaign_manifest_sha256,
        field="expected_campaign_manifest_sha256",
    ):
        raise VerifiedMetricError("certificate campaign manifest digest mismatch")
    if (
        expected_selected_candidate is not None
        and certificate.get("selected_candidate") != expected_selected_candidate
    ):
        raise VerifiedMetricError(
            "certificate selected candidate does not match promoted candidate"
        )

    candidates = certificate.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise VerifiedMetricError("certificate has no derived candidates")
    selected = str(certificate.get("selected_candidate", ""))
    if (
        sum(row.get("id") == selected for row in candidates if isinstance(row, dict))
        != 1
    ):
        raise VerifiedMetricError("selected candidate is not uniquely certified")

    try:
        completed = subprocess.run(
            [
                _resolve_checker(checker),
                "verify",
                str(evidence_file),
                str(certificate_file),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=INTERRUPT_AFTER_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise VerifiedMetricError("LeverProof replay exceeded the run cap") from exc
    if completed.returncode != 0 or completed.stdout.strip() != "verified":
        detail = completed.stderr.strip() or completed.stdout.strip() or "rejected"
        raise VerifiedMetricError(f"LeverProof replay failed: {detail}")
    return certificate


def default_metric_paths(artifact_root: Path | str) -> tuple[Path, Path]:
    root = Path(artifact_root)
    return root / DEFAULT_EVIDENCE_NAME, root / DEFAULT_CERTIFICATE_NAME


__all__ = [
    "CERTIFICATE_SCHEMA",
    "CHECKER_ID",
    "DEFAULT_CERTIFICATE_NAME",
    "DEFAULT_EVIDENCE_NAME",
    "EVIDENCE_SCHEMA",
    "VerifiedMetricError",
    "build_metric_evidence",
    "default_metric_paths",
    "sha256_file",
    "verify_metric_certificate",
    "write_metric_evidence",
]
