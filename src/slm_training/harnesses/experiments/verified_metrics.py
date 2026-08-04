"""LeverProof evidence export and fail-closed certificate verification.

The Lean project owns metric arithmetic and selection. This adapter owns the
explicitly trusted boundary around JSON, files, SHA-256, and process execution.
Promotion callers must replay a certificate with the kernel-backed checker; a
plausible-looking JSON document is never sufficient.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS
from slm_training.formal.checkers import (
    FormalProjectLock,
    ProcessOutcome,
    formal_process_budget,
    run_formal_process,
)

EVIDENCE_SCHEMA = "metric_evidence/v1"
CERTIFICATE_SCHEMA = "metric_certificate/v1"
CHECKER_ID = "leverproof-lean/v1"
EVIDENCE_SCHEMA_V2 = "metric_evidence/v2"
CERTIFICATE_SCHEMA_V2 = "metric_certificate/v2"
CHECKER_ID_V2 = "leverproof-lean/v2"
EXPECTATIONS_SCHEMA = "metric_expectations/v1"
OBSERVATIONS_SCHEMA = "metric_observations/v1"
DEFAULT_EVIDENCE_NAME = "metric-evidence.json"
DEFAULT_CERTIFICATE_NAME = "metric-certificate.json"
REPO_ROOT = Path(__file__).resolve().parents[4]
IN_REPO_CHECKER = (
    REPO_ROOT
    / "src"
    / "leverproof_lean"
    / ".lake"
    / "build"
    / "bin"
    / "leverproof-lean"
)


def _checker_project_root(checker: Path) -> Path:
    """Resolve the Lake project root for an in-repo or test checker binary."""

    resolved = checker.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "lakefile.toml").is_file():
            return parent
    return resolved.parent


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
    expectation_manifest_path: Path | str | None = None,
    observations_path: Path | str | None = None,
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

    result = {
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
    if expectation_manifest_path is None and observations_path is None:
        return result
    if expectation_manifest_path is None or observations_path is None:
        raise VerifiedMetricError(
            "band-aware evidence requires both expectations and observations"
        )
    expectations = _load_object(
        Path(expectation_manifest_path), schema=EXPECTATIONS_SCHEMA
    )
    observations = _load_object(Path(observations_path), schema=OBSERVATIONS_SCHEMA)
    metrics = expectations.get("metrics")
    observed = observations.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise VerifiedMetricError("metric expectations require a non-empty metrics list")
    if not isinstance(observed, dict):
        raise VerifiedMetricError("metric observations require a metrics object")
    embedded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            raise VerifiedMetricError(f"metrics[{index}] must be an object")
        metric_id = str(metric.get("metric_id", ""))
        if not metric_id or metric_id in seen:
            raise VerifiedMetricError("metric expectation ids must be non-empty and unique")
        seen.add(metric_id)
        samples = _natural_samples(
            observed.get(metric_id), field=f"observations.metrics.{metric_id}"
        )
        embedded.append({**metric, "observations": samples})
    extras = set(observed) - seen
    if extras:
        raise VerifiedMetricError(
            f"observations contain undeclared metrics: {sorted(extras)}"
        )
    result["schema"] = EVIDENCE_SCHEMA_V2
    result["source"]["metric_expectations_sha256"] = sha256_file(
        expectation_manifest_path
    )
    result["expectations"] = embedded
    return result


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
    payload = _load_json_object(path)
    if payload.get("schema") != schema:
        raise VerifiedMetricError(f"{path} is not a {schema} document")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerifiedMetricError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerifiedMetricError(f"{path} must contain a JSON object")
    return payload


def _resolve_checker(checker: Path | str | None) -> str:
    requested = Path(checker) if checker is not None else IN_REPO_CHECKER
    if not requested.is_file():
        raise VerifiedMetricError(
            "in-repo LeverProof checker is unavailable; run "
            "`make -C src/leverproof_lean build`"
        )
    return str(requested)


def optimum_feedback(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the cycle disposition without inventing a causal diagnosis."""
    if certificate.get("schema") != CERTIFICATE_SCHEMA_V2:
        return {
            "policy": "historical_only",
            "breaches": [],
            "diagnosis_lanes": [],
        }
    assessments = certificate.get("assessments")
    if not isinstance(assessments, list) or not assessments:
        raise VerifiedMetricError("band certificate has no assessments")
    breaches = [
        {
            "metric_id": str(row.get("metric_id", "")),
            "authority": row.get("authority"),
            "relation": row.get("relation"),
        }
        for row in assessments
        if isinstance(row, dict) and row.get("relation") != "in_band"
    ]
    if not breaches:
        policy = "continue"
    elif any(row.get("authority") == "theorem" for row in breaches):
        policy = "stop"
    else:
        policy = "block_promotion_and_diagnose"
    return {
        "policy": policy,
        "breaches": breaches,
        "diagnosis_lanes": (
            [
                "measurement_control",
                "training_method",
                "architecture",
                "lean_model",
                "assumptions",
            ]
            if breaches
            else []
        ),
    }


def verify_metric_certificate(
    *,
    evidence_path: Path | str,
    certificate_path: Path | str,
    expected_campaign_manifest_sha256: str | None = None,
    expected_metric_expectations_sha256: str | None = None,
    expected_selected_candidate: str | None = None,
    checker: Path | str | None = None,
    require_band_certificate: bool = False,
) -> dict[str, Any]:
    """Replay a whitelisted kernel-backed certificate and check bindings."""
    evidence_file = Path(evidence_path)
    certificate_file = Path(certificate_path)
    raw_evidence = _load_json_object(evidence_file)
    evidence_schema = raw_evidence.get("schema")
    expected_certificate_schema = {
        EVIDENCE_SCHEMA: CERTIFICATE_SCHEMA,
        EVIDENCE_SCHEMA_V2: CERTIFICATE_SCHEMA_V2,
    }.get(evidence_schema)
    if expected_certificate_schema is None:
        raise VerifiedMetricError(f"unsupported metric evidence schema {evidence_schema}")
    evidence = _load_object(evidence_file, schema=evidence_schema)
    certificate = _load_object(
        certificate_file, schema=expected_certificate_schema
    )
    expected_checker = (
        CHECKER_ID_V2
        if expected_certificate_schema == CERTIFICATE_SCHEMA_V2
        else CHECKER_ID
    )
    if certificate.get("checker") != expected_checker or certificate.get("verified") is not True:
        raise VerifiedMetricError("certificate is not a verified LeverProof result")
    expected_assurance = (
        "calculated_bands_and_observed_raw_samples"
        if expected_certificate_schema == CERTIFICATE_SCHEMA_V2
        else "observed_raw_samples"
    )
    if certificate.get("assurance") != expected_assurance:
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
    if expected_certificate_schema == CERTIFICATE_SCHEMA_V2:
        expected_digest = source.get("metric_expectations_sha256")
        if certificate.get("metric_expectations_sha256") != expected_digest:
            raise VerifiedMetricError(
                "certificate metric expectation digest does not match evidence"
            )
    if expected_campaign_manifest_sha256 is not None and certificate.get(
        "campaign_manifest_sha256"
    ) != _sha256(
        expected_campaign_manifest_sha256,
        field="expected_campaign_manifest_sha256",
    ):
        raise VerifiedMetricError("certificate campaign manifest digest mismatch")
    if expected_metric_expectations_sha256 is not None and certificate.get(
        "metric_expectations_sha256"
    ) != _sha256(
        expected_metric_expectations_sha256,
        field="expected_metric_expectations_sha256",
    ):
        raise VerifiedMetricError("certificate metric expectation digest mismatch")
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
    feedback = optimum_feedback(certificate)
    if require_band_certificate:
        if expected_certificate_schema != CERTIFICATE_SCHEMA_V2:
            raise VerifiedMetricError(
                "new promotion requires a calculated-band v2 certificate"
            )
        if feedback["policy"] == "stop":
            raise VerifiedMetricError(
                "theorem-backed metric is outside its calculated range"
            )
        if feedback["policy"] == "block_promotion_and_diagnose":
            raise VerifiedMetricError(
                "assumption-backed metric is outside its calculated range"
            )

    checker_path = Path(_resolve_checker(checker))
    project_root = _checker_project_root(checker_path)
    total, _interrupt_after, _grace = formal_process_budget(
        INTERRUPT_AFTER_SECONDS
    )
    try:
        with FormalProjectLock(project_root, timeout_seconds=total) as lock:
            completed = run_formal_process(
                [
                    str(checker_path),
                    "verify",
                    str(evidence_file),
                    str(certificate_file),
                ],
                cwd=project_root,
                timeout_seconds=max(0.001, total - lock.wait_seconds),
            )
    except TimeoutError as exc:
        raise VerifiedMetricError(f"LeverProof replay lock timed out: {exc}") from exc
    if completed.timed_out:
        raise VerifiedMetricError("LeverProof replay exceeded the run cap")
    if completed.outcome is ProcessOutcome.LAUNCH_FAILED:
        raise VerifiedMetricError(
            f"LeverProof replay could not start: {completed.launch_error}"
        )
    if completed.returncode != 0 or completed.stdout.strip() != "verified":
        detail = completed.stderr.strip() or completed.stdout.strip() or "rejected"
        raise VerifiedMetricError(f"LeverProof replay failed: {detail}")
    return certificate


def default_metric_paths(artifact_root: Path | str) -> tuple[Path, Path]:
    root = Path(artifact_root)
    return root / DEFAULT_EVIDENCE_NAME, root / DEFAULT_CERTIFICATE_NAME


__all__ = [
    "CERTIFICATE_SCHEMA",
    "CERTIFICATE_SCHEMA_V2",
    "CHECKER_ID",
    "CHECKER_ID_V2",
    "DEFAULT_CERTIFICATE_NAME",
    "DEFAULT_EVIDENCE_NAME",
    "EVIDENCE_SCHEMA",
    "EVIDENCE_SCHEMA_V2",
    "EXPECTATIONS_SCHEMA",
    "IN_REPO_CHECKER",
    "OBSERVATIONS_SCHEMA",
    "VerifiedMetricError",
    "build_metric_evidence",
    "default_metric_paths",
    "optimum_feedback",
    "sha256_file",
    "verify_metric_certificate",
    "write_metric_evidence",
]
