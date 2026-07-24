"""SLM-262 (VSD0-03): durable GPU train->checkpoint->eval reference-run harness.

This module defines the provider-neutral :class:`AcceleratorRunManifestV1`, a
small provider-adapter contract, and helper routines for no-spend validation,
local CPU compatibility smoke, and dry-run planning.  It does **not** by itself
submit paid GPU jobs; the CLI in ``scripts/run_gpu_reference.py`` composes the
existing ``scripts.hf_jobs_train`` / ``scripts.remote_train`` entry points and
records the result in the manifest.
"""

from __future__ import annotations

import abc
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from slm_training.harness_core.lineage.records import content_sha
from slm_training.versioning import build_version_stamp

MANIFEST_SCHEMA = "accelerator_run_manifest/v1"
EXPERIMENT_ID = "slm262-gpu-reference-run"
MATRIX_SET = "slm262_gpu_reference_run"
MATRIX_VERSION = "vsd0-03-v1"
ALLOWED_PROVIDERS = ("hf_jobs", "remote_pod", "dry_run")
DISPOSITIONS = (
    "gpu_path_qualified",
    "provider_adapter_defect_fixed",
    "credentials_or_quota_blocked",
    "environment_incompatible",
    "inconclusive",
)
UNKNOWN = "UNKNOWN"

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SECRET_KEY_RE = re.compile(r"(token|secret|password|credential|private_key|api_key)", re.I)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def _bucket_from_prefix(prefix: str) -> str:
    """Extract the bucket URI from a run-specific checkpoint prefix."""
    prefix = prefix.rstrip("/")
    if prefix.startswith("hf://buckets/"):
        # Prefix looks like hf://buckets/<owner>/<name>/checkpoints/<run_id>
        parts = prefix.removeprefix("hf://buckets/").split("/")
        if len(parts) >= 2:
            return f"hf://buckets/{parts[0]}/{parts[1]}"
    return prefix


def _to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple(item) for item in value)
    if isinstance(value, dict):
        return {k: _to_tuple(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class AcceleratorRunManifestV1:
    """Immutable, JSON-safe manifest for one accelerator train/eval run.

    The manifest is intentionally strict: any value the caller does not actually
    have stays ``UNKNOWN`` or empty, and :meth:`check_ready` fails closed rather
    than fabricating provenance.
    """

    schema_version: str = MANIFEST_SCHEMA
    run_id: str = ""
    track: str = "twotower"
    parent_lineage: tuple[str, ...] = ()
    repo_url: str = "https://github.com/Tyler-R-Kendrick/slm-training.git"
    source_commit: str = UNKNOWN
    dirty_tree_ok: bool = False
    container_image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    environment_lock_hash: str = UNKNOWN
    provider: str = "dry_run"
    region: str | None = None
    instance_type: str = "a10g-large"
    gpu_model: str = "A10G"
    gpu_count: int = 1
    gpu_memory_gb: int | None = None
    framework: str = "pytorch"
    torch_version: str = ""
    cuda_version: str = ""
    driver_version: str | None = None
    secrets_refs: tuple[str, ...] = ()
    data_snapshot_id: str = UNKNOWN
    data_snapshot_sha: str = UNKNOWN
    eval_snapshot_id: str = UNKNOWN
    eval_snapshot_sha: str = UNKNOWN
    model_config_hash: str = UNKNOWN
    tokenizer_hash: str = UNKNOWN
    codec_hash: str = UNKNOWN
    optimizer_settings: Mapping[str, Any] = field(default_factory=dict)
    scheduler_settings: Mapping[str, Any] = field(default_factory=dict)
    precision: str = "fp32"
    distribution: str | None = None
    seed: int = 0
    rng_sampler_contract: str = "pytorch_default"
    target_decisions: int = 0
    max_wall_minutes: int = 3
    checkpoint_cadence_decisions: int = 0
    expected_artifacts: tuple[str, ...] = ()
    remote_uri_prefix: str = ""
    provider_request_id: str | None = None
    provider_job_id: str | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    timestamps: Mapping[str, str] = field(default_factory=dict)
    utilization: Mapping[str, Any] = field(default_factory=dict)
    energy_wh: float | None = None
    billable_seconds: float | None = None
    provider_cost_usd: float | None = None
    cost_estimate_usd: float | None = None
    cost_confidence: str | None = None
    checkpoint_inventory: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    full_state_inventory: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    evaluation_report_refs: tuple[str, ...] = ()
    disposition: str | None = None
    notes: tuple[str, ...] = ()
    version_stamp: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported accelerator run manifest schema {self.schema_version!r}"
            )
        if self.provider not in ALLOWED_PROVIDERS:
            raise ValueError(
                f"unsupported provider {self.provider!r}; expected one of {ALLOWED_PROVIDERS}"
            )
        if self.disposition is not None and self.disposition not in DISPOSITIONS:
            raise ValueError(
                f"unsupported disposition {self.disposition!r}; expected one of {DISPOSITIONS}"
            )
        if self.source_commit != UNKNOWN and not _GIT_SHA_RE.fullmatch(self.source_commit):
            raise ValueError(
                f"source_commit must be a 40-char lowercase git SHA or {UNKNOWN!r}"
            )
        if self.max_wall_minutes < 1:
            raise ValueError("max_wall_minutes must be at least 1")
        if self.target_decisions < 0:
            raise ValueError("target_decisions must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, default=str
        )

    def write_json(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AcceleratorRunManifestV1":
        mapped = dict(data)
        mapped["parent_lineage"] = tuple(mapped.get("parent_lineage", ()))
        mapped["secrets_refs"] = tuple(mapped.get("secrets_refs", ()))
        mapped["expected_artifacts"] = tuple(mapped.get("expected_artifacts", ()))
        mapped["evaluation_report_refs"] = tuple(mapped.get("evaluation_report_refs", ()))
        mapped["notes"] = tuple(mapped.get("notes", ()))
        for key in (
            "optimizer_settings",
            "scheduler_settings",
            "provider_options",
            "timestamps",
            "utilization",
            "checkpoint_inventory",
            "full_state_inventory",
            "version_stamp",
        ):
            mapped[key] = _to_tuple(mapped.get(key, {}))
        # Drop unknown fields so the dataclass constructor does not fail on
        # future-compatible reading.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        mapped = {k: v for k, v in mapped.items() if k in known}
        return cls(**mapped)

    @classmethod
    def load_json(cls, path: Path | str) -> "AcceleratorRunManifestV1":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Validation / readiness
    # ------------------------------------------------------------------
    def check_ready(self, *, require_gpu: bool = False) -> list[str]:
        """Return a list of blockers; empty means ready to describe/submit."""
        errors: list[str] = []
        if not self.run_id:
            errors.append("run_id is required")
        if self.source_commit == UNKNOWN:
            errors.append("source_commit is required")
        if self.data_snapshot_sha == UNKNOWN:
            errors.append("data_snapshot_sha is required")
        if self.eval_snapshot_sha == UNKNOWN:
            errors.append("eval_snapshot_sha is required")
        if not self.expected_artifacts:
            errors.append("expected_artifacts must list at least one artifact")
        if not self.remote_uri_prefix:
            errors.append("remote_uri_prefix is required")
        if self.target_decisions <= 0:
            errors.append("target_decisions must be positive")
        if self.checkpoint_cadence_decisions <= 0:
            errors.append("checkpoint_cadence_decisions must be positive")

        if not self.dirty_tree_ok:
            dirty = _git_dirty()
            if dirty is True:
                errors.append("worktree is dirty; commit or set dirty_tree_ok=true")
            head = _git_head()
            if head is not None and self.source_commit != UNKNOWN and head != self.source_commit:
                errors.append(
                    f"source_commit {self.source_commit[:8]} does not match HEAD {head[:8]}"
                )

        if self.provider == "hf_jobs":
            if not self.instance_type:
                errors.append("hf_jobs provider requires instance_type/flavor")
            if not _hf_authenticated():
                errors.append(
                    "hf_jobs provider requires Hugging Face auth (HF_TOKEN or hf auth login)"
                )
        elif self.provider == "remote_pod":
            host = (self.provider_options or {}).get("host")
            if not host:
                errors.append("remote_pod provider requires provider_options.host")

        if require_gpu and self.provider in {"dry_run"}:
            errors.append("real GPU execution requires provider=hf_jobs or remote_pod")
        return errors

    def describe(self) -> dict[str, Any]:
        """Return a human-readable plan dict without mutating the manifest."""
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "instance_type": self.instance_type,
            "gpu": {"model": self.gpu_model, "count": self.gpu_count, "memory_gb": self.gpu_memory_gb},
            "source_commit": self.source_commit,
            "data_snapshot": {"id": self.data_snapshot_id, "sha": self.data_snapshot_sha},
            "eval_snapshot": {"id": self.eval_snapshot_id, "sha": self.eval_snapshot_sha},
            "target_decisions": self.target_decisions,
            "max_wall_minutes": self.max_wall_minutes,
            "checkpoint_cadence_decisions": self.checkpoint_cadence_decisions,
            "remote_uri_prefix": self.remote_uri_prefix,
            "expected_artifacts": list(self.expected_artifacts),
            "ready_blockers": self.check_ready(),
            "version_stamp": self.version_stamp,
        }

    @staticmethod
    def redact_secrets(value: Any) -> Any:
        """Recursively redact values whose keys look like secrets."""
        if isinstance(value, dict):
            return {
                k: "***" if _SECRET_KEY_RE.search(str(k)) else AcceleratorRunManifestV1.redact_secrets(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [AcceleratorRunManifestV1.redact_secrets(item) for item in value]
        return value


def _hf_authenticated() -> bool:
    """Best-effort check for usable Hugging Face auth without printing secrets."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        proc = subprocess.run(
            ["hf", "auth", "whoami"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return proc.returncode == 0 and "Logged in" in proc.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def build_default_manifest(
    run_id: str,
    *,
    provider: str = "dry_run",
    source_commit: str | None = None,
    data_snapshot_id: str = UNKNOWN,
    data_snapshot_sha: str = UNKNOWN,
    eval_snapshot_id: str = UNKNOWN,
    eval_snapshot_sha: str = UNKNOWN,
    target_decisions: int = 50_000,
    remote_uri_prefix: str | None = None,
    train_version: str = "e530_visible_semantic_roles_r1_20260719",
) -> AcceleratorRunManifestV1:
    """Build a default SLM-262 manifest with current-repo provenance prefilled."""
    head = source_commit or _git_head() or UNKNOWN
    dirty = _git_dirty() or False
    if remote_uri_prefix is None:
        remote_uri_prefix = f"hf://buckets/TKendrick/OpenUI/checkpoints/{run_id}"
    return AcceleratorRunManifestV1(
        run_id=run_id,
        provider=provider,
        source_commit=head,
        dirty_tree_ok=dirty,  # If already dirty, do not block further dry-run work.
        container_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        instance_type="a10g-large" if provider == "hf_jobs" else "",
        gpu_model="A10G" if provider == "hf_jobs" else "",
        gpu_count=1,
        secrets_refs=("HF_TOKEN",),
        data_snapshot_id=data_snapshot_id,
        data_snapshot_sha=data_snapshot_sha,
        eval_snapshot_id=eval_snapshot_id,
        eval_snapshot_sha=eval_snapshot_sha,
        target_decisions=target_decisions,
        checkpoint_cadence_decisions=max(1, target_decisions // 10),
        expected_artifacts=(
            "last.pt",
            "last.tokenizer.json",
            "last.meta.json",
            "last_full_state.pt",
            "train_summary.json",
        ),
        remote_uri_prefix=remote_uri_prefix,
        provider_options={"train_version": train_version},
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm262_gpu_reference_run",
        ),
    )


# ----------------------------------------------------------------------
# Local CPU compatibility smoke + artifact verification
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_artifacts(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    """Build ``{filename: {size_bytes, sha256, role}}`` for every artifact file."""
    inventory: dict[str, dict[str, Any]] = {}
    if not checkpoint_dir.is_dir():
        return inventory
    for path in sorted(checkpoint_dir.iterdir()):
        if not path.is_file():
            continue
        inventory[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "role": Path(path.name).stem,
        }
    return inventory


def run_local_smoke(
    manifest: AcceleratorRunManifestV1,
    *,
    steps: int = 5,
    resume_steps: int = 2,
    train_version: str = "e530_visible_semantic_roles_r1_20260719",
    device: str = "cpu",
    context_backend: str = "scratch",
    run_root: Path | str = "outputs/runs",
) -> dict[str, Any]:
    """Run a short CPU train->resume loop to prove compatibility of the manifest.

    This is Phase-A no-spend evidence only: it does not touch a GPU and does not
    claim durable persistence.  It does prove that the exact run_id/config can
    execute, produce a full-state checkpoint, and resume from it.
    """
    errors = manifest.check_ready()
    if errors:
        raise ValueError("manifest not ready: " + "; ".join(errors))

    run_id = manifest.run_id
    smoke_run_id = f"{run_id}_cpu_smoke"
    run_dir = Path(run_root) / smoke_run_id
    ckpt_dir = run_dir / "checkpoints"
    python = sys.executable

    def _run_train(train_steps: int, *extra_args: str) -> subprocess.CompletedProcess[str]:
        cmd = [
            python,
            "-m",
            "scripts.train_model",
            "--train-version",
            train_version,
            "--run-id",
            smoke_run_id,
            "--steps",
            str(train_steps),
            "--device",
            device,
            "--context-backend",
            context_backend,
            "--no-sync-checkpoints",
            "--fast-train",
        ] + list(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    initial = _run_train(steps)
    if initial.returncode != 0:
        return {
            "ok": False,
            "stage": "initial_train",
            "returncode": initial.returncode,
            "stderr": initial.stderr[-2000:],
            "stdout": initial.stdout[-2000:],
        }

    full_state = ckpt_dir / "last_full_state.pt"
    if not full_state.is_file():
        return {
            "ok": False,
            "stage": "full_state_check",
            "error": f"full-state checkpoint not found: {full_state}",
        }

    # train_loop resumes global_step from the checkpoint and trains while
    # step < steps, so request enough total steps for the resume leg.
    resumed = _run_train(steps + max(1, resume_steps), "--resume-from", str(full_state))
    if resumed.returncode != 0:
        return {
            "ok": False,
            "stage": "resume_train",
            "returncode": resumed.returncode,
            "stderr": resumed.stderr[-2000:],
            "stdout": resumed.stdout[-2000:],
        }

    inventory = hash_artifacts(ckpt_dir)
    return {
        "ok": True,
        "run_id": smoke_run_id,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "inventory": inventory,
        "last_full_state_exists": (ckpt_dir / "last_full_state.pt").is_file(),
        "last_serving_exists": (ckpt_dir / "last.pt").is_file(),
        "train_summary_exists": (run_dir / "train_summary.json").is_file(),
    }


# ----------------------------------------------------------------------
# Provider adapters
# ----------------------------------------------------------------------
class GPUProviderAdapter(abc.ABC):
    """Minimal contract that every accelerator provider must satisfy."""

    @property
    @abc.abstractmethod
    def provider(self) -> str:
        """Provider slug (``hf_jobs``, ``remote_pod``, ...)."""

    def capability(self) -> dict[str, Any]:
        return {"provider": self.provider, "supports_dry_run": True}

    @abc.abstractmethod
    def submit(
        self, manifest: AcceleratorRunManifestV1, *, dry_run: bool = False
    ) -> dict[str, Any]:
        """Submit or plan a run; return a dict that the caller merges into the manifest."""

    @abc.abstractmethod
    def status(self, manifest: AcceleratorRunManifestV1) -> dict[str, Any]:
        """Fetch provider status for ``manifest.provider_job_id``."""

    @abc.abstractmethod
    def logs(self, manifest: AcceleratorRunManifestV1) -> str:
        """Fetch recent logs for the job."""

    @abc.abstractmethod
    def cancel(self, manifest: AcceleratorRunManifestV1) -> dict[str, Any]:
        """Cancel the provider job if possible."""

    @abc.abstractmethod
    def cost(self, raw_records: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize measured cost/telemetry from raw provider records."""

    @abc.abstractmethod
    def reconcile(
        self,
        manifest: AcceleratorRunManifestV1,
        payload: Mapping[str, Any],
    ) -> AcceleratorRunManifestV1:
        """Return a new manifest updated with provider outcome + checkpoint refs."""


def _manifest_with(
    manifest: AcceleratorRunManifestV1, **changes: Any
) -> AcceleratorRunManifestV1:
    """Return a copy of ``manifest`` with the supplied field overrides."""
    data = manifest.to_dict()
    data.update(changes)
    return AcceleratorRunManifestV1.from_dict(data)


class HFJobsAdapter(GPUProviderAdapter):
    """Adapter around ``scripts.hf_jobs_train``."""

    @property
    def provider(self) -> str:
        return "hf_jobs"

    def _build_cli_args(
        self, manifest: AcceleratorRunManifestV1, *, dry_run: bool
    ) -> list[str]:
        bucket = _bucket_from_prefix(manifest.remote_uri_prefix)
        # Bounded by the repository-wide 3-minute hard cap.
        steps = max(1, min(manifest.target_decisions, 200))
        args: list[str] = [
            sys.executable,
            "-m",
            "scripts.hf_jobs_train",
            "--run-id",
            manifest.run_id,
            "--steps",
            str(steps),
            "--flavor",
            manifest.instance_type or "a10g-large",
            "--checkpoint-bucket",
            bucket,
            "--context-backend",
            "scratch",
            "--skip-eval",
        ]
        if dry_run:
            args.append("--dry-run")
        return args

    def submit(
        self, manifest: AcceleratorRunManifestV1, *, dry_run: bool = False
    ) -> dict[str, Any]:
        cmd = self._build_cli_args(manifest, dry_run=dry_run)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            return {
                "ok": False,
                "provider": self.provider,
                "returncode": proc.returncode,
                "stderr": proc.stderr[-4000:],
            }
        try:
            plan = json.loads(proc.stdout)
        except json.JSONDecodeError:
            plan = {"raw_stdout": proc.stdout[-4000:]}
        result: dict[str, Any] = {
            "ok": True,
            "provider": self.provider,
            "dry_run": dry_run,
            "provider_request_id": "dry-run" if dry_run else None,
            "provider_job_id": None,
            "plan": plan,
            "timestamp_submitted": _utc_now(),
        }
        if not dry_run:
            # Last non-empty line is expected to be the HF Jobs job id.
            job_id = next(
                (line.strip() for line in reversed(proc.stdout.splitlines()) if line.strip()),
                None,
            )
            result["provider_job_id"] = job_id
            result["provider_request_id"] = job_id
        return result

    def status(self, manifest: AcceleratorRunManifestV1) -> dict[str, Any]:
        job_id = manifest.provider_job_id
        if not job_id:
            return {"ok": False, "error": "no provider_job_id"}
        proc = subprocess.run(
            ["hf", "jobs", "inspect", job_id, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr[-2000:]}
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"invalid JSON: {exc}"}
        if isinstance(payload, list) and payload:
            payload = payload[0]
        return {"ok": True, "payload": payload}

    def logs(self, manifest: AcceleratorRunManifestV1) -> str:
        job_id = manifest.provider_job_id
        if not job_id:
            return ""
        proc = subprocess.run(
            ["hf", "jobs", "logs", job_id],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.stdout if proc.returncode == 0 else proc.stderr

    def cancel(self, manifest: AcceleratorRunManifestV1) -> dict[str, Any]:
        job_id = manifest.provider_job_id
        if not job_id:
            return {"ok": False, "error": "no provider_job_id"}
        proc = subprocess.run(
            ["hf", "jobs", "cancel", job_id],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stderr": proc.stderr,
        }

    def cost(self, raw_records: Mapping[str, Any]) -> dict[str, Any]:
        # HF Jobs does not expose itemized billing in the CLI; surface what we can.
        return {
            "provider": self.provider,
            "billable_seconds": raw_records.get("billable_seconds"),
            "provider_cost_usd": raw_records.get("cost_usd"),
            "flavor": raw_records.get("flavor"),
            "instance_type": raw_records.get("flavor"),
        }

    @staticmethod
    def _normalize_status(payload: Mapping[str, Any]) -> str:
        for key in ("status", "stage", "state"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.lower()
            if isinstance(value, Mapping):
                for nested in ("stage", "status", "name"):
                    if isinstance(value.get(nested), str):
                        return str(value[nested]).lower()
        return "unknown"

    def reconcile(
        self,
        manifest: AcceleratorRunManifestV1,
        payload: Mapping[str, Any],
    ) -> AcceleratorRunManifestV1:
        status = self._normalize_status(payload)
        timestamps = dict(manifest.timestamps)
        timestamps["reconciled_at"] = _utc_now()
        notes = list(manifest.notes)
        disposition = manifest.disposition

        if status in {"completed", "finished", "success", "succeeded"}:
            notes.append(f"hf_jobs job {manifest.provider_job_id} completed")
            # Try to confirm checkpoint inventory from the bucket-side summary.
            bucket = _bucket_from_prefix(manifest.remote_uri_prefix)
            remote_summary = f"{bucket}/checkpoints/{manifest.run_id}/train_summary.json"
            summary = _fetch_bucket_json(remote_summary)
            if summary is not None:
                notes.append("train_summary.json located in bucket")
        elif status in {"cancelled", "canceled", "error", "failed"}:
            disposition = disposition or "inconclusive"
            notes.append(f"hf_jobs job {manifest.provider_job_id} terminal status: {status}")
        else:
            notes.append(f"hf_jobs job {manifest.provider_job_id} status: {status}")

        return _manifest_with(
            manifest,
            timestamps=timestamps,
            notes=tuple(notes),
            disposition=disposition,
            utilization={**dict(manifest.utilization), "last_status": status},
        )


class RemotePodAdapter(GPUProviderAdapter):
    """Adapter around ``scripts.remote_train``."""

    @property
    def provider(self) -> str:
        return "remote_pod"

    def _host(self, manifest: AcceleratorRunManifestV1) -> str:
        return str((manifest.provider_options or {}).get("host", ""))

    def _build_cli_args(
        self, manifest: AcceleratorRunManifestV1, *, dry_run: bool
    ) -> list[str]:
        host = self._host(manifest)
        bucket = _bucket_from_prefix(manifest.remote_uri_prefix)
        args: list[str] = [
            sys.executable,
            "-m",
            "scripts.remote_train",
            "--host",
            host,
            "--run-id",
            manifest.run_id,
            "--steps",
            str(max(1, min(manifest.target_decisions, 200))),
            "--checkpoint-bucket",
            bucket,
        ]
        if dry_run:
            args.append("--dry-run")
        return args

    def submit(
        self, manifest: AcceleratorRunManifestV1, *, dry_run: bool = False
    ) -> dict[str, Any]:
        cmd = self._build_cli_args(manifest, dry_run=dry_run)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        try:
            plan = json.loads(proc.stdout)
        except json.JSONDecodeError:
            plan = {"raw_stdout": proc.stdout[-4000:]}
        return {
            "ok": proc.returncode == 0,
            "provider": self.provider,
            "dry_run": dry_run,
            "provider_request_id": "dry-run" if dry_run else manifest.run_id,
            "provider_job_id": manifest.run_id if not dry_run else None,
            "plan": plan,
            "timestamp_submitted": _utc_now(),
            "returncode": proc.returncode,
            "stderr": proc.stderr[-4000:] if proc.returncode != 0 else "",
        }

    def status(self, manifest: AcceleratorRunManifestV1) -> dict[str, Any]:
        return {"ok": True, "payload": {"status": "unknown", "note": "remote_pod status requires SSH polling"}}

    def logs(self, manifest: AcceleratorRunManifestV1) -> str:
        return ""

    def cancel(self, manifest: AcceleratorRunManifestV1) -> dict[str, Any]:
        return {"ok": False, "error": "cancel not implemented for remote_pod"}

    def cost(self, raw_records: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "billable_seconds": raw_records.get("billable_seconds"),
            "provider_cost_usd": raw_records.get("cost_usd"),
        }

    def reconcile(
        self,
        manifest: AcceleratorRunManifestV1,
        payload: Mapping[str, Any],
    ) -> AcceleratorRunManifestV1:
        status = str(payload.get("status", "unknown")).lower()
        timestamps = dict(manifest.timestamps)
        timestamps["reconciled_at"] = _utc_now()
        notes = list(manifest.notes) + [f"remote_pod reconcile status: {status}"]
        return _manifest_with(
            manifest,
            timestamps=timestamps,
            notes=tuple(notes),
            utilization={**dict(manifest.utilization), "last_status": status},
        )


class DryRunAdapter(GPUProviderAdapter):
    """Adapter that only plans; it never submits real hardware."""

    @property
    def provider(self) -> str:
        return "dry_run"

    def submit(
        self, manifest: AcceleratorRunManifestV1, *, dry_run: bool = False
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "provider": self.provider,
            "dry_run": True,
            "provider_request_id": "dry-run",
            "provider_job_id": None,
            "plan": manifest.describe(),
            "timestamp_submitted": _utc_now(),
        }

    def status(self, manifest: AcceleratorRunManifestV1) -> dict[str, Any]:
        return {"ok": True, "payload": {"status": "dry_run"}}

    def logs(self, manifest: AcceleratorRunManifestV1) -> str:
        return "dry-run: no logs"

    def cancel(self, manifest: AcceleratorRunManifestV1) -> dict[str, Any]:
        return {"ok": True}

    def cost(self, raw_records: Mapping[str, Any]) -> dict[str, Any]:
        return {"provider": self.provider, "provider_cost_usd": None, "billable_seconds": None}

    def reconcile(
        self,
        manifest: AcceleratorRunManifestV1,
        payload: Mapping[str, Any],
    ) -> AcceleratorRunManifestV1:
        notes = list(manifest.notes) + ["dry-run reconcile: no durable artifact produced"]
        return _manifest_with(
            manifest,
            notes=tuple(notes),
            disposition=manifest.disposition or "inconclusive",
        )


def adapter_for(provider: str) -> GPUProviderAdapter:
    if provider == "hf_jobs":
        return HFJobsAdapter()
    if provider == "remote_pod":
        return RemotePodAdapter()
    if provider == "dry_run":
        return DryRunAdapter()
    raise ValueError(f"unsupported provider {provider!r}")


def _fetch_bucket_json(uri: str) -> dict[str, Any] | None:
    """Best-effort fetch of a JSON object from an hf:// bucket URI."""
    if not uri.startswith("hf://"):
        return None
    with tempfile.TemporaryDirectory(prefix="slm262-") as tmp:
        local = Path(tmp) / "payload.json"
        proc = subprocess.run(
            ["hf", "buckets", "cp", uri, str(local)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 or not local.is_file():
            return None
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


def summarize_telemetry(
    manifest: AcceleratorRunManifestV1,
    train_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate measured telemetry from the manifest and an optional train summary."""
    utilization = dict(manifest.utilization)
    if train_summary:
        utilization["train_seconds"] = train_summary.get("train_time_seconds")
        utilization["tokens_seen"] = train_summary.get("target_tokens_seen")
        utilization["steps"] = train_summary.get("steps")
    return {
        "run_id": manifest.run_id,
        "provider": manifest.provider,
        "billable_seconds": manifest.billable_seconds,
        "provider_cost_usd": manifest.provider_cost_usd,
        "cost_estimate_usd": manifest.cost_estimate_usd,
        "cost_confidence": manifest.cost_confidence,
        "energy_wh": manifest.energy_wh,
        "utilization": utilization,
        "target_decisions": manifest.target_decisions,
        "max_wall_minutes": manifest.max_wall_minutes,
    }
