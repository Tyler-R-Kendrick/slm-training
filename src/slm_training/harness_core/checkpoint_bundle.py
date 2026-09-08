"""Shared immutable checkpoint bundles; controller-owned, fenced pointer updates.

The caller owns authority and the event journal. A successful CAS is reconciled
by its bundle digest after a crash; staging alone never publishes anything.
This is a local POSIX filesystem protocol, not a multi-host lease service.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from slm_training.harness_core.lineage.store import _atomic_write


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: dict) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"bundle:missing_or_linked_component:{path.name}")
    return path


def _components(checkpoint: Path, full_state: Path | None) -> dict[str, Path]:
    meta_path = _regular(checkpoint.with_suffix(".meta.json"))
    meta = json.loads(meta_path.read_text())
    if meta.get("kind") != "twotower" or meta.get("output_contract_version") != 2:
        raise ValueError("bundle:unsupported_model_contract")
    files = {"last.pt": _regular(checkpoint), "last.meta.json": meta_path}
    files["last.tokenizer.json"] = _regular(checkpoint.with_suffix(".tokenizer.json"))
    context = checkpoint.with_name(checkpoint.stem + ".context.tokenizer.json")
    if meta.get("context_tokenizer"):
        if meta["context_tokenizer"] != context.name:
            raise ValueError("bundle:context_tokenizer_name_mismatch")
        files["last.context.tokenizer.json"] = _regular(context)
    if full_state is not None:
        files["last_full_state.pt"] = _regular(full_state)
    return files


def stage_checkpoint_bundle(
    root: Path, checkpoint: Path, metadata: dict, *, full_state: Path | None = None
) -> str:
    """Copy a complete snapshot into a private stage and seal by content identity.

    Resume-state presence is recorded, never interpreted as an exact-resume
    certificate. ``full_state.validate_resume_contract`` owns that decision.
    """
    root = Path(root)
    files = _components(Path(checkpoint), full_state)
    source_dir = Path(checkpoint).parent
    if source_dir.parent.name == "bundles":
        _, source_manifest = validate_bundle(source_dir.parent.parent, source_dir.name)
        provenance = source_manifest["metadata"]
        metadata = {
            **metadata,
            **{
                key: provenance[key]
                for key in ("exposure", "ancestor_exposure")
                if key in provenance
            },
        }
    bundles = root / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=bundles))
    try:
        entries = {}
        for name, source in files.items():
            before = _digest(source.read_bytes())
            target = stage / name
            shutil.copyfile(source, target)
            with target.open("rb") as handle:
                os.fsync(handle.fileno())
            if (
                _digest(target.read_bytes()) != before
                or _digest(source.read_bytes()) != before
            ):
                raise ValueError("bundle:source_changed_during_stage")
            entries[name] = {"sha256": before, "bytes": target.stat().st_size}
        manifest = {
            "schema": "checkpoint_bundle/v1",
            "files": entries,
            "metadata": metadata,
            "resume_state_present": full_state is not None,
        }
        data = _json(manifest)
        digest = _digest(data)
        with (stage / "manifest.json").open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(stage)
        destination = bundles / digest
        try:
            os.rename(stage, destination)
        except OSError:
            if not destination.is_dir():
                raise
            validate_bundle(root, digest)
        _sync_directory(bundles)
        return digest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def validate_bundle(root: Path, digest: str) -> tuple[Path, dict]:
    """Resolve and validate every component, with no fallback to old sidecars."""
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("bundle:invalid_digest")
    directory = Path(root) / "bundles" / digest
    if directory.is_symlink():
        raise ValueError("bundle:linked_directory")
    data = _regular(directory / "manifest.json").read_bytes()
    if _digest(data) != digest:
        raise ValueError("bundle:manifest_digest_mismatch")
    manifest = json.loads(data)
    _validate_manifest_header(manifest)
    files = manifest["files"]
    required = {"last.pt", "last.meta.json", "last.tokenizer.json"}
    if not required <= files.keys():
        raise ValueError("bundle:missing_required_components")
    for name, entry in files.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("bundle:invalid_component_path")
        _validate_component_entry(entry)
        path = _regular(directory / name)
        if (
            path.stat().st_size != entry["bytes"]
            or _digest(path.read_bytes()) != entry["sha256"]
        ):
            raise ValueError("bundle:component_digest_mismatch")
    meta = json.loads((directory / "last.meta.json").read_text())
    if meta.get("kind") != "twotower" or meta.get("output_contract_version") != 2:
        raise ValueError("bundle:unsupported_model_contract")
    if meta.get("context_tokenizer") and "last.context.tokenizer.json" not in files:
        raise ValueError("bundle:missing_context_tokenizer")
    return directory, manifest


def _validate_component_entry(entry: dict) -> None:
    if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}:
        raise ValueError("bundle:invalid_component_fields")
    if type(entry["bytes"]) is not int or entry["bytes"] < 0:
        raise ValueError("bundle:invalid_component_bytes")
    digest = entry["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("bundle:invalid_component_digest")


def _validate_manifest_header(manifest: dict) -> None:
    required = {"schema", "files", "metadata", "resume_state_present"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("bundle:invalid_manifest_fields")
    if manifest["schema"] != "checkpoint_bundle/v1":
        raise ValueError("bundle:unsupported_schema")
    if not isinstance(manifest["metadata"], dict) or not isinstance(
        manifest["files"], dict
    ):
        raise ValueError("bundle:invalid_manifest_objects")
    files = manifest["files"]
    allowed = {
        "last.pt",
        "last.meta.json",
        "last.tokenizer.json",
        "last.context.tokenizer.json",
        "last_full_state.pt",
    }
    if not files.keys() <= allowed:
        raise ValueError("bundle:unexpected_component")
    if type(manifest["resume_state_present"]) is not bool or manifest[
        "resume_state_present"
    ] != ("last_full_state.pt" in files):
        raise ValueError("bundle:resume_component_mismatch")


def resolve_bundle(root: Path) -> tuple[Path, dict] | None:
    """Read the pointer once, then use only that immutable bundle."""
    pointer = Path(root) / "current.json"
    if not pointer.exists():
        return None
    payload = json.loads(_regular(pointer).read_text())
    if payload.get("schema") != "checkpoint_pointer/v1":
        raise ValueError("bundle:unsupported_pointer")
    return validate_bundle(root, payload["bundle_digest"])


def current_bundle_digest(root: Path) -> str | None:
    bundle = resolve_bundle(root)
    return bundle[0].name if bundle else None


def publish_bundle(
    root: Path,
    digest: str,
    *,
    expected_digest: str | None,
    fence: str,
    validate_fence: Callable[[str], bool],
) -> Path:
    """Controller-only CAS. Revoked leases cannot publish even idempotent output.

    ``validate_fence`` must check the authoritative current lease, not a worker's
    assertion. The controller must serialize lease revocation with this critical
    section. No signing key or scientific authority is stored in this module.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not fence or validate_fence(fence) is not True:
            raise ValueError("bundle:stale_fence")
        validate_bundle(root, digest)
        current = current_bundle_digest(root)
        if current == digest:
            return root / "current.json"
        if current != expected_digest:
            raise ValueError("bundle:cas_conflict")
        pointer = _atomic_write(
            root / "current.json",
            {
                "schema": "checkpoint_pointer/v1",
                "bundle_digest": digest,
                "previous_digest": current,
                "fence": fence,
            },
        )
        _sync_directory(root)
        return pointer


def seal_trial_checkpoint(
    checkpoint: Path, config, manifest_sha: str, step: int, exposure: list[dict]
) -> Path:
    """Seal a trial cursor in its own run namespace, without champion publication."""
    full_state = checkpoint.parent / "last_full_state.pt"
    full_state = full_state if config.full_state_checkpoint else None
    root = checkpoint.parent / "trial"
    parent = config.initialize_from or config.resume_from
    ancestry = None if parent else []
    if parent and (Path(parent).parent / "manifest.json").is_file():
        parent_dir = Path(parent).parent
        _, parent_manifest = validate_bundle(parent_dir.parent.parent, parent_dir.name)
        parent_meta = parent_manifest["metadata"]
        ancestry = parent_meta.get("ancestor_exposure")
        if config.initialize_from and ancestry is not None:
            ancestry = [*ancestry, *parent_meta.get("exposure", [])]
    digest = stage_checkpoint_bundle(
        root,
        checkpoint,
        {
            "role": "trial_cursor",
            "run_id": config.run_id,
            "optimizer_updates": step,
            "data_manifest_sha": manifest_sha,
            "claimed_promotion": False,
            "exposure": exposure,
            "ancestor_exposure": ancestry,
        },
        full_state=full_state,
    )
    directory, _manifest = validate_bundle(root, digest)
    _atomic_write(
        checkpoint.parent / "trial_cursor.json",
        {
            "schema": "trial_cursor/v1",
            "bundle_digest": digest,
            "checkpoint": str(directory / "last.pt"),
            "resume_from": str(directory / "last_full_state.pt")
            if full_state
            else None,
            "optimizer_updates": step,
            "role": "trial_cursor",
        },
    )
    _sync_directory(checkpoint.parent)
    return directory / "last.pt"


def published_resume_state(state: Path | None) -> Path | None:
    """Resolve a byte-identical published state without inferring legacy ancestry.

    A newer unsealed periodic state remains resumable through the existing loader;
    it must not inherit the older bundle's exposure. Corrupt cursors fail closed.
    """
    if state is None:
        return None
    state = _regular(Path(state))
    cursor = state.parent / "trial_cursor.json"
    if not cursor.exists() and not cursor.is_symlink():
        return state
    payload = json.loads(_regular(cursor).read_text())
    fields = {
        "schema",
        "role",
        "optimizer_updates",
        "bundle_digest",
        "checkpoint",
        "resume_from",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload["schema"] != "trial_cursor/v1"
        or payload["role"] != "trial_cursor"
        or type(payload["optimizer_updates"]) is not int
        or payload["optimizer_updates"] < 0
        or not isinstance(payload["bundle_digest"], str)
    ):
        raise ValueError("trial_cursor:invalid_resume_pointer")
    directory, manifest = validate_bundle(
        state.parent / "trial", payload["bundle_digest"]
    )
    published = directory / "last_full_state.pt"
    if (
        not manifest["resume_state_present"]
        or manifest["metadata"].get("role") != "trial_cursor"
        or manifest["metadata"].get("optimizer_updates") != payload["optimizer_updates"]
        or payload.get("checkpoint") != str(directory / "last.pt")
        or payload.get("resume_from") != str(published)
    ):
        raise ValueError("trial_cursor:resume_identity_mismatch")
    return (
        published
        if _digest(state.read_bytes())
        == manifest["files"]["last_full_state.pt"]["sha256"]
        else state
    )


def publish_activity_bundle(
    runtime, lease, root: Path, digest: str, expected_digest: str | None
) -> Path:
    """Publish while the canonical runtime holds lease revocation exclusion."""
    with runtime.publication(lease):
        return publish_bundle(
            root,
            digest,
            expected_digest=expected_digest,
            fence=lease.token,
            validate_fence=lambda token: token == lease.token,
        )
