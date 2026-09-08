"""Trusted checkpoint publication scope; lease authority stays in ActivityRuntime.

This is controller composition, not a sandbox or worker authentication system.
Only the trusted driver receives the canonical runtime's delegated publisher;
training/decode workloads must never receive this object or controller storage.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.harness_core.checkpoint_bundle import (
    current_bundle_digest,
    publish_bundle,
    stage_checkpoint_bundle,
)


@dataclass(frozen=True)
class _PublicationScope:
    publisher: Any  # Canonical RUNTIME owner with publication(lease).
    lease: Any
    root: Path
    artifact_root: Path
    pid: int


_SCOPE: ContextVar[_PublicationScope | None] = ContextVar(
    "champion_publication", default=None
)


@contextmanager
def champion_publication_scope(publisher, lease, *, loop_dir: Path):
    """Bind all nested champion writes to a canonical live publication guard.

    Pass the canonical DelegatedPublisher for a trusted child, or ActivityRuntime
    for an in-process controller.
    The delegated guard is created/authorized by RUNTIME, not from an env flag.
    RUNTIME's publication guard validates delegation/epoch/process/lease and holds
    the common lock across CAS. A revoked or expired guard raises; no fallback.
    """
    if _SCOPE.get() is not None:
        raise ValueError("champion_publication:nested_scope")
    with publisher.publication(lease):
        scope = _PublicationScope(
            publisher,
            lease,
            (Path(loop_dir) / "champion").resolve(),
            (loop_dir.parent.parent if loop_dir.parent.name == "loops" else loop_dir).resolve(),
            os.getpid(),
        )
    token = _SCOPE.set(scope)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def has_champion_publication_scope() -> bool:
    return _SCOPE.get() is not None


@contextmanager
def controller_artifact_publication(destination: Path):
    """Fence trusted driver projections under the same runtime publication owner.

    A direct synchronous CLI has no lease claim; the supervised driver always
    enters the scope. This grants no filesystem access to untrusted workers.
    """
    scope = _SCOPE.get()
    if scope is None:
        yield None
        return
    if os.getpid() != scope.pid:
        raise ValueError("controller_publication:foreign_process")
    destination.resolve().relative_to(scope.artifact_root)
    with scope.publisher.publication(scope.lease):
        yield scope.lease.token


def publish_champion_checkpoint(
    root: Path,
    checkpoint: Path,
    metadata: dict,
    *,
    expected_digest: str | None = None,
    fence: str | None = None,
    validate_fence=None,
) -> Path:
    scope = _SCOPE.get()
    if scope is not None:
        if os.getpid() != scope.pid or root.resolve() != scope.root:
            raise ValueError("champion_publication:foreign_process_or_destination")
        if fence is not None or validate_fence is not None:
            raise ValueError("champion_publication:scope_override_forbidden")
        previous = (
            current_bundle_digest(root) if expected_digest is None else expected_digest
        )
        digest = stage_checkpoint_bundle(root, checkpoint, metadata)
        with scope.publisher.publication(scope.lease):
            return publish_bundle(
                root,
                digest,
                expected_digest=previous,
                fence=scope.lease.token,
                validate_fence=scope.lease.token.__eq__,
            )
    # Explicit legacy synchronous-controller compatibility. Supervised driver
    # dispatch must enter the trusted scope; this branch is not lease evidence.
    if fence is None:
        expected_digest = current_bundle_digest(root)
        fence = "local-controller:" + uuid.uuid4().hex
        validate_fence = fence.__eq__
    if validate_fence is None:
        raise ValueError("champion_publication:missing_fence_validator")
    digest = stage_checkpoint_bundle(root, checkpoint, metadata)
    return publish_bundle(
        root,
        digest,
        expected_digest=expected_digest,
        fence=fence,
        validate_fence=validate_fence,
    )
