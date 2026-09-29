"""Private staging and durable local directory publication for DataStore.

Local Linux filesystem only. The lock serializes cooperating publishers; this
is not an OS sandbox and does not authenticate a worker's scientific claims.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_tree(source: Path, destination: Path, *, metadata: dict) -> None:
    from slm_training.data.store import (
        MAX_GIT_FILE_BYTES, _relocate_manifest_paths, write_common_manifest,
    )

    source = source.resolve()
    for path in source.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
            raise ValueError("data publication refuses symlinks/hardlinks")
        if path.is_file() and path.stat().st_size >= MAX_GIT_FILE_BYTES:
            raise ValueError("data file exceeds publication cap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ".publication.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            raise FileExistsError(f"published dataset is immutable: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=".data-stage-", dir=destination.parent))
        try:
            shutil.copytree(source, staging, dirs_exist_ok=True)
            manifest = json.loads((staging / "manifest.json").read_text())
            manifest = _relocate_manifest_paths(manifest, source=source,
                                                destination=destination)
            manifest.update(metadata)
            (staging / "manifest.json").write_text(json.dumps(manifest) + "\n")
            write_common_manifest(staging, kind=manifest["kind"],
                                  dataset_id=manifest["dataset_id"], immutable=True,
                                  trace_id=manifest.get("trace_id"))
            for path in staging.rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            for directory in sorted((p for p in staging.rglob("*") if p.is_dir()),
                                    key=lambda p: len(p.parts), reverse=True):
                _fsync_directory(directory)
            _fsync_directory(staging)
            os.rename(staging, destination)
            _fsync_directory(destination.parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)  # Private directory created by this invocation.

