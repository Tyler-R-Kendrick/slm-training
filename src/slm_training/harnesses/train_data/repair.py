"""Publish bounded data-builder output only as an admitted immutable successor."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.data.readiness_contract import DataReadinessRequest, scoped_path
from slm_training.data.store import DataStore, write_common_manifest
from slm_training.harnesses.train_data.readiness import check_readiness, snapshot_input


def repair_metadata(request: DataReadinessRequest) -> dict:
    """Shared candidate identity for preparation and immutable publication."""
    return {"dataset_id": request.successor_id, "kind": request.original.kind,
            "successor_of": request.original.manifest_sha256,
            "readiness_request_sha256": request.sha256,
            "preprocessing_identity": request.preprocessing_identity,
            "sampling_policy_digest": request.sampling_policy_digest}


def publish_repair(
    request: DataReadinessRequest, *, root: Path, staged: Path,
) -> dict:
    """Consume a private builder namespace; publish no adverse sealed replacement.

    The original data is never edited. Invalid rows remain in staging together
    with line/digest/reason evidence. A rejected hard family is not silently
    dropped. Only a wholly ready candidate is copied into DataStore.
    """
    root, staged = root.resolve(), staged.resolve()
    staged.relative_to(root / "outputs" / "runs")
    scoped_path(root, staged.relative_to(root).as_posix())
    if staged.name != request.successor_id:
        raise ValueError("private candidate namespace must use the locked successor id")
    metadata = repair_metadata(request)
    manifest_path = staged / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if existing.get("immutable"):
        raise ValueError("cannot edit an immutable candidate snapshot")
    manifest_path.write_text(json.dumps({**existing, **metadata}) + "\n")
    write_common_manifest(staged, kind=request.original.kind, dataset_id=request.successor_id)
    candidate = snapshot_input(root, staged, exposure=request.original.exposure)
    observation = check_readiness(request, root=root, candidate=candidate)
    (staged / "data_readiness.json").write_text(json.dumps(observation, indent=2) + "\n")
    (staged / "readiness_rejected.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in observation["rejected"]))
    if not observation["ready"]:
        return {"published": False, "evidence": observation}
    store = DataStore(root)
    published = store.publish_successor(request.original.kind, request.successor_id,
                                        source=staged, metadata=metadata)
    candidate = snapshot_input(root, published.path, exposure=request.original.exposure)
    current = check_readiness(request, root=root, candidate=candidate)
    return {"published": True, "candidate": candidate.model_dump(), "evidence": current}
