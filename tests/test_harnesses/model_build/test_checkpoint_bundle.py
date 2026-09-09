"""Failure boundaries of the real local publisher (no promotion gate mocks)."""

import hashlib
import json
from pathlib import Path

import pytest

from slm_training.harness_core import checkpoint_bundle as bundles


def _checkpoint(root: Path, marker: bytes = b"weights") -> Path:
    root.mkdir(parents=True)
    path = root / "last.pt"
    path.write_bytes(marker)
    path.with_suffix(".tokenizer.json").write_text('{"token_to_id": {"a": 0}}')
    path.with_name("last.context.tokenizer.json").write_text(
        '{"token_to_id": {"b": 0}}'
    )
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "kind": "twotower",
                "output_contract_version": 2,
                "context_tokenizer": "last.context.tokenizer.json",
            }
        )
    )
    return path


def _publish(root, checkpoint, expected=None, fence="current"):
    digest = bundles.stage_checkpoint_bundle(root, checkpoint, {"claim": "fixture"})
    bundles.publish_bundle(
        root,
        digest,
        expected_digest=expected,
        fence=fence,
        validate_fence=lambda value: value == "current",
    )
    return digest


def test_bundle_rejects_missing_new_sidecar_and_preserves_old(tmp_path):
    root = tmp_path / "store"
    first = _publish(root, _checkpoint(tmp_path / "first"))
    second = _checkpoint(tmp_path / "second", b"new")
    second.with_name("last.context.tokenizer.json").unlink()
    with pytest.raises(ValueError, match="missing_or_linked"):
        _publish(root, second, first)
    directory, _ = bundles.resolve_bundle(root)
    assert directory.name == first
    assert (directory / "last.pt").read_bytes() == b"weights"


@pytest.mark.parametrize("value", [True, False, 1.0, -1, "1", None])
def test_component_bytes_reject_noninteger_counts_without_publication(tmp_path, value):
    root = tmp_path / "store"
    first = _publish(root, _checkpoint(tmp_path / "source", b"x"))
    directory, manifest = bundles.validate_bundle(root, first)
    manifest["files"]["last.pt"]["bytes"] = value
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    successor = directory.parent / digest
    successor.mkdir()
    for name in manifest["files"]:
        (successor / name).write_bytes((directory / name).read_bytes())
    (successor / "manifest.json").write_bytes(raw)
    with pytest.raises(ValueError, match="invalid_component_bytes"):
        bundles.publish_bundle(
            root,
            digest,
            expected_digest=first,
            fence="current",
            validate_fence=lambda value: value == "current",
        )
    assert bundles.current_bundle_digest(root) == first


def test_bundle_stale_fence_and_cas_conflict(tmp_path):
    root = tmp_path / "store"
    first = _publish(root, _checkpoint(tmp_path / "first"))
    second = _checkpoint(tmp_path / "second", b"new")
    with pytest.raises(ValueError, match="stale_fence"):
        _publish(root, second, first, fence="revoked")
    with pytest.raises(ValueError, match="cas_conflict"):
        _publish(root, second, None)
    assert bundles.current_bundle_digest(root) == first
    digest = _publish(root, second, first)
    # Publication may have succeeded just before a terminal event failed.
    assert _publish(root, second, first) == digest
    with pytest.raises(ValueError, match="stale_fence"):
        _publish(root, second, first, fence="revoked")


@pytest.mark.parametrize(
    "boundary", ["copy", "file_fsync", "pointer", "directory_fsync"]
)
def test_publication_failure_is_previous_or_complete(tmp_path, monkeypatch, boundary):
    root = tmp_path / "store"
    first = _publish(root, _checkpoint(tmp_path / "first"))
    second = _checkpoint(tmp_path / "second", b"new")

    def fail(*args, **kwargs):
        raise OSError("injected disk failure")

    with monkeypatch.context() as patch:
        if boundary == "copy":
            patch.setattr(bundles.shutil, "copyfile", fail)
        elif boundary == "file_fsync":
            patch.setattr(bundles.os, "fsync", fail)
        elif boundary == "pointer":
            patch.setattr(bundles, "_atomic_write", fail)
        else:
            original = bundles._sync_directory
            patch.setattr(
                bundles,
                "_sync_directory",
                lambda path: fail() if path == root else original(path),
            )
        with pytest.raises(OSError, match="injected"):
            _publish(root, second, first)
    directory, manifest = bundles.resolve_bundle(root)
    assert (directory / "last.pt").read_bytes() in {b"weights", b"new"}
    assert set(manifest["files"]) == {
        "last.pt",
        "last.meta.json",
        "last.tokenizer.json",
        "last.context.tokenizer.json",
    }


def test_corruption_and_linked_input_refused(tmp_path):
    root = tmp_path / "store"
    source = _checkpoint(tmp_path / "first")
    digest = _publish(root, source)
    directory, _ = bundles.resolve_bundle(root)
    (directory / "last.tokenizer.json").write_text("corrupt")
    with pytest.raises(ValueError, match="digest_mismatch"):
        bundles.resolve_bundle(root)
    linked = tmp_path / "linked.pt"
    linked.symlink_to(source)
    with pytest.raises(ValueError):
        bundles.stage_checkpoint_bundle(root, linked, {})
    with pytest.raises(ValueError, match="invalid_digest"):
        bundles.validate_bundle(root, "../" + digest)


def test_real_runtime_fence_blocks_epoch_replacement(tmp_path):
    from slm_training.autoresearch.runtime.activity_runtime import (
        ActivityRuntime,
        StaleLease,
    )
    from slm_training.autoresearch.storage import CampaignStore
    from tests.test_autoresearch.test_activity_runtime import spec

    root = tmp_path / "bundles"
    digest = bundles.stage_checkpoint_bundle(root, _checkpoint(tmp_path / "source"), {})
    store = CampaignStore("artifact-integration", tmp_path / "campaigns")
    with ActivityRuntime(store) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        bundles.publish_activity_bundle(runtime, lease, root, digest, None)
    with ActivityRuntime(store) as runtime:
        with pytest.raises(StaleLease):
            bundles.publish_activity_bundle(runtime, lease, root, digest, None)
    assert bundles.current_bundle_digest(root) == digest


@pytest.mark.parametrize("authorized", [True, False])
def test_trusted_child_champion_scope_uses_real_delegated_guard(tmp_path, authorized):
    import sys

    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.harness_core.activity_contract import ActivitySpec, ResourceGrant
    from slm_training.autoresearch.storage import CampaignStore

    checkpoint = _checkpoint(tmp_path / "source")
    store = CampaignStore("delegate", tmp_path / "campaigns")
    caps = (
        ("local_process", "controller_publication")
        if authorized
        else ("local_process",)
    )
    spec = ActivitySpec(
        activity_id="publish",
        family="publication",
        kind="control",
        source_digest="a" * 64,
        environment_digest="b" * 64,
        input_digest="c" * 64,
        output_namespace="attempt",
        capabilities=caps,
        # This tests ownership, not a five-second import-latency endpoint.
        # Leave CPU scheduling headroom while retaining a bounded real child.
        grant=ResourceGrant(
            interrupt_seconds=30, kill_grace_seconds=1, total_seconds=32
        ),
    )
    child = """
import json,sys
from pathlib import Path
from slm_training.autoresearch.runtime.activity_publication import DelegatedPublisher
from slm_training.harness_core.activity_contract import ActivityLease
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.checkpoint_publication import champion_publication_scope
from slm_training.autoresearch.hillclimb import ClimbChampionSidecar,write_climb_champion,load_climb_champion
store=CampaignStore("delegate",Path(sys.argv[1]))
lease=ActivityLease.model_validate_json(sys.argv[2])
publisher=DelegatedPublisher(store,source_digest="a"*64)
loop=Path(sys.argv[3])
with champion_publication_scope(publisher,lease,loop_dir=loop):
    sidecar=ClimbChampionSidecar(source_campaign="integrity_fixture",cumulative_steps=0,train_data_manifest_sha="",cumulative_epochs=0.0,status="baseline_seed")
    write_climb_champion(loop,checkpoint=Path(sys.argv[4]),sidecar=sidecar)
    assert load_climb_champion(loop)==sidecar
"""
    with ActivityRuntime(store) as runtime:
        runtime.register(spec)
        lease = runtime.claim_next(capabilities=set(caps))
        result = runtime.run(
            lease,
            [
                sys.executable,
                "-c",
                child,
                str(tmp_path / "campaigns"),
                lease.model_dump_json(),
                str(tmp_path / "loop"),
                str(checkpoint),
            ],
            cwd=Path.cwd(),
        )
    assert (result.returncode == 0) is authorized, result.stderr
    assert ((tmp_path / "loop/champion/current.json").exists()) is authorized
    if authorized:
        pointer = json.loads((tmp_path / "loop/champion/current.json").read_text())
        assert pointer["fence"] == lease.token
        assert (
            bundles.resolve_bundle(tmp_path / "loop/champion")[1]["metadata"][
                "source_campaign"
            ]
            == "integrity_fixture"
        )


def test_champion_scope_cannot_fallback_after_lease_revocation(tmp_path):
    from slm_training.autoresearch.runtime.activity_runtime import (
        ActivityRuntime,
        StaleLease,
    )
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.checkpoint_publication import (
        champion_publication_scope,
        publish_champion_checkpoint,
    )
    from tests.test_autoresearch.test_activity_runtime import spec

    checkpoint = _checkpoint(tmp_path / "source")
    with ActivityRuntime(CampaignStore("scope", tmp_path / "campaigns")) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        with champion_publication_scope(runtime, lease, loop_dir=tmp_path):
            runtime.cancel(lease.activity_id, reason="revocation fixture")
            with pytest.raises(StaleLease):
                publish_champion_checkpoint(tmp_path / "champion", checkpoint, {})
    assert not (tmp_path / "champion/current.json").exists()


@pytest.mark.parametrize("after_pointer", [False, True])
def test_real_process_death_at_pointer_boundary(tmp_path, after_pointer):
    import multiprocessing
    import os

    root = tmp_path / "store"
    first = _publish(root, _checkpoint(tmp_path / "first"))
    second = _checkpoint(tmp_path / "second", b"new")

    def crash():
        original = bundles._atomic_write

        def interrupted(*args, **kwargs):
            if after_pointer:
                original(*args, **kwargs)
            os._exit(70)

        bundles._atomic_write = interrupted
        _publish(root, second, first)

    process = multiprocessing.get_context("fork").Process(target=crash)
    process.start()
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("bounded publisher fixture hung")
    assert process.exitcode == 70
    directory, _ = bundles.resolve_bundle(root)
    assert (directory / "last.pt").read_bytes() == (
        b"new" if after_pointer else b"weights"
    )


def test_write_climb_champion_carries_tokenizer_sidecars(tmp_path: Path) -> None:
    """A warm start reads the tokenizer sidecars beside the checkpoint.

    Copying only ``last.pt`` left the context sidecar missing, so
    ``TwoTowerModel.load`` fell back to the output tokenizer and refused the
    warm start with 'scratch-context warm starts require OpenUITokenizer
    sidecars' — the champion existed, the regime said climb, and every arm
    still died.
    """
    from slm_training.autoresearch.hillclimb import (
        CLIMB_CHAMPION_STATUS_BASELINE_SEED,
        ClimbChampionSidecar,
        climb_champion_checkpoint_path,
        write_climb_champion,
    )

    run = tmp_path / "run" / "checkpoints"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"weights")
    (run / "last.tokenizer.json").write_text('{"output": 1}', encoding="utf-8")
    (run / "last.context.tokenizer.json").write_text('{"context": 1}', encoding="utf-8")
    (run / "last.meta.json").write_text(
        json.dumps(
            {
                "kind": "twotower",
                "output_contract_version": 2,
                "context_tokenizer": "last.context.tokenizer.json",
            }
        )
    )

    loop_dir = tmp_path / "loop"
    sidecar = ClimbChampionSidecar(
        source_campaign="c1",
        cumulative_steps=53,
        train_data_manifest_sha="",
        cumulative_epochs=0.0,
        status=CLIMB_CHAMPION_STATUS_BASELINE_SEED,
    )
    write_climb_champion(loop_dir, checkpoint=run / "last.pt", sidecar=sidecar)

    ckpt = climb_champion_checkpoint_path(loop_dir)
    assert ckpt.read_bytes() == b"weights"
    assert (
        ckpt.with_name("last.tokenizer.json").read_text(encoding="utf-8")
        == '{"output": 1}'
    )
    assert (
        ckpt.with_name("last.context.tokenizer.json").read_text(encoding="utf-8")
        == '{"context": 1}'
    )


def test_write_climb_champion_without_sidecars_refuses_publication(
    tmp_path: Path,
) -> None:
    """A missing deployment component cannot inherit an earlier sidecar."""
    from slm_training.autoresearch.hillclimb import (
        CLIMB_CHAMPION_STATUS_BASELINE_SEED,
        ClimbChampionSidecar,
        climb_champion_checkpoint_path,
        write_climb_champion,
    )

    run = tmp_path / "run" / "checkpoints"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"weights")
    loop_dir = tmp_path / "loop"
    with pytest.raises(ValueError, match="missing_or_linked_component"):
        write_climb_champion(
            loop_dir,
            checkpoint=run / "last.pt",
            sidecar=ClimbChampionSidecar(
                source_campaign="c1",
                cumulative_steps=1,
                train_data_manifest_sha="",
                cumulative_epochs=0.0,
                status=CLIMB_CHAMPION_STATUS_BASELINE_SEED,
            ),
        )
    ckpt = climb_champion_checkpoint_path(loop_dir)
    assert not ckpt.is_file()
    assert not ckpt.with_name("last.tokenizer.json").exists()
    assert not ckpt.with_name("last.context.tokenizer.json").exists()
