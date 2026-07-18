"""HF Bucket checkpoint sync helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_training.harnesses.model_build.checkpoint_bucket import (
    DEFAULT_CHECKPOINT_BUCKET_URI,
    maybe_sync_train_checkpoints,
    normalize_bucket_uri,
    remote_run_prefix,
    resolve_sync_checkpoints,
    sync_run_checkpoints,
)


def test_normalize_bucket_uri_variants() -> None:
    assert normalize_bucket_uri("TKendrick/OpenUI") == "hf://buckets/TKendrick/OpenUI"
    assert (
        normalize_bucket_uri("https://huggingface.co/buckets/TKendrick/OpenUI")
        == "hf://buckets/TKendrick/OpenUI"
    )
    assert normalize_bucket_uri(None) == DEFAULT_CHECKPOINT_BUCKET_URI


def test_remote_run_prefix() -> None:
    assert (
        remote_run_prefix("hf://buckets/TKendrick/OpenUI", "twotower_v1")
        == "hf://buckets/TKendrick/OpenUI/checkpoints/twotower_v1"
    )


def test_resolve_sync_auto() -> None:
    # Default / explicit off stays local-only.
    assert (
        resolve_sync_checkpoints(
            sync_checkpoints=False, context_backend="hf", explicit_bucket=None
        )
        is False
    )
    # Full-train CLI forces True.
    assert (
        resolve_sync_checkpoints(
            sync_checkpoints=True, context_backend="hf", explicit_bucket=None
        )
        is True
    )
    # Legacy None auto: HF on, scratch off, empty bucket disables.
    assert (
        resolve_sync_checkpoints(
            sync_checkpoints=None, context_backend="hf", explicit_bucket=None
        )
        is True
    )
    assert (
        resolve_sync_checkpoints(
            sync_checkpoints=None, context_backend="scratch", explicit_bucket=None
        )
        is False
    )
    assert (
        resolve_sync_checkpoints(
            sync_checkpoints=None, context_backend="hf", explicit_bucket=""
        )
        is False
    )


def test_staging_and_dry_run_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("huggingface_hub")

    class FakePlan:
        def __init__(self) -> None:
            self.source = "stage"
            self.dest = "remote"
            self.timestamp = "t"
            self.operations = [SimpleNamespace(action="upload", path="last.pt", size=1)]

    calls: dict[str, object] = {}

    class FakeApi:
        def __init__(self, token=None) -> None:  # noqa: ANN001
            calls["token"] = token

        def sync_bucket(self, **kwargs):  # noqa: ANN003
            calls["sync"] = kwargs
            return FakePlan()

        def create_bucket(self, *args, **kwargs):  # noqa: ANN002, ANN003
            calls["create"] = kwargs
            return "https://huggingface.co/buckets/TKendrick/OpenUI"

        def bucket_info(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return SimpleNamespace(private=False)

    import slm_training.harnesses.model_build.checkpoint_bucket as cb

    monkeypatch.setattr(cb, "_require_hub", lambda: (FakeApi, lambda: "tok"))

    run_dir = tmp_path / "runs" / "demo"
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "last.pt").write_bytes(b"weights")
    (ckpt / "last.tokenizer.json").write_text("{}", encoding="utf-8")
    (ckpt / "last.meta.json").write_text("{}", encoding="utf-8")
    (run_dir / "train_summary.json").write_text("{}", encoding="utf-8")

    report = sync_run_checkpoints(
        ckpt,
        run_id="demo",
        bucket="TKendrick/OpenUI",
        run_dir=run_dir,
        dry_run=True,
        ensure_bucket=False,
    )
    assert report["ok"] is True
    assert report["remote_uri"].endswith("/checkpoints/demo")
    assert "last.pt" in report["files"]
    assert "train_summary.json" in report["files"]
    assert calls["sync"]["dry_run"] is True


def test_cli_sync_ignores_progress_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    import slm_training.harnesses.model_build.checkpoint_bucket as cb

    output = "\n".join(
        [
            "Syncing...",
            '{"type":"header","summary":{"uploads":1,"skips":0}}',
            '{"type":"operation","action":"upload","path":"last.pt","size":7}',
            "Sync completed.",
        ]
    )
    monkeypatch.setattr(
        cb.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    plan = cb._cli_sync_bucket(
        source="checkpoints",
        dest="hf://buckets/TKendrick/OpenUI/checkpoints/demo",
        dry_run=False,
        token=None,
    )
    assert plan["uploads"] == 1
    assert plan["operations"][0]["path"] == "last.pt"


def test_maybe_sync_respects_disabled(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        sync_checkpoints=False,
        context_backend="hf",
        checkpoint_bucket=None,
        run_id="x",
        run_dir=tmp_path,
        checkpoint_bucket_dry_run=False,
    )
    assert maybe_sync_train_checkpoints(cfg, tmp_path) is None


class _Plan:
    def __init__(self, ops: list[SimpleNamespace]) -> None:
        self.operations = ops


class _FakeApi:
    """HF api stub. Real syncs 'upload'; verify dry-runs report nothing pending
    unless ``fail_verify`` is set (simulating an upload that did not land)."""

    def __init__(self, token=None, *, fail_verify: bool = False) -> None:  # noqa: ANN001
        self.fail_verify = fail_verify
        self.dry_run_calls = 0
        self.real_calls = 0

    def sync_bucket(self, **kwargs):  # noqa: ANN003
        if kwargs.get("dry_run"):
            self.dry_run_calls += 1
            if self.fail_verify:
                return _Plan([SimpleNamespace(action="upload", path="last.pt", size=7)])
            return _Plan([])
        self.real_calls += 1
        return _Plan([SimpleNamespace(action="upload", path="last.pt", size=7)])

    def create_bucket(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return "https://huggingface.co/buckets/TKendrick/OpenUI"

    def bucket_info(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return SimpleNamespace(private=False)


def _make_checkpoint(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "demo"
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "last.pt").write_bytes(b"weights")
    (ckpt / "last.tokenizer.json").write_text("{}", encoding="utf-8")
    (run_dir / "train_summary.json").write_text("{}", encoding="utf-8")
    return ckpt


def _patch_api(monkeypatch: pytest.MonkeyPatch, api: _FakeApi) -> None:
    import slm_training.harnesses.model_build.checkpoint_bucket as cb

    monkeypatch.setattr(
        cb, "_require_hub", lambda: (lambda token=None: api, lambda: "tok")
    )


def test_dry_run_hashes_but_leaves_references_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_api(monkeypatch, _FakeApi())
    report = sync_run_checkpoints(
        _make_checkpoint(tmp_path),
        run_id="demo",
        bucket="TKendrick/OpenUI",
        run_dir=tmp_path / "runs" / "demo",
        dry_run=True,
        ensure_bucket=False,
        claim_class="frontier",
    )
    assert report["verification"] is None
    # Every artifact is hashed before upload.
    assert report["inventory"]["last.pt"]["sha256"]
    assert report["inventory"]["last.pt"]["size_bytes"] == len(b"weights")
    ref = report["references"][0]
    assert ref["verification_timestamp"] is None
    # A dry run can never back a durable claim.
    from slm_training.harnesses.model_build.checkpoint_reference import (
        CheckpointReferenceV1,
    )

    assert CheckpointReferenceV1.from_dict(ref).is_publishable is False


def test_real_sync_verifies_and_stamps_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeApi()
    _patch_api(monkeypatch, api)
    report = sync_run_checkpoints(
        _make_checkpoint(tmp_path),
        run_id="demo",
        bucket="TKendrick/OpenUI",
        run_dir=tmp_path / "runs" / "demo",
        dry_run=False,
        ensure_bucket=False,
        claim_class="frontier",
        provenance={
            "training_source_commit": "c" * 40,
            "evaluation_source_commit": "d" * 40,
            "model_config_hash": "m",
            "tokenizer_hash": "t",
            "output_codec_hash": "o",
            "corpus_manifest_hash": "cm",
            "data_version": "v1",
        },
    )
    assert report["verification"]["verified"] is True
    assert report["verification"]["method"] == "resync_dry_run"
    assert api.real_calls == 2  # initial upload + verified sidecar push
    ref = report["references"][0]
    assert ref["verification_timestamp"]
    assert ref["verifier_version"]
    # Fully-provenanced verified frontier reference is publishable.
    from slm_training.harnesses.model_build.checkpoint_reference import (
        CheckpointReferenceV1,
    )

    assert CheckpointReferenceV1.from_dict(ref).is_publishable is True


def test_real_sync_fails_closed_when_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_api(monkeypatch, _FakeApi(fail_verify=True))
    with pytest.raises(RuntimeError, match="verification failed"):
        sync_run_checkpoints(
            _make_checkpoint(tmp_path),
            run_id="demo",
            bucket="TKendrick/OpenUI",
            run_dir=tmp_path / "runs" / "demo",
            dry_run=False,
            ensure_bucket=False,
        )
