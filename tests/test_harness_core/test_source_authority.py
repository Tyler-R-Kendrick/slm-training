"""Authenticated local journal + real Git object hashes; no provider calls."""

import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest

from scripts.github_source_authority import initial_source_request, record_initial_release
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.runtime.operations_delivery import DeliveryHost
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core import execution_release as release
from slm_training.harness_core.activity_contract import ActivityOutcome, ActivitySpec, contract_digest
from slm_training.harness_core.github_delivery_tree import git_object, source_entries, tree_sha
from slm_training.harness_core.source_authority import resolve_source_authority, require_main_source_authority


@pytest.fixture
def initial(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("# initial source\n")
    (source / "alias").symlink_to("module.py")
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(release, "_checkout_provenance", lambda _: None)
    manifest = release.prepare_release(source, tmp_path / "snapshot-release", snapshot, tmp_path / "snapshot-outputs")
    entries = source_entries(snapshot, manifest["files"])
    tree = tree_sha(entries)
    raw = f"tree {tree}\nauthor Fixture <fixture@invalid> 0 +0000\ncommitter Fixture <fixture@invalid> 0 +0000\n\ninitial\n"
    commit = git_object("commit", raw.encode())
    host = DeliveryHost(command=("/fixture/writer",), executable_sha256="a" * 64,
        repository="owner/repo", base_ref=commit, source_digest=manifest["source_digest"],
        expires_at=time.time() + 600, authorized=True,
        verification_plan={"initial_release": {"tree": tree, "source": str(snapshot),
            "release": str(tmp_path / "release"), "execution": str(tmp_path / "execution"),
            "outputs": str(tmp_path / "outputs")}})
    calls = []
    async def connector(tool, arguments):
        assert tool == "github_fetch"
        calls.append(arguments["url"])
        suffix = arguments["url"].split("/repos/owner/repo/", 1)[1]
        if suffix.startswith("git/ref/heads/"):
            value = {"ref": "refs/heads/" + host.base_branch, "object": {"type": "commit", "sha": commit}}
        elif suffix == "git/commits/" + commit:
            person = {"name": "Fixture", "email": "fixture@invalid", "date": "1970-01-01T00:00:00Z"}
            value = {"tree": {"sha": tree}, "parents": [], "author": person, "committer": person, "message": "initial\n"}
        else:
            assert suffix == "git/trees/" + tree + "?recursive=1"
            value = {"sha": tree, "truncated": False, "tree": [
                {"path": name, "mode": mode, "type": "blob", "sha": sha}
                for name, (mode, sha) in entries.items()]}
        return {"structuredContent": {"content": json.dumps(value)}}
    with ActivityRuntime(CampaignStore("runtime", tmp_path / "trusted")) as runtime:
        yield host, runtime, connector, calls


def _claim(host, runtime):
    request = initial_source_request(host)
    runtime.register(ActivitySpec(activity_id="initial", family="source", kind="verify",
        source_digest=host.source_digest, environment_digest="a" * 64,
        input_digest=contract_digest(request), output_namespace="initial",
        capabilities=("authorized_github_connector_read",)))
    return runtime.claim_next(activity_id="initial", capabilities={"authorized_github_connector_read"})


def _finish(runtime, lease):
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.attempt_dir(lease).glob("*.json")}
    runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED, spent_seconds=0, outputs=outputs)


def _args(host):
    return dict(execution=Path(host.verification_plan["initial_release"]["execution"]),
                expected_repository=host.repository, expected_commit=host.base_ref,
                expected_source_digest=host.source_digest)


@pytest.mark.parametrize("branch", ["main", "acceptance-only/fault"])
def test_initial_membership_needs_committed_producer_and_preserves_branch(initial, branch, monkeypatch):
    host, runtime, connector, calls = initial
    host.base_branch = branch
    lease = _claim(host, runtime)
    reference = asyncio.run(record_initial_release(runtime, lease, host, connector))
    args = _args(host)
    with pytest.raises(ValueError, match="committed_success"):
        resolve_source_authority(runtime.store, reference, **args)
    _finish(runtime, lease)
    result = resolve_source_authority(runtime.store, reference, **args)
    assert result["ref"] == "refs/heads/" + branch and len(calls) == 4
    assert release.runtime_git_provenance(args["execution"])["code_dirty"] is False
    if branch == "main":
        assert require_main_source_authority(runtime.store, reference, **args) == result
    else:
        with pytest.raises(ValueError, match="main_membership_required"):
            require_main_source_authority(runtime.store, reference, **args)
    from slm_training.harness_core import versioning
    monkeypatch.chdir(args["execution"])
    assert versioning.build_version_stamp()["source_authority"] == reference


@pytest.mark.parametrize("fault", ["repository", "source", "commit", "copy", "artifact", "output", "journal", "marker"])
def test_copied_or_tampered_authority_cannot_grant_main(initial, tmp_path, fault):
    host, runtime, connector, _ = initial
    lease = _claim(host, runtime)
    reference = asyncio.run(record_initial_release(runtime, lease, host, connector))
    _finish(runtime, lease)
    args, store = _args(host), runtime.store
    if fault in {"repository", "source", "commit"}:
        key, value = {"repository": ("expected_repository", "other/repo"),
                      "source": ("expected_source_digest", "f" * 64),
                      "commit": ("expected_commit", "f" * 40)}[fault]
        args[key] = value
    elif fault == "copy":
        copied = tmp_path / "copied-execution"
        shutil.copytree(args["execution"], copied, symlinks=True)
        args["execution"] = copied
    elif fault == "artifact":
        (store.root / "artifacts/source_authorities" / (reference["artifact_sha256"] + ".json")).write_text("{}")
    elif fault == "output":
        (runtime.store.root / "initial" / lease.attempt_id / "source-authority.json").write_text("{}")
    elif fault == "journal":
        store = CampaignStore("runtime", tmp_path / "untrusted-copy")
        shutil.copytree(runtime.store.root / "artifacts", store.root / "artifacts")
    else:
        for directory in (args["execution"], Path(host.verification_plan["initial_release"]["release"])):
            marker = directory / release.MARKER
            data = json.loads(marker.read_text())
            data["source_authority"] = {**reference, "main_verified": True}
            marker.chmod(0o644)
            marker.write_text(json.dumps(data))
    with pytest.raises((ValueError, KeyError)):
        require_main_source_authority(store, reference, **args)


@pytest.mark.parametrize("fault", ["ref", "tree", "source", "expired"])
def test_initial_readback_rejects_mismatch_before_authority(initial, fault):
    host, runtime, connector, _ = initial
    lease = _claim(host, runtime)
    if fault == "expired":
        host.expires_at = time.time() - 1
    if fault == "source":
        (Path(host.verification_plan["initial_release"]["source"]) / "module.py").write_text("changed")
    count = 0
    async def broken(tool, arguments):
        nonlocal count
        result = await connector(tool, arguments)
        data = json.loads(result["structuredContent"]["content"])
        if "git/ref/" in arguments["url"]:
            count += 1
            if fault == "ref" and count == 2:
                data["object"]["sha"] = "f" * 40
        if fault == "tree" and "git/trees/" in arguments["url"]:
            data["truncated"] = True
        return {"structuredContent": {"content": json.dumps(data)}}
    with pytest.raises(ValueError):
        asyncio.run(record_initial_release(runtime, lease, host, broken))
    assert not any(e["event_type"] == "source_authority_recorded" for e in runtime.store.verify_event_chain())


def test_legacy_clean_diagnostic_has_no_inferred_membership(initial, tmp_path):
    host, runtime, connector, _ = initial
    lease = _claim(host, runtime)
    reference = asyncio.run(record_initial_release(runtime, lease, host, connector))
    _finish(runtime, lease)
    # A clean historical marker cannot borrow a new producer's proof.
    args = _args(host)
    for directory in (args["execution"], Path(host.verification_plan["initial_release"]["release"])):
        marker = directory / release.MARKER
        data = json.loads(marker.read_text())
        del data["source_authority"]
        marker.chmod(0o644)
        marker.write_text(json.dumps(data))
    assert release.runtime_git_provenance(args["execution"])["code_dirty"] is False
    assert release.source_authority_reference(args["execution"]) is None
    with pytest.raises(ValueError, match="materialization_binding"):
        require_main_source_authority(runtime.store, reference, **args)


@pytest.mark.parametrize("fault", [None, "request_branch", "config_branch", "repository", "tool", "ref"])
def test_initial_adapter_enforces_host_and_lease_read_scope(initial, tmp_path, fault):
    from scripts import github_delivery_adapter as adapter
    from slm_training.harness_core.github_connector import ConnectorConfig

    host, runtime, connector, calls = initial
    lease = _claim(host, runtime)
    config = adapter.AdapterConfig(host=host, transport=ConnectorConfig(),
        runtime_root=str(runtime.store.root), state_dir=str(tmp_path / "adapter"),
        reviewers=(), review_note_authors=("fixture",))
    request = {"schema_version": "connector_read_request/v1", "lease": lease.model_dump(mode="json"),
               "source_request": initial_source_request(host), "tool": "github_fetch",
               "arguments": {"url": "https://api.github.com/repos/owner/repo/git/ref/heads/main"}}
    if fault == "request_branch":
        request["source_request"]["base_branch"] = "other"
    elif fault == "config_branch":
        config.host.base_branch = "other"
        request["source_request"] = initial_source_request(config.host)
    elif fault == "repository":
        request["arguments"]["url"] = request["arguments"]["url"].replace("owner/repo", "other/repo")
    elif fault == "tool":
        request["tool"] = "github_merge_pull_request"
    elif fault == "ref":
        request["arguments"]["url"] += "?ref=other"
    if fault:
        with pytest.raises(ValueError):
            asyncio.run(adapter.read_request(config, request, connector))
        assert not calls
    else:
        result = asyncio.run(adapter.read_request(config, request, connector))
        assert result["provider"] == "github_connector" and result["read_only"] is True
        assert result["request_digest"] == contract_digest(request)
        assert len(calls) == 1


@pytest.mark.parametrize("mutate", [False, True])
def test_initial_retry_reuses_only_exact_materialization(initial, mutate):
    host, runtime, connector, calls = initial
    lease = _claim(host, runtime)
    reference = asyncio.run(record_initial_release(runtime, lease, host, connector))
    target = _args(host)["execution"] / "module.py"
    if mutate:
        target.write_text("changed since observation")
        with pytest.raises(ValueError, match="execution_source_drift"):
            asyncio.run(record_initial_release(runtime, lease, host, connector))
    else:
        assert asyncio.run(record_initial_release(runtime, lease, host, connector)) == reference
        _finish(runtime, lease)
        assert require_main_source_authority(runtime.store, reference, **_args(host))["commit"] == host.base_ref
    assert len(calls) == 8


def test_configured_initial_entrypoint_uses_pinned_read_transport(initial, tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from scripts import github_delivery_adapter as adapter
    from slm_training.harness_core.github_connector import ConnectorConfig

    host, runtime, connector, calls = initial
    lease = _claim(host, runtime)
    transport = ConnectorConfig(endpoint="https://fixture.invalid/mcp")
    config = adapter.AdapterConfig(host=host, transport=transport,
        runtime_root=str(runtime.store.root), state_dir=str(tmp_path / "adapter"),
        reviewers=(), review_note_authors=("fixture",))
    @asynccontextmanager
    async def configured(actual, *, allowed_tools, timeout_seconds):
        assert actual is transport and allowed_tools == {"github_fetch"}
        assert 0 < timeout_seconds <= 170
        yield connector
    monkeypatch.setattr(adapter, "host_connector", configured)
    reference = asyncio.run(adapter.prepare_initial_release(runtime, lease, config))
    _finish(runtime, lease)
    assert require_main_source_authority(runtime.store, reference, **_args(host))["ref"] == "refs/heads/main"
    assert len(calls) == 4
