"""A single explicit host grant covers accepted lineage without manual repinning."""

import asyncio
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import github_source_authority as owner, github_source_delivery, github_source_preparation
from slm_training.harness_core import execution_release, github_delivery_tree
from slm_training.harness_core.github_delivery_tree import base_entries as real_base_entries
from tests.test_scripts.test_github_source_delivery import accepted as accepted, ReadRemote


def root_host(host, subject, tmp_path):
    host.accepted_source_successors = True
    host.source_digest = subject["predecessor"]["runtime_source_digest"]
    host.verification_plan = {"source_delivery_root": str(tmp_path / "delivery-inputs"),
        "repository_source": str(tmp_path / "initial-base"), "grant": {"total_seconds": 180, "interrupt_seconds": 90, "max_attempts": 2}}
    host.model_copy = lambda update: SimpleNamespace(**{**vars(host), **update})
    return host


def test_explicit_host_policy_derives_first_accepted_source_only(accepted, tmp_path):
    store, wait, host, subject = accepted
    host = root_host(host, subject, tmp_path)
    derived = owner.successor_host(store, wait, host)
    assert derived.source_digest == subject["successor_source_digest"]
    assert derived.base_ref == host.base_ref
    assert derived.verification_plan["execution"] == subject["successor_execution"]
    assert Path(derived.verification_plan["source"]).parts[-2:] == (wait["publication_id"], "candidate")
    assert host.source_digest == subject["predecessor"]["runtime_source_digest"]
    host.accepted_source_successors = False
    assert owner.successor_host(store, wait, host) is host


def test_unaccepted_predecessor_cannot_widen_root_source_grant(accepted, tmp_path):
    store, wait, host, subject = accepted
    host = root_host(host, subject, tmp_path)
    subject["predecessor"]["runtime_source_digest"] = "7" * 64
    subject["predecessor_publication_id"] = "8" * 64
    with pytest.raises(ValueError, match="predecessor_not_accepted"):
        owner.successor_host(store, wait, host)


def test_second_repair_derives_real_merged_base_from_fenced_proof(accepted, tmp_path, monkeypatch):
    store, wait, host, subject = accepted
    first = copy.deepcopy(subject)
    binding = github_source_delivery.source_binding(store, wait, host)
    remote = ReadRemote(binding)
    runtime = SimpleNamespace(store=store, publication=lambda lease: nullcontext())
    asyncio.run(github_source_delivery.reconcile_source_delivery(runtime, SimpleNamespace(activity_id="first"), wait,
                                                                 remote.proposal, host, remote))
    store.append_event("repair_release_accepted", detail={"publication_id": wait["publication_id"], "handoff": {"source_delivery": wait}})
    source = tmp_path / "next-source"
    source.mkdir()
    (source / "module.py").write_text("next repair\n")
    release = execution_release.prepare_release(source, tmp_path / "next-release", tmp_path / "next-execution", tmp_path / "outputs")
    next_wait = {**wait, "publication_id": "9" * 64, "artifact_sha256": "8" * 64}
    successor = {"successor_source_digest": release["source_digest"], "successor_execution": str(tmp_path / "next-execution"),
        "successor": {"verified_source": str(source)}, "predecessor_publication_id": wait["publication_id"],
        "predecessor": {"runtime_source_digest": first["successor_source_digest"], "execution": first["successor_execution"]},
        "scope": {"changes": {"module.py": {}}}}
    monkeypatch.setattr(owner.repair_delivery, "resolve_source_delivery",
                        lambda store, given: first if given == wait else successor)
    host = root_host(host, first, tmp_path)
    derived = owner.successor_host(store, next_wait, host)
    assert derived.base_ref == remote.proposal["merge_sha"]
    assert derived.source_digest == release["source_digest"]
    assert "repository_source" not in derived.verification_plan
    assert derived.verification_plan["grant"] == host.verification_plan["grant"]
    # Prove the second preparation requires neither refresh of the original
    # checkout nor a handwritten source/base/gate identity in host config.
    monkeypatch.setattr(github_delivery_tree, "base_entries", real_base_entries)
    monkeypatch.setattr(github_source_preparation, "verification_binding", lambda root, base, *a, **kw: {"base": base})
    prepared = github_source_preparation.resolve_source_host(store, next_wait, derived)
    candidate = Path(prepared.verification_plan["source"])
    assert real_base_entries(candidate, derived.base_ref) == {
        "module.py": ("100644", github_delivery_tree.git_object("blob", b"fixed\n"))}
    assert (candidate / "module.py").read_text() == "next repair\n"
    assert host.base_ref == "a" * 40 and not Path(host.verification_plan["repository_source"]).exists()
