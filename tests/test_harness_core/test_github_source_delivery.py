"""Real tree algorithms and durable transaction across accepted source changes."""

import asyncio
import copy
import shutil
from types import SimpleNamespace

import pytest

from slm_training.harness_core import github_delivery_tree as tree
from slm_training.harness_core.github_connector import ConnectorConfig, ConnectorRejected, DeliveryWaiting
from slm_training.harness_core.github_document_delivery import DocumentDelivery, WRITE_TOOLS
from tests.test_harness_core.test_github_document_delivery import Remote, BASE, HEAD, MERGE


@pytest.fixture
def source_change(tmp_path, monkeypatch):
    predecessor = tmp_path / "predecessor"
    predecessor.mkdir()
    (predecessor / "module.py").write_text("broken\n")
    (predecessor / "remove.py").write_text("obsolete\n")
    before = tree.source_entries(predecessor, ["module.py", "remove.py"])
    raw = b"".join(f"{mode} blob {digest}\t{name}\0".encode()
                   for name, (mode, digest) in before.items())
    monkeypatch.setattr(tree.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=raw))
    candidate = tmp_path / "candidate"
    shutil.copytree(predecessor, candidate)
    (candidate / "module.py").write_text("fixed\n")
    (candidate / "module.py").chmod(0o755)
    (candidate / "remove.py").unlink()
    (candidate / "link").symlink_to("module.py")
    source = tmp_path / "gate"
    shutil.copytree(candidate, source, symlinks=True)
    paths = ["module.py", "remove.py", "link"]
    arguments = dict(predecessor=(predecessor, ["module.py", "remove.py"]),
                     candidate=(candidate, ["module.py", "link"]),
                     allowed_paths=paths, paths=paths)
    return source, arguments


def test_complete_source_operations_preserve_modes_links_and_deletions(source_change):
    source, arguments = source_change
    binding = tree.source_tree(source, BASE, **arguments)
    assert binding["tree_elements"] == [
        {"path": "link", "mode": "120000", "type": "blob", "content": "module.py"},
        {"path": "module.py", "mode": "100755", "type": "blob", "content": "fixed\n"},
        {"path": "remove.py", "mode": "100644", "type": "blob", "sha": None},
    ]
    assert binding["files"] == {"module.py": "fixed\n"}
    assert binding["candidate_git_tree"] == tree.tree_sha(tree.source_entries(source, arguments["paths"]))


@pytest.mark.parametrize("change", ["predecessor", "candidate", "scope", "mode", "link", "deletion"])
def test_accepted_patch_cannot_authorize_other_base_or_gate_bytes(source_change, change):
    source, arguments = source_change
    if change == "predecessor":
        (arguments["predecessor"][0] / "module.py").write_text("unpublished work\n")
    elif change == "candidate":
        (source / "module.py").write_text("unverified\n")
    elif change == "scope":
        arguments["allowed_paths"] = ["module.py"]
    elif change == "mode":
        (source / "module.py").chmod(0o644)
    elif change == "link":
        (source / "link").unlink()
        (source / "link").symlink_to("other.py")
    else:
        (source / "remove.py").write_text("obsolete\n")
    with pytest.raises(ValueError, match="mismatch|outside_verified_scope"):
        tree.source_tree(source, BASE, **arguments)


def test_unrepresentable_binary_waits_before_dispatch(source_change):
    source, arguments = source_change
    for root in (source, arguments["candidate"][0]):
        (root / "module.py").write_bytes(b"\xff\x00")
    with pytest.raises(DeliveryWaiting, match="utf8_representation_required"):
        tree.source_tree(source, BASE, **arguments)


class SourceRemote(Remote):
    def fetch(self, suffix):
        if suffix.startswith("git/trees/"):
            return {"sha": self.binding["candidate_git_tree"], "truncated": False,
                    "tree": [{"path": name, "mode": mode, "type": "blob", "sha": digest}
                             for name, (mode, digest) in self.entries.items()]}
        value = super().fetch(suffix)
        if suffix.startswith("git/commits/"):
            value["tree"]["sha"] = (self.binding["base_git_tree"] if suffix.endswith(BASE)
                                     else self.binding["candidate_git_tree"])
        return value

    def write(self, tool, args):
        if tool == "github_create_tree":
            assert args["tree_elements"] == self.binding["tree_elements"]
            return {"sha": self.binding["candidate_git_tree"]}
        return super().write(tool, args)


@pytest.fixture
def source_writer(source_change):
    source, arguments = source_change
    binding = {**tree.source_tree(source, BASE, **arguments),
               "repository": "owner/repo", "base_ref": BASE, "branch": "autotrain/source",
               "marker": "source-request", "message": "accepted source\n\nsource-request",
               "required_checks": ["full-gate"], "reviewers": [], "review_note_authors": ["writer"]}
    remote = SourceRemote(binding)
    remote.entries = tree.source_entries(source, arguments["paths"])
    disk = {}

    async def verify(proposal):
        assert proposal["merge_sha"] == MERGE and remote.merged

    def writer():
        return DocumentDelivery(binding, copy.deepcopy(disk), connector=remote,
            transport=ConnectorConfig(), save=lambda value: disk.update(copy.deepcopy(value)),
            validate=lambda **kwargs: None, verify=verify)

    return writer, remote, disk


@pytest.mark.parametrize("crash", sorted(WRITE_TOOLS))
def test_source_transaction_reconciles_each_lost_mutation_response(source_writer, crash):
    writer, remote, _ = source_writer
    remote.crash = crash
    with pytest.raises(TimeoutError):
        asyncio.run(writer().run())
    assert asyncio.run(writer().run()) == {"pr_number": 7, "verified_head_sha": HEAD, "merge_sha": MERGE}
    assert sum(tool == crash for tool, _ in remote.calls) == (2 if crash == "github_create_tree" else 1)


def test_source_payload_denial_never_fragments_or_falls_back(source_writer):
    writer, remote, _ = source_writer
    remote.binding["tree_elements"][0]["content"] = "x" * 200001
    with pytest.raises(ConnectorRejected, match="review_limit"):
        asyncio.run(writer().run())
    assert not remote.calls


def test_remote_tree_content_drift_prevents_branch_and_merge(source_writer):
    writer, remote, _ = source_writer
    remote.entries["module.py"] = ("100644", "f" * 40)
    with pytest.raises(ValueError, match="tree_content_mismatch"):
        asyncio.run(writer().run())
    assert not remote.branch and not remote.merged


@pytest.mark.parametrize("mismatch", [False, True])
def test_non_main_source_transaction_and_independent_readback(source_writer, mismatch):
    from slm_training.autoresearch.runtime.operations_reconciliation import _remote_delivery

    writer, remote, _ = source_writer
    remote.binding["base_branch"] = "acceptance-only/agentv-fault"
    if mismatch:
        remote.pr_base = "main"
    receipts = []

    async def verify(proposal):
        receipts.append(await _remote_delivery(remote, remote.binding["repository"],
            proposal, remote.binding["files"], remote.binding["required_checks"],
            base_branch=remote.binding["base_branch"]))

    instance = writer()
    instance.verify = verify
    if mismatch:
        with pytest.raises(ValueError, match="pr_identity_mismatch"):
            asyncio.run(instance.run())
        assert not remote.merged and not receipts
    else:
        assert asyncio.run(instance.run())["merge_sha"] == MERGE
        assert receipts[0]["base_branch"] == remote.binding["base_branch"]
        assert receipts[0]["head_sha"] == HEAD
        writes = sum(tool in WRITE_TOOLS for tool, _ in remote.calls)
        asyncio.run(instance.run())
        assert sum(tool in WRITE_TOOLS for tool, _ in remote.calls) == writes
        remote.binding["base_branch"] = "main"
        with pytest.raises(ValueError, match="journal_binding_changed"):
            writer()
