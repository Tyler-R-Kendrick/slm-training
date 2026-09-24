"""Source adapter joins accepted publication, pinned host and independent proof."""

import asyncio
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts import github_source_delivery as owner
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import contract_digest
from slm_training.harness_core import execution_release, github_delivery_tree as tree


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    from slm_training.autoresearch.heal import repair_delivered, repair_delivery
    from scripts import merge_verification_evidence

    before, candidate = tmp_path / "before", tmp_path / "candidate"
    for path, text in ((before, "broken\n"), (candidate, "fixed\n")):
        path.mkdir()
        (path / "module.py").write_text(text)
    monkeypatch.setattr(execution_release, "_checkout_provenance", lambda _: None)
    old = execution_release.prepare_release(before, tmp_path / "old-release", tmp_path / "old-execution", tmp_path / "outputs")
    new = execution_release.prepare_release(candidate, tmp_path / "new-release", tmp_path / "new-execution", tmp_path / "outputs")
    subject = {
        "successor_execution": str(tmp_path / "new-execution"),
        "successor_source_digest": new["source_digest"],
        "successor": {"verified_source": str(candidate)},
        "predecessor": {"execution": str(tmp_path / "old-execution"), "runtime_source_digest": old["source_digest"]},
        "scope": {"changes": {"module.py": {}}},
    }
    monkeypatch.setattr(repair_delivery, "resolve_source_delivery", lambda *args: copy.deepcopy(subject))
    # The heal owner separately exercises real accepted-publication activation.
    # These connector fixtures isolate the trusted in-lease publication seam.
    monkeypatch.setattr(repair_delivered, "record_delivered_activation",
                        lambda runtime, lease, wait, proof: {"publication_id": wait["publication_id"]})
    monkeypatch.setattr(repair_delivered, "resolve_delivered_activation", lambda store, ref: ref)
    monkeypatch.setattr(merge_verification_evidence, "source_paths", lambda _: ["module.py"])
    monkeypatch.setattr(tree, "base_entries", lambda *args: {"module.py": ("100644", tree.git_object("blob", b"broken\n"))})
    host = SimpleNamespace(repository="owner/repo", base_ref="a" * 40, source_digest=new["source_digest"],
        required_checks=(), verification_plan={"source": str(candidate), "execution": subject["successor_execution"],
        "identity": "b" * 64, "state_dir": str(tmp_path / "cache")})
    wait = {"kind": "verified_repair_source", "publication_id": "c" * 64,
            "artifact_sha256": "d" * 64, "required_capability": "authorized_github_connector_delivery"}
    store = CampaignStore("runtime", tmp_path / "loops/lab")
    return store, wait, host, subject


def test_source_binding_uses_accepted_successor_and_whole_actual_base(accepted):
    store, wait, host, _ = accepted
    binding = owner.source_binding(store, wait, host)
    assert binding["source_digest"] == host.source_digest
    assert binding["publication_id"] == wait["publication_id"]
    assert binding["verification_identity"] == host.verification_plan["identity"]
    assert binding["files"] == {"module.py": "fixed\n"}
    assert binding["tree_elements"] == [{"path": "module.py", "mode": "100644", "type": "blob", "content": "fixed\n"}]


@pytest.mark.parametrize("change", ["source", "execution", "documents", "scope", "candidate"])
def test_host_cannot_retarget_accepted_source(accepted, change):
    store, wait, host, subject = accepted
    if change == "source":
        host.source_digest = "f" * 64
    elif change == "execution":
        host.verification_plan["execution"] = subject["predecessor"]["execution"]
    elif change == "documents":
        host.verification_plan["delivery_documents_sha256"] = {"README.md": "e" * 64}
    elif change == "scope":
        subject["scope"]["changes"] = {}
    else:
        from pathlib import Path
        (Path(host.verification_plan["source"]) / "module.py").write_text("other\n")
    with pytest.raises(ValueError):
        owner.source_binding(store, wait, host)


class ReadRemote:
    def __init__(self, binding, failure=None):
        self.binding, self.failure, self.calls = binding, failure, []
        self.commits = {}
        for message in ("repair head", "repair squash"):
            person = {"name": "Reviewer", "email": "reviewer@invalid", "date": "2000-01-01T00:00:00Z"}
            commit = {"tree": {"sha": binding["candidate_git_tree"]}, "parents": [{"sha": "a" * 40}],
                      "author": person, "committer": person, "message": message}
            raw = (f"tree {binding['candidate_git_tree']}\nparent {'a' * 40}\n"
                   "author Reviewer <reviewer@invalid> 946684800 +0000\n"
                   "committer Reviewer <reviewer@invalid> 946684800 +0000\n\n" + message)
            self.commits[tree.git_object("commit", raw.encode())] = commit
        head, merge = self.commits
        self.proposal = {"pr_number": 7, "verified_head_sha": head, "merge_sha": merge}

    async def __call__(self, tool, arguments):
        self.calls.append(tool)
        if tool == "github_get_pr_info":
            value = {"merged": True, "base": "main", "head_sha": self.proposal["verified_head_sha"], "merge_commit_sha": self.proposal["merge_sha"]}
        elif tool == "github_get_commit_combined_status":
            value = {"statuses": []}
        elif tool == "github_list_pull_request_review_threads":
            value = {"review_threads": []}
        elif tool == "github_list_pull_request_reviews":
            value = {"reviews": []}
        elif tool == "github_compare_commits":
            value = {"status": "identical", "merge_base_commit": {"sha": self.proposal["merge_sha"]}}
        elif tool == "github_fetch_file":
            value = {"encoding": "utf-8", "content": "fixed\n"}
        else:
            assert tool == "github_fetch"
            if "/git/commits/" in arguments["url"]:
                revision = arguments["url"].rsplit("/", 1)[1]
                value = copy.deepcopy(self.commits.get(revision, {"tree": {"sha": self.binding["base_git_tree"]}}))
                if self.failure == "parent":
                    value["parents"] = [{"sha": "b" * 40}]
            else:
                value = {"sha": self.binding["candidate_git_tree"], "truncated": self.failure == "truncated",
                         "tree": [{"path": "module.py", "type": "blob", "mode": "100644",
                                   "sha": tree.git_object("blob", b"fixed\n")}]}
                if self.failure == "mode":
                    value["tree"][0]["mode"] = "100755"
            value = {"content": json.dumps(value)}
        return {"structuredContent": value}


@pytest.mark.parametrize("failure", [None, "parent", "mode", "truncated"])
def test_independent_source_reconciliation_requires_complete_tree(accepted, failure):
    store, wait, host, _ = accepted
    binding = owner.source_binding(store, wait, host)
    connector = ReadRemote(binding, failure)
    runtime = SimpleNamespace(store=store, publication=lambda lease: nullcontext())
    proposal = connector.proposal
    invoke = owner.reconcile_source_delivery(runtime, SimpleNamespace(activity_id="delivery"), wait, proposal, host, connector)
    if failure:
        with pytest.raises(ValueError):
            asyncio.run(invoke)
        assert not store.verify_event_chain()
    else:
        result = asyncio.run(invoke)
        proof = owner.source_completion(store, wait, result)
        assert proof["publication_id"] == wait["publication_id"]
        assert proof["candidate_git_tree"] == binding["candidate_git_tree"]
        assert proof["verification_plan_sha256"] == contract_digest(host.verification_plan)
        forged = copy.deepcopy(result)
        forged["receipt"]["remote"]["source_digest"] = "0" * 64
        with pytest.raises(ValueError, match="artifact_mismatch"):
            owner.source_completion(store, wait, forged)


@pytest.mark.parametrize("suffix", ["git/commits/" + "a" * 40, "git/trees/" + "a" * 40 + "?recursive=1"])
def test_reader_grants_only_required_immutable_git_gets(suffix):
    url = "https://api.github.com/repos/owner/repo/" + suffix
    assert owner.reader_url_allowed("owner/repo", {"url": url})
    assert not owner.reader_url_allowed("owner/repo", {"url": url.replace("owner/repo", "other/repo")})
    assert not owner.reader_url_allowed("owner/repo", {"url": url + "#suffix"})
    assert not owner.reader_url_allowed("owner/repo", {"url": url, "method": "POST"})
    assert not owner.reader_url_allowed("owner/repo", {"url": url.replace("git/", "git/../")})
