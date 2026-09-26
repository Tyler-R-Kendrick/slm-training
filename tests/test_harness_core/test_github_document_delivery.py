"""Exercise the production algorithm against connector-shaped crash boundaries."""

import asyncio
import copy
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from slm_training.harness_core.github_connector import (
    ConnectorConfig,
    ConnectorRejected,
    DeliveryWaiting,
)
from slm_training.harness_core.github_document_delivery import (
    DocumentDelivery,
    WRITE_TOOLS,
)


BASE, TREE, HEAD, MERGE = (c * 40 for c in "abcd")


class Remote:
    def __init__(self, binding, *, crash=None, deny=None):
        self.binding, self.crash, self.deny = binding, crash, deny
        self.calls, self.branch, self.pr, self.merged, self.committed = (
            [],
            False,
            False,
            False,
            False,
        )
        self.bad_check, self.bad_review, self.bad_content, self.other_head = (
            False,
            False,
            False,
            False,
        )

    def pull(self):
        return {
            "number": 7,
            "head": {
                "sha": "f" * 40 if self.other_head else HEAD,
                "ref": self.binding["branch"],
                "repo": {"full_name": "owner/repo"},
            },
            "base": {"sha": BASE, "ref": getattr(self, "pr_base", self.binding.get("base_branch", "main")), "repo": {"full_name": "owner/repo"}},
            "body": self.binding["marker"],
            "user": {"login": "writer"},
            "draft": False,
            "state": "closed" if self.merged else "open",
            "mergeable": True,
            "mergeable_state": "clean",
            "merged": self.merged,
            "merge_commit_sha": MERGE,
        }

    async def __call__(self, tool, args):
        self.calls.append((tool, copy.deepcopy(args)))
        if tool == self.deny:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "approval denied"}],
            }
        data = self.respond(tool, args)
        if tool == self.crash:
            self.crash = None
            raise TimeoutError("response lost after remote write")
        return {"structuredContent": data}

    def git_read(self, tool, args):
        if tool == "github_fetch":
            url = urlsplit(args["url"])
            if url.path.endswith("/pulls"):
                assert parse_qs(url.query)["base"] == [self.binding.get("base_branch", "main")]
            return {
                "content": json.dumps(
                    self.fetch(
                        url.path.split("/repos/owner/repo/")[1]
                    )
                )
            }
        if tool == "github_get_pr_info":
            return {"merged": self.merged, "base": self.pull()["base"]["ref"],
                    "head_sha": HEAD, "merge_commit_sha": MERGE}
        if tool == "github_compare_commits":
            assert args["head"] == self.binding.get("base_branch", "main")
            return {"status": "identical", "merge_base_commit": {"sha": MERGE}}
        raise AssertionError(tool)

    def respond(self, tool, args):
        if tool in {"github_fetch", "github_get_pr_info", "github_compare_commits"}:
            return self.git_read(tool, args)
        if tool in WRITE_TOOLS:
            return self.write(tool, args)
        if tool == "github_search_commits":
            return {"commits": [{"sha": HEAD}] if self.committed else []}
        if tool == "github_fetch_file":
            return {
                "encoding": "utf-8",
                "content": "tampered"
                if self.bad_content
                else self.binding["files"][args["path"]],
            }
        if tool == "github_get_commit_combined_status":
            return {
                "statuses": [
                    {
                        "context": "full-gate",
                        "state": "failure" if self.bad_check else "success",
                    }
                ]
            }
        if tool == "github_list_pull_request_review_threads":
            data = {"review_threads": []}
        elif tool == "github_list_pull_request_reviews":
            data = {
                "reviews": [
                    {
                        "id": 1,
                        "state": "APPROVED",
                        "user": {"login": "reviewer"},
                        "commit_id": BASE if self.bad_review else HEAD,
                    }
                ]
            }
        else:
            raise AssertionError(tool)
        return data

    def write(self, tool, args):
        if tool == "github_create_tree":
            assert len(args["tree_elements"]) == len(self.binding["files"])
            return {"sha": TREE}
        if tool == "github_create_commit":
            self.committed = True
            return {"sha": HEAD}
        if tool == "github_create_branch":
            self.branch = True
            return {"ref": "refs/heads/" + self.binding["branch"]}
        if tool == "github_create_pull_request":
            assert args["base"] == self.binding.get("base_branch", "main")
            self.pr = True
            return {"number": 7}
        assert args["expected_head_sha"] == HEAD and args["merge_method"] == "squash"
        self.merged = True
        return {"sha": MERGE, "merged": True}

    def fetch(self, suffix):
        if suffix.startswith("git/commits/"):
            return {
                "tree": {"sha": TREE},
                "parents": [{"sha": BASE}],
                "message": self.binding["message"],
            }
        if suffix.startswith("compare/"):
            return {
                "files": [
                    {"filename": name, "status": "modified"}
                    for name in self.binding["files"]
                ]
            }
        if suffix.startswith("git/matching-refs/"):
            return (
                [
                    {
                        "ref": "refs/heads/" + self.binding["branch"],
                        "object": {"sha": HEAD},
                    }
                ]
                if self.branch
                else []
            )
        if suffix == "issues/7/comments":
            note = {
                "schema_version": "connector_review_notes/v1",
                "request_marker": self.binding["marker"],
                "head_sha": HEAD,
                "rubber_duck": "reviewed producer to consumer",
                "adversarial": "checked failure boundaries",
                "verdict": "pass",
            }
            return (
                []
                if getattr(self, "missing_notes", False)
                else [{"id": 1, "user": {"login": "writer"}, "body": json.dumps(note)}]
            )
        fixed = {
            "pulls": [self.pull()] if self.pr else [],
            "pulls/7": self.pull(),
            "git/ref/heads/" + self.binding.get("base_branch", "main"): {"object": {"sha": BASE}},
            f"commits/{HEAD}/check-runs": {
                "check_runs": getattr(self, "check_runs", [])
            },
        }
        return fixed[suffix]


@pytest.fixture
def delivery():
    binding = {
        "repository": "owner/repo",
        "base_ref": BASE,
        "branch": "autotrain/request",
        "marker": "request",
        "candidate_git_tree": TREE,
        "message": "documents request",
        "required_checks": ["full-gate"],
        "reviewers": ["reviewer"],
        "review_note_authors": ["writer"],
        "files": {
            "docs/design/result.md": "# Results\n",
            "docs/design/result.json": "{}\n",
        },
    }
    remote, disk = Remote(binding), {}
    validations, verifications = [], []

    def writer():
        async def verify(proposal):
            assert remote.merged and proposal["merge_sha"] == MERGE
            verifications.append(proposal)

        return DocumentDelivery(
            binding,
            copy.deepcopy(disk),
            connector=remote,
            transport=ConnectorConfig(),
            save=lambda state: disk.update(copy.deepcopy(state)),
            validate=lambda **kw: validations.append(kw),
            verify=verify,
        )

    return writer, remote, disk, validations, verifications


@pytest.mark.parametrize("crash", sorted(WRITE_TOOLS))
def test_crash_after_remote_write_reconciles_without_duplicate_mutation(
    delivery, crash
):
    writer, remote, disk, validations, verifications = delivery
    remote.crash = crash
    with pytest.raises(TimeoutError):
        asyncio.run(writer().run())
    assert any(row["state"] == "pending" for row in disk["writes"].values())
    result = asyncio.run(writer().run())
    assert result == {"pr_number": 7, "verified_head_sha": HEAD, "merge_sha": MERGE}
    count = sum(tool == crash for tool, _ in remote.calls)
    assert count == (2 if crash == "github_create_tree" else 1)
    before = sum(tool in WRITE_TOOLS for tool, _ in remote.calls)
    asyncio.run(writer().run())
    assert sum(tool in WRITE_TOOLS for tool, _ in remote.calls) == before
    assert validations and len(verifications) == 2


def test_rejection_persists_and_has_no_alternate_write_or_retry(delivery):
    writer, remote, disk, _, _ = delivery
    remote.deny = "github_create_tree"
    for _ in range(2):
        with pytest.raises(ConnectorRejected):
            asyncio.run(writer().run())
    assert [tool for tool, _ in remote.calls if tool in WRITE_TOOLS] == [
        "github_create_tree"
    ]
    assert disk["rejected"]


def test_full_payload_rejected_before_any_remote_write(delivery):
    writer, remote, _, _, _ = delivery
    remote.binding["files"]["docs/design/result.md"] = "x" * 200001
    with pytest.raises(ConnectorRejected, match="review_limit"):
        asyncio.run(writer().run())
    assert not remote.calls


@pytest.mark.parametrize(
    "fault", ["bad_check", "bad_review", "bad_content", "other_head"]
)
def test_premerge_failures_never_merge(delivery, fault):
    writer, remote, _, _, _ = delivery
    setattr(remote, fault, True)
    with pytest.raises((ValueError, DeliveryWaiting)):
        asyncio.run(writer().run())
    assert not remote.merged


def test_ambiguous_unindexed_commit_does_not_create_another(delivery):
    writer, remote, _, _, _ = delivery
    remote.crash = "github_create_commit"
    with pytest.raises(TimeoutError):
        asyncio.run(writer().run())
    remote.committed = False  # Models GitHub search lag, not proven write failure.
    with pytest.raises(DeliveryWaiting, match="not_uniquely_indexed"):
        asyncio.run(writer().run())
    assert sum(tool == "github_create_commit" for tool, _ in remote.calls) == 1


def test_mutated_binding_cannot_reuse_journal(delivery):
    writer, remote, _, _, _ = delivery
    asyncio.run(writer().run())
    remote.binding["files"]["docs/design/result.md"] = "changed"
    with pytest.raises(ValueError, match="journal_binding_changed"):
        writer()


def test_current_gate_failure_precedes_any_remote_call(delivery):
    writer, remote, _, _, _ = delivery
    instance = writer()

    def invalid_gate(**kwargs):
        raise ValueError("authenticated_local_merge_gate_rejected")

    instance.validate = invalid_gate
    with pytest.raises(ValueError, match="local_merge_gate"):
        asyncio.run(instance.run())
    assert remote.calls == []


def test_connector_tree_must_match_full_verified_candidate(delivery):
    writer, remote, _, _, _ = delivery
    remote.binding["candidate_git_tree"] = "f" * 40
    with pytest.raises(ValueError, match="published_tree_differs"):
        asyncio.run(writer().run())
    assert [tool for tool, _ in remote.calls if tool in WRITE_TOOLS] == [
        "github_create_tree"
    ]


def test_lost_merge_reply_still_requires_exact_head_review(delivery):
    writer, remote, _, _, verifications = delivery
    remote.crash = "github_merge_pull_request"
    with pytest.raises(TimeoutError):
        asyncio.run(writer().run())
    remote.bad_review = True
    with pytest.raises(DeliveryWaiting, match="exact_head_review"):
        asyncio.run(writer().run())
    assert verifications == []
    assert sum(tool == "github_merge_pull_request" for tool, _ in remote.calls) == 1


def test_success_status_does_not_hide_failed_check_run(delivery):
    writer, remote, _, _, _ = delivery
    remote.check_runs = [
        {
            "head_sha": HEAD,
            "name": "Python full gate",
            "status": "completed",
            "conclusion": "failure",
        }
    ]
    with pytest.raises(DeliveryWaiting, match="check_runs_incomplete"):
        asyncio.run(writer().run())
    assert not remote.merged


def test_explicit_empty_approval_policy_still_requires_review_notes(delivery):
    writer, remote, _, _, _ = delivery
    remote.binding["reviewers"] = []
    remote.missing_notes = True
    with pytest.raises(DeliveryWaiting, match="rubber_duck_notes_required"):
        asyncio.run(writer().run())
    assert not remote.merged
    remote.missing_notes = False
    assert asyncio.run(writer().run())["merge_sha"] == MERGE
