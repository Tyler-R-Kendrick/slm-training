"""Document delivery through a trusted connector; no domain or transport defaults."""

from __future__ import annotations

from types import SimpleNamespace

from .activity_contract import contract_digest
from .github_connector import (
    ConnectorRejected,
    DeliveryWaiting,
    check_payload,
    checked_read,
)
from .github_delivery_remote import (
    fetch,
    find_pr,
    mutate,
    ready_to_merge,
    sha,
    verify_commit,
    verify_pr_identity,
    verify_checks_reviews,
)


WRITE_TOOLS = frozenset(
    {
        "github_create_tree",
        "github_create_commit",
        "github_create_branch",
        "github_create_pull_request",
        "github_merge_pull_request",
    }
)
WRITER_READ_TOOLS = frozenset(
    {
        "github_fetch",
        "github_search_commits",
        "github_get_pr_info",
        "github_get_commit_combined_status",
        "github_list_pull_request_review_threads",
        "github_list_pull_request_reviews",
        "github_compare_commits",
        "github_fetch_file",
    }
)


class DocumentDelivery:
    """Callbacks belong to the trusted host, never to a request or repair output.

    save must fsync/replace the journal under an exclusive host lock. validate
    rechecks the current lease, immutable materialization and authenticated gate.
    verify performs independent domain verification after the squash merge.
    """

    def __init__(
        self, binding, journal, *, connector, transport, save, validate, verify
    ):
        self.binding, self.journal = binding, journal
        self.connector, self.transport = connector, transport
        self.save, self.validate, self.verify = save, validate, verify
        if journal and journal.get("binding") != binding:
            raise ValueError("delivery_journal_binding_changed")
        if not journal:
            journal.update(binding=binding, writes={})
            save(journal)
        self.repository = binding["repository"]

    async def run(self):
        self.validate(gate=True)
        if self.journal.get("rejected"):
            raise ConnectorRejected(self.journal["rejected"])
        elements = self.binding.get("tree_elements") or [
            {"path": name, "mode": "100644", "type": "blob", "content": value}
            for name, value in sorted(self.binding["files"].items())
        ]
        # Preflight the WHOLE materialization before any mutation. No chunking.
        check_payload(
            self.transport,
            "github_create_tree",
            {
                "repository_full_name": self.repository,
                "base_tree_sha": "0" * 40,
                "tree_elements": elements,
            },
        )
        base = await fetch(
            self.connector, self.repository, "git/commits/" + self.binding["base_ref"]
        )
        if self.binding.get("base_git_tree", base["tree"]["sha"]) != base["tree"]["sha"]:
            raise ValueError("remote_base_tree_differs_from_verified_lineage")
        tree = await self._write(
            "tree",
            "github_create_tree",
            {
                "repository_full_name": self.repository,
                "base_tree_sha": sha(base["tree"]["sha"]),
                "tree_elements": elements,
            },
        )
        if sha(tree["sha"]) != self.binding["candidate_git_tree"]:
            raise ValueError("published_tree_differs_from_verified_candidate")
        head = await self._commit(tree["sha"])
        await verify_commit(self.connector, self.binding, head, tree["sha"])
        await self._branch(head)
        pr = await self._pull_request(head)
        return await self._merge(pr, head)

    async def _write(self, label, tool, arguments):
        """Persist intent BEFORE dispatch; ambiguous outcomes never imply failure."""
        self.validate()
        writes = self.journal["writes"]
        row = writes.get(label)
        digest = contract_digest(arguments)
        if row and row["arguments_digest"] != digest:
            raise ValueError("delivery_write_intent_changed")
        if row and "result" in row:
            return row["result"]
        # Git trees are content addressed: an identical retry creates no new
        # logical object. Other ambiguous writes must reconcile before retry.
        if row and label != "tree":
            raise DeliveryWaiting(
                "ambiguous_" + label + "_requires_remote_reconciliation"
            )
        writes[label] = {"tool": tool, "arguments_digest": digest, "state": "pending"}
        self.save(self.journal)
        try:
            check_payload(self.transport, tool, arguments)
            result = await mutate(self.connector, tool, arguments)
        except ConnectorRejected as error:
            self.journal["rejected"] = str(error)
            self.save(self.journal)
            raise
        writes[label].update(state="completed", result=result)
        self.save(self.journal)
        return result

    def _reconciled(self, label, result):
        row = self.journal["writes"][label]
        row.update(state="reconciled", result=result)
        self.save(self.journal)

    async def _commit(self, tree):
        row = self.journal["writes"].get("commit")
        if row and "result" not in row:
            found = await checked_read(
                self.connector,
                "github_search_commits",
                repository_full_name=self.repository,
                query=self.binding["marker"],
                topn=100,
            )
            commits = found["commits"]
            if len(commits) != 1:
                # GitHub search can lag, and unreachable commits may be absent.
                # Never create a second timestamped commit based on absence.
                raise DeliveryWaiting("ambiguous_commit_not_uniquely_indexed")
            head = sha(commits[0]["sha"])
            await verify_commit(self.connector, self.binding, head, tree)
            self._reconciled("commit", {"sha": head})
        result = await self._write(
            "commit",
            "github_create_commit",
            {
                "repository_full_name": self.repository,
                "tree_sha": tree,
                "parent_sha": self.binding["base_ref"],
                "message": self.binding["message"],
            },
        )
        return sha(result["sha"])

    async def _branch(self, head):
        # matching-refs is a documented GET collection and avoids treating a
        # transport error/404 as proof that a branch is absent.
        refs = await fetch(
            self.connector,
            self.repository,
            "git/matching-refs/heads/" + self.binding["branch"],
        )
        exact = [r for r in refs if r["ref"] == "refs/heads/" + self.binding["branch"]]
        if exact:
            if len(exact) != 1 or exact[0]["object"]["sha"] != head:
                raise ValueError("delivery_branch_collision")
            if "branch" in self.journal["writes"]:
                self._reconciled("branch", {"sha": head})
            return
        await self._write(
            "branch",
            "github_create_branch",
            {
                "repository_full_name": self.repository,
                "branch_name": self.binding["branch"],
                "sha": head,
            },
        )
        refs = await fetch(
            self.connector,
            self.repository,
            "git/matching-refs/heads/" + self.binding["branch"],
        )
        if not any(
            r["ref"] == "refs/heads/" + self.binding["branch"]
            and r["object"]["sha"] == head
            for r in refs
        ):
            raise DeliveryWaiting("created_branch_not_observed")

    async def _pull_request(self, head):
        pr = await find_pr(self.connector, self.repository, self.binding["branch"],
                               base_branch=self.binding.get("base_branch", "main"))
        if pr is None:
            await self._write(
                "pr",
                "github_create_pull_request",
                {
                    "repository_full_name": self.repository,
                    "base": self.binding.get("base_branch", "main"),
                    "head": self.binding["branch"],
                    "title": self.binding.get("title", "Publish measured campaign documents"),
                    "body": self.binding["marker"],
                    "draft": False,
                    "maintainer_can_modify": False,
                },
            )
            pr = await find_pr(self.connector, self.repository, self.binding["branch"],
                               base_branch=self.binding.get("base_branch", "main"))
        if pr is None:
            raise DeliveryWaiting("created_pr_not_observed")
        verify_pr_identity(pr, self.binding, head)
        if "pr" in self.journal["writes"]:
            self._reconciled("pr", {"number": pr["number"]})
        return pr

    async def _merge(self, pr, head):
        number = pr["number"]
        pr = await fetch(self.connector, self.repository, f"pulls/{number}")
        verify_pr_identity(pr, self.binding, head)
        policy = SimpleNamespace(
            repository=self.repository,
            required_checks=self.binding["required_checks"],
            reviewers=self.binding["reviewers"],
            review_note_authors=self.binding["review_note_authors"],
            marker=self.binding["marker"],
        )
        if pr.get("merged") is not True:
            await ready_to_merge(self.connector, policy, self.binding, pr)
            await verify_commit(
                self.connector,
                self.binding,
                head,
                self.journal["writes"]["tree"]["result"]["sha"],
            )
            self.validate(gate=True)
            await self._write(
                "merge",
                "github_merge_pull_request",
                {
                    "repository_full_name": self.repository,
                    "pr_number": number,
                    "expected_head_sha": head,
                    "merge_method": "squash",
                },
            )
            pr = await fetch(self.connector, self.repository, f"pulls/{number}")
        verify_pr_identity(pr, self.binding, head)
        if pr.get("merged") is not True:
            raise DeliveryWaiting("merge_not_observed")
        await verify_checks_reviews(self.connector, policy, pr)
        merge = sha(pr["merge_commit_sha"])
        commit = await fetch(self.connector, self.repository, "git/commits/" + merge)
        if [p["sha"] for p in commit["parents"]] != [self.binding["base_ref"]]:
            raise ValueError("squash_parent_or_base_changed")
        # A squash must preserve the exact validated tree, not just target docs.
        if commit["tree"]["sha"] != self.journal["writes"]["tree"]["result"]["sha"]:
            raise ValueError("squash_tree_changed")
        proposal = {"pr_number": number, "verified_head_sha": head, "merge_sha": merge}
        await self.verify(proposal)
        self.validate()
        if "merge" in self.journal["writes"]:
            self._reconciled("merge", proposal)
        return proposal
