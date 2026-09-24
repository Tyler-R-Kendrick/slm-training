"""Exact GitHub document, review and check predicates over connector reads."""

from __future__ import annotations

import json
import re
from urllib.parse import urlencode

from .github_connector import ConnectorRejected, DeliveryWaiting, checked_read


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value):
        raise ValueError("connector_invalid_git_sha")
    return value


async def fetch(connector, repository, suffix):
    """Use the connector's documented GET-only tool; never HTTP to GitHub."""
    result = await checked_read(
        connector,
        "github_fetch",
        url=f"https://api.github.com/repos/{repository}/{suffix}",
    )
    return json.loads(result["content"])


async def pages(connector, repository, suffix, *, key=None):
    rows, page = [], 1
    while True:
        separator = "&" if "?" in suffix else "?"
        data = await fetch(
            connector, repository, f"{suffix}{separator}per_page=100&page={page}"
        )
        batch = data[key] if key else data
        if not isinstance(batch, list):
            raise ValueError("connector_invalid_page")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
        page += 1


async def find_pr(connector, repository, branch):
    owner = repository.split("/")[0]
    query = urlencode({"state": "all", "head": f"{owner}:{branch}", "base": "main"})
    rows = await pages(connector, repository, "pulls?" + query)
    if len(rows) > 1:
        raise ValueError("ambiguous_delivery_pull_request")
    return rows[0] if rows else None


async def verify_commit(connector, binding, head, tree):
    repository, base = binding["repository"], binding["base_ref"]
    commit = await fetch(connector, repository, "git/commits/" + sha(head))
    if (
        [p["sha"] for p in commit["parents"]] != [base]
        or commit["tree"]["sha"] != tree
        or commit["message"] != binding["message"]
    ):
        raise ValueError("delivery_commit_binding_mismatch")
    if "tree_elements" in binding:
        await verify_tree(connector, repository, tree)
        return
    changes = await pages(
        connector, repository, f"compare/{base}...{head}", key="files"
    )
    names = [row["filename"] for row in changes]
    if len(names) != len(set(names)) or not set(names) <= set(binding["files"]):
        raise ValueError("delivery_contains_unrequested_changes")
    if any(row["status"] not in {"added", "modified"} for row in changes):
        raise ValueError("delivery_unexpected_file_operation")
    await verify_contents(connector, repository, head, binding["files"])


async def verify_tree(connector, repository, expected):
    """Independent complete tree read covers deletion, links and executable bits."""
    from .github_delivery_tree import tree_sha

    result = await fetch(connector, repository, "git/trees/" + sha(expected) + "?recursive=1")
    if result.get("truncated") is not False or result.get("sha") != expected:
        raise DeliveryWaiting("complete_remote_git_tree_required")
    entries = {}
    for row in result["tree"]:
        if row["type"] == "tree":
            continue
        name = row["path"]
        if name in entries or row["type"] != "blob" or row["mode"] not in {"100644", "100755", "120000"}:
            raise ValueError("remote_source_tree_representation_mismatch")
        entries[name] = (row["mode"], sha(row["sha"]))
    if tree_sha(entries) != expected:
        raise ValueError("remote_source_tree_content_mismatch")


async def verify_contents(connector, repository, head, files):
    for name, content in files.items():
        result = await checked_read(
            connector,
            "github_fetch_file",
            repository_full_name=repository,
            path=name,
            ref=head,
            encoding="utf-8",
        )
        if result.get("encoding") != "utf-8" or result.get("content") != content:
            raise ValueError("remote_document_content_mismatch")


def verify_pr_identity(pr, binding, head):
    if (
        pr["head"]["sha"] != head
        or pr["head"]["ref"] != binding["branch"]
        or pr["head"]["repo"]["full_name"] != binding["repository"]
        or pr["base"]["repo"]["full_name"] != binding["repository"]
        or pr["base"]["ref"] != "main"
        or binding["marker"] not in (pr.get("body") or "")
    ):
        raise ValueError("delivery_pr_identity_mismatch")


async def ready_to_merge(connector, config, binding, pr):
    repo = binding["repository"]
    if (
        pr["base"]["sha"] != binding["base_ref"]
        or pr.get("draft") is not False
        or pr.get("state") != "open"
        or pr.get("mergeable") is not True
        or pr.get("mergeable_state") != "clean"
    ):
        raise DeliveryWaiting("remote_base_or_mergeability_not_ready")
    main = await fetch(connector, repo, "git/ref/heads/main")
    if main["object"]["sha"] != binding["base_ref"]:
        raise DeliveryWaiting("remote_base_changed_requires_successor")
    await verify_checks_reviews(connector, config, pr)


async def _checks(connector, config, head):
    repo = config.repository
    required = config.required_checks
    if required is None:
        raise DeliveryWaiting("trusted_required_checks_missing")
    combined = await checked_read(
        connector,
        "github_get_commit_combined_status",
        repo_full_name=repo,
        commit_sha=head,
    )
    statuses = {r["context"]: r["state"] for r in combined["statuses"]}
    runs = await pages(
        connector, repo, f"commits/{head}/check-runs?filter=latest", key="check_runs"
    )
    if any(
        r["head_sha"] != head
        or r["status"] != "completed"
        or r["conclusion"] != "success"
        for r in runs
    ):
        raise DeliveryWaiting("remote_check_runs_incomplete")
    if any(value != "success" for value in statuses.values()):
        raise DeliveryWaiting("remote_status_checks_incomplete")
    # Keep the canonical domain validator authoritative: required status names
    # must exist there too; check-run success alone cannot forge a status.
    if any(statuses.get(name) != "success" for name in required):
        raise DeliveryWaiting("remote_required_checks_incomplete")


async def _reviews(connector, config, pr):
    arguments = {"repo_full_name": config.repository, "pr_number": pr["number"]}
    threads = await checked_read(
        connector, "github_list_pull_request_review_threads", **arguments
    )
    reviews = await checked_read(
        connector, "github_list_pull_request_reviews", **arguments
    )
    if any(
        row.get("isResolved") is not True for row in threads["review_threads"]
    ) or changes_requested(reviews["reviews"]):
        raise DeliveryWaiting("remote_reviews_unresolved")
    latest = {}
    for row in sorted(
        reviews["reviews"], key=lambda r: (r.get("submitted_at", ""), r.get("id", 0))
    ):
        if row.get("state") not in {"COMMENTED", "PENDING"}:
            latest[row["user"]["login"].casefold()] = row
    for reviewer in config.reviewers:
        row = latest.get(reviewer.casefold(), {})
        if (
            row.get("state") != "APPROVED"
            or row.get("commit_id") != pr["head"]["sha"]
            or reviewer.casefold() == pr["user"]["login"].casefold()
        ):
            raise DeliveryWaiting("independent_exact_head_review_required")


async def mutate(connector, tool, arguments):
    result = await connector(tool, arguments)
    if result.get("isError"):
        raise ConnectorRejected("connector_rejected_full_request")
    payload = result.get("structuredContent")
    if not isinstance(payload, dict):
        raise DeliveryWaiting("connector_write_outcome_ambiguous")
    return payload


def changes_requested(reviews):
    latest = {}
    for row in sorted(
        reviews, key=lambda r: (r.get("submitted_at", ""), r.get("id", 0))
    ):
        if row.get("state") in {"COMMENTED", "PENDING"}:
            continue
        author = (row.get("user") or {}).get("login")
        if not author or row.get("state") not in {
            "APPROVED",
            "CHANGES_REQUESTED",
            "DISMISSED",
        }:
            raise ValueError("remote_review_identity_or_state_missing")
        latest[author.casefold()] = row["state"]
    return "CHANGES_REQUESTED" in latest.values()


async def verify_checks_reviews(connector, config, pr):
    await _checks(connector, config, pr["head"]["sha"])
    await _reviews(connector, config, pr)
    await _review_notes(connector, config, pr)


async def _review_notes(connector, config, pr):
    comments = await pages(
        connector, config.repository, f"issues/{pr['number']}/comments"
    )
    authors = {name.casefold() for name in config.review_note_authors}
    matching = []
    for row in comments:
        if (row.get("user") or {}).get("login", "").casefold() not in authors:
            continue
        try:
            note = json.loads(row.get("body", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(note, dict) and all(
            note.get(key) == value
            for key, value in {
                "schema_version": "connector_review_notes/v1",
                "request_marker": config.marker,
                "head_sha": pr["head"]["sha"],
            }.items()
        ):
            matching.append((row["id"], note))
    if not matching:
        raise DeliveryWaiting("exact_head_adversarial_and_rubber_duck_notes_required")
    note = max(matching, key=lambda pair: pair[0])[1]
    if note.get("verdict") != "pass" or any(
        not isinstance(note.get(key), str) or not note[key].strip()
        for key in ("rubber_duck", "adversarial")
    ):
        raise DeliveryWaiting("exact_head_review_notes_not_passed")
