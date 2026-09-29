"""Hash-authenticated Git objects for a private verifier, never publication.

The remote connector supplies commit metadata; accepted immutable local source
supplies all blobs. A reconstructed object is usable only when its Git SHA
equals the observed remote SHA. Unsupported representations fail closed.
"""

import os
import subprocess
import zlib
from datetime import datetime
from itertools import product

from .github_connector import DeliveryWaiting
from .github_delivery_tree import git_object, source_entries, tree_sha


def commit_object(commit, expected):
    proof = commit.get("verification") or {}
    payload, signature = proof.get("payload"), proof.get("signature")
    if payload and signature:
        headers, message = payload.split("\n\n", 1)
        contents = [headers + "\ngpgsig " + signature.rstrip("\n").replace("\n", "\n ") + "\n\n" + message]
    else:
        headers = ["tree " + commit["tree"]["sha"]]
        headers.extend("parent " + parent["sha"] for parent in commit["parents"])
        people = [(commit[role], int(datetime.fromisoformat(commit[role]["date"].replace("Z", "+00:00")).timestamp()))
                  for role in ("author", "committer")]
        # GitHub normalizes dates to UTC; the original commit object retains each offset.
        zones = ["+0000", *(f"{'+' if minutes >= 0 else '-'}{abs(minutes) // 60:02d}{abs(minutes) % 60:02d}"
                            for minutes in range(-12 * 60, 14 * 60 + 1, 15) if minutes)]
        contents = ("\n".join([*headers, *(f"{role} {person['name']} <{person['email']}> {timestamp} {zone}"
                     for role, (person, timestamp), zone in zip(("author", "committer"), people, pair))])
                    + "\n\n" + commit["message"] for pair in product(zones, repeat=2))
    # GitHub metadata sometimes omits the final newline from its message. Only
    # an exact cryptographic match is accepted; neither form is assumed correct.
    for content in contents:
        for candidate in (content, content + "\n"):
            if git_object("commit", candidate.encode()) == expected:
                lines = candidate.split("\n\n", 1)[0].splitlines()
                if ([line for line in lines if line.startswith("tree ")] != ["tree " + commit["tree"]["sha"]]
                        or [line for line in lines if line.startswith("parent ")] != ["parent " + row["sha"] for row in commit["parents"]]):
                    raise ValueError("remote_commit_payload_metadata_mismatch")
                return candidate
    raise DeliveryWaiting("exact_remote_git_commit_representation_required")


def private_git_snapshot(destination, source, paths, *, commit, expected):
    """Create new isolated metadata rooted at an already observed remote commit.

    Does not change source metadata, create a new commit, contact GitHub, or
    update a shared ref. The only index command targets this new private repo.
    """
    raw = commit.encode()
    if git_object("commit", raw) != expected:
        raise ValueError("remote_base_commit_object_hash_mismatch")
    entries = source_entries(source, paths)
    lines = raw.split(b"\n\n", 1)[0].splitlines()
    if lines[0] != b"tree " + tree_sha(entries).encode():
        raise ValueError("remote_base_commit_tree_mismatch")
    metadata = destination / ".git"
    metadata.mkdir()
    (metadata / "objects").mkdir()
    (metadata / "refs").mkdir()

    def write(kind, content):
        digest = git_object(kind, content)
        path = metadata / "objects" / digest[:2] / digest[2:]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(zlib.compress(kind.encode() + b" " + str(len(content)).encode() + b"\0" + content))

    for name, (mode, _) in entries.items():
        path = source / name
        content = os.fsencode(os.readlink(path)) if mode == "120000" else path.read_bytes()
        write("blob", content)
    tree_sha(entries, write=write)
    write("commit", raw)
    (metadata / "HEAD").write_text(expected + "\n")
    # Parents may intentionally be absent; canonical clone/selection sees this
    # observed commit as the boundary, not a fabricated predecessor history.
    (metadata / "shallow").write_text(expected + "\n")
    (metadata / "config").write_text("[core]\nrepositoryformatversion = 0\nbare = false\n")
    subprocess.run(["/usr/bin/git", "-c", "core.hooksPath=/dev/null", "read-tree", expected],
                   cwd=destination, check=True, capture_output=True, timeout=10,
                   env={"PATH": "/usr/bin:/bin", "HOME": str(destination),
                        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"})
