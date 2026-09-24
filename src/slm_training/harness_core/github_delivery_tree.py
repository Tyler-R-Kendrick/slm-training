"""Read-only Git tree proof: verified source must equal base plus exact documents."""

import hashlib
import os
import stat
import subprocess
from pathlib import PurePosixPath

from .github_connector import DeliveryWaiting


def git_object(kind, content):
    return hashlib.sha1(
        kind.encode() + b" " + str(len(content)).encode() + b"\0" + content
    ).hexdigest()


def tree_sha(entries, *, write=None):
    directories = {"": {}}
    for name, (mode, digest) in entries.items():
        parts = PurePosixPath(name).parts
        parent = ""
        for part in parts[:-1]:
            child = parent + "/" + part if parent else part
            directories.setdefault(child, {})
            directories[parent][part] = ("40000", child)
            parent = child
        directories[parent][parts[-1]] = (mode, digest)
    hashes = {}
    for name in sorted(
        directories, key=lambda p: len(PurePosixPath(p).parts), reverse=True
    ):
        rows = directories[name]
        order = sorted(
            rows, key=lambda n: (n + ("/" if rows[n][0] == "40000" else "")).encode()
        )
        content = b""
        for filename in order:
            mode, value = rows[filename]
            digest = hashes[value] if mode == "40000" else value
            content += (
                mode.encode() + b" " + filename.encode() + b"\0" + bytes.fromhex(digest)
            )
        hashes[name] = git_object("tree", content)
        if write is not None:
            write("tree", content)
    return hashes[""]


def base_entries(source, base):
    """Read the exact pinned Git base, without updating repository metadata."""
    result = subprocess.run(
        ["git", "ls-tree", "-rz", "--full-tree", base],
        cwd=source,
        capture_output=True,
        check=True,
        timeout=10,
    )
    expected = {}
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, name = raw.split(b"\t", 1)
        mode, kind, digest = metadata.decode().split()
        if kind != "blob" or mode not in {"100644", "100755", "120000"}:
            raise DeliveryWaiting("unsupported_git_entry_requires_verified_successor")
        expected[name.decode()] = (mode, digest)
    return expected


def source_entries(source, paths):
    """The caller supplies the complete authenticated file universe."""
    actual = {}
    for name in paths:
        path = source / name
        if path.is_symlink():
            actual[name] = (
                "120000",
                git_object("blob", os.fsencode(os.readlink(path))),
            )
        elif path.is_file():
            mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
            actual[name] = (mode, git_object("blob", path.read_bytes()))
        elif path.exists():
            raise DeliveryWaiting(
                "unsupported_source_entry_requires_verified_successor"
            )
    return actual


def document_tree(source, base, files, paths):
    """Caller authenticates the gate over this same full source path universe."""
    expected = base_entries(source, base)
    expected.update({name: ("100644", git_object("blob", value.encode()))
                     for name, value in files.items()})
    actual = source_entries(source, paths)
    if actual != expected:
        raise DeliveryWaiting(
            "immutable_document_successor_full_gate_and_tree_mapping_required"
        )
    return tree_sha(actual)


def source_tree(source, base, *, predecessor, candidate, allowed_paths, paths):
    """Prove the whole remote-base successor; no dirty inventory is authority.

    predecessor/candidate are authenticated immutable (path, full file universe)
    pairs. Only the accepted delta may differ from the true remote base.
    """
    before = base_entries(source, base)
    if source_entries(*predecessor) != before:
        raise ValueError("accepted_predecessor_remote_base_tree_mismatch")
    after = source_entries(*candidate)
    changed = {name for name in before.keys() | after.keys()
               if before.get(name) != after.get(name)}
    if not changed or not changed <= set(allowed_paths):
        raise ValueError("accepted_source_delta_outside_verified_scope")
    if source_entries(source, paths) != after:
        raise ValueError("source_delivery_gate_candidate_tree_mismatch")
    elements = []
    files = {}
    for name in sorted(changed):
        mode = (after.get(name) or before[name])[0]
        entry = {"path": name, "mode": mode, "type": "blob"}
        if name not in after:
            entry["sha"] = None
        else:
            path = candidate[0] / name
            raw = os.fsencode(os.readlink(path)) if mode == "120000" else path.read_bytes()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise DeliveryWaiting("source_delivery_utf8_representation_required") from error
            entry["content"] = content
            if mode != "120000":
                files[name] = content
        elements.append(entry)
    return {"base_git_tree": tree_sha(before), "candidate_git_tree": tree_sha(after),
            "tree_elements": elements, "files": files}
