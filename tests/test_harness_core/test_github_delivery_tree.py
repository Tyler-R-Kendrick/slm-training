"""Whole-tree proof includes unauthorized code, deletions, modes and symlinks."""

import os
from types import SimpleNamespace

import pytest

from slm_training.harness_core import github_delivery_tree as owner
from slm_training.harness_core.github_connector import DeliveryWaiting


@pytest.fixture
def source(tmp_path, monkeypatch):
    files = {"module.py": b"pass\n", "docs/result.md": b"old\n"}
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    raw = b"".join(
        b"100644 blob "
        + owner.git_object("blob", data).encode()
        + b"\t"
        + name.encode()
        + b"\0"
        for name, data in files.items()
    )

    def ls_tree(command, **kwargs):
        assert command == ["git", "ls-tree", "-rz", "--full-tree", "a" * 40]
        return SimpleNamespace(stdout=raw)

    monkeypatch.setattr(owner.subprocess, "run", ls_tree)
    (tmp_path / "docs/result.md").write_text("new\n")
    return tmp_path, {"docs/result.md": "new\n"}, list(files)


def test_exact_candidate_tree_has_stable_git_identity(source):
    root, docs, paths = source
    sha = owner.document_tree(root, "a" * 40, docs, paths)
    blob = owner.git_object("blob", b"new\n")
    docs_tree = owner.git_object("tree", b"100644 result.md\0" + bytes.fromhex(blob))
    expected = owner.git_object(
        "tree",
        b"40000 docs\0"
        + bytes.fromhex(docs_tree)
        + b"100644 module.py\0"
        + bytes.fromhex(owner.git_object("blob", b"pass\n")),
    )
    assert sha == expected


@pytest.mark.parametrize(
    "change", ["code", "deleted", "executable", "symlink", "extra", "old_docs"]
)
def test_source_gate_cannot_authorize_different_proposed_tree(source, change):
    root, docs, paths = source
    code = root / "module.py"
    if change == "code":
        code.write_text("raise RuntimeError\n")
    elif change == "deleted":
        code.unlink()
    elif change == "executable":
        code.chmod(0o755)
    elif change == "symlink":
        code.unlink()
        code.symlink_to("docs/result.md")
    elif change == "extra":
        (root / "unrelated.py").write_text("pass\n")
        paths.append("unrelated.py")
    else:
        (root / "docs/result.md").write_text("old\n")
    with pytest.raises(DeliveryWaiting, match="immutable_document_successor"):
        owner.document_tree(root, "a" * 40, docs, paths)


def test_git_tree_order_and_symlink_bytes():
    entries = {"a.b": ("100755", "b" * 40), "a/link": ("120000", "c" * 40)}
    child = owner.git_object("tree", b"120000 link\0" + bytes.fromhex("c" * 40))
    expected = owner.git_object(
        "tree",
        b"100755 a.b\0" + bytes.fromhex("b" * 40) + b"40000 a\0" + bytes.fromhex(child),
    )
    assert owner.tree_sha(entries) == expected
    assert os.fsencode("relative/target") == b"relative/target"
