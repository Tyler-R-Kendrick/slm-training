"""Exact remote objects become private comparison metadata, never new commits."""

import subprocess

import pytest

from slm_training.harness_core.github_git_snapshot import commit_object, private_git_snapshot
from slm_training.harness_core.github_delivery_tree import git_object, source_entries, tree_sha


def commit(tree, *, signature=None):
    raw = (f"tree {tree}\nparent {'a' * 40}\n"
           "author Reviewer <reviewer@invalid> 946684800 +0000\n"
           "committer Reviewer <reviewer@invalid> 946684800 +0000\n\nmerged\n")
    person = {"name": "Reviewer", "email": "reviewer@invalid", "date": "2000-01-01T00:00:00Z"}
    value = {"tree": {"sha": tree}, "parents": [{"sha": "a" * 40}], "author": person,
             "committer": person, "message": "merged\n"}
    if signature:
        value["verification"] = {"payload": raw, "signature": signature}
        raw = raw.replace("\n\n", "\ngpgsig " + signature.rstrip("\n").replace("\n", "\n ") + "\n\n", 1)
    return value, raw, git_object("commit", raw.encode())


@pytest.mark.parametrize("signed", [False, True])
def test_remote_commit_reconstruction_requires_exact_git_hash(signed):
    value, raw, sha = commit("b" * 40, signature="-----BEGIN PGP SIGNATURE-----\nopaque\n-----END PGP SIGNATURE-----\n" if signed else None)
    assert commit_object(value, sha) == raw
    value["tree"]["sha"] = "c" * 40
    with pytest.raises(ValueError, match="exact_remote_git_commit|payload_metadata_mismatch"):
        commit_object(value, sha)


def test_observed_merge_can_be_next_gate_base_without_checkout_refresh(tmp_path):
    import time
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    (base / "module.py").write_text("accepted previous repair\n")
    (candidate / "module.py").write_text("next repair\n")
    tree = tree_sha(source_entries(base, ["module.py"]))
    value, raw, sha = commit(tree)
    private_git_snapshot(candidate, base, ["module.py"], commit=commit_object(value, sha), expected=sha)
    result = subprocess.run(["git", "show", sha + ":module.py"], cwd=candidate, check=True, capture_output=True, text=True, timeout=10)
    assert result.stdout == "accepted previous repair\n"
    from scripts.merge_verification import changed_paths
    assert changed_paths(candidate, sha) == (tree, ["module.py"])
    assert (candidate / "module.py").read_text() == "next repair\n"
    assert not (base / ".git").exists()
    from scripts.merge_verification_git import _private_git
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    _private_git(candidate, isolated / ".git", 30, time.monotonic())
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=isolated,
                          check=True, capture_output=True, text=True, timeout=10).stdout.strip() == sha
    with pytest.raises(FileExistsError):
        private_git_snapshot(candidate, base, ["module.py"], commit=raw, expected=sha)


def test_wrong_remote_object_never_creates_private_metadata(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    (base / "module.py").write_text("different\n")
    _, raw, sha = commit("b" * 40)
    with pytest.raises(ValueError, match="commit_tree_mismatch"):
        private_git_snapshot(candidate, base, ["module.py"], commit=raw, expected=sha)
    assert not (candidate / ".git").exists()
