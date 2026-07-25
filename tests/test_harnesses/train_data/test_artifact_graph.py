import hashlib
import json
from pathlib import Path

import pytest

from scripts.explain_artifact_overlaps import main
from slm_training.harnesses.train_data.artifact_graph import (
    ArtifactGraphStore,
    ArtifactNodeV1,
    OverlapCode,
    find_cross_split_overlaps,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _root_for(split: str) -> str:
    policy = RootFamilySplitPolicyV1()
    for index in range(10_000):
        value = f"family-{split}-{index}"
        if policy.assign(value) == split:
            return value
    raise AssertionError(f"no root for {split}")


def _node(
    name: str,
    root: str,
    *,
    parents: tuple[str, ...] = (),
    shared: str | None = None,
) -> ArtifactNodeV1:
    split = RootFamilySplitPolicyV1().assign(root)
    return ArtifactNodeV1(
        artifact_id=_sha(name),
        artifact_type="answer",
        root_family_id=root,
        split_group_id=root,
        split=split,
        parent_ids=parents,
        surface_sha256=_sha(shared or f"{name}-surface"),
        alpha_sha256=_sha(shared or f"{name}-alpha"),
        canonical_ast_sha256=_sha(shared or f"{name}-ast"),
        near_template_sha256=_sha(shared or f"{name}-template"),
        payload={"name": name},
    )


def test_root_split_is_assigned_before_expansion_and_inherited() -> None:
    policy = RootFamilySplitPolicyV1()
    root = _root_for("validation")
    assert policy.assign(root) == "validation"
    policy.require_inherited(
        root_family_id=root,
        split_group_id=root,
        split="validation",
        parent_splits=("validation",),
    )
    with pytest.raises(ValueError, match="split_group_id"):
        policy.require_inherited(
            root_family_id=root,
            split_group_id="fork",
            split="validation",
        )


def test_append_is_idempotent_and_lineage_is_traversable(tmp_path: Path) -> None:
    store = ArtifactGraphStore(tmp_path)
    root = _root_for("train")
    first = _node("first", root)
    second = _node("second", root, parents=(first.artifact_id,))
    third = _node("third", root, parents=(second.artifact_id,))

    assert store.append(first) == store.append(first)
    store.append(second)
    store.append(third)
    assert store.ancestors(third.artifact_id) == tuple(
        sorted((first.artifact_id, second.artifact_id))
    )
    assert len(store.load_nodes()) == 3


def test_mixed_split_composition_fails_closed(tmp_path: Path) -> None:
    store = ArtifactGraphStore(tmp_path)
    train = _node("train", _root_for("train"))
    test = _node("test", _root_for("test"))
    store.append(train)
    store.append(test)
    child = _node(
        "child",
        train.root_family_id,
        parents=(train.artifact_id, test.artifact_id),
    )
    with pytest.raises(ValueError, match="incompatible source splits"):
        store.append(child)


def test_cross_split_overlap_is_quarantined_with_all_reason_codes(
    tmp_path: Path,
) -> None:
    store = ArtifactGraphStore(tmp_path)
    shared = "same-template"
    train = _node("train", _root_for("train"), shared=shared)
    test = _node("test", _root_for("test"), shared=shared)
    store.append(train)
    quarantine = store.append(test)

    assert quarantine.parent.name == "quarantine"
    payload = json.loads(quarantine.read_text())
    assert set(payload["reason_codes"]) == {
        OverlapCode.EXACT.value,
        OverlapCode.ALPHA.value,
        OverlapCode.CANONICAL_AST.value,
        OverlapCode.NEAR_TEMPLATE.value,
    }
    assert test.artifact_id not in store.load_nodes()


def test_shared_parent_overlap_is_explained_and_family_drift_rejects() -> None:
    parent = _sha("parent")
    left = _node("left", _root_for("train"), parents=(parent,))
    right = _node("right", _root_for("test"), parents=(parent,))
    findings = find_cross_split_overlaps((left, right))
    assert OverlapCode.PARENT in {item.code for item in findings}
    with pytest.raises(ValueError, match="belongs to"):
        ArtifactNodeV1(**{**left.__dict__, "split": "test"})


def test_explain_cli_reports_clean_graph(tmp_path: Path, capsys) -> None:
    store = ArtifactGraphStore(tmp_path)
    store.append(_node("only", _root_for("train")))
    assert main([str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_explain_cli_includes_quarantined_candidates(tmp_path: Path, capsys) -> None:
    store = ArtifactGraphStore(tmp_path)
    store.append(_node("train", _root_for("train"), shared="same"))
    store.append(_node("test", _root_for("test"), shared="same"))

    assert main([str(tmp_path), "--json"]) == 1
    codes = {row["code"] for row in json.loads(capsys.readouterr().out)}
    assert OverlapCode.CANONICAL_AST.value in codes
