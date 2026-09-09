"""Release coverage and evidence falsifiers; actual pytest subprocess exercises."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.casefiles import case_values
from hypothesis import given
from hypothesis import strategies as st

from scripts.check_changed import select_tests
from scripts.merge_verification import plan_shards, run_release_gate, run_workload
from scripts.merge_verification_evidence import (
    ReceiptCache,
    authorize_release,
    digest,
    environment_identity,
    source_identity,
    validate_workload,
)


def _covered(targets: list[str], node: str) -> bool:
    path = node.split("::", 1)[0]
    return any(path == target or path.startswith(target + "/") for target in targets)


@given(
    st.sets(
        st.sampled_from(
            [
                "src/slm_training/models/grammar.py",
                "tests/test_dsl/test_parser.py",
                "src/slm_training/web/routes.py",
                "tests/test_web/conftest.py",
                "unknown/settings.toml",
                "package-lock.json",
                "src/slm_training/resources/test_cases/test_dsl/test_parser.json",
            ]
        )
    )
)
def test_source_resource_coverage_is_monotone(extra) -> None:
    base = ["src/slm_training/models/grammar.py"]
    before, after = select_tests(base), select_tests([*base, *extra])
    for node in [
        "tests/test_models/test_any.py::test_a",
        "tests/test_harnesses/model_build/test_a.py::test_a",
        "tests/test_dsl/test_tokenizer_grammar_invariants.py::test_a",
    ]:
        assert not _covered(before, node) or _covered(after, node)


@pytest.mark.parametrize(
    "unknown",
    case_values(__file__, "test_unknown_alongside_known_target_is_conservative"),
)
def test_unknown_alongside_known_target_is_conservative(unknown) -> None:
    assert select_tests([unknown, "tests/test_dsl/test_parser.py"]) == ["tests"]


def test_extracted_test_helpers_and_new_nll_keep_required_coverage():
    from scripts.check_changed import TEST_HELPER_DEPENDENTS

    for helper, consumers in TEST_HELPER_DEPENDENTS.items():
        assert set(consumers) <= set(select_tests([helper]))
        resource = (
            "src/slm_training/resources/test_cases/"
            + helper.removeprefix("tests/").removesuffix(".py")
            + ".json"
        )
        assert set(consumers) <= set(select_tests([resource]))
    assert "tests/test_scripts" in select_tests(["scripts/autotrain_nll.py"])
    assert "tests/test_autoresearch" in select_tests(
        ["src/slm_training/autoresearch/campaign_events.py"]
    )


def _fixture(tmp_path: Path, source: str) -> tuple[Path, Path]:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "tests").mkdir()
    (root / "tests/test_case.py").write_text(source)
    (root / "pytest.ini").write_text(
        "[pytest]\naddopts = -m 'not training and not slow'\nmarkers =\n training\n slow\n"
    )
    control = tmp_path / "controller"
    control.mkdir(mode=0o700)
    return root, control


def test_real_collection_and_shard_include_default_excluded_markers(
    tmp_path, monkeypatch
) -> None:
    root, control = _fixture(
        tmp_path,
        "import pytest\n@pytest.mark.training\ndef test_train(): pass\n@pytest.mark.slow\ndef test_chaos(): pass\n",
    )
    monkeypatch.setenv("PYTEST_ADDOPTS", "-m 'not training and not slow'")
    collection = run_workload(
        root, ["tests"], collect_only=True, seconds=15, directory=control
    )
    assert collection["status"] == "ok", collection
    nodes = collection["nodes"]
    assert len(nodes) == 2
    assert any(
        "training" in names for names in collection["workload"]["markers"].values()
    )
    result = run_workload(
        root, nodes, collect_only=False, seconds=15, directory=control
    )
    assert result["status"] == "ok", result
    assert len(result["workload"]["reports"]) == 6


def test_actual_source_resource_and_changed_test_node_union(tmp_path) -> None:
    root, control = _fixture(tmp_path, "def test_extra(): pass\n")
    web = root / "tests/test_web"
    web.mkdir()
    (web / "test_web.py").write_text("def test_owned(): pass\n")
    source = ["src/slm_training/web/routes.py"]
    extra = [
        "tests/test_case.py",
        "src/slm_training/resources/test_cases/test_web/test_web.json",
    ]
    before = run_workload(
        root,
        select_tests(source, root=root),
        collect_only=True,
        seconds=15,
        directory=control,
    )
    after = run_workload(
        root,
        select_tests(source + extra, root=root),
        collect_only=True,
        seconds=15,
        directory=control,
    )
    assert before["status"] == after["status"] == "ok"
    assert set(before["nodes"]) < set(after["nodes"])
    assert len(after["nodes"]) == 2


@pytest.mark.parametrize(
    "source",
    case_values(__file__, "test_real_empty_or_skipped_workload_cannot_pass"),
)
def test_real_empty_or_skipped_workload_cannot_pass(tmp_path, source) -> None:
    root, control = _fixture(tmp_path, source)
    result = run_workload(
        root,
        [
            "tests/test_case.py::test_skip"
            if "test_skip" in source
            else "tests/test_case.py::test_xfail"
            if "test_xfail" in source
            else "tests"
        ],
        collect_only=not source,
        seconds=15,
        directory=control,
    )
    assert result["status"] == "failed"


def test_real_timeout_has_no_passing_evidence(tmp_path) -> None:
    root, control = _fixture(tmp_path, "import time\ndef test_hang(): time.sleep(30)\n")
    result = run_workload(
        root,
        ["tests/test_case.py::test_hang"],
        collect_only=False,
        seconds=0.2,
        directory=control,
    )
    assert result["status"] == "timeout"
    assert "workload_sha256" not in result


def _valid_workload() -> dict:
    node = "tests/test_case.py::test_case"
    return {
        "schema": "merge_test_workload/v1",
        "request_digest": "r",
        "exit_code": 0,
        "nodes": [node],
        "deselected": [],
        "collection_errors": [],
        "reports": [
            {
                "nodeid": node,
                "when": phase,
                "outcome": "passed",
                "duration_seconds": 0.1,
            }
            for phase in ("setup", "call", "teardown")
        ],
    }


def test_workload_exact_node_and_phase_accounting() -> None:
    valid = _valid_workload()
    assert (
        validate_workload(valid, request_digest="r", expected=valid["nodes"])
        == valid["nodes"]
    )
    with pytest.raises(ValueError, match="locked selection"):
        validate_workload(
            valid, request_digest="r", expected=["tests/test_other.py::test_case"]
        )
    valid["reports"].pop()
    with pytest.raises(ValueError, match="phases"):
        validate_workload(valid, request_digest="r", expected=valid["nodes"])


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1, True])
def test_invalid_workload_numbers_rejected(duration) -> None:
    valid = _valid_workload()
    valid["reports"][0]["duration_seconds"] = duration
    with pytest.raises(ValueError, match="duration"):
        validate_workload(valid, request_digest="r", expected=valid["nodes"])


def test_no_candidate_cache_or_tampered_receipt(tmp_path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        ReceiptCache(root / "cache", root)
    cache = ReceiptCache(tmp_path / "controller", root)
    identity = digest("tree1")
    cache.save({"identity": identity, "passed_nodes": []})
    assert cache.load(digest("tree2")) is None
    path = cache.directory / f"{identity}.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["passed_nodes"] = ["forged"]
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="unauthenticated"):
        cache.load(identity)


def test_fixture_lock_and_runner_content_change_tree_identity(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    fixture = tmp_path / "fixture.json"
    fixture.write_text("old")
    before = source_identity(tmp_path)
    fixture.write_text("new")
    assert source_identity(tmp_path) != before
    before = source_identity(tmp_path)
    (tmp_path / "uv.lock").write_text("new lock")
    assert source_identity(tmp_path) != before
    before = source_identity(tmp_path)
    (tmp_path / "runner.py").write_text("new runner")
    assert source_identity(tmp_path) != before


def test_environment_binding_has_runtime_and_distribution_identity(monkeypatch) -> None:
    identity = environment_identity()
    assert identity["python"] == sys.version
    assert len(identity["executable_sha256"]) == 64
    assert len(identity["distributions_sha256"]) == 64
    assert len(identity["dependency_files_sha256"]) == 64
    monkeypatch.setenv("PATH", "/unavailable-test-toolchain")
    assert (
        environment_identity()["execution_environment_sha256"]
        != identity["execution_environment_sha256"]
    )


def test_dependency_edit_invalidates_environment_without_metadata_change(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from scripts import merge_verification_evidence

    dependency = tmp_path / "dependency.py"
    dependency.write_text("before")
    monkeypatch.setattr(
        merge_verification_evidence.shutil, "which", lambda _: str(dependency)
    )
    distribution = SimpleNamespace(
        metadata={"Name": "fixture", "Version": "1"},
        read_text=lambda _: "unchanged metadata",
        files=["dependency.py"],
        locate_file=lambda path: tmp_path / path,
    )
    monkeypatch.setattr(
        merge_verification_evidence.importlib.metadata,
        "distributions",
        lambda: [distribution],
    )
    before = environment_identity()
    dependency.write_text("after!")
    after = environment_identity()
    assert before["distributions_sha256"] == after["distributions_sha256"]
    assert before["dependency_files_sha256"] != after["dependency_files_sha256"]
    assert before["command_files"] != after["command_files"]


def test_shards_are_nonempty_and_exact() -> None:
    nodes = [f"tests/test_case.py::test_{n}" for n in range(100)]
    shards = plan_shards(nodes, 10)
    assert all(shards)
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)
    with pytest.raises(ValueError):
        plan_shards([], 10)


def test_real_workload_resumes_verified_collection_without_duplicate_tests(
    tmp_path, monkeypatch
) -> None:
    from functools import partial

    from scripts import merge_verification
    from scripts.verify_merge_ready import Step, run_step

    root, control = _fixture(tmp_path, "def test_case(): pass\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    # The disposable repo has no commits. Only base-tree lookup is a fixture;
    # candidate hashing, pytest collection/execution, signing and restart are real.
    monkeypatch.setattr(
        merge_verification,
        "changed_paths",
        lambda *_: ("base-fixture", ["tests/test_case.py"]),
    )
    steps = (Step("static-canary", (sys.executable, "-c", "pass")),)
    execute = merge_verification._run_shards
    monkeypatch.setattr(merge_verification, "_run_shards", lambda *args: None)
    verify = partial(
        run_release_gate,
        steps,
        root=root,
        base_ref="HEAD",
        state_dir=control,
        step_seconds=15,
        run_step=run_step,
        local_feedback=True,
    )
    first = verify()
    assert first["status"] == "pending"
    assert first["node_counts"]["pending"] == 1
    monkeypatch.setattr(merge_verification, "_run_shards", execute)
    second = verify()
    assert second["verification_complete"] is True, second
    assert second["release_authorized"] is False
    assert [row.get("kind") for row in second["steps"]].count("collection") == 1
    assert [row.get("kind") for row in second["steps"]].count("shard") == 1
    third = verify(step_seconds=12)
    assert third["identity"] == second["identity"]
    assert third["spent_seconds"] == second["spent_seconds"]
    (root / "fixture.json").write_text("new fixture identity")
    fourth = verify()
    assert fourth["identity"] != third["identity"]
    assert fourth["verification_complete"] is True


def test_local_evidence_is_not_independent_release_authority() -> None:
    evidence = {
        "identity": "i",
        "verification_complete": True,
        "evidence_class": "isolated_process",
    }
    assert not authorize_release(
        evidence, expected_identity="i", independent_verification={}
    )
    independent = {
        "verification_identity": "i",
        "evidence_sha256": digest(evidence),
        "scope_passed": True,
        "original_reproducer_passed": True,
        "isolation_enforced": True,
    }
    assert authorize_release(
        evidence, expected_identity="i", independent_verification=independent
    )
    assert not authorize_release(
        evidence, expected_identity="other", independent_verification=independent
    )
    independent["original_reproducer_passed"] = False
    assert not authorize_release(
        evidence, expected_identity="i", independent_verification=independent
    )
