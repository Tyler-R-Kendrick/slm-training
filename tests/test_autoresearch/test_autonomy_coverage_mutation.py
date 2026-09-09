"""A14: changed-test shortcut mutation, checked over actual collected pytest nodes.

Only a disposable copy of the loaded selector function is mutated. This proves
selection semantics, not an OS sandbox or a passing execution of collected tests.
"""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

import pytest

from scripts import check_changed as owner
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS

SOURCE = "src/slm_training/harnesses/model_build/ship_gates.py"
CHANGED_TEST = "tests/test_scripts/test_check_changed.py"


def _covered(targets, nodes):
    return {
        node
        for node in nodes
        if any(
            node.partition("::")[0] == target
            or node.partition("::")[0].startswith(target + "/")
            for target in targets
        )
    }


def _coverage_oracle(select, nodes, required):
    actual = _covered(select([SOURCE, CHANGED_TEST]), nodes)
    if not required <= actual:
        raise AssertionError("source_owned_nodes_lost")
    return actual


def test_changed_test_only_mutant_loses_source_owned_collected_nodes(
    tmp_path, record_property
):
    source = textwrap.dedent(inspect.getsource(owner.select_tests))
    tree = ast.parse(source)
    function = tree.body[0]
    # Insert the historical non-monotone shortcut into the real current owner.
    shortcut = ast.parse(
        "changed = [p for p in paths if p.startswith('tests/') and p.endswith('.py')]\n"
        "if changed:\n"
        "    mutation_reached.append(tuple(paths))\n"
        "    return sorted(changed)\n"
    ).body
    function.body[1:1] = shortcut
    mutated = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    root = Path(__file__).resolve().parents[2]
    targets = owner.select_tests([SOURCE, CHANGED_TEST])
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-o",
        "addopts=",
        "-m",
        "",
        "-p",
        "no:cacheprovider",
        *targets,
    ]
    collection = run_bounded_process(
        argv,
        cwd=root,
        interrupt_after_seconds=INTERRUPT_AFTER_SECONDS,
        kill_grace_seconds=KILL_GRACE_SECONDS,
        max_output_bytes=2_000_000,
    )
    assert collection.returncode == 0 and not collection.timed_out, collection.stderr
    assert not collection.stdout_truncated, "truncated collection is not evidence"
    nodes = {
        line
        for line in collection.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }
    required = _covered(owner.select_tests([SOURCE]), nodes)
    assert required and _covered([CHANGED_TEST], nodes), (
        "empty collection is not evidence"
    )
    namespace = dict(owner.select_tests.__globals__, mutation_reached=[])
    baseline_path, mutant_path = tmp_path / "baseline.py", tmp_path / "mutant.py"
    baseline_path.write_text(source)
    mutant_path.write_text(mutated)
    exec(compile(baseline_path.read_text(), str(baseline_path), "exec"), namespace)
    baseline = _coverage_oracle(namespace["select_tests"], nodes, required)
    exec(compile(mutant_path.read_text(), str(mutant_path), "exec"), namespace)
    with pytest.raises(AssertionError, match="^source_owned_nodes_lost$") as failed:
        _coverage_oracle(namespace["select_tests"], nodes, required)
    assert namespace["mutation_reached"] == [(SOURCE, CHANGED_TEST)]
    assert not _covered([CHANGED_TEST], nodes) & required
    exec(compile(baseline_path.read_text(), str(baseline_path), "exec"), namespace)
    assert _coverage_oracle(namespace["select_tests"], nodes, required) == baseline
    receipt = {
        "mutation": "restore_changed_test_only_shortcut",
        "evidence_class": "isolated_function_copy_actual_owner_and_pytest_collection",
        "source_sha256": hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
        "baseline_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "mutated_function_sha256": hashlib.sha256(mutated.encode()).hexdigest(),
        "collection_argv": argv,
        "collection_seconds": collection.duration_seconds,
        "collection_exit": collection.returncode,
        "required_nodes": sorted(required),
        "collected_nodes": sorted(nodes),
        "mutated_branch_calls": len(namespace["mutation_reached"]),
        "oracle_failure": str(failed.value),
        "baseline": "passed",
        "mutant": "killed_by_intended_oracle",
        "restored": "passed",
    }
    record_property("mutation_receipt", json.dumps(receipt, sort_keys=True))


def test_collection_oracle_does_not_count_unrelated_errors_as_killed_mutants():
    def unrelated_failure(_paths):
        raise ValueError("unrelated fixture error")

    with pytest.raises(ValueError, match="^unrelated fixture error$"):
        _coverage_oracle(unrelated_failure, set(), set())
