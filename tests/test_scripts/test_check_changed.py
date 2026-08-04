import json
import subprocess
import sys
from types import SimpleNamespace

from scripts import check_changed
from scripts.check_changed import hook_test_targets, select_changed_tests, select_tests


def test_select_tests_is_scoped_and_conservative() -> None:
    assert select_tests(["src/slm_training/web/routes.py"]) == ["tests/test_web"]
    assert select_tests(["src/slm_training/harnesses/model_build/ship_gates.py"]) == [
        "tests/test_harnesses/model_build/test_eval_gates.py"
    ]
    assert select_tests(["tests/test_dsl/test_parser.py"]) == [
        "tests/test_dsl/test_parser.py"
    ]
    assert select_tests(["docs/design/note.md"]) == []
    assert select_tests([".github/workflows/ci.yml"]) == []
    assert select_tests(["pyproject.toml"]) == ["tests"]
    assert select_tests(["unknown/tool.ts"]) == ["tests"]
    assert select_tests(["package-lock.json"]) == []


def test_select_tests_skips_a_deleted_test_file() -> None:
    """A pure test-file deletion must not select a nonexistent pytest target."""
    deleted = "tests/test_dsl/test_a_file_that_was_deleted_and_never_existed.py"
    assert select_tests([deleted]) == []
    assert select_changed_tests([deleted]) == []
    # Alongside its now-deleted source module, the source's own suite mapping
    # still resolves -- the deleted test path is dropped, not substituted.
    assert select_tests(
        ["src/slm_training/harnesses/distill/some_removed_module.py", deleted]
    ) == ["tests/test_harnesses/distill"]


def test_select_tests_deduplicates_nested_targets() -> None:
    assert select_tests(
        ["src/slm_training/dsl/parser.py", "tests/test_dsl/test_parser.py"]
    ) == ["tests/test_dsl", "tests/test_harnesses/model_build"]


def test_autotrain_skill_reference_edits_run_the_cli_parity_suite() -> None:
    assert select_tests([".agents/skills/autotrain/references/sft.md"]) == [
        "tests/test_scripts/test_slm_cli.py"
    ]


def test_autoresearch_skill_and_brains_edits_run_the_skill_guard_suite() -> None:
    assert select_tests([".agents/skills/autoresearch/references/loop.md"]) == [
        "tests/test_scripts/test_autoresearch_skill.py"
    ]
    assert select_tests(["docs/brains/repo/MOC.md"]) == [
        "tests/test_scripts/test_autoresearch_skill.py"
    ]


def test_script_changes_include_their_domain_suite() -> None:
    assert select_tests(["scripts/train_model.py"]) == [
        "tests/test_harnesses/model_build",
        "tests/test_harnesses/quality",
        "tests/test_harnesses/rl",
        "tests/test_scripts",
    ]
    # Exact path ownership: listed script paths do not also pull scripts/.
    assert select_tests(["scripts/autoresearch.py"]) == [
        "tests/test_autoresearch",
    ]
    assert select_tests(["scripts/verify_agent_surfaces.py"]) == [
        "tests/test_scripts/test_verify_agent_surfaces.py",
    ]
    assert select_tests(["scripts/check_changed.py"]) == [
        "tests/test_scripts/test_check_changed.py",
    ]


def test_hook_prefers_explicit_changed_regressions() -> None:
    assert select_changed_tests(
        [
            "src/slm_training/models/grammar.py",
            "tests/casefiles.py",
            "tests/test_dsl/test_grammar_fastpath.py",
        ]
    ) == ["tests/test_dsl/test_grammar_fastpath.py"]
    assert select_changed_tests(["src/slm_training/web/routes.py"]) == [
        "tests/test_web"
    ]


def test_hook_defers_pytest_for_large_diffs() -> None:
    paths = [f"docs/design/run-{i}.json" for i in range(101)]
    paths.append("tests/test_dsl/test_parser.py")
    assert hook_test_targets(paths) == []


def test_version_registry_changes_run_versioning_suite() -> None:
    assert select_tests(["src/slm_training/resources/versions.json"]) == [
        "tests/test_versioning"
    ]
    assert select_tests(["src/slm_training/versioning.py"]) == ["tests/test_versioning"]


def test_case_resource_selects_only_its_mirrored_test() -> None:
    assert select_tests(
        [
            "src/slm_training/resources/test_cases/"
            "test_harness_core/test_gate_engine_golden.json"
        ]
    ) == ["tests/test_harness_core/test_gate_engine_golden.py"]


def test_runtime_eval_resources_select_their_consumers() -> None:
    assert select_tests(["src/slm_training/resources/evals/loss_suite_v1.json"]) == [
        "tests/test_evals",
        "tests/test_harnesses/model_build",
    ]
    assert select_tests(
        ["src/slm_training/resources/evals/openui_ship_gates_v5.json"]
    ) == [
        "tests/test_harness_core/test_gate_engine_golden.py",
        "tests/test_harnesses/model_build/test_eval_gates.py",
    ]


def test_changed_files_can_compare_a_ci_base(monkeypatch) -> None:
    commands = []

    def fake_git(command):
        commands.append(command)
        return "tests/test_b.py\nsrc/a.py\n"

    monkeypatch.setattr(check_changed, "_git", fake_git)
    assert check_changed.changed_files(staged=False, base_ref="base-sha") == [
        "src/a.py",
        "tests/test_b.py",
    ]
    assert commands == [
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMRD",
            "base-sha...HEAD",
            "--",
        ]
    ]


def test_changed_tests_are_collected_once_and_hash_balanced(monkeypatch) -> None:
    commands = []

    def fake_collect(command, **kwargs):
        assert command == [
            check_changed.sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/test_a.py",
            "tests/test_b.py",
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="tests/test_a.py::test_one\ntests/test_a.py::test_two\ntests/test_b.py::test_three\n",
            stderr="",
        )

    def fake_run(command):
        commands.append(command)
        return 0

    monkeypatch.setattr(check_changed, "CHANGED_TEST_WORKERS", 2)
    monkeypatch.setattr(check_changed.subprocess, "run", fake_collect)
    monkeypatch.setattr(check_changed, "_run", fake_run)

    assert check_changed._run_changed_tests_parallel(
        ["tests/test_a.py", "tests/test_b.py"]
    ) == 0
    assert len(commands) == 3
    assert all(len(command) - 4 == 1 for command in commands)
    assert {
        node
        for command in commands
        for node in command[4:]
    } == {
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two",
        "tests/test_b.py::test_three",
    }


def test_changed_test_shards_balance_each_source_file() -> None:
    nodes = [
        *(f"tests/test_slow.py::test_{index}" for index in range(11)),
        *(f"tests/test_fast.py::test_{index}" for index in range(5)),
    ]

    batches = check_changed._shard_test_nodes(nodes, 4)

    for path in ("tests/test_slow.py", "tests/test_fast.py"):
        counts = [
            sum(node.startswith(f"{path}::") for node in batch)
            for batch in batches
        ]
        assert max(counts) - min(counts) <= 1
    assert {node for batch in batches for node in batch} == set(nodes)


def test_parallel_runner_overdecomposes_for_work_stealing(monkeypatch) -> None:
    collected_nodes = "\n".join(
        f"tests/test_slow.py::test_{index}" for index in range(12)
    )
    commands = []

    def fake_collect(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=collected_nodes, stderr="")

    monkeypatch.setattr(check_changed, "CHANGED_TEST_WORKERS", 2)
    monkeypatch.setattr(check_changed.subprocess, "run", fake_collect)
    monkeypatch.setattr(check_changed, "_run", lambda command: commands.append(command) or 0)

    assert check_changed._run_changed_tests_parallel(["tests/test_slow.py"]) == 0
    assert len(commands) == 4
    assert {node for command in commands for node in command[4:]} == set(
        collected_nodes.splitlines()
    )


def test_ci_test_shards_are_disjoint_and_complete(monkeypatch) -> None:
    collected_nodes = "\n".join(
        [
            *(f"tests/test_slow.py::test_{index}" for index in range(16)),
            *(f"tests/test_fast.py::test_{index}" for index in range(8)),
        ]
    )
    shard_nodes = []

    def fake_collect(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=collected_nodes, stderr="")

    monkeypatch.setattr(check_changed.subprocess, "run", fake_collect)
    monkeypatch.setattr(
        check_changed,
        "_run",
        lambda command: shard_nodes.extend(command[4:]) or 0,
    )

    for shard_index in range(8):
        before = set(shard_nodes)
        assert (
            check_changed._run_changed_tests_parallel(
                ["tests/test_slow.py", "tests/test_fast.py"],
                shard_index=shard_index,
                shard_count=8,
            )
            == 0
        )
        current = set(shard_nodes) - before
        assert len(current) == 3

    assert set(shard_nodes) == set(collected_nodes.splitlines())
    assert len(shard_nodes) == len(set(shard_nodes))


def test_parallel_pytest_workers_limit_native_thread_pools(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")

    env = check_changed._pytest_worker_env()

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_list_mode_needs_no_installed_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "scripts.check_changed",
            "--changed-tests-only",
            "--base-ref",
            "HEAD",
            "--list",
        ],
        cwd=check_changed.ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _uniform_nodes(files: int, per_file: int) -> list[str]:
    return [
        f"tests/test_uniform_{file_index}.py::test_{node_index}"
        for file_index in range(files)
        for node_index in range(per_file)
    ]


def test_duration_aware_shards_are_deterministic() -> None:
    nodes = _uniform_nodes(5, 4)
    table = {"tests/test_uniform_1.py": 30.0, "tests/test_uniform_3.py": 2.5}

    first = check_changed._shard_test_nodes(nodes, 3, durations=table)
    second = check_changed._shard_test_nodes(list(reversed(nodes)), 3, durations=table)

    assert first == second
    assert {node for batch in first for node in batch} == set(nodes)


def test_duration_aware_shards_isolate_a_heavy_file() -> None:
    heavy = "tests/test_heavy.py"
    light = [f"tests/test_light_{index}.py" for index in range(10)]
    nodes = [
        *(f"{heavy}::test_{index}" for index in range(3)),
        *(f"{path}::test_{index}" for path in light for index in range(2)),
    ]
    table = {heavy: 100.0, **{path: 1.0 for path in light}}

    for shard_count in (2, 4, 8):
        batches = check_changed._shard_test_nodes(nodes, shard_count, durations=table)
        heavy_batches = [
            batch for batch in batches if any(node.startswith(f"{heavy}::") for node in batch)
        ]

        assert len(heavy_batches) == 1, "a measured file must stay on one shard"
        assert sorted(heavy_batches[0]) == sorted(
            node for node in nodes if node.startswith(f"{heavy}::")
        )
        assert {node for batch in batches for node in batch} == set(nodes)


def test_empty_duration_table_reduces_to_node_count_balancing() -> None:
    nodes = _uniform_nodes(6, 5)

    batches = check_changed._shard_test_nodes(nodes, 4, durations={})

    counts = [len(batch) for batch in batches]
    assert max(counts) - min(counts) <= 1
    assert {node for batch in batches for node in batch} == set(nodes)
    # Per-file balance is preserved too, which is what the count-balanced
    # predecessor guaranteed.
    for file_index in range(6):
        prefix = f"tests/test_uniform_{file_index}.py::"
        per_file = [sum(node.startswith(prefix) for node in batch) for batch in batches]
        assert max(per_file) - min(per_file) <= 1


def test_committed_test_duration_table_is_schema_valid() -> None:
    payload = json.loads(check_changed.TEST_DURATIONS_PATH.read_text(encoding="utf-8"))

    assert payload["schema"] == "test_file_durations/v1"
    assert payload["unit"] == "seconds"
    assert payload["generated_by"]
    durations = payload["durations"]
    assert durations
    for path, seconds in durations.items():
        assert path.startswith("tests/") and path.endswith(".py"), path
        assert isinstance(seconds, (int, float)) and seconds > 0, path

    loaded = check_changed._test_file_durations()
    assert loaded == {path: float(value) for path, value in durations.items()}


def test_missing_duration_table_degrades_to_an_empty_weighting(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        check_changed, "TEST_DURATIONS_PATH", tmp_path / "absent.json"
    )
    check_changed._test_file_durations.cache_clear()
    try:
        assert check_changed._test_file_durations() == {}
    finally:
        check_changed._test_file_durations.cache_clear()
