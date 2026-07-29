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
    assert select_tests(["scripts/autoresearch.py"]) == [
        "tests/test_autoresearch",
        "tests/test_scripts",
    ]


def test_hook_prefers_explicit_changed_regressions() -> None:
    assert select_changed_tests(
        [
            "src/slm_training/models/grammar.py",
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
    assert select_tests(["src/slm_training/versioning.py"]) == [
        "tests/test_versioning"
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


def test_check_passes_push_base_to_version_stamp_verifier(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(check_changed, "validate_repository", lambda: [])
    monkeypatch.setattr(
        check_changed,
        "_run",
        lambda command: commands.append(command) or 0,
    )

    assert check_changed.check([], base_ref="base-sha") == 0
    assert commands == [
        [
            check_changed.sys.executable,
            "-m",
            "scripts.verify_version_stamps",
            "--check",
            "--base",
            "base-sha",
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
