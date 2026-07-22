from types import SimpleNamespace

from scripts import check_changed
from scripts.check_changed import hook_test_targets, select_changed_tests, select_tests


def test_select_tests_is_scoped_and_conservative() -> None:
    assert select_tests(["src/slm_training/web/routes.py"]) == ["tests/test_web"]
    assert select_tests(["tests/test_dsl/test_parser.py"]) == [
        "tests/test_dsl/test_parser.py"
    ]
    assert select_tests(["docs/design/note.md"]) == []
    assert select_tests([".github/workflows/ci.yml"]) == []
    assert select_tests(["pyproject.toml"]) == ["tests"]
    assert select_tests(["unknown/tool.ts"]) == ["tests"]


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


def test_changed_tests_are_collected_once_and_partitioned_by_node(monkeypatch) -> None:
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
    assert sorted(commands) == sorted(
        [
            [
                check_changed.sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_a.py::test_one",
                "tests/test_b.py::test_three",
            ],
            [
                check_changed.sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_a.py::test_two",
            ],
        ]
    )
