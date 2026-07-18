from scripts.check_changed import hook_test_targets, select_changed_tests, select_tests


def test_select_tests_is_scoped_and_conservative() -> None:
    assert select_tests(["src/slm_training/web/routes.py"]) == ["tests/test_web"]
    assert select_tests(["tests/test_dsl/test_parser.py"]) == [
        "tests/test_dsl/test_parser.py"
    ]
    assert select_tests(["docs/design/note.md"]) == []
    assert select_tests(["pyproject.toml"]) == ["tests"]
    assert select_tests(["unknown/tool.ts"]) == ["tests"]


def test_select_tests_deduplicates_nested_targets() -> None:
    assert select_tests(
        ["src/slm_training/dsl/parser.py", "tests/test_dsl/test_parser.py"]
    ) == ["tests/test_dsl", "tests/test_harnesses/model_build"]


def test_train_skill_reference_edits_run_the_cli_parity_suite() -> None:
    assert select_tests([".agents/skills/train/references/sft.md"]) == [
        "tests/test_scripts/test_slm_cli.py"
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
