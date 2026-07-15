from scripts.check_changed import select_tests


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


def test_script_changes_include_their_domain_suite() -> None:
    assert select_tests(["scripts/train_model.py"]) == [
        "tests/test_harnesses/model_build",
        "tests/test_harnesses/quality",
        "tests/test_harnesses/rl",
        "tests/test_scripts",
    ]
