"""Runner tests."""

from scripts import run_research_19_lean_rademacher


def test_default_off_json(capsys) -> None:
    assert run_research_19_lean_rademacher.main(["--json"]) == 0
    assert "skipped_default_off" in capsys.readouterr().out
