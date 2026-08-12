"""Runner tests."""

from scripts import run_research_20_cost_compare


def test_default_off_json(capsys) -> None:
    assert run_research_20_cost_compare.main(["--json"]) == 0
    assert "skipped_default_off" in capsys.readouterr().out
