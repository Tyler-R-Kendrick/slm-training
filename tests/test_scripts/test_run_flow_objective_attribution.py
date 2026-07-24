"""CLI surface tests for SLM-200."""

from __future__ import annotations

import json
from scripts.run_flow_objective_attribution import main


def test_describe_lists_modes_and_bounded_matrix(capsys) -> None:
    assert main(["--describe"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matrix_set"] == "slm200_flow_objective_attribution"
    assert payload["max_wall_minutes"] <= 3
    assert payload["modes"] == [
        "describe",
        "plan",
        "screen",
        "confirm",
        "resume",
        "analyze",
    ]
