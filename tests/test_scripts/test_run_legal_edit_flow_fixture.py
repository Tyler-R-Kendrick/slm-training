from __future__ import annotations

import json

from scripts.train_legal_edit_flow import main as train_main


def test_fixture_train_describe_is_default_off(capsys) -> None:
    assert train_main(["--describe"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["default_off"]
    assert payload["fidelity"] == "adapted_path_approximation"
