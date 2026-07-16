from pathlib import Path

import pytest

from scripts.run_quality_matrix import _v10_experiments, main


def test_v10_registers_exact_state_ablation_rows() -> None:
    rows = _v10_experiments(Path("outputs/data/train/v1"))
    assert [row.eid for row in rows] == [
        "E248",
        "E249",
        "E250",
        "E251",
        "E252",
        "E253",
        "E254",
    ]
    assert rows[0].local_parent_control is True
    assert [row.local_preference_objective for row in rows[1:]] == [
        "ce_margin",
        "unlikelihood",
        "ftpo_single",
        "ftpo_set",
        "ftpo_set",
        "ftpo_set",
    ]
    assert rows[-2].local_preference_reference_tether is True
    assert rows[-1].local_preference_balanced is True
    assert all(row.compiler_decode_mode == "tree" for row in rows)


def test_v10_list_needs_no_parent_or_event_file(capsys) -> None:
    assert main(["--matrix", "v10", "--list"]) == 0
    assert '"id": "E248"' in capsys.readouterr().out


def test_v10_intervention_execution_requires_events() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--matrix",
                "v10",
                "--only",
                "E249",
                "--parent",
                "parent.pt",
            ]
        )
