from pathlib import Path

from scripts.run_quality_matrix import _v9_experiments


def test_v9_registers_only_planned_lattice_rows() -> None:
    rows = _v9_experiments(Path("outputs/data/train/v1"))
    assert [row.eid for row in rows] == [
        "E240",
        "E241",
        "E242",
        "E243",
        "E244",
        "E245",
        "E246",
        "E247",
    ]
    assert rows[0].compiler_search_mode == "greedy"
    assert rows[-1].compiler_search_width == 8
    assert all(row.compiler_decode_mode == "tree" for row in rows)
