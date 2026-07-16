from pathlib import Path

from scripts.run_quality_matrix import _apply_eval_checkpoint, _v9_experiments


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


def test_v9_rows_declare_eval_only_lineage() -> None:
    rows = _v9_experiments(Path("outputs/data/train/v1"))
    assert all(row.initialization == "eval_only" for row in rows)


def test_apply_eval_checkpoint_routes_declared_eval_only_rows() -> None:
    rows = _v9_experiments(Path("outputs/data/train/v1"))
    ckpt = Path("outputs/runs/qx_e240_compiler_tree_control/checkpoints/last.pt")
    routed = _apply_eval_checkpoint(rows, ckpt)
    assert all(row.eval_from_checkpoint == str(ckpt) for row in routed)
    # Without a checkpoint the rows are returned untouched (classifier would
    # then require --parent/--scratch-control rather than silently retraining).
    assert all(
        row.eval_from_checkpoint is None for row in _apply_eval_checkpoint(rows, None)
    )
