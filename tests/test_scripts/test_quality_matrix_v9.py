import json
from pathlib import Path

from scripts.run_quality_matrix import _v9_experiments, main


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
    assert all(row.initialization == "eval_only" for row in rows)
    assert all(row.honest_slot_contract for row in rows)
    assert all(not row.allow_unconstrained_fallback for row in rows)
    assert rows[1].compiler_search_local_nogoods is False
    assert all(row.compiler_search_local_nogoods for row in rows[2:])


def test_v9_uses_parent_read_only_without_training(tmp_path, monkeypatch) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "manifest.json").write_text('{"content_fingerprint":"fixture"}')
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"checkpoint")
    run_root = tmp_path / "runs"

    def fail_train(*_args, **_kwargs):
        raise AssertionError("eval-only V9 row must not train")

    def fake_evaluate(_cfg, _suites, *, checkpoint, write_gates):
        assert checkpoint == parent
        assert write_gates is True
        return {
            "suites": {
                "smoke": {
                    "n": 1,
                    "parse_rate": 1.0,
                    "placeholder_fidelity": 1.0,
                    "structural_similarity": 1.0,
                    "reward_score": 1.0,
                }
            }
        }

    monkeypatch.setattr("scripts.run_quality_matrix.train", fail_train)
    monkeypatch.setattr("scripts.run_quality_matrix.evaluate_suites", fake_evaluate)

    assert main(
        [
            "--matrix", "v9", "--only", "E240", "--parent", str(parent),
            "--train-dir", str(train_dir), "--test-dir", str(tmp_path / "eval"),
            "--run-root", str(run_root), "--docs-out", str(tmp_path / "results.json"),
            "--suites", "smoke",
        ]
    ) == 0
    result = json.loads((run_root / "qx_e240_compiler_tree_control" / "matrix_result.json").read_text())
    assert result["initialization"] == "eval_only"
    assert result["checkpoint"] == str(parent)
    assert not (run_root / "qx_e240_compiler_tree_control" / "checkpoints").exists()
