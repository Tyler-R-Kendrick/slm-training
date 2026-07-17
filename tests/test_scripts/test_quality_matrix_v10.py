import json
from pathlib import Path

import pytest

from scripts.run_quality_matrix import _maybe_local_preference, _v10_experiments, main


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
        "E262",
    ]
    assert rows[0].local_parent_control is True
    assert [row.local_preference_objective for row in rows[1:]] == [
        "ce_margin",
        "unlikelihood",
        "ftpo_single",
        "ftpo_set",
        "ftpo_set",
        "ftpo_set",
        "ftpo_set",
    ]
    by_id = {row.eid: row for row in rows}
    assert by_id["E253"].local_preference_reference_tether is True
    assert by_id["E254"].local_preference_balanced is True
    assert by_id["E262"].local_preference_reference_tether is False
    assert by_id["E262"].local_preference_balanced is False
    assert all(row.compiler_decode_mode == "tree" for row in rows)


def test_v10_uses_the_v9_strict_compiler_tree_control() -> None:
    from scripts.run_quality_matrix import _v9_experiments

    v9 = _v9_experiments(Path("outputs/data/train/v1"))[0]
    v10 = _v10_experiments(Path("outputs/data/train/v1"))[0]
    for field in (
        "runtime_override_fields",
        "output_tokenizer",
        "grammar_ltr_primary",
        "grammar_finalize_validate",
        "compiler_decode_mode",
        "schema_in_context",
        "slot_contract_in_context",
        "slot_contract_constrained_decode",
        "honest_slot_contract",
        "design_md_in_context",
        "allow_unconstrained_fallback",
    ):
        assert getattr(v10, field) == getattr(v9, field)


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


def test_v10_parent_control_is_read_only(tmp_path, monkeypatch) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "manifest.json").write_text('{"content_fingerprint":"fixture"}')
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"checkpoint")
    run_root = tmp_path / "runs"

    def fake_evaluate(_cfg, _suites, *, checkpoint, write_gates):
        assert checkpoint == parent
        assert write_gates is True
        return {"suites": {}}

    monkeypatch.setattr("scripts.run_quality_matrix.evaluate_suites", fake_evaluate)

    assert main(
        [
            "--matrix", "v10", "--only", "E248", "--parent", str(parent),
            "--train-dir", str(train_dir), "--test-dir", str(tmp_path / "eval"),
            "--run-root", str(run_root), "--docs-out", str(tmp_path / "results.json"),
            "--suites", "smoke",
        ]
    ) == 0
    result = json.loads(
        (run_root / "qx_e248_local_parent_control" / "matrix_result.json").read_text()
    )
    assert result["initialization"] == "eval_only"
    assert result["training_executed"] is False
    assert result["checkpoint"] == str(parent)
    assert not (run_root / "qx_e248_local_parent_control" / "checkpoints").exists()


def test_local_preference_resume_reuses_only_matching_stage(tmp_path) -> None:
    import hashlib
    from types import SimpleNamespace

    from slm_training.harnesses.preference.local_decisions import DecisionEventV1

    exp = next(row for row in _v10_experiments(tmp_path) if row.eid == "E262")
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    trained = tmp_path / "trained.pt"
    trained.write_bytes(b"trained")
    for suffix in (".tokenizer.json", ".meta.json"):
        trained.with_suffix(suffix).write_text("{}")
    events = tmp_path / "events.jsonl"
    event = DecisionEventV1(
        version=1,
        event_id="event",
        trajectory_id="trajectory",
        group_id="group",
        split="train",
        seed=0,
        context_text="context",
        canvas_ids=(1, 3),
        position=1,
        good_token_ids=(2,),
        bad_token_ids=(4,),
        legal_token_ids=(2, 4),
        decision_kind="sym",
        evidence_kind="counterfactual",
        evidence_confidence=1.0,
        policy_checkpoint_sha="policy",
        tokenizer_sha="tokenizer",
        decode_config_hash="decode",
        source_suite=None,
    )
    events.write_text(json.dumps(event.to_dict()) + "\n")
    run_root = tmp_path / "runs"
    summary_path = run_root / exp.run_id / "local_preference/local_preference_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "objective": "ftpo_set",
                "steps": 30,
                "balanced": False,
                "reference_tethered": False,
                "source_checkpoint_sha": hashlib.sha256(parent.read_bytes()).hexdigest(),
                "train_events": 1,
                "held_out_events": 0,
                "checkpoint": str(trained),
            }
        )
    )
    args = SimpleNamespace(
        decision_events=events,
        run_root=run_root,
        resume=True,
        pref_steps=30,
    )

    checkpoint, summary = _maybe_local_preference(exp, parent, args)

    assert checkpoint == run_root / exp.run_id / "checkpoints/last.pt"
    assert summary["checkpoint"] == str(trained)
