"""G5 (SLM-37): meta-trace schema, harvest, retention, and replay proof."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.distill.meta_trace import (
    MetaTraceRecord,
    harvest_run_dir,
    load_corpus,
    record_live_decode,
    replay_trace,
    write_corpus,
)

PROGRAM = (
    'root = Stack([title, cta], "column")\n'
    'title = TextContent(":hero.title")\n'
    'cta = Button(":cta.label")'
)


def test_schema_is_strict_and_round_trips() -> None:
    record = MetaTraceRecord(
        run_id="r1", record_id="a", prompt="p", prediction="x = 1"
    )
    clone = MetaTraceRecord.model_validate_json(record.model_dump_json())
    assert clone == record
    with pytest.raises(Exception):
        MetaTraceRecord(run_id="r1", record_id="a", prompt="p", bogus_field=1)


def test_harvest_joins_eval_details_gates_and_trace_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trace.json").write_text(
        json.dumps({"trace_id": "t" * 32, "traceparent": "00-tp-01"}),
        encoding="utf-8",
    )
    (run_dir / "train_summary.json").write_text(
        json.dumps(
            {"run_id": "qx_demo", "model": "twotower", "recipe": {"seed": 3}}
        ),
        encoding="utf-8",
    )
    (run_dir / "matrix_result.json").write_text(
        json.dumps({"pass": False, "failures": ["smoke:meaningful_program_rate"]}),
        encoding="utf-8",
    )
    (run_dir / "eval_smoke.json").write_text(
        json.dumps(
            {
                "details": [
                    {
                        "id": "rec1",
                        "prediction": PROGRAM,
                        "parse_ok": True,
                        "structural_similarity": 0.4,
                        "reward_score": 0.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    records = harvest_run_dir(run_dir)
    assert len(records) == 1
    record = records[0]
    assert record.run_id == "qx_demo"
    assert record.trace_id == "t" * 32
    assert record.seed == 3
    assert record.verdicts["suite"] == "smoke"
    assert record.verdicts["run_gates"]["pass"] is False
    assert record.verdicts["parse_ok"] is True
    assert record.prediction == PROGRAM


def test_corpus_retention_is_append_only_with_checksums(tmp_path: Path) -> None:
    first = [
        MetaTraceRecord(run_id="r1", record_id="a", prompt="p", prediction="1")
    ]
    second = [
        MetaTraceRecord(run_id="r1", record_id="b", prompt="q", prediction="2")
    ]
    write_corpus(first, tmp_path, "g5_unit")
    manifest = write_corpus(second, tmp_path, "g5_unit")
    assert manifest["n_records"] == 2
    assert len(manifest["line_sha256"]) == 2
    loaded = load_corpus(tmp_path, "g5_unit")
    assert [r.record_id for r in loaded] == ["a", "b"]
    # sync_campaign compatibility: the campaign.json gate exists.
    assert (tmp_path / "g5_unit" / "campaign.json").exists()


def test_replay_reproduces_deterministic_decode(tmp_path: Path) -> None:
    """The replay proof: a stored trace re-decodes to the identical output
    for the deterministic tree-edit decoder, and fails closed on checkpoint
    mismatch or non-deterministic model kinds."""
    torch = pytest.importorskip("torch")  # noqa: F841

    from slm_training.dsl.schema import ExampleRecord
    from slm_training.harnesses.model_build.plugin import GenerationRequest
    from slm_training.models.tree_edit_diffusion import (
        TreeEditDiffusionConfig,
        TreeEditDiffusionModel,
    )

    records = [
        ExampleRecord(
            id="a",
            prompt="Hero card with a title and a CTA.",
            openui=PROGRAM,
            placeholders=[":hero.title", ":cta.label"],
        )
    ]
    cfg = TreeEditDiffusionConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        beam_width=2,
        expand_per_state=2,
        max_search_steps=3,
        seed=11,
    )
    model = TreeEditDiffusionModel.from_records(records, config=cfg, device="cpu")
    checkpoint = tmp_path / "ckpt.pt"
    model.save(checkpoint)
    slot_contract = (":hero.title", ":cta.label")
    prediction = model.generate_batch_requests(
        [GenerationRequest(prompt=records[0].prompt, slot_contract=slot_contract)]
    )[0]
    trace = record_live_decode(
        run_id="g5_fixture",
        record_id="a",
        dsl_id="openui",
        prompt=records[0].prompt,
        slot_contract=slot_contract,
        model_kind="tree_edit_diffusion",
        checkpoint_path=checkpoint,
        prediction=prediction,
        deterministic_decode=True,
    )
    assert replay_trace(trace, checkpoint) == prediction

    # Fail closed: tampered checkpoint sha.
    tampered = trace.model_copy(update={"checkpoint_sha": "0" * 64})
    with pytest.raises(ValueError, match="sha mismatch"):
        replay_trace(tampered, checkpoint)
    # Fail closed: non-deterministic model kinds refuse bit-exact replay.
    maskgit = trace.model_copy(update={"model_kind": "twotower"})
    with pytest.raises(ValueError, match="deterministic"):
        replay_trace(maskgit, checkpoint)
