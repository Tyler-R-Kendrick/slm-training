from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.data.flow.bridge_corpus import (
    ExactLegalEditCandidateSetV1,
    LegalEditBridgeRowV1,
    RequestEditContractV1,
    build_bridge_rows,
    canonical_json,
    enumerate_live_candidates,
    load_corpus,
    stable_candidate_id,
    validate_rows,
)
from slm_training.harnesses.experiments.slm188_edit_algebra import CanonicalEdit
from slm_training.data.store import DataStore

FIXTURE = Path("tests/fixtures/slm196_legal_edit_bridge")
PUBLISHED = Path(
    "src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture"
)


def _record(index: int = 0) -> dict:
    return json.loads(
        FIXTURE.joinpath("records.jsonl").read_text(encoding="utf-8").splitlines()[
            index
        ]
    )


def test_candidate_id_ignores_ephemeral_edit_metadata() -> None:
    left = CanonicalEdit(
        edit_id="planner-7",
        action="BindSlotPointer",
        target_name="title",
        slot=":title",
        affected_node_ids=("title",),
        cost={"edits": 99},
    )
    right = CanonicalEdit(
        edit_id="other",
        action="BindSlotPointer",
        target_name="title",
        slot=":title",
    )
    assert stable_candidate_id(left) == stable_candidate_id(right)


def test_exact_candidate_set_is_canonical_and_permutation_invariant() -> None:
    record = _record()
    candidate_set = enumerate_live_candidates(
        record["source_program"],
        RequestEditContractV1.from_dict(record["request_contract"]),
    )
    reversed_set = ExactLegalEditCandidateSetV1(
        state_fingerprint=candidate_set.state_fingerprint,
        candidates=tuple(reversed(candidate_set.candidates)),
    )
    assert reversed_set.candidate_set_digest == candidate_set.candidate_set_digest
    assert [item.candidate_id for item in reversed_set.candidates] == sorted(
        item.candidate_id for item in candidate_set.candidates
    )


def test_bridge_rows_replay_and_keep_unknown_out_of_negatives() -> None:
    rows, candidate_sets = build_bridge_rows(
        _record(), version_pins={"planner": "fixture"}
    )
    result = validate_rows(rows, candidate_sets)
    assert result["replay_rate"] == 1.0
    assert result["candidate_reconstruction_rate"] == 1.0
    assert all(row.positive_candidate_ids for row in rows)
    assert any(len(row.positive_candidate_ids) > 1 for row in rows)
    assert all(
        not set(row.unknown_candidate_ids) & set(row.unsupported_candidate_ids)
        for row in rows
    )
    for row in rows:
        model_input = canonical_json(
            row.model_input(candidate_sets[row.candidate_set_digest])
        )
        assert "planner_selected" not in model_input
        assert row.focus == {"node_id": None, "source": "whole_state"}


def test_schema_rejects_unknown_fields_and_split_leakage() -> None:
    rows, candidate_sets = build_bridge_rows(
        _record(), version_pins={"planner": "fixture"}
    )
    payload = rows[0].to_dict()
    payload["future_witness"] = "forbidden"
    with pytest.raises(ValueError, match="unknown bridge row fields"):
        LegalEditBridgeRowV1.from_dict(payload)
    leaked = replace(rows[0], row_id="leaked", split="dev")
    with pytest.raises(ValueError, match="split-group"):
        validate_rows([rows[0], leaked], candidate_sets)


def test_model_input_recursively_rejects_forbidden_fields() -> None:
    rows, candidate_sets = build_bridge_rows(
        _record(), version_pins={"planner": "fixture"}
    )
    candidate_set = candidate_sets[rows[0].candidate_set_digest]
    poisoned = replace(
        rows[0],
        state_summary={"safe": {"confirmation_token": "leak"}},
    )
    with pytest.raises(ValueError, match="forbidden model-input"):
        poisoned.model_input(candidate_set)


def test_content_addressed_fixture_loads_and_tampering_fails(tmp_path: Path) -> None:
    rows, candidate_sets, manifest = load_corpus(PUBLISHED)
    assert manifest["schema"] == "LegalEditBridgeCorpusManifestV1"
    assert validate_rows(rows, candidate_sets)["rows"] == 4
    assert (
        DataStore(root=Path("."))
        .verify("train", "slm196_legal_edit_bridge_fixture")
        .storage
        == "git"
    )

    copied = tmp_path / "fixture"
    shutil.copytree(PUBLISHED, copied)
    candidate_path = next((copied / "candidate_sets").glob("*.json"))
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["state_fingerprint"] = "tampered"
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_corpus(copied)
