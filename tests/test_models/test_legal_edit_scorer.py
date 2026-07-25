from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from slm_training.data.flow.bridge_corpus import (
    RequestEditContractV1,
    load_corpus,
)
from slm_training.flow.termination import FixedKPolicy
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_scorer import (
    DirectLegalEditPolicy,
    LegalEditScorer,
    LegalEditScorerConfig,
    multi_positive_set_loss,
)

CORPUS = Path("src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture")


def _batch() -> LegalEditBatch:
    rows, candidate_sets, _ = load_corpus(CORPUS)
    return LegalEditBatch.pack(rows, candidate_sets)


def test_set_mass_loss_and_unknown_telemetry() -> None:
    batch = _batch()
    scorer = LegalEditScorer()
    logits = scorer(batch)
    loss, metrics = multi_positive_set_loss(logits, batch)
    assert loss.item() >= 0.0
    assert 0.0 < metrics["positive_mass"] <= 1.0
    assert metrics["unknown_mass"] > 0.0
    assert torch.isfinite(loss)


def test_no_time_ignores_gold_progress_and_parameter_parity() -> None:
    batch = _batch()
    changed = replace(
        batch,
        state_features=batch.state_features.clone(),
    )
    changed.state_features[:, 2:] = 1.0
    configs = [
        LegalEditScorerConfig(time_encoding=mode, seed=7)
        for mode in ("no_time", "linear", "fourier")
    ]
    scorers = [LegalEditScorer(config) for config in configs]
    assert len({sum(p.numel() for p in scorer.parameters()) for scorer in scorers}) == 1
    no_time = scorers[0]
    assert torch.equal(no_time(batch), no_time(changed))
    progress = torch.linspace(0.0, 1.0, len(batch.row_ids))
    assert not torch.equal(
        scorers[1](batch, schedule_progress=progress),
        scorers[2](batch, schedule_progress=progress),
    )


def test_unknown_is_never_an_explicit_negative() -> None:
    batch = _batch()
    assert not bool((batch.unknown_mask & batch.unsupported_mask).any())
    assert not bool((batch.positive_mask & batch.unknown_mask).any())


def test_direct_policy_checkpoint_round_trip_and_migration(tmp_path: Path) -> None:
    policy = DirectLegalEditPolicy(
        LegalEditScorer(LegalEditScorerConfig(time_encoding="linear", seed=3))
    )
    checkpoint = tmp_path / "policy.pt"
    policy.save(checkpoint, metadata={"dataset": "slm196-fixture"})
    loaded = DirectLegalEditPolicy.from_checkpoint(checkpoint)
    batch = _batch()
    progress = torch.zeros(len(batch.row_ids))
    assert torch.equal(
        policy.scorer(batch, schedule_progress=progress),
        loaded.scorer(batch, schedule_progress=progress),
    )
    payload = torch.load(checkpoint, weights_only=False)
    payload["config"].pop("time_encoding")
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    migrated = DirectLegalEditPolicy.from_checkpoint(legacy)
    assert migrated.scorer.config.time_encoding == "no_time"


def test_exact_decode_reenumerates_and_replays() -> None:
    record = json.loads(
        Path("tests/fixtures/slm196_legal_edit_bridge/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    policy = DirectLegalEditPolicy(LegalEditScorer())
    trace = policy.decode_exact(
        record["source_program"],
        RequestEditContractV1.from_dict(record["request_contract"]),
        termination=FixedKPolicy(k=1, max_steps=2),
        max_steps=2,
    )
    assert len(trace.decisions) == 1
    assert trace.stop_reason == "FIXED_K_END"
    assert trace.decisions[0]["selected_candidate_id"] in trace.decisions[0]["candidate_ids"]
    assert trace.final_fingerprint == trace.decisions[0]["successor_fingerprint"]
