"""Operational timeouts never retire scientific hypotheses or erase identity."""

from pathlib import Path

import pytest

from slm_training.autoresearch.climb_policy import (
    load_climb_policy,
    load_loop_exhausted_ledger,
    loop_data_eval_identity,
    save_loop_exhausted_ledger,
)
from slm_training.autoresearch.hillclimb import (
    ExhaustedKnobEntry,
    ExhaustedKnobLedger,
    HillClimbError,
    assert_matrix_knobs_not_exhausted,
    knob_signature_sha256,
)

_GENERATION = {"unique_root_target": 8, "data_only": True}


def _identity(generation):
    return loop_data_eval_identity(
        load_climb_policy(), train_version="wf_smoke_v2", eval_version="e_test",
        primary_metric="smoke.structural_similarity", direction="increase",
        claim_class="diagnostic", extra={"data_generation": generation},
    )


@pytest.mark.parametrize("reason", [
    "reproduced_decode_timeout_retirement", "primary_lcb_within_noise",
])
def test_retirement_preserves_generation_and_rejects_timeout_authority(tmp_path: Path, reason: str):
    identity, other = _identity(_GENERATION), _identity(None)
    assert identity != other
    signature = knob_signature_sha256({"binder_arity_loss_weight": 1.0})
    entry = ExhaustedKnobEntry(
        knob_signature_sha256=signature, data_eval_identity=identity,
        claim_class="diagnostic", reason=reason,
    )
    ledger = ExhaustedKnobLedger(entries=[entry])
    path = save_loop_exhausted_ledger(ledger, tmp_path, "loop-1")
    original = path.read_bytes()
    restored = load_loop_exhausted_ledger(tmp_path, "loop-1")
    kwargs = dict(knob_signatures=[signature], ledger=restored, claim_class="diagnostic")
    if reason == "primary_lcb_within_noise":
        with pytest.raises(HillClimbError, match="exhausted"):
            assert_matrix_knobs_not_exhausted(data_eval_identity=identity, **kwargs)
    else:
        assert_matrix_knobs_not_exhausted(data_eval_identity=identity, **kwargs)
    assert_matrix_knobs_not_exhausted(data_eval_identity=other, **kwargs)
    assert restored.entries == [entry]
    assert path.read_bytes() == original


def test_operational_timeout_cannot_be_recorded_as_scientific_null():
    ledger = ExhaustedKnobLedger()
    with pytest.raises(ValueError, match="operational timeout is not scientific null"):
        ledger.record_null(
            knob_signature_sha256="a" * 64, data_eval_identity=_identity(_GENERATION),
            claim_class="diagnostic", reason="reproduced_decode_timeout_retirement",
        )
    assert not ledger.entries
