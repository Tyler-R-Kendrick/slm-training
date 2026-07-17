"""A2 (SLM-38): single-step ASAp distribution-aware constrained-decode primitives.

Covers the pure tensor/set primitives that back the MaskGIT ASAp correction:
``AsapLedger``, ``removed_mass``, ``asap_reweight`` (in
``dsl/grammar/fastpath/gate.py``) and the ``asap_filter_commits`` commit gate
(in ``models/parallel_decode.py``). See
docs/design/iter-a2-asap-constrained-decode-20260717.md and Grammar-Aligned
Decoding / ASAp (Park et al., NeurIPS 2024, arXiv:2405.21047).
"""

from __future__ import annotations

import torch

from slm_training.dsl.grammar.fastpath.gate import (
    AsapLedger,
    asap_reweight,
    removed_mass,
)
from slm_training.models.parallel_decode import asap_filter_commits


def _probs(weights: list[float]) -> torch.Tensor:
    t = torch.tensor(weights, dtype=torch.float32)
    return t / t.sum()


def test_removed_mass_records_masked_out_probability() -> None:
    # Legal = {0, 1}; illegal = {2, 3} carry 0.3 + 0.1 = 0.4 removed mass.
    p = _probs([0.4, 0.2, 0.3, 0.1])
    m = removed_mass(p, [0, 1])
    assert abs(m - 0.4) < 1e-6
    # No tokens removed → zero distortion.
    assert abs(removed_mass(p, [0, 1, 2, 3])) < 1e-6


def test_removed_mass_fails_closed_on_empty_legal_set() -> None:
    p = _probs([0.5, 0.5])
    assert removed_mass(p, []) == 1.0
    # Out-of-range ids are dropped → still empty → fail closed.
    assert removed_mass(p, [7, 9]) == 1.0


def test_asap_reweight_is_valid_distribution_and_differs_from_plain() -> None:
    p = _probs([0.5, 0.25, 0.2, 0.05])  # legal {0,1}, M = 0.25
    legal, q = asap_reweight(p, [0, 1], alpha=1.0)
    assert legal == [0, 1]
    # Valid probability distribution.
    assert torch.all(q >= 0)
    assert abs(float(q.sum().item()) - 1.0) < 1e-6
    # Deterministically differs from the plain hard-mask renormalization.
    plain = p[[0, 1]] / p[[0, 1]].sum()
    assert not torch.allclose(q, plain, atol=1e-4)
    # Order-preserving: the winning legal token never changes.
    assert int(q.argmax().item()) == int(plain.argmax().item())


def test_asap_reweight_alpha_zero_equals_plain_renormalization() -> None:
    p = _probs([0.5, 0.25, 0.2, 0.05])
    _legal, q = asap_reweight(p, [0, 1], alpha=0.0)
    plain = p[[0, 1]] / p[[0, 1]].sum()
    assert torch.allclose(q, plain, atol=1e-6)


def test_asap_reweight_no_removed_mass_is_identity() -> None:
    p = _probs([0.6, 0.4])
    _legal, q = asap_reweight(p, [0, 1], alpha=1.0)  # M = 0 → gamma = 1
    assert torch.allclose(q, p, atol=1e-6)


def test_asap_reweight_empty_legal_set_returns_empty() -> None:
    p = _probs([0.5, 0.5])
    legal, q = asap_reweight(p, [], alpha=1.0)
    assert legal == []
    assert q.numel() == 0


def test_asap_ledger_aggregates_removed_mass() -> None:
    ledger = AsapLedger()
    for m in (0.0, 0.4, 0.6):
        ledger.record(m)
    assert ledger.positions == 3
    assert ledger.nonzero_removed == 2
    assert abs(ledger.removed_mass_sum - 1.0) < 1e-6
    assert abs(ledger.max_removed_mass - 0.6) < 1e-6
    assert abs(ledger.mean_removed_mass - (1.0 / 3.0)) < 1e-6
    # Out-of-range values clamp instead of poisoning the ledger.
    ledger.record(1.5)
    ledger.record(-0.3)
    assert ledger.max_removed_mass == 1.0
    assert ledger.positions == 5


def test_asap_filter_commits_defers_high_distortion_positions() -> None:
    # length = 4, batch 0. Position 1 keeps almost all mass legal (low removed);
    # position 2 has almost all mass removed (high distortion → defer).
    length = 4
    probs = torch.zeros(1, length, 5)
    probs[0, 1] = _probs([0.9, 0.05, 0.03, 0.02, 0.0])  # legal {0} → M ~= 0.1
    probs[0, 2] = _probs([0.1, 0.1, 0.1, 0.1, 0.6])  # legal {0} → M = 0.9

    def legal_fn(t: int):
        return {0}

    ledger = AsapLedger()
    kept = asap_filter_commits(
        [1, 2],
        probs,
        length=length,
        legal_ids_fn=legal_fn,
        ledger=ledger,
        alpha=1.0,
        defer_mass=0.5,
        last_step=False,
    )
    assert kept == [1]  # low-distortion committed, high-distortion deferred
    assert ledger.positions == 2
    assert ledger.nonzero_removed == 2


def test_asap_filter_commits_last_step_keeps_all() -> None:
    length = 4
    probs = torch.zeros(1, length, 5)
    probs[0, 1] = _probs([0.1, 0.1, 0.1, 0.1, 0.6])  # high removed mass

    ledger = AsapLedger()
    kept = asap_filter_commits(
        [1],
        probs,
        length=length,
        legal_ids_fn=lambda t: {0},
        ledger=ledger,
        alpha=1.0,
        defer_mass=0.5,
        last_step=True,
    )
    assert kept == [1]  # final step must terminate — nothing deferred


def test_asap_filter_commits_progress_guarantee_keeps_lowest() -> None:
    # Every candidate is high-distortion; keep the single lowest so decode
    # always makes progress (no dead step).
    length = 4
    probs = torch.zeros(1, length, 5)
    probs[0, 1] = _probs([0.2, 0.0, 0.0, 0.0, 0.8])  # M = 0.8
    probs[0, 2] = _probs([0.4, 0.0, 0.0, 0.0, 0.6])  # M = 0.6 (lowest)

    ledger = AsapLedger()
    kept = asap_filter_commits(
        [1, 2],
        probs,
        length=length,
        legal_ids_fn=lambda t: {0},
        ledger=ledger,
        alpha=1.0,
        defer_mass=0.5,
        last_step=False,
    )
    assert kept == [2]


def test_asap_filter_commits_unknown_legality_passes_through() -> None:
    # legal_ids_fn returning None (broad/unknown terminal set) must not be
    # recorded and must not be deferred.
    length = 4
    probs = torch.zeros(1, length, 5)
    probs[0, 1] = _probs([0.2, 0.2, 0.2, 0.2, 0.2])

    ledger = AsapLedger()
    kept = asap_filter_commits(
        [1],
        probs,
        length=length,
        legal_ids_fn=lambda t: None,
        ledger=ledger,
        alpha=1.0,
        defer_mass=0.5,
        last_step=False,
    )
    assert kept == [1]
    assert ledger.positions == 0


def test_asap_filter_commits_empty_is_noop() -> None:
    ledger = AsapLedger()
    assert asap_filter_commits(
        [],
        torch.zeros(1, 4, 5),
        length=4,
        legal_ids_fn=lambda t: {0},
        ledger=ledger,
        alpha=1.0,
        defer_mass=0.5,
    ) == []
    assert ledger.positions == 0
