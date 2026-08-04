"""Property tests for the transparent restart-diffusion operator."""

from __future__ import annotations

import numpy as np

from slm_training.data.progspec.semantic_evidence import (
    FactorRepresentation,
    build_factor_representation,
    project_program_factors,
)
from slm_training.models.semantic_factor_propagation import (
    PropagationConfig,
    build_incidence_matrix,
    column_stochastic_S,
    propagate_restart,
)


def test_S_column_stochastic_and_mass_preserved() -> None:
    snap = project_program_factors(
        program_id="prop",
        components=(("v1", "Card"), ("v2", "Button"), ("v3", "Input")),
        binder_edges=(("v1", "v2", "USE"), ("v1", "v3", "USE")),
        legal_candidate_ids=("v2", "v3"),
    )
    view = build_factor_representation(snap, FactorRepresentation.FACTOR_NODE)
    assert view is not None
    B = build_incidence_matrix(view)
    S, column_sums, _notes = column_stochastic_S(B)
    assert np.all(S >= -1e-12)
    assert np.allclose(column_sums, 1.0, atol=1e-9)
    c = np.ones(S.shape[0]) / S.shape[0]
    Sc = S @ c
    assert abs(float(Sc.sum()) - 1.0) < 1e-9


def test_restart_converges_and_respects_contraction_bound() -> None:
    # Two vertices, one factor connecting both.
    B = np.array([[1.0], [1.0]], dtype=np.float64)
    r = np.array([1.0, 0.0], dtype=np.float64)
    cfg = PropagationConfig(lambda_restart=0.2, max_steps=50, tol=1e-12)
    result = propagate_restart(B, r, config=cfg)
    assert result.converged
    assert abs(float(result.confidence.sum()) - 1.0) < 1e-8
    # Empirical contraction: one step distance shrinks by ≤ (1-λ)
    S, _, _ = column_stochastic_S(B)
    x = np.array([0.9, 0.1])
    y = np.array([0.1, 0.9])
    lam = cfg.lambda_restart
    Tx = lam * (r / r.sum()) + (1 - lam) * (S @ x)
    Ty = lam * (r / r.sum()) + (1 - lam) * (S @ y)
    assert float(np.abs(Tx - Ty).sum()) <= (1 - lam) * float(np.abs(x - y).sum()) + 1e-9


def test_manual_vertex_factor_vertex_transition() -> None:
    # One hyperedge {a,b}: vertex→factor→vertex equals uniform over members.
    B = np.array([[1.0], [1.0]], dtype=np.float64)
    S, _, _ = column_stochastic_S(B)
    # From a: De=2, Dv=1 each → S should map e0 mass to both vertices equally.
    e0 = np.array([1.0, 0.0])
    out = S @ e0
    assert np.allclose(out, np.array([0.5, 0.5]), atol=1e-9)
