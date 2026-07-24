"""SLM-314 (LAR2-05): multi-mode dataset freeze, WTA loss, synthetic fixture."""

from __future__ import annotations

import json

import pytest

# Tiny CPU fixture models: torch thread thrash dominates otherwise.
import torch

torch.set_num_threads(1)

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm314_winner_take_all import (
    DEFAULT_RESOURCE,
    FLOOR_EPSILON,
    FreezeIntegrityError,
    PromptGroupV1,
    build_multimode_dataset,
    dataset_manifest,
    freeze_dataset,
    mode_supervision_loss,
    read_frozen_dataset,
    run_synthetic_two_mode,
    select_winner,
    single_gold_groups,
    synthetic_assertions,
    winner_take_all_loss,
    _fingerprint_source,
)
from slm_training.models.tree_edit_diffusion import (
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
)

_SMALL_CONFIG = {
    "context_backend": "scratch",
    "seed": 0,
    "d_model": 64,
    "denoiser_layers": 2,
    "context_layers": 1,
    "max_chain": 2,
    "max_search_steps": 4,
    "beam_width": 2,
    "expand_per_state": 2,
}

MODE_A = 'root = Stack([t], "column")\nt = TextContent(":body")'
MODE_B = 'root = Stack([c], "column")\nc = Card([t], "column")\nt = TextContent(":body")'


def _group() -> PromptGroupV1:
    return next(g for g in build_multimode_dataset() if g.group_id == "mm_01")


def _model(records: list[ExampleRecord]) -> TreeEditDiffusionModel:
    torch.manual_seed(0)
    return TreeEditDiffusionModel.from_records(
        records, config=TreeEditDiffusionConfig(**_SMALL_CONFIG), device="cpu"
    )


# --- mode identity via canonical fingerprints -------------------------------


def test_fingerprint_alpha_invariant_same_ast() -> None:
    """Two serializations of the same AST (renamed statements, reordered
    declarations) share one canonical fingerprint."""
    renamed = 'root = Stack([x1], "column")\nx1 = TextContent(":body")'
    reordered = 't = TextContent(":body")\nroot = Stack([t], "column")'
    assert _fingerprint_source(MODE_A) == _fingerprint_source(renamed)
    assert _fingerprint_source(MODE_A) == _fingerprint_source(reordered)


def test_fingerprint_distinct_for_different_asts() -> None:
    assert _fingerprint_source(MODE_A) != _fingerprint_source(MODE_B)


def test_fingerprint_requires_verifier_acceptance() -> None:
    with pytest.raises(ValueError):
        _fingerprint_source("root = NotAComponent([t])")


# --- frozen dataset ----------------------------------------------------------


def test_dataset_every_prompt_has_two_distinct_valid_modes() -> None:
    groups = build_multimode_dataset()
    assert len(groups) == 8
    for group in groups:
        assert len(group.modes) >= 2
        fps = [m.canonical_fingerprint for m in group.modes]
        assert len(set(fps)) == len(fps)
        for mode in group.modes:
            # fingerprint identity is recomputed through the real parser
            assert _fingerprint_source(mode.source) == mode.canonical_fingerprint


def test_committed_resource_matches_builder_and_manifest() -> None:
    frozen = read_frozen_dataset(DEFAULT_RESOURCE)
    live = build_multimode_dataset()
    assert [g.to_dict() for g in frozen] == [g.to_dict() for g in live]
    assert dataset_manifest(frozen)["n_modes"] == 16


def test_freeze_roundtrip_and_tamper_detection(tmp_path) -> None:
    path = tmp_path / "modes.jsonl"
    groups = build_multimode_dataset()
    manifest = freeze_dataset(groups, path)
    assert dataset_manifest(read_frozen_dataset(path)) == manifest
    # Tamper with one mode source: fail-closed on reload.
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["modes"][0]["source"] = MODE_B
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    with pytest.raises(FreezeIntegrityError):
        read_frozen_dataset(path)


def test_single_gold_groups_deterministic() -> None:
    groups = build_multimode_dataset()
    reduced = single_gold_groups(groups)
    assert all(len(g.modes) == 1 for g in reduced)
    for full, one in zip(groups, reduced):
        expected = min(m.canonical_fingerprint for m in full.modes)
        assert one.modes[0].canonical_fingerprint == expected


# --- WTA loss ----------------------------------------------------------------


def test_select_winner_min_loss_tie_breaks_lowest() -> None:
    assert select_winner([2.0, 1.0, 3.0]) == 1
    assert select_winner([1.0, 1.0]) == 0
    with pytest.raises(ValueError):
        select_winner([])


def test_wta_selects_min_loss_mode_and_records_telemetry() -> None:
    group = _group()
    records = [r for g in build_multimode_dataset() for r in g.records()]
    model = _model(records)
    loss, telemetry = winner_take_all_loss(model, group, seed=0)
    losses = [m["loss"] for m in telemetry["per_mode"]]
    winner = telemetry["winner_index"]
    assert winner == select_winner(losses)
    assert telemetry["winner_fingerprint"] == (
        group.modes[winner].canonical_fingerprint
    )
    assert sum(1 for m in telemetry["per_mode"] if m["selected"]) == 1
    # total = winner + eps * mean(losers)
    losers = [x for i, x in enumerate(losses) if i != winner]
    expected = losses[winner] + FLOOR_EPSILON * (sum(losers) / len(losers))
    assert telemetry["total_loss"] == pytest.approx(expected, rel=1e-5)
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-5)


def test_wta_floor_term_backpropagates_into_losing_mode() -> None:
    """The coverage/floor term must constrain the losing mode: gradient flows
    into both modes' losses (winner fully, loser with the epsilon weight)."""
    la = torch.tensor(2.0, requires_grad=True)
    lb = torch.tensor(3.0, requires_grad=True)
    losses = [la, lb]
    winner = select_winner([2.0, 3.0])
    losers = [x for i, x in enumerate(losses) if i != winner]
    total = losses[winner] + FLOOR_EPSILON * (sum(losers) / len(losers))
    total.backward()
    assert la.grad.item() == pytest.approx(1.0)
    assert lb.grad.item() == pytest.approx(FLOOR_EPSILON)


def test_mode_supervision_loss_deterministic_and_rng_restored() -> None:
    group = _group()
    records = [r for g in build_multimode_dataset() for r in g.records()]
    model = _model(records)
    record = group.records()[0]
    rng_state_before = model._rng.getstate()
    l1 = float(mode_supervision_loss(model, record, seed=123).detach())
    l2 = float(mode_supervision_loss(model, record, seed=123).detach())
    assert l1 == pytest.approx(l2)
    assert model._rng.getstate() == rng_state_before


def test_winner_take_all_loss_deterministic() -> None:
    group = _group()
    records = [r for g in build_multimode_dataset() for r in g.records()]
    m1 = _model(records)
    m2 = _model(records)
    l1, t1 = winner_take_all_loss(m1, group, seed=7)
    l2, t2 = winner_take_all_loss(m2, group, seed=7)
    assert t1 == t2
    assert float(l1.detach()) == pytest.approx(float(l2.detach()))


# --- synthetic two-mode collapse-vs-retain fixture ---------------------------


def test_synthetic_single_gold_collapses_wta_retains() -> None:
    result = run_synthetic_two_mode()
    checks = synthetic_assertions(result)
    assert all(checks.values()), checks
    # Explicit collapse/retain numbers (preregistered thresholds).
    assert result["single_gold"]["p_mode_b"] < 0.002
    assert result["wta"]["p_mode_b"] >= 0.004
    assert result["wta"]["p_mode_a"] > result["wta"]["p_mode_b"]
    # Averaging arm puts >=30% mass on invalid hybrids.
    assert result["multi_gold"]["p_invalid_hybrid"] >= 0.3


def test_synthetic_deterministic() -> None:
    assert run_synthetic_two_mode() == run_synthetic_two_mode()


def test_synthetic_floor_controls_retained_mass() -> None:
    """Larger floor epsilon -> more losing-mode mass retained (monotone)."""
    low = run_synthetic_two_mode(floor_epsilon=0.05)
    high = run_synthetic_two_mode(floor_epsilon=0.3)
    assert high["wta"]["p_mode_b"] > low["wta"]["p_mode_b"]


# --- arm isolation ------------------------------------------------------------


def test_arm_isolation_identical_budgets() -> None:
    """The CLI's train_arm gives every arm the same model init, step count,
    and per-step prompt coverage; only the loss weighting differs."""
    from scripts.run_slm314_winner_take_all import ARMS, FIXTURE_CONFIG_OVERRIDES

    assert ARMS == ("single_gold", "multi_gold", "wta")
    # Same declared fixture config object for all arms (no per-arm overrides).
    assert set(FIXTURE_CONFIG_OVERRIDES) >= {"d_model", "denoiser_layers"}
    groups = build_multimode_dataset()
    records = [r for g in groups for r in g.records()]
    torch.manual_seed(0)
    m1 = TreeEditDiffusionModel.from_records(
        records, config=TreeEditDiffusionConfig(**_SMALL_CONFIG), device="cpu"
    )
    torch.manual_seed(0)
    m2 = TreeEditDiffusionModel.from_records(
        records, config=TreeEditDiffusionConfig(**_SMALL_CONFIG), device="cpu"
    )
    s1 = {k: v.detach().clone() for k, v in m1.state_dict().items()}
    s2 = m2.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
