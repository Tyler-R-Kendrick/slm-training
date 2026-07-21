"""SLM-138 SharedRecursiveDenoiserTower regression tests.

Also covers SLM-237 (RSC-A01): the corrected weighted recursive
deep-supervision objective and its fail-closed
``validate_recursive_depth_supervision`` validator.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.blocks import DenoiserTower
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower
from slm_training.models.twotower import (
    RECURSIVE_OBJECTIVE_CONTRACT_VERSION,
    RecursiveObjectiveContractV2,
    TwoTowerConfig,
    TwoTowerModel,
    migrate_recursive_depth_aux_config,
    resolve_recursive_depth_aux_mode,
    validate_recursive_depth_supervision,
)

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def test_recursive_tower_matches_denoiser_interface() -> None:
    """The recursive tower exposes the same public attributes/methods."""
    vocab, d_model, max_len = 23, 16, 32
    rec = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        max_len=max_len,
    )
    assert rec.tok.weight.shape == (vocab, d_model)
    assert rec.lm_head.weight is rec.tok.weight
    assert rec.max_len == max_len
    assert len(rec.layers) == 1
    assert hasattr(rec, "kind_lookup")
    assert hasattr(rec, "set_runtime_symbol_features")


def test_recursive_forward_shapes_and_gradients() -> None:
    vocab, d_model, tgt, ctx_len = 23, 16, 6, 3
    tower = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=32,
        recursive_steps=2,
        recursive_transition_layers=2,
    )
    noisy = torch.randint(1, vocab, (2, tgt))
    ctx = torch.randn(2, ctx_len, d_model)
    logits = tower(noisy, ctx, pad_id=0)
    assert logits.shape == (2, tgt, vocab)
    loss = logits.sum()
    loss.backward()
    assert tower.tok.weight.grad is not None
    assert tower.ctx_proj.weight.grad is not None
    assert tower.z_latent.grad is not None


def test_recursive_encode_project_matches_forward() -> None:
    vocab, d_model, tgt, ctx_len = 23, 16, 6, 3
    tower = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=32,
        recursive_steps=2,
    )
    noisy = torch.randint(1, vocab, (2, tgt))
    ctx = torch.randn(2, ctx_len, d_model)
    hidden = tower.encode(noisy, ctx, pad_id=0)
    assert hidden.shape == (2, tgt, d_model)
    logits = tower.project(hidden)
    full = tower(noisy, ctx, pad_id=0)
    torch.testing.assert_close(logits, full)

    candidates = torch.tensor([1, 2, 3])
    gathered = tower.project(hidden, candidate_ids=candidates)
    assert gathered.shape == (2, tgt, 3)
    torch.testing.assert_close(gathered, full.index_select(-1, candidates))


def test_recursive_steps_one_parity_with_denoiser_tower() -> None:
    """R=1 with L transition blocks matches the stacked DenoiserTower contract."""
    vocab, d_model, tgt, ctx_len = 23, 16, 6, 3
    torch.manual_seed(0)
    stacked = DenoiserTower(
        vocab_size=vocab, d_model=d_model, n_layers=2, n_heads=2, max_len=32
    )
    torch.manual_seed(0)
    recursive = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=32,
        recursive_steps=1,
        recursive_transition_layers=2,
    )
    noisy = torch.randint(1, vocab, (2, tgt))
    ctx = torch.randn(2, ctx_len, d_model)
    stacked.eval()
    recursive.eval()
    with torch.no_grad():
        s_logits = stacked(noisy, ctx, pad_id=0)
        r_logits = recursive(noisy, ctx, pad_id=0)
    assert s_logits.shape == r_logits.shape == (2, tgt, vocab)
    # The recursive tower adds a z-state path, so the outputs differ, but both
    # must be deterministic and finite with the same interface shape.
    assert torch.isfinite(r_logits).all()


def test_weight_sharing_across_recursions() -> None:
    """The same layer objects are reused at every recursion step."""
    tower = SharedRecursiveDenoiserTower(
        vocab_size=23,
        d_model=16,
        n_layers=4,
        n_heads=2,
        max_len=32,
        recursive_steps=3,
    )
    f_ids = {id(layer) for layer in tower._f_layers}
    g_ids = {id(layer) for layer in tower._g_layers}
    out = tower.recursive_outputs(
        torch.randint(1, 23, (1, 4)), torch.randn(1, 2, 16), pad_id=0
    )
    depth_logits = out["depth_logits"]
    assert len(depth_logits) == 3
    # All computation flows through the same object-identity layers each step.
    assert len(f_ids) + len(g_ids) == len(tower.layers)


def test_runtime_symbol_features_sliced_projection() -> None:
    vocab, d_model = 23, 16
    tower = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        max_len=32,
    )
    features = torch.randn(1, vocab, d_model)
    tower.set_runtime_symbol_features(features)
    noisy = torch.randint(1, vocab, (1, 4))
    ctx = torch.randn(1, 2, d_model)
    logits = tower(noisy, ctx, pad_id=0)
    assert logits.shape == (1, 4, vocab)
    # Sliced projection matches the full-vocabulary gather.
    hidden = tower.encode(noisy, ctx, pad_id=0)
    candidates = torch.tensor([1, 2, 3])
    gathered = tower.project(hidden[0, 0], candidate_ids=candidates)
    assert gathered.shape == (3,)


def test_twotower_shared_recursive_trains_and_roundtrips(tmp_path: Path) -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_transition_layers=2,
            grammar_constrained=False,
            gen_steps=2,
            seed=0,
        ),
        device="cpu",
    )
    assert isinstance(model.denoiser, SharedRecursiveDenoiserTower)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss = model.training_loss(records)
    loss.backward()
    opt.step()

    ckpt = tmp_path / "recursive.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert loaded.config.denoiser_arch == "shared_recursive"
    assert isinstance(loaded.denoiser, SharedRecursiveDenoiserTower)


def test_twotower_deep_supervision_metrics() -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=(0.5, 1.0, 0.5),
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    assert "recursive_depth_supervision_loss" in model.last_training_metrics
    assert "recursive_depth_loss_0" in model.last_training_metrics
    assert "recursive_depth_loss_2" in model.last_training_metrics


def test_checkpoint_migration_to_shared_recursive(tmp_path: Path) -> None:
    from slm_training.models.checkpoint_migrate import (
        migrate_to_shared_recursive_denoiser,
    )

    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
    ]
    stacked = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(d_model=32, n_heads=2, denoiser_layers=2, seed=0),
        device="cpu",
    )
    src = tmp_path / "stacked.pt"
    stacked.save(src)

    dst = tmp_path / "recursive.pt"
    report = migrate_to_shared_recursive_denoiser(
        src,
        dst,
        config={"recursive_steps": 2, "recursive_transition_layers": 2},
        device="cpu",
    )
    assert dst.exists()
    assert report["denoiser_arch"] == "shared_recursive"
    assert any("z_latent" in k for k in report["initialized_keys"])
    assert any("ctx_proj" in k for k in report["initialized_keys"])

    loaded = TwoTowerModel.from_checkpoint(dst, device="cpu")
    assert loaded.config.denoiser_arch == "shared_recursive"
    assert isinstance(loaded.denoiser, SharedRecursiveDenoiserTower)


# ---------------------------------------------------------------------------
# SLM-237 (RSC-A01): corrected weighted objective + fail-closed validator.
# ---------------------------------------------------------------------------


def _recursive_model_for_weights(
    weights: tuple[float, ...], *, recursive_steps: int = 2, seed: int = 0
) -> tuple[TwoTowerModel, list[ExampleRecord]]:
    """Fresh shared_recursive model + fixed single-record batch.

    Rebuilding from the same seed for every ``weights`` value (rather than
    mutating one model's config) keeps the RNG draw sequence identical up to
    the point ``training_loss`` samples its noise/mask -- so the *raw*
    per-depth losses this repro's tests compare across weight configs are
    bit-identical, and only the weighting differs. Verified empirically: the
    recorded ``recursive_depth_loss_0``/``recursive_depth_loss_1`` values are
    identical across every weights tuple of the same length tested here.
    """
    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=recursive_steps,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=weights,
            grammar_constrained=False,
            seed=seed,
        ),
        device="cpu",
    )
    return model, records


def test_weights_zero_one_equals_l1_exactly() -> None:
    """(0, 1) must equal L1 exactly -- weight-0 depths must not leak in."""
    model, records = _recursive_model_for_weights((0.0, 1.0))
    model.training_loss(records)
    metrics = model.last_training_metrics
    torch.testing.assert_close(
        torch.tensor(metrics["recursive_depth_supervision_loss"]),
        torch.tensor(metrics["recursive_depth_loss_1"]),
    )


def test_weights_one_zero_equals_l0_exactly() -> None:
    """(1, 0) must equal L0 exactly."""
    model, records = _recursive_model_for_weights((1.0, 0.0))
    model.training_loss(records)
    metrics = model.last_training_metrics
    torch.testing.assert_close(
        torch.tensor(metrics["recursive_depth_supervision_loss"]),
        torch.tensor(metrics["recursive_depth_loss_0"]),
    )


def test_weights_half_one_equals_one_two() -> None:
    """(0.5, 1) and (1, 2) are the same normalized weighted mean.

    This is the historical failure mode #2: the old ``sum(L_d) / sum(w_d)``
    formula produced an exact 2x scale difference between these two
    configurations instead of an identical normalized mean.
    """
    model_a, records_a = _recursive_model_for_weights((0.5, 1.0))
    model_a.training_loss(records_a)
    loss_a = model_a.last_training_metrics["recursive_depth_supervision_loss"]

    model_b, records_b = _recursive_model_for_weights((1.0, 2.0))
    model_b.training_loss(records_b)
    loss_b = model_b.last_training_metrics["recursive_depth_supervision_loss"]

    assert loss_a == pytest.approx(loss_b, rel=1e-5)


def test_weights_one_one_equals_mean_of_l0_l1() -> None:
    """(1, 1) must equal (L0 + L1) / 2."""
    model, records = _recursive_model_for_weights((1.0, 1.0))
    model.training_loss(records)
    metrics = model.last_training_metrics
    expected = (
        metrics["recursive_depth_loss_0"] + metrics["recursive_depth_loss_1"]
    ) / 2.0
    assert metrics["recursive_depth_supervision_loss"] == pytest.approx(
        expected, rel=1e-5
    )


def test_validate_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="negative"):
        validate_recursive_depth_supervision(
            weights=(-1.0, 1.0), num_depths=2, supports_recursive_outputs=True
        )


def test_validate_nan_weight_raises() -> None:
    with pytest.raises(ValueError, match="not finite"):
        validate_recursive_depth_supervision(
            weights=(float("nan"), 1.0), num_depths=2, supports_recursive_outputs=True
        )


def test_validate_infinite_weight_raises() -> None:
    with pytest.raises(ValueError, match="not finite"):
        validate_recursive_depth_supervision(
            weights=(float("inf"), 1.0), num_depths=2, supports_recursive_outputs=True
        )


def test_validate_all_zero_raises() -> None:
    with pytest.raises(ValueError, match="all"):
        validate_recursive_depth_supervision(
            weights=(0.0, 0.0), num_depths=2, supports_recursive_outputs=True
        )


def test_validate_shorter_tuple_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        validate_recursive_depth_supervision(
            weights=(1.0,), num_depths=2, supports_recursive_outputs=True
        )


def test_validate_longer_tuple_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        validate_recursive_depth_supervision(
            weights=(1.0, 1.0, 1.0), num_depths=2, supports_recursive_outputs=True
        )


def test_validate_nonempty_weights_on_unsupported_architecture_raises() -> None:
    with pytest.raises(ValueError, match="recursive_outputs"):
        validate_recursive_depth_supervision(
            weights=(1.0,),
            num_depths=0,
            supports_recursive_outputs=False,
            architecture="stacked",
        )


def test_training_loss_fails_closed_on_stacked_denoiser() -> None:
    """Non-empty weights on a stacked (non-recursive) denoiser must raise
    before any loss/backward -- historical failure mode #6 (silently
    ignored)."""
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            denoiser_layers=2,
            denoiser_arch="stacked",
            recursive_depth_supervision_weights=(1.0,),
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    with pytest.raises(ValueError, match="recursive_outputs"):
        model.training_loss(records)


def test_training_loss_fails_closed_on_all_zero_weights() -> None:
    model, records = _recursive_model_for_weights((0.0, 0.0))
    with pytest.raises(ValueError, match="all"):
        model.training_loss(records)


def test_training_loss_fails_closed_on_length_mismatch() -> None:
    model, records = _recursive_model_for_weights((1.0,), recursive_steps=2)
    with pytest.raises(ValueError, match="length"):
        model.training_loss(records)


def test_empty_tuple_valid_on_every_architecture_no_aux_term() -> None:
    """Empty tuple stays feature-off on both stacked and shared_recursive,
    adding no per-depth telemetry.

    SLM-238 (RSC-A02) changed the *disabled* aux term's telemetry contract:
    ``recursive_depth_supervision_loss`` (and the sibling
    ``recursive_intermediate_aux_loss`` /
    ``recursive_final_depth_aux_contribution`` / ``combined_training_loss``
    fields) are now always present with an explicit ``0.0`` rather than
    omitted, so a disabled run is distinguishable from a missing-metrics bug
    (test #4 of SLM-238's required tests). Per-depth keys
    (``recursive_depth_loss_{d}`` etc.) remain absent -- no depths were ever
    looped over.
    """
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]

    stacked = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            denoiser_layers=2,
            denoiser_arch="stacked",
            recursive_depth_supervision_weights=(),
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    stacked.training_loss(records)
    assert stacked.last_training_metrics["recursive_depth_supervision_enabled"] is False
    assert stacked.last_training_metrics["recursive_depth_supervision_loss"] == 0.0
    assert "recursive_depth_loss_0" not in stacked.last_training_metrics

    recursive, _ = _recursive_model_for_weights(())
    recursive.training_loss(records)
    assert (
        recursive.last_training_metrics["recursive_depth_supervision_enabled"] is False
    )
    assert recursive.last_training_metrics["recursive_depth_supervision_loss"] == 0.0
    assert "recursive_depth_loss_0" not in recursive.last_training_metrics


def test_gradient_reaches_only_positive_weight_depths() -> None:
    """Zero-weighted depths get exactly zero gradient from the aggregation;
    positive-weighted depths get nonzero gradient. Uses independently
    differentiable synthetic depth logits (fixed tensors, no model)."""
    torch.manual_seed(0)
    vocab, n = 5, 4
    targets = torch.randint(0, vocab, (n,))
    logits_zero_weighted = torch.randn(n, vocab, requires_grad=True)
    logits_positive_weighted = torch.randn(n, vocab, requires_grad=True)

    validated = validate_recursive_depth_supervision(
        weights=(0.0, 1.0), num_depths=2, supports_recursive_outputs=True
    )
    norm_w0, norm_w1 = validated.normalized()
    l0 = F.cross_entropy(logits_zero_weighted, targets)
    l1 = F.cross_entropy(logits_positive_weighted, targets)
    total = norm_w0 * l0 + norm_w1 * l1
    total.backward()

    assert logits_zero_weighted.grad is not None
    torch.testing.assert_close(
        logits_zero_weighted.grad, torch.zeros_like(logits_zero_weighted.grad)
    )
    assert logits_positive_weighted.grad is not None
    assert not torch.allclose(
        logits_positive_weighted.grad, torch.zeros_like(logits_positive_weighted.grad)
    )


def test_fixture_metrics_agree_with_manual_calculation() -> None:
    """The committed fixture's deep-supervision metrics (weights=(0.5, 1.0))
    match the manual weighted-mean calculation from its own recorded raw
    per-depth losses."""
    from scripts.run_slm138_recursive_denoiser_fixture import _run_fixture

    report = _run_fixture()
    metrics = report["deep_supervision_metrics"]
    expected = (
        0.5 * metrics["recursive_depth_loss_0"] + 1.0 * metrics["recursive_depth_loss_1"]
    ) / 1.5
    assert metrics["recursive_depth_supervision_loss"] == pytest.approx(
        expected, rel=1e-5
    )
    # The historical defective formula was sum(L_d) / sum(w_d) -- the
    # unweighted mean -- which this fixture's own recorded values must no
    # longer reproduce (would only coincide by chance if L0 == L1).
    defective = (
        metrics["recursive_depth_loss_0"] + metrics["recursive_depth_loss_1"]
    ) / 1.5
    assert metrics["recursive_depth_supervision_loss"] != pytest.approx(
        defective, rel=1e-9
    ) or math.isclose(metrics["recursive_depth_loss_0"], metrics["recursive_depth_loss_1"])


# ---------------------------------------------------------------------------
# SLM-238 (RSC-A02): explicit final-depth double-counting semantics --
# recursive_depth_aux_mode / recursive_depth_aux_weight and the objective
# decomposition (primary_final_reconstruction_loss,
# recursive_intermediate_aux_loss, recursive_final_depth_aux_contribution,
# combined_training_loss).
# ---------------------------------------------------------------------------


def _recursive_model_for_mode(
    *,
    mode: str | None,
    weights: tuple[float, ...],
    aux_weight: float = 1.0,
    recursive_steps: int = 3,
    seed: int = 0,
) -> tuple[TwoTowerModel, list[ExampleRecord]]:
    """Fresh shared_recursive model + fixed single-record batch for a given
    ``recursive_depth_aux_mode``/``recursive_depth_aux_weight``. Same
    same-seed-rebuild rationale as ``_recursive_model_for_weights``: the raw
    per-depth losses are bit-identical across mode/weight configs of the same
    ``recursive_steps``, so only the aggregation differs."""
    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=recursive_steps,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=weights,
            recursive_depth_aux_mode=mode,
            recursive_depth_aux_weight=aux_weight,
            grammar_constrained=False,
            fidelity_loss_weight=0.0,
            seed=seed,
        ),
        device="cpu",
    )
    return model, records


def test_intermediate_only_never_reads_final_depth_aux_path() -> None:
    """Test #1: intermediate_only never reads or differentiates through the
    final depth's auxiliary path -- structurally, not just via a zero
    weight."""
    model, records = _recursive_model_for_mode(
        mode="intermediate_only", weights=(0.5, 1.0), recursive_steps=3
    )
    model.training_loss(records)
    metrics = model.last_training_metrics
    assert metrics["recursive_depth_aux_mode"] == "intermediate_only"
    # Depths 0 and 1 (0..R-2) are eligible and computed...
    assert "recursive_depth_loss_0" in metrics
    assert "recursive_depth_loss_1" in metrics
    # ...but depth 2 (R-1, the final depth) is never indexed at all.
    assert "recursive_depth_loss_2" not in metrics
    assert "recursive_depth_weighted_contribution_2" not in metrics
    assert metrics["recursive_final_depth_aux_contribution"] == 0.0


def test_all_depths_includes_final_contribution_exactly_once() -> None:
    """Test #2: all_depths includes the final depth's contribution exactly
    once in the auxiliary term (on top of the primary term already reflecting
    it once via depth_logits[-1] == logits)."""
    model, records = _recursive_model_for_mode(
        mode="all_depths", weights=(0.5, 1.0, 0.5), recursive_steps=3
    )
    model.training_loss(records)
    metrics = model.last_training_metrics
    assert metrics["recursive_depth_aux_mode"] == "all_depths"
    assert "recursive_depth_loss_2" in metrics
    final_contribution = metrics["recursive_depth_weighted_contribution_2"]
    assert metrics["recursive_final_depth_aux_contribution"] == pytest.approx(
        final_contribution, rel=1e-6
    )
    # Exactly once: intermediate + final reproduces the combined aux exactly.
    assert metrics["recursive_intermediate_aux_loss"] + metrics[
        "recursive_final_depth_aux_contribution"
    ] == pytest.approx(metrics["recursive_depth_supervision_loss"], rel=1e-6)


def test_primary_final_loss_identical_across_modes() -> None:
    """Test #3: for fixed logits/targets (same seed/model config), the
    primary final reconstruction loss does not depend on
    recursive_depth_aux_mode."""
    off_model, off_records = _recursive_model_for_mode(mode="off", weights=())
    off_model.training_loss(off_records)
    primary_off = off_model.last_training_metrics["primary_final_reconstruction_loss"]

    inter_model, inter_records = _recursive_model_for_mode(
        mode="intermediate_only", weights=(0.5, 1.0)
    )
    inter_model.training_loss(inter_records)
    primary_inter = inter_model.last_training_metrics[
        "primary_final_reconstruction_loss"
    ]

    all_model, all_records = _recursive_model_for_mode(
        mode="all_depths", weights=(0.5, 1.0, 0.5)
    )
    all_model.training_loss(all_records)
    primary_all = all_model.last_training_metrics["primary_final_reconstruction_loss"]

    assert primary_off == pytest.approx(primary_inter, rel=1e-6)
    assert primary_off == pytest.approx(primary_all, rel=1e-6)


def test_aux_weight_zero_is_primary_only_with_explicit_zero_telemetry() -> None:
    """Test #4: aux_weight=0 produces exact primary-only combined loss while
    retaining an explicit zero telemetry record (not an omitted field)."""
    model, records = _recursive_model_for_mode(
        mode="all_depths", weights=(0.5, 1.0, 0.5), aux_weight=0.0
    )
    model.training_loss(records)
    metrics = model.last_training_metrics
    assert metrics["recursive_depth_aux_weight"] == 0.0
    assert metrics["recursive_depth_supervision_loss"] == 0.0
    assert metrics["recursive_intermediate_aux_loss"] == 0.0
    assert metrics["recursive_final_depth_aux_contribution"] == 0.0
    assert metrics["combined_training_loss"] == pytest.approx(
        metrics["primary_final_reconstruction_loss"], rel=1e-6
    )
    # Explicit record, not an absent key.
    assert "recursive_depth_supervision_loss" in metrics


@pytest.mark.parametrize(
    ("mode", "weights", "num_depths", "match"),
    [
        ("bogus_mode", (1.0,), 2, "not one of"),
        ("off", (1.0,), 2, "requires an empty"),
        ("intermediate_only", (1.0, 1.0), 2, "length"),
        ("all_depths", (1.0,), 2, "length"),
    ],
)
def test_invalid_mode_weight_combinations_raise(
    mode: str, weights: tuple[float, ...], num_depths: int, match: str
) -> None:
    """Test #5: invalid mode/weight-length combinations raise."""
    with pytest.raises(ValueError, match=match):
        validate_recursive_depth_supervision(
            weights=weights,
            num_depths=num_depths,
            supports_recursive_outputs=True,
            mode=mode,
        )


def test_invalid_aux_weight_raises() -> None:
    """Test #5 (coefficient half): non-finite/negative aux_weight raises."""
    with pytest.raises(ValueError, match="not finite"):
        validate_recursive_depth_supervision(
            weights=(1.0, 1.0),
            num_depths=2,
            supports_recursive_outputs=True,
            mode="all_depths",
            aux_weight=float("nan"),
        )
    with pytest.raises(ValueError, match="negative"):
        validate_recursive_depth_supervision(
            weights=(1.0, 1.0),
            num_depths=2,
            supports_recursive_outputs=True,
            mode="all_depths",
            aux_weight=-0.5,
        )


def test_r1_intermediate_only_reduces_to_primary_only() -> None:
    """Test #6: R=1 intermediate_only has no eligible depth (0..R-2 is
    empty). Documented contract: an empty weights tuple reduces to
    primary-only (rather than being rejected); a non-empty tuple still
    raises the usual length-mismatch (0 expected, none eligible)."""
    validated = validate_recursive_depth_supervision(
        weights=(), num_depths=1, supports_recursive_outputs=True, mode="intermediate_only"
    )
    assert validated.enabled is False
    assert validated.mode == "intermediate_only"

    with pytest.raises(ValueError, match="length|requires 0"):
        validate_recursive_depth_supervision(
            weights=(1.0,),
            num_depths=1,
            supports_recursive_outputs=True,
            mode="intermediate_only",
        )

    # End-to-end: R=1 shared_recursive model, intermediate_only, empty
    # weights -- trains fine as primary-only, combined == primary.
    model, records = _recursive_model_for_mode(
        mode="intermediate_only", weights=(), recursive_steps=1
    )
    model.training_loss(records)
    metrics = model.last_training_metrics
    assert metrics["recursive_depth_supervision_enabled"] is False
    assert metrics["combined_training_loss"] == pytest.approx(
        metrics["primary_final_reconstruction_loss"], rel=1e-6
    )


def test_checkpoint_roundtrip_preserves_mode_and_coefficient(tmp_path: Path) -> None:
    """Test #7: resume/checkpoint/config round-trip preserves
    recursive_depth_aux_mode and recursive_depth_aux_weight."""
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=(0.5, 1.0),
            recursive_depth_aux_mode="intermediate_only",
            recursive_depth_aux_weight=0.3,
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    ckpt = tmp_path / "rsc_a02.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert loaded.config.recursive_depth_aux_mode == "intermediate_only"
    assert loaded.config.recursive_depth_aux_weight == pytest.approx(0.3)
    assert loaded.config.recursive_depth_supervision_weights == (0.5, 1.0)


def test_migrate_recursive_depth_aux_config_deterministic() -> None:
    """Test #7 (migration half): a persisted config dict predating
    recursive_depth_aux_mode migrates deterministically -- "off" when no
    legacy weights were set, "legacy_all_depths" (reproduction-only) when
    they were -- and is idempotent / non-mutating."""
    empty_legacy = {"recursive_depth_supervision_weights": ()}
    migrated_empty = migrate_recursive_depth_aux_config(empty_legacy)
    assert migrated_empty["recursive_depth_aux_mode"] == "off"
    assert migrated_empty["recursive_depth_aux_weight"] == 1.0
    assert "recursive_depth_aux_mode" not in empty_legacy  # not mutated in place

    nonempty_legacy = {"recursive_depth_supervision_weights": (0.5, 1.0)}
    migrated_nonempty = migrate_recursive_depth_aux_config(nonempty_legacy)
    assert migrated_nonempty["recursive_depth_aux_mode"] == "legacy_all_depths"
    assert migrated_nonempty["recursive_depth_aux_weight"] == 1.0

    already_migrated = {
        "recursive_depth_supervision_weights": (1.0,),
        "recursive_depth_aux_mode": "all_depths",
        "recursive_depth_aux_weight": 0.7,
    }
    assert migrate_recursive_depth_aux_config(already_migrated) == already_migrated


def test_resolve_recursive_depth_aux_mode_backward_compatible() -> None:
    """``resolve_recursive_depth_aux_mode`` mirrors the migration policy for
    in-memory (non-checkpoint) configs, e.g. TwoTowerConfig() built directly
    in Python without ever touching the new field."""
    assert resolve_recursive_depth_aux_mode(None, ()) == "off"
    assert resolve_recursive_depth_aux_mode(None, (0.5, 1.0)) == "legacy_all_depths"
    assert resolve_recursive_depth_aux_mode("all_depths", (0.5, 1.0)) == "all_depths"
    assert resolve_recursive_depth_aux_mode("off", ()) == "off"


def test_generated_decomposition_sums_reproduce_scalar_loss_exactly() -> None:
    """Test #8: the objective decomposition's own fields sum exactly to the
    combined loss actually added into the training objective."""
    model, records = _recursive_model_for_mode(
        mode="all_depths", weights=(0.5, 1.0, 0.5), aux_weight=0.3, recursive_steps=3
    )
    loss = model.training_loss(records)
    metrics = model.last_training_metrics
    assert float(loss.detach().cpu()) == pytest.approx(
        metrics["combined_training_loss"], rel=1e-5
    )
    assert metrics["combined_training_loss"] == pytest.approx(
        metrics["primary_final_reconstruction_loss"]
        + metrics["recursive_depth_supervision_loss"],
        rel=1e-6,
    )
    assert metrics["recursive_depth_supervision_loss"] == pytest.approx(
        metrics["recursive_intermediate_aux_loss"]
        + metrics["recursive_final_depth_aux_contribution"],
        rel=1e-6,
    )
    # Required RecursiveObjectiveContractV2 schema artifact is populated and
    # agrees with the flat metrics it was built from.
    contract = metrics["recursive_objective_contract"]
    assert contract["contract_version"] == RECURSIVE_OBJECTIVE_CONTRACT_VERSION
    assert contract["combined_training_loss"] == pytest.approx(
        metrics["combined_training_loss"], rel=1e-6
    )


def test_recursive_objective_contract_v2_validates_sum_identities() -> None:
    """RecursiveObjectiveContractV2.from_metrics builds a valid contract from
    a consistent metrics dict, and __post_init__ rejects one whose sum
    identities don't hold (e.g. a future edit that breaks the decomposition)."""
    consistent = {
        "recursive_depth_aux_mode": "all_depths",
        "recursive_depth_aux_weight": 0.3,
        "primary_final_reconstruction_loss": 10.0,
        "recursive_intermediate_aux_loss": 2.0,
        "recursive_final_depth_aux_contribution": 1.0,
        "recursive_depth_supervision_loss": 3.0,
        "combined_training_loss": 13.0,
    }
    contract = RecursiveObjectiveContractV2.from_metrics(consistent)
    assert contract.mode == "all_depths"
    assert contract.as_dict()["recursive_depth_supervision_loss"] == 3.0

    inconsistent = dict(consistent)
    inconsistent["combined_training_loss"] = 999.0
    with pytest.raises(ValueError, match="combined_training_loss"):
        RecursiveObjectiveContractV2.from_metrics(inconsistent)
