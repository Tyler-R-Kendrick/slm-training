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
from slm_training.models.recursive_denoiser import (
    ArchitectureComparisonReportV1,
    SharedRecursiveDenoiserTower,
    StackedMatchedStateDenoiserTower,
    compare_denoiser_architectures,
    recursive_zstate_parameter_delta,
)
from slm_training.models.recursive_control_arms import (
    ALL_ARM_IDS,
    ARM_DENOISER_ARCH,
    BUILT_ARM_IDS,
    DEFERRED_ARM_IDS,
    build_control_arm_table,
    construct_arm_tower,
)
from slm_training.models.rng_contract import (
    build_recursive_control_initialization,
    derive_seed,
)
from slm_training.models.twotower import (
    KNOWN_DENOISER_ARCHES,
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


def test_recursive_r1_preserves_denoiser_interface_and_finite_shapes() -> None:
    """R=1 with L transition blocks preserves the ``DenoiserTower`` public
    interface (matching forward shape, finite outputs) -- NOT output
    equivalence.

    SLM-240 (RSC-A04): renamed from the misleadingly-named
    ``test_recursive_steps_one_parity_with_denoiser_tower``, which asserted
    only shape/finiteness while its own comment already noted the outputs
    differ -- i.e. it tested interface compatibility, not parity. See
    ``test_recursive_r1_output_not_behaviorally_equivalent_to_stacked``
    below for the independently-tested non-equivalence claim, and
    ``ArchitectureComparisonReportV1`` for the full multi-dimensional
    comparison this issue requires in place of a single ``parity`` claim.
    """
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
    assert torch.isfinite(r_logits).all()


def test_recursive_r1_output_not_behaviorally_equivalent_to_stacked() -> None:
    """R=1 never reproduces ``DenoiserTower``'s outputs: the z-state path
    (``z_latent``/``ctx_proj``) is active at every ``R``, including 1.

    SLM-240 (RSC-A04): no true-degeneracy mode exists today (a documented
    non-goal); this pins that fact so a future "R=1 reduces exactly to
    DenoiserTower" claim must land its own passing test rather than being
    silently assumed from interface compatibility.
    """
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
    assert not torch.allclose(s_logits, r_logits)


@pytest.mark.parametrize(
    "d_model,max_len",
    [(16, 32), (16, 256), (32, 32), (32, 256), (64, 128)],
)
def test_recursive_parameter_delta_formula_matches_constructed_towers(
    d_model: int, max_len: int
) -> None:
    """``recursive_zstate_parameter_delta`` matches the real measured
    parameter-count delta between constructed towers, for several
    ``(d_model, max_len)`` pairs -- the delta is a function of those two
    values only (independent of ``vocab_size``/``n_layers``/``recursive_steps``).

    ``(d_model=32, max_len=256)`` is the SLM-138 fixture's own denoiser
    config (``TwoTowerConfig`` defaults ``max_target_len=256``,
    ``scripts/run_slm138_recursive_denoiser_fixture.py`` uses ``d_model=32``):
    the formula reproduces the checked-in fixture's 9,248-parameter delta
    exactly, from the formula -- never hard-coded.
    """
    vocab = 23
    stacked = DenoiserTower(
        vocab_size=vocab, d_model=d_model, n_layers=2, n_heads=2, max_len=max_len
    )
    recursive = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=max_len,
        recursive_steps=1,
        recursive_transition_layers=2,
    )
    measured_delta = sum(p.numel() for p in recursive.parameters()) - sum(
        p.numel() for p in stacked.parameters()
    )
    assert measured_delta == recursive_zstate_parameter_delta(
        d_model=d_model, max_len=max_len
    )
    if d_model == 32 and max_len == 256:
        assert measured_delta == 9248


def test_recursive_parameter_count_independent_of_recursive_steps() -> None:
    """Shared-transition parameters are reused by object identity every
    recursion, so total parameter count does not scale with
    ``recursive_steps`` -- only per-forward compute (block evaluations) does."""
    vocab, d_model, max_len = 23, 16, 32
    counts = {
        steps: sum(
            p.numel()
            for p in SharedRecursiveDenoiserTower(
                vocab_size=vocab,
                d_model=d_model,
                n_layers=2,
                n_heads=2,
                max_len=max_len,
                recursive_steps=steps,
                recursive_transition_layers=2,
            ).parameters()
        )
        for steps in (1, 2, 5)
    }
    assert len(set(counts.values())) == 1, counts


def test_transition_layer_names_and_shapes_map_onto_stacked_1to1() -> None:
    """When ``recursive_transition_layers == stacked.n_layers``, every stacked
    transition-layer parameter name/shape has an exact recursive counterpart,
    and the only recursive-specific parameters are ``z_latent``/``ctx_proj``."""
    vocab, d_model, max_len = 23, 16, 32
    stacked = DenoiserTower(
        vocab_size=vocab, d_model=d_model, n_layers=2, n_heads=2, max_len=max_len
    )
    recursive = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=max_len,
        recursive_steps=2,
        recursive_transition_layers=2,
    )
    noisy = torch.randint(1, vocab, (2, 6))
    ctx = torch.randn(2, 3, d_model)
    report = compare_denoiser_architectures(
        stacked, recursive, noisy_ids=noisy, context=ctx, pad_id=0
    )

    stacked_layer_names = {
        name: list(p.shape)
        for name, p in stacked.named_parameters()
        if name.startswith("layers.")
    }
    assert stacked_layer_names  # sanity: stacked actually has transition layers
    for name, shape in stacked_layer_names.items():
        assert report.common_parameter_names_and_shapes.get(name) == shape

    specific = report.architecture_specific_parameter_names_and_shapes
    assert specific["stacked_only"] == {}
    assert set(specific["recursive_only"]) == {
        "z_latent",
        "ctx_proj.weight",
        "ctx_proj.bias",
    }


def test_architecture_comparison_report_block_evaluations_match_real_hook_counts() -> (
    None
):
    """``block_evaluations_per_forward`` must equal real ``TransformerBlock``
    invocation counts (measured via forward hooks), not an assumed formula."""
    vocab, d_model, max_len = 23, 16, 32
    stacked = DenoiserTower(
        vocab_size=vocab, d_model=d_model, n_layers=2, n_heads=2, max_len=max_len
    )
    recursive = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=max_len,
        recursive_steps=3,
        recursive_transition_layers=2,
    )
    noisy = torch.randint(1, vocab, (2, 6))
    ctx = torch.randn(2, 3, d_model)

    counts = {"stacked": 0, "recursive": 0}

    def _make_hook(key: str):
        def _hook(module, inp, out):
            counts[key] += 1

        return _hook

    handles = [
        layer.register_forward_hook(_make_hook("stacked")) for layer in stacked.layers
    ] + [
        layer.register_forward_hook(_make_hook("recursive"))
        for layer in recursive.layers
    ]
    try:
        report = compare_denoiser_architectures(
            stacked, recursive, noisy_ids=noisy, context=ctx, pad_id=0
        )
    finally:
        for handle in handles:
            handle.remove()

    # compare_denoiser_architectures runs two forwards per tower (one
    # no_grad shape/equivalence check, one grad-enabled active-parameter
    # measurement), so real hook counts are 2x the per-forward figure.
    assert counts["stacked"] == 2 * report.block_evaluations_per_forward["stacked"]
    assert counts["recursive"] == 2 * report.block_evaluations_per_forward["recursive"]


def test_architecture_comparison_report_consistent_with_measured_counts() -> None:
    """Delta/formula/checkpoint fields are internally consistent and derived
    from real measurements -- and no field anywhere collapses into a single
    ``parity`` boolean."""
    vocab, d_model, max_len = 23, 32, 256
    stacked = DenoiserTower(
        vocab_size=vocab, d_model=d_model, n_layers=2, n_heads=2, max_len=max_len
    )
    recursive = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=max_len,
        recursive_steps=2,
        recursive_transition_layers=2,
    )
    noisy = torch.randint(1, vocab, (2, 6))
    ctx = torch.randn(2, 3, d_model)
    report = compare_denoiser_architectures(
        stacked, recursive, noisy_ids=noisy, context=ctx, pad_id=0
    )

    def _all_keys(obj: object) -> set[str]:
        if isinstance(obj, dict):
            keys = set(obj.keys())
            for value in obj.values():
                keys |= _all_keys(value)
            return keys
        return set()

    assert "parity" not in _all_keys(report.as_dict())
    assert report.parameter_count_delta == 9248
    assert report.parameter_count_delta_matches_formula is True
    assert (
        report.parameter_count_total["recursive"]
        - report.parameter_count_total["stacked"]
        == report.parameter_count_delta
    )
    assert (
        report.parameter_count_denoiser["stacked"]
        == report.parameter_count_denoiser["recursive"]
    )
    assert report.checkpoint_bytes["recursive"] > report.checkpoint_bytes["stacked"]
    assert report.interface_compatible is True
    assert report.output_shape_compatible is True
    assert report.behaviorally_equivalent_under_declared_degeneracy is False
    assert report.claim_class == "wiring"


def _base_report_kwargs() -> dict:
    return dict(
        contract_version="ArchitectureComparisonReportV1",
        claim_class="wiring",
        d_model=16,
        max_len=32,
        recursive_steps=1,
        recursive_transition_layers=2,
        interface_compatible=True,
        output_shape_compatible=True,
        parameter_count_total={"stacked": 100, "recursive": 200},
        parameter_count_denoiser={"stacked": 50, "recursive": 50},
        active_parameter_count={"stacked": 80, "recursive": 150},
        checkpoint_bytes={"stacked": 400, "recursive": 800},
        common_parameter_names_and_shapes={},
        architecture_specific_parameter_names_and_shapes={
            "stacked_only": {},
            "recursive_only": {},
        },
        parameter_count_delta=100,
        parameter_count_delta_pct=100.0,
        parameter_count_delta_matches_formula=(
            recursive_zstate_parameter_delta(d_model=16, max_len=32) == 100
        ),
        block_evaluations_per_forward={"stacked": 2, "recursive": 2},
        estimated_forward_flops={"stacked": 1.0, "recursive": 1.0},
        behaviorally_equivalent_under_declared_degeneracy=False,
    )


def test_architecture_comparison_report_rejects_bad_contract_version() -> None:
    kwargs = _base_report_kwargs()
    kwargs["contract_version"] = "bogus"
    with pytest.raises(ValueError, match="contract_version"):
        ArchitectureComparisonReportV1(**kwargs)


def test_architecture_comparison_report_rejects_non_wiring_claim_class() -> None:
    kwargs = _base_report_kwargs()
    kwargs["claim_class"] = "quality"
    with pytest.raises(ValueError, match="wiring-only"):
        ArchitectureComparisonReportV1(**kwargs)


def test_architecture_comparison_report_rejects_inconsistent_delta() -> None:
    kwargs = _base_report_kwargs()
    kwargs["parameter_count_delta"] = 999
    with pytest.raises(ValueError, match="parameter_count_delta"):
        ArchitectureComparisonReportV1(**kwargs)


def test_architecture_comparison_report_rejects_mislabeled_formula_match() -> None:
    kwargs = _base_report_kwargs()
    kwargs["parameter_count_delta_matches_formula"] = not kwargs[
        "parameter_count_delta_matches_formula"
    ]
    with pytest.raises(ValueError, match="parameter_count_delta_matches_formula"):
        ArchitectureComparisonReportV1(**kwargs)


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
    # SLM-240 (RSC-A04): the migration report must list *every* new z-state
    # key explicitly -- these have no DenoiserTower counterpart, so
    # checkpoint layer-name compatibility applies only to the mapped/common
    # transition-layer keys, never to these.
    expected_new_keys = {
        "denoiser.z_latent",
        "denoiser.ctx_proj.weight",
        "denoiser.ctx_proj.bias",
    }
    assert expected_new_keys.issubset(set(report["initialized_keys"]))

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


def test_fixture_architecture_comparison_delta_reproduced_from_formula() -> None:
    """SLM-240 (RSC-A04): the committed fixture's whole-model parameter delta
    (74,242 - 64,994 = 9,248) equals both the tower-level
    ``ArchitectureComparisonReportV1.parameter_count_delta`` and the pure
    ``recursive_zstate_parameter_delta`` formula -- reproduced, never
    hard-coded, and never collapsed into a single ``parity`` field."""
    from scripts.run_slm138_recursive_denoiser_fixture import _run_fixture

    report = _run_fixture()
    comparison = report["architecture_comparison"]
    whole_model_delta = report["recursive_params"] - report["stacked_params"]

    assert whole_model_delta == comparison["parameter_count_delta"]
    assert comparison["parameter_count_delta_matches_formula"] is True
    assert comparison["parameter_count_delta"] == recursive_zstate_parameter_delta(
        d_model=comparison["d_model"], max_len=comparison["max_len"]
    )
    assert comparison["interface_compatible"] is True
    assert comparison["behaviorally_equivalent_under_declared_degeneracy"] is False
    assert comparison["claim_class"] == "wiring"
    assert "parity" not in comparison


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


# ---------------------------------------------------------------------------
# SLM-239 (RSC-A03): bit-reproducible RNG contract for the SLM-138 fixture --
# explicit disjoint RNG namespaces, deterministic call-order-independent
# shape probes, restored-checkpoint repeated evaluation, a clean-tree
# evidence gate, and FixtureDeterminismReportV1.
# ---------------------------------------------------------------------------

from slm_training.models.rng_contract import (
    NAMESPACE_OFFSETS,
    RngCheckpoint,
    isolated_draw,
    seed_training_corruption,
)
from scripts.run_slm138_recursive_denoiser_fixture import (
    _canonical_json,
    _clean_tree_gate,
    _compare_reports,
    _run_fixture,
)


def test_rsc_a03_derive_seed_disjoint_and_fails_closed_on_unknown_namespace() -> None:
    seeds = {ns: derive_seed(7, ns) for ns in NAMESPACE_OFFSETS}
    assert len(set(seeds.values())) == len(seeds)  # pairwise disjoint
    assert derive_seed(7, "model_initialization") == 7  # offset 0, unchanged
    with pytest.raises(ValueError, match="unknown RNG namespace"):
        derive_seed(7, "bogus_namespace")


def test_rsc_a03_isolated_draw_leaves_outer_rng_state_untouched() -> None:
    """isolated_draw's fork_rng contract: the outer global stream is
    byte-identical whether or not (or how many times) a probe runs."""
    torch.manual_seed(123)
    before = RngCheckpoint.capture().digest()
    isolated_draw(0, "shape_probe_inputs", lambda: torch.randn(3, 3))
    isolated_draw(0, "shape_probe_context", lambda: torch.randn(5, 5))
    after = RngCheckpoint.capture().digest()
    assert before == after
    # A subsequent global draw is identical to what an untouched stream would
    # have produced -- proof the probes never advanced it.
    torch.manual_seed(123)
    expected = torch.randn(2, 2)
    torch.manual_seed(123)
    isolated_draw(0, "shape_probe_inputs", lambda: torch.randn(3, 3))
    actual = torch.randn(2, 2)
    torch.testing.assert_close(actual, expected)


# Test 1: two isolated fixture executions produce byte-identical JSON.
def test_rsc_a03_two_fixture_runs_byte_identical_json() -> None:
    run_a = _run_fixture(allow_dirty=True)
    run_b = _run_fixture(allow_dirty=True)
    assert _canonical_json(run_a) == _canonical_json(run_b)


# Test 2: inserting/reordering shape-probe random calls does not change
# training-loss or deep-supervision values.
def test_rsc_a03_probe_order_permutation_does_not_change_training_values() -> None:
    stacked_first = _run_fixture(probe_order="stacked_first", allow_dirty=True)
    recursive_first = _run_fixture(probe_order="recursive_first", allow_dirty=True)
    assert stacked_first["losses"] == recursive_first["losses"]
    assert (
        stacked_first["post_update_verification"]
        == recursive_first["post_update_verification"]
    )
    assert (
        stacked_first["deep_supervision_metrics"]
        == recursive_first["deep_supervision_metrics"]
    )
    assert stacked_first["forward_shapes"] == recursive_first["forward_shapes"]


def test_rsc_a03_extra_harmless_probe_does_not_change_training_values() -> None:
    without_extra = _run_fixture(insert_extra_probe=False, allow_dirty=True)
    with_extra = _run_fixture(insert_extra_probe=True, allow_dirty=True)
    assert without_extra["losses"] == with_extra["losses"]
    assert (
        without_extra["deep_supervision_metrics"]
        == with_extra["deep_supervision_metrics"]
    )
    assert (
        without_extra["post_update_verification"] == with_extra["post_update_verification"]
    )


# Test 3: repeated objective evaluation with a restored generator state is
# identical.
def test_rsc_a03_restored_generator_state_repeated_evaluation_identical() -> None:
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=(0.5, 1.0),
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    checkpoint = seed_training_corruption(0, model)
    loss_1 = model.training_loss(records)
    metrics_1 = dict(model.last_training_metrics)

    checkpoint.restore(model)
    loss_2 = model.training_loss(records)
    metrics_2 = dict(model.last_training_metrics)

    torch.testing.assert_close(loss_1, loss_2)
    assert metrics_1["recursive_depth_loss_0"] == metrics_2["recursive_depth_loss_0"]
    assert metrics_1["recursive_depth_loss_1"] == metrics_2["recursive_depth_loss_1"]
    assert (
        metrics_1["recursive_depth_supervision_loss"]
        == metrics_2["recursive_depth_supervision_loss"]
    )


def test_rsc_a03_restoring_only_torch_rng_is_insufficient_without_model() -> None:
    """Documents the real second RNG source this module accounts for:
    ``TwoTowerModel`` keeps a persistent per-instance ``self._rng``
    (``random.Random(config.seed)``) that ``_mask_targets`` also reads (the
    "ensure at least one predictable token per row" fallback / mixed-pattern
    span selection). Restoring *only* the global torch RNG state (by not
    passing ``model`` to ``seed_training_corruption``/``RngCheckpoint``)
    reproduces a *different* loss on the second call -- proving both RNG
    sources must be captured/restored together, which is exactly what
    passing ``model`` does in the sibling
    ``test_rsc_a03_restored_generator_state_repeated_evaluation_identical``."""
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=(0.5, 1.0),
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    torch_only_checkpoint = seed_training_corruption(0)  # no model -> torch only
    loss_1 = model.training_loss(records)
    torch_only_checkpoint.restore()  # torch RNG restored; model._rng is not
    loss_2 = model.training_loss(records)
    assert not torch.allclose(loss_1, loss_2)


# Test 4: different declared training-corruption seeds change the intended
# corruption-dependent fields and no others.
def test_rsc_a03_different_training_corruption_seed_changes_only_corruption_fields() -> None:
    default_seed = _run_fixture(allow_dirty=True)
    other_seed = _run_fixture(training_corruption_seed=999_999, allow_dirty=True)

    assert default_seed["losses"] != other_seed["losses"]

    classification = _compare_reports(default_seed, other_seed)
    corruption_only_fields = {
        "losses",
        "post_update_verification",
        "deep_supervision_metrics",
    }
    unexpected = {
        k: v
        for k, v in classification.items()
        if v != "exact" and k not in corruption_only_fields
    }
    assert unexpected == {}
    assert default_seed["stacked_params"] == other_seed["stacked_params"]
    assert default_seed["forward_shapes"] == other_seed["forward_shapes"]
    assert (
        default_seed["recursive_weight_sharing"] == other_seed["recursive_weight_sharing"]
    )
    assert (
        default_seed["checkpoint_roundtrip_ok"] == other_seed["checkpoint_roundtrip_ok"]
    )
    assert (
        default_seed["provenance_hashes"] == other_seed["provenance_hashes"]
    )


# Test 5: global RNG state before/after the fixture changes only according to
# an explicit, tested contract -- here, the fixture's exit state is a
# deterministic function of its own base_seed, independent of the caller's
# prior global RNG state (every RNG-consuming step inside it reseeds from
# base_seed-derived namespaces before it draws).
def test_rsc_a03_fixture_exit_rng_state_independent_of_caller_entry_state() -> None:
    torch.manual_seed(1)
    _run_fixture(allow_dirty=True)
    exit_state_a = RngCheckpoint.capture().digest()

    torch.manual_seed(999_999)
    _run_fixture(allow_dirty=True)
    exit_state_b = RngCheckpoint.capture().digest()

    assert exit_state_a == exit_state_b


# Test 6: dirty-tree debug artifacts are marked non-comparable and cannot
# satisfy the clean-tree evidence gate.
def test_rsc_a03_clean_tree_gate_marks_dirty_non_comparable() -> None:
    assert _clean_tree_gate(code_dirty=False, allow_dirty=False) == {
        "comparable": True,
        "claim_grade": True,
    }
    assert _clean_tree_gate(code_dirty=True, allow_dirty=False) == {
        "comparable": False,
        "claim_grade": False,
    }
    # --allow-dirty permits the artifact to be written but never launders it
    # into "comparable" -- it stays non-comparable regardless.
    assert _clean_tree_gate(code_dirty=True, allow_dirty=True) == {
        "comparable": False,
        "claim_grade": False,
    }
    # Unknowable git state fails closed the same as dirty.
    assert _clean_tree_gate(code_dirty=None, allow_dirty=True) == {
        "comparable": False,
        "claim_grade": False,
    }


def test_rsc_a03_fixture_evidence_gate_reflects_dirty_tree(monkeypatch) -> None:
    """End-to-end: when the embedded version_stamp reports a dirty tree, the
    fixture's own evidence_gate honestly marks itself non-comparable --
    exercised via a forced-dirty stamp rather than depending on this
    session's actual git status."""
    import scripts.run_slm138_recursive_denoiser_fixture as fixture_mod

    def _fake_stamp(*_component_ids: str) -> dict:
        return {
            "stamp_schema": "version_stamp/v1",
            "code_commit": "deadbeef",
            "code_dirty": True,
            "components": {},
            "stamped_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(fixture_mod, "build_version_stamp", _fake_stamp)
    report = fixture_mod._run_fixture(allow_dirty=True)
    assert report["evidence_gate"]["comparable"] is False
    assert report["evidence_gate"]["code_dirty"] is True


# Test 7: clean-tree artifact passes version-stamp verification -- the
# fixture's embedded version_stamp component version matches the current
# registry entry for model.recursive_denoiser.
def test_rsc_a03_fixture_version_stamp_matches_registry() -> None:
    from slm_training.versioning import STAMP_SCHEMA, component_version

    report = _run_fixture(allow_dirty=True)
    stamp = report["version_stamp"]
    assert stamp["stamp_schema"] == STAMP_SCHEMA
    assert (
        stamp["components"]["model.recursive_denoiser"]
        == component_version("model.recursive_denoiser")
    )


# Test 8: checkpoint save/load output digest is identical before/after
# round-trip.
def test_rsc_a03_checkpoint_roundtrip_state_dict_digest_identical(
    tmp_path: Path,
) -> None:
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_transition_layers=2,
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )

    def _state_digest(m: TwoTowerModel) -> str:
        import hashlib

        hasher = hashlib.sha256()
        for key in sorted(m.state_dict()):
            tensor = m.state_dict()[key]
            hasher.update(key.encode("utf-8"))
            hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()

    before = _state_digest(model)
    ckpt = tmp_path / "roundtrip.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    after = _state_digest(loaded)
    assert before == after


def test_rsc_a03_determinism_report_verdict_bit_exact_and_namespace_isolated() -> None:
    """The FixtureDeterminismReportV1 built from real repeated executions +
    call-order permutations is bit_exact, and the different-training-
    corruption-seed comparison changes only the declared corruption-dependent
    fields (namespace isolation holds)."""
    from scripts.run_slm138_recursive_denoiser_fixture import _determinism_report

    report = _determinism_report(base_seed=0)
    assert report["run_a_digest"] == report["run_b_digest"]
    assert report["verdict"] == "bit_exact"
    assert report["namespace_isolation_ok"] is True
    assert report["different_training_corruption_seed_unexpected_changes"] == {}


# ---------------------------------------------------------------------------
# SLM-241 (RSC-A05): matched recursive control arms (A/B/C/D/G built;
# E/F/H explicitly deferred).
# ---------------------------------------------------------------------------


def test_zstate_mode_rejects_unknown_value() -> None:
    """Fail closed on a typo'd z_state_mode -- never silently fall back."""
    with pytest.raises(ValueError, match="z_state_mode"):
        SharedRecursiveDenoiserTower(
            vocab_size=23, d_model=16, n_layers=1, n_heads=2, max_len=32,
            z_state_mode="bogus",  # type: ignore[arg-type]
        )


def test_denoiser_arch_rejects_unknown_value() -> None:
    """TwoTowerModel construction fails closed on an unrecognized
    denoiser_arch instead of silently falling back to 'stacked'."""
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    with pytest.raises(ValueError, match="denoiser_arch"):
        TwoTowerModel.from_records(
            records,
            config=TwoTowerConfig(
                d_model=16, n_heads=2, denoiser_layers=1,
                denoiser_arch="bogus_arch",  # type: ignore[arg-type]
                grammar_constrained=False, seed=0,
            ),
            device="cpu",
        )


@pytest.mark.parametrize("arm_id", ["C", "D"])
def test_arm_c_d_have_no_zstate_parameters(arm_id: str) -> None:
    """Requirement #4: C/D must contain no undeclared z-state parameters --
    no ``z_latent``/``ctx_proj`` name at all, not merely a zeroed tensor."""
    tower = construct_arm_tower(
        arm_id, vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
        recursive_steps=2, recursive_transition_layers=2,
    )
    names = dict(tower.named_parameters())
    assert not any(name.split(".")[0] in {"z_latent", "ctx_proj"} for name in names)
    assert not hasattr(tower, "z_latent")
    assert not hasattr(tower, "ctx_proj")


@pytest.mark.parametrize("arm_id", ["C", "D"])
def test_arm_c_d_parameter_count_matches_stacked_baseline(arm_id: str) -> None:
    """Requirement #2: analytical parameter-count formula. When
    recursive_transition_layers == the stacked baseline's n_layers, C/D add
    exactly zero parameters over A (no z-state bank of any kind) -- verified
    against real constructed towers, not assumed."""
    stacked = construct_arm_tower(
        "A", vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
    )
    arm = construct_arm_tower(
        arm_id, vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
        recursive_steps=3, recursive_transition_layers=2,
    )
    stacked_total = sum(p.numel() for p in stacked.parameters())
    arm_total = sum(p.numel() for p in arm.parameters())
    assert arm_total == stacked_total


def test_arm_g_is_r1_shared_recursive_and_not_behaviorally_equivalent() -> None:
    """Arm G: R=1 shared architecture control -- same denoiser_arch as B,
    recursive_steps forced to 1 regardless of the requested value, and (per
    SLM-240's already-established framing) not behaviorally equivalent to A."""
    g_tower = construct_arm_tower(
        "G", vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
        recursive_steps=5, recursive_transition_layers=2,
    )
    assert isinstance(g_tower, SharedRecursiveDenoiserTower)
    assert g_tower.recursive_steps == 1
    assert g_tower.z_state_mode == "full"

    torch.manual_seed(0)
    stacked = construct_arm_tower(
        "A", vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
    )
    torch.manual_seed(0)
    g_again = construct_arm_tower(
        "G", vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
        recursive_steps=1, recursive_transition_layers=2,
    )
    noisy = torch.randint(1, 23, (2, 6))
    ctx = torch.randn(2, 3, 16)
    stacked.eval()
    g_again.eval()
    with torch.no_grad():
        s_logits = stacked(noisy, ctx, pad_id=0)
        g_logits = g_again(noisy, ctx, pad_id=0)
    assert s_logits.shape == g_logits.shape
    assert not torch.allclose(s_logits, g_logits)


@pytest.mark.parametrize("arm_id", DEFERRED_ARM_IDS)
def test_deferred_arms_fail_closed_not_silently_built(arm_id: str) -> None:
    """E/F/H must never silently construct something -- they raise
    NotImplementedError until a future iteration actually builds them."""
    with pytest.raises(NotImplementedError, match=arm_id):
        construct_arm_tower(
            arm_id, vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
        )


def test_construct_arm_tower_rejects_unknown_arm_id() -> None:
    with pytest.raises(ValueError, match="unknown control arm"):
        construct_arm_tower(
            "Z", vocab_size=23, d_model=16, n_layers=1, n_heads=2, max_len=32,
        )


def test_control_arm_table_reports_every_built_arm_no_parity_or_winner() -> None:
    """Requirement #11: the fixture emits a complete comparison table for
    every built arm, with no raw-loss winner language -- these reports never
    even compute a loss."""
    vocab, d_model, n_layers, max_len = 23, 16, 2, 32
    noisy = torch.randint(1, vocab, (2, 6))
    ctx = torch.randn(2, 3, d_model)
    reports = build_control_arm_table(
        BUILT_ARM_IDS,
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len, recursive_steps=3, recursive_transition_layers=n_layers,
        noisy_ids=noisy, context=ctx, pad_id=0,
    )
    assert {r.arm_id for r in reports} == set(BUILT_ARM_IDS)
    by_id = {r.arm_id: r for r in reports}

    for report in reports:
        assert report.claim_class == "wiring"
        assert "parity" not in report.as_dict()
        assert "winner" not in report.as_dict()

    # A is the declared reference: zero delta against itself.
    assert by_id["A"].parameter_count_delta_vs_baseline == 0
    # C/D match A's parameter count exactly at this (n_layers ==
    # recursive_transition_layers) configuration.
    assert by_id["C"].parameter_count_delta_vs_baseline == 0
    assert by_id["D"].parameter_count_delta_vs_baseline == 0
    assert by_id["C"].within_matching_tolerance
    assert by_id["D"].within_matching_tolerance
    # B/G add the z_latent/ctx_proj delta (same formula as SLM-240).
    expected_delta = recursive_zstate_parameter_delta(d_model=d_model, max_len=max_len)
    assert by_id["B"].parameter_count_delta_vs_baseline == expected_delta
    assert by_id["G"].parameter_count_delta_vs_baseline == expected_delta
    # Block-evaluation accounting: stacked does n_layers evals; B does
    # recursive_steps * recursive_transition_layers; G forces steps=1.
    assert by_id["A"].block_evaluations_per_forward == n_layers
    assert by_id["B"].block_evaluations_per_forward == 3 * n_layers
    assert by_id["G"].block_evaluations_per_forward == 1 * n_layers
    assert (
        by_id["B"].self_attn_calls_per_forward
        == by_id["B"].cross_attn_calls_per_forward
        == by_id["B"].mlp_calls_per_forward
        == by_id["B"].block_evaluations_per_forward
    )
    # F: block-evaluation-matched against B by construction, but MORE real
    # measured parameters than B -- never reported as parameter-matched.
    assert by_id["F"].block_evaluations_per_forward == by_id["B"].block_evaluations_per_forward
    assert by_id["F"].parameter_count_total > by_id["B"].parameter_count_total
    assert not by_id["F"].within_matching_tolerance
    assert by_id["F"].undeclared_zstate_parameter_names == ()
    # E: unshared, non-recursive -- same block-evaluation count as A -- but
    # its parameter delta over A equals recursive_zstate_parameter_delta
    # exactly (same formula B/G's delta matches), never A's own zero-delta
    # target (that's C's/D's kind of match, not E's).
    assert by_id["E"].block_evaluations_per_forward == by_id["A"].block_evaluations_per_forward
    assert by_id["E"].parameter_count_delta_vs_baseline == expected_delta
    assert by_id["E"].undeclared_zstate_parameter_names == ()


def test_recursive_control_arm_report_rejects_bad_contract_version() -> None:
    from slm_training.models.recursive_control_arms import RecursiveControlArmReportV1

    kwargs = dict(
        contract_version="bogus",
        claim_class="wiring",
        arm_id="A",
        label="x",
        denoiser_arch="stacked",
        z_state_mode=None,
        recursive_steps=0,
        recursive_transition_layers=2,
        d_model=16,
        max_len=32,
        parameter_count_total=100,
        parameter_count_denoiser=50,
        active_parameter_count=80,
        checkpoint_bytes=400,
        undeclared_zstate_parameter_names=(),
        block_evaluations_per_forward=2,
        self_attn_calls_per_forward=2,
        cross_attn_calls_per_forward=2,
        mlp_calls_per_forward=2,
        estimated_forward_flops=1.0,
        matching_target="baseline",
        matching_tolerance_params=0,
        parameter_count_delta_vs_baseline=0,
        parameter_count_delta_vs_baseline_pct=0.0,
        residual_matching_error_params=0,
        within_matching_tolerance=True,
        notes=(),
    )
    with pytest.raises(ValueError, match="contract_version"):
        RecursiveControlArmReportV1(**kwargs)


def test_deep_supervision_works_for_arm_c_and_d() -> None:
    """C/D also expose recursive_outputs (duck-typed by
    TwoTowerModel.training_loss), so deep supervision is not special-cased
    to arm B only."""
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]
    for arch, weights in (
        ("shared_recursive_y_only", (0.5, 1.0)),
        ("shared_recursive_no_extra_capacity", (0.5, 1.0)),
    ):
        model = TwoTowerModel.from_records(
            records,
            config=TwoTowerConfig(
                d_model=32, n_heads=2, context_layers=1, denoiser_layers=2,
                denoiser_arch=arch,  # type: ignore[arg-type]
                recursive_steps=2, recursive_transition_layers=2,
                recursive_depth_supervision_weights=weights,
                grammar_constrained=False, seed=0,
            ),
            device="cpu",
        )
        loss = model.training_loss(records)
        assert torch.isfinite(loss)
        assert "recursive_depth_supervision_loss" in model.last_training_metrics


@pytest.mark.parametrize(
    "arch", ["shared_recursive_y_only", "shared_recursive_no_extra_capacity"]
)
def test_arm_c_d_train_one_step_and_roundtrip_checkpoint(
    arch: str, tmp_path: Path
) -> None:
    """Requirement #1: every implemented arm constructs through the
    canonical factory, trains one step, and round-trips its checkpoint."""
    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32, n_heads=2, context_layers=1, denoiser_layers=2,
            denoiser_arch=arch,  # type: ignore[arg-type]
            recursive_steps=2, recursive_transition_layers=2,
            grammar_constrained=False, gen_steps=2, seed=0,
        ),
        device="cpu",
    )
    assert isinstance(model.denoiser, SharedRecursiveDenoiserTower)
    assert not hasattr(model.denoiser, "z_latent")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss = model.training_loss(records)
    loss.backward()
    opt.step()

    ckpt = tmp_path / f"{arch}.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert loaded.config.denoiser_arch == arch
    assert isinstance(loaded.denoiser, SharedRecursiveDenoiserTower)
    assert not hasattr(loaded.denoiser, "z_latent")


def test_known_denoiser_arches_matches_control_arm_registry() -> None:
    """KNOWN_DENOISER_ARCHES (twotower.py's fail-closed allowlist) and
    ARM_DENOISER_ARCH (the control-arm registry) must name the same set of
    real denoiser_arch strings -- no shadow arch string in either direction."""
    assert set(KNOWN_DENOISER_ARCHES) == set(ARM_DENOISER_ARCH.values())


def test_recursive_control_initialization_common_tensors_match_and_seeds_disjoint() -> (
    None
):
    """RecursiveControlInitializationV1: reseeding to the same
    model_initialization seed immediately before each arm's construction (the
    same discipline TwoTowerModel.__init__ already applies) produces
    bit-identical common tensors across A/B/C/D, and the declared
    architecture-specific seeds are pairwise disjoint."""
    base_seed = 0
    vocab, d_model, n_layers, max_len = 23, 16, 2, 32
    arm_ids = ("A", "B", "C", "D")
    towers = {}
    for arm_id in arm_ids:
        torch.manual_seed(derive_seed(base_seed, "model_initialization"))
        towers[arm_id] = construct_arm_tower(
            arm_id, vocab_size=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=2, max_len=max_len, recursive_steps=2,
            recursive_transition_layers=n_layers,
        )
    report = build_recursive_control_initialization(
        base_seed=base_seed,
        arm_towers=towers,
        arm_denoiser_arch={arm_id: ARM_DENOISER_ARCH[arm_id] for arm_id in arm_ids},
    )
    assert report.common_tensor_hashes_match_across_arms is True
    # Shared token/position embeddings and transition-block weights are
    # common across every arm at this (recursive_transition_layers ==
    # n_layers) configuration.
    assert "tok.weight" in report.common_tensor_names
    assert "pos.weight" in report.common_tensor_names
    assert any(name.startswith("layers.") for name in report.common_tensor_names)
    # B's z_latent/ctx_proj are architecture-specific, never common.
    assert "z_latent" in report.architecture_specific_tensor_names_and_shapes["B"]
    assert "z_latent" not in report.architecture_specific_tensor_names_and_shapes["C"]
    assert "z_latent" not in report.architecture_specific_tensor_names_and_shapes["D"]
    seeds = list(report.architecture_specific_seeds.values())
    assert len(seeds) == len(set(seeds))


def test_recursive_control_initialization_rejects_mismatched_common_tensors() -> None:
    """Fail closed: if common tensors were NOT actually initialized
    identically (e.g. the caller forgot to reseed between arms), the
    contract must refuse to report success."""
    vocab, d_model, n_layers, max_len = 23, 16, 2, 32
    torch.manual_seed(0)
    tower_a = construct_arm_tower(
        "A", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len,
    )
    torch.manual_seed(1)  # deliberately different seed -- no reseed discipline
    tower_b = construct_arm_tower(
        "B", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len, recursive_steps=2, recursive_transition_layers=n_layers,
    )
    with pytest.raises(ValueError, match="common_tensor_hashes_match_across_arms"):
        build_recursive_control_initialization(
            base_seed=0,
            arm_towers={"A": tower_a, "B": tower_b},
            arm_denoiser_arch={"A": "stacked", "B": "shared_recursive"},
        )


# ---------------------------------------------------------------------------
# SLM-241 (RSC-A05) follow-up: arm F -- unshared depth-matched tower.
# ---------------------------------------------------------------------------


def test_arm_f_is_unshared_depth_matched_tower_with_no_zstate() -> None:
    """Arm F is a plain DenoiserTower (no weight sharing, no z state) built
    with recursive_steps * recursive_transition_layers independent blocks --
    not the shared-object SharedRecursiveDenoiserTower."""
    tower = construct_arm_tower(
        "F", vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
        recursive_steps=3, recursive_transition_layers=2,
    )
    assert isinstance(tower, DenoiserTower)
    assert not hasattr(tower, "recursive_steps")
    assert not hasattr(tower, "z_latent")
    assert not hasattr(tower, "ctx_proj")
    assert len(tower.layers) == 3 * 2
    # No weight sharing: every block is a distinct parameterized object.
    block_ids = [id(layer) for layer in tower.layers]
    assert len(set(block_ids)) == len(block_ids)
    names = dict(tower.named_parameters())
    assert not any(name.split(".")[0] in {"z_latent", "ctx_proj"} for name in names)


def test_arm_f_block_evaluations_match_arm_b_verified_by_hook_count() -> None:
    """Requirement #6: concretely instrument/count real forward calls into
    the transition-block module during one forward pass -- not just a
    structural len(layers) claim."""
    vocab, d_model, n_layers, max_len = 23, 16, 2, 32
    recursive_steps, recursive_transition_layers = 3, 2

    f_tower = construct_arm_tower(
        "F", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len, recursive_steps=recursive_steps,
        recursive_transition_layers=recursive_transition_layers,
    )
    b_tower = construct_arm_tower(
        "B", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len, recursive_steps=recursive_steps,
        recursive_transition_layers=recursive_transition_layers,
    )

    def _count_block_calls(tower: torch.nn.Module) -> int:
        calls = {"n": 0}

        def _hook(module: torch.nn.Module, inp: object, out: object) -> None:
            calls["n"] += 1

        handles = [layer.register_forward_hook(_hook) for layer in tower.layers]
        noisy = torch.randint(1, vocab, (2, 6))
        ctx = torch.randn(2, 3, d_model)
        tower(noisy, ctx, pad_id=0)
        for handle in handles:
            handle.remove()
        return calls["n"]

    f_calls = _count_block_calls(f_tower)
    b_calls = _count_block_calls(b_tower)
    expected = recursive_steps * recursive_transition_layers
    assert f_calls == expected
    assert b_calls == expected
    assert f_calls == b_calls == len(f_tower.layers)


def test_arm_f_parameter_count_exceeds_arm_b_real_measured() -> None:
    """F's block-evaluation-matched construction has strictly MORE real
    measured parameters than B -- nothing is shared, so this must never be
    reported as parameter-matched."""
    vocab, d_model, n_layers, max_len = 23, 32, 2, 256
    b_tower = construct_arm_tower(
        "B", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len, recursive_steps=2, recursive_transition_layers=n_layers,
    )
    f_tower = construct_arm_tower(
        "F", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len, recursive_steps=2, recursive_transition_layers=n_layers,
    )
    b_total = sum(p.numel() for p in b_tower.parameters())
    f_total = sum(p.numel() for p in f_tower.parameters())
    assert f_total > b_total


def test_build_arm_f_dual_view_reports_honest_residuals() -> None:
    """build_arm_f_dual_view reports both matching views for arm F, each
    with an explicit, nonzero residual on whichever dimension is not
    matched -- never a bare 'matched' claim on both dimensions at once."""
    from slm_training.models.recursive_control_arms import build_arm_f_dual_view

    vocab, d_model, n_layers, max_len = 23, 32, 2, 256
    recursive_steps = 2
    noisy = torch.randint(1, vocab, (2, 6))
    ctx = torch.randn(2, 3, d_model)

    dual = build_arm_f_dual_view(
        vocab_size=vocab, d_model=d_model, n_heads=2, max_len=max_len,
        recursive_steps=recursive_steps, recursive_transition_layers=n_layers,
        noisy_ids=noisy, context=ctx, pad_id=0,
    )

    target_block_evals = recursive_steps * n_layers
    block_matched = dual["block_evaluation_matched"]
    nearest = dual["parameter_nearest"]

    # Block-evaluation-matched view: exact block-eval match, nonzero
    # parameter residual vs B (never hidden).
    assert block_matched["report"]["block_evaluations_per_forward"] == target_block_evals
    assert block_matched["parameter_count_delta_vs_target_arm_b"] > 0

    # Parameter-nearest view: real measured total closer to B than the
    # block-matched view's total, but NOT block-evaluation-matched.
    assert abs(nearest["parameter_count_delta_vs_target_arm_b"]) < abs(
        block_matched["parameter_count_delta_vs_target_arm_b"]
    )
    assert nearest["block_evaluations_delta_vs_target_arm_b"] != 0

    for view in (block_matched, nearest):
        assert view["report"]["claim_class"] == "wiring"
        assert "parity" not in view["report"]
        assert "winner" not in view["report"]

    # Per-layer cost is measured from real constructed towers, not hard-coded.
    formula = dual["per_layer_parameter_cost_formula"]
    assert formula["per_layer_parameters"] > 0


def test_arm_f_denoiser_arch_wired_through_twotower_config_and_roundtrips(
    tmp_path: Path,
) -> None:
    """Requirement #1: F constructs through the canonical factory
    (TwoTowerConfig.denoiser_arch) and round-trips a checkpoint, same as
    every other built arm."""
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32, n_heads=2, context_layers=1, denoiser_layers=2,
            denoiser_arch="stacked_depth_matched",  # type: ignore[arg-type]
            recursive_steps=2, recursive_transition_layers=2,
            grammar_constrained=False, gen_steps=2, seed=0,
        ),
        device="cpu",
    )
    assert isinstance(model.denoiser, DenoiserTower)
    assert len(model.denoiser.layers) == 4
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    loss.backward()

    ckpt = tmp_path / "arm_f.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert loaded.config.denoiser_arch == "stacked_depth_matched"
    assert isinstance(loaded.denoiser, DenoiserTower)
    assert len(loaded.denoiser.layers) == 4


def test_recursive_control_initialization_includes_arm_f_with_disjoint_seed() -> None:
    """The fairness contract covers arm F too: common tensors (tok/pos/the
    shared-prefix transition blocks) match across A/B/C/D/F, F's extra
    unshared blocks are its own architecture-specific tensors, and its
    reserved arch_specific:stacked_depth_matched seed is disjoint from every
    other arm's."""
    base_seed = 0
    vocab, d_model, n_layers, max_len = 23, 16, 2, 32
    arm_ids = ("A", "B", "C", "D", "F")
    towers = {}
    for arm_id in arm_ids:
        torch.manual_seed(derive_seed(base_seed, "model_initialization"))
        towers[arm_id] = construct_arm_tower(
            arm_id, vocab_size=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=2, max_len=max_len, recursive_steps=2,
            recursive_transition_layers=n_layers,
        )
    report = build_recursive_control_initialization(
        base_seed=base_seed,
        arm_towers=towers,
        arm_denoiser_arch={arm_id: ARM_DENOISER_ARCH[arm_id] for arm_id in arm_ids},
    )
    assert report.common_tensor_hashes_match_across_arms is True
    assert "tok.weight" in report.common_tensor_names
    # F's extra unshared blocks (layers.2/.3, beyond the shared prefix) are
    # architecture-specific to F alone.
    f_specific = report.architecture_specific_tensor_names_and_shapes["F"]
    assert any(name.startswith("layers.2") or name.startswith("layers.3") for name in f_specific)
    seeds = report.architecture_specific_seeds
    assert seeds["F"] not in {v for k, v in seeds.items() if k != "F"}
    assert len(set(seeds.values())) == len(seeds)


# ---------------------------------------------------------------------------
# SLM-241 (RSC-A05) follow-up: arm E -- stacked + matched state capacity.
# ---------------------------------------------------------------------------


def test_arm_e_is_unshared_non_recursive_tower_with_matched_state() -> None:
    """Arm E is a plain, unshared, non-recursive tower (same block-evaluation
    count as arm A) plus its own state/state_ctx_proj -- never B's
    z_latent/ctx_proj names, never a SharedRecursiveDenoiserTower."""
    tower = construct_arm_tower(
        "E", vocab_size=23, d_model=16, n_layers=2, n_heads=2, max_len=32,
    )
    assert isinstance(tower, StackedMatchedStateDenoiserTower)
    assert not hasattr(tower, "recursive_steps")
    assert not hasattr(tower, "z_latent")
    assert not hasattr(tower, "ctx_proj")
    assert hasattr(tower, "state")
    assert hasattr(tower, "state_ctx_proj")
    assert tuple(tower.state.shape) == (32, 16)
    assert tuple(tower.state_ctx_proj.weight.shape) == (16, 16)
    assert len(tower.layers) == 2
    # No weight sharing: every block is a distinct parameterized object.
    block_ids = [id(layer) for layer in tower.layers]
    assert len(set(block_ids)) == len(block_ids)
    names = dict(tower.named_parameters())
    assert not any(name.split(".")[0] in {"z_latent", "ctx_proj"} for name in names)


def test_arm_e_block_evaluations_match_arm_a_verified_by_hook_count() -> None:
    """Arm E's block-evaluation count must equal arm A's exactly (no
    recurrence added) -- verified with a real forward-hook call counter, not
    just a structural len(layers) claim."""
    vocab, d_model, n_layers, max_len = 23, 16, 2, 32
    e_tower = construct_arm_tower(
        "E", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len,
    )
    a_tower = construct_arm_tower(
        "A", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=2,
        max_len=max_len,
    )

    def _count_block_calls(tower: torch.nn.Module) -> int:
        calls = {"n": 0}

        def _hook(module: torch.nn.Module, inp: object, out: object) -> None:
            calls["n"] += 1

        handles = [layer.register_forward_hook(_hook) for layer in tower.layers]
        noisy = torch.randint(1, vocab, (2, 6))
        ctx = torch.randn(2, 3, d_model)
        tower(noisy, ctx, pad_id=0)
        for handle in handles:
            handle.remove()
        return calls["n"]

    e_calls = _count_block_calls(e_tower)
    a_calls = _count_block_calls(a_tower)
    assert e_calls == a_calls == n_layers == len(e_tower.layers)


def test_arm_e_parameter_count_matches_zstate_delta_formula_exactly() -> None:
    """Requirement #2: arm E's total-parameter delta over a same-n_layers
    arm A equals recursive_zstate_parameter_delta(d_model, max_len) exactly
    -- the same formula/target arm B's delta matches -- since its
    state/state_ctx_proj tensors are shape-matched to B's z_latent/ctx_proj."""
    vocab, d_model, n_layers, n_heads, max_len = 23, 32, 3, 2, 256
    stacked = construct_arm_tower(
        "A", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        max_len=max_len,
    )
    e_tower = construct_arm_tower(
        "E", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        max_len=max_len,
    )
    b_tower = construct_arm_tower(
        "B", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        max_len=max_len, recursive_steps=2, recursive_transition_layers=n_layers,
    )
    stacked_total = sum(p.numel() for p in stacked.parameters())
    e_total = sum(p.numel() for p in e_tower.parameters())
    b_total = sum(p.numel() for p in b_tower.parameters())
    formula = recursive_zstate_parameter_delta(d_model=d_model, max_len=max_len)

    assert e_total - stacked_total == formula
    assert b_total - stacked_total == formula
    # E and B therefore add the identical parameter delta over the same
    # stacked baseline, by construction -- even though E has no recurrence
    # and B's transition blocks are recursion_steps-shared, not independent.
    assert e_total - stacked_total == b_total - stacked_total


def test_arm_e_consumes_matched_state_and_receives_gradients() -> None:
    """Required test: arm E's added state/state_ctx_proj is not dead
    padding. Zeroing it changes the forward output, and both tensors receive
    real, nonzero gradient from a backward pass through the actual forward
    computation (not merely declared with the right shape)."""
    vocab, d_model, n_layers, n_heads, max_len = 23, 16, 2, 2, 32
    tower = construct_arm_tower(
        "E", vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        max_len=max_len,
    )
    noisy = torch.randint(1, vocab, (2, 6))
    ctx = torch.randn(2, 3, d_model)

    # --- gradient consumption ---
    tower.zero_grad(set_to_none=True)
    out = tower(noisy, ctx, pad_id=0)
    out.float().sum().backward()
    assert tower.state.grad is not None
    assert torch.any(tower.state.grad != 0)
    assert tower.state_ctx_proj.weight.grad is not None
    assert torch.any(tower.state_ctx_proj.weight.grad != 0)

    # --- ablation: zeroing the matched state changes the forward output ---
    with torch.no_grad():
        zeroed = construct_arm_tower(
            "E", vocab_size=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, max_len=max_len,
        )
        zeroed.load_state_dict(tower.state_dict())
        zeroed.state.zero_()
        zeroed.state_ctx_proj.weight.zero_()
        zeroed.state_ctx_proj.bias.zero_()
    tower.eval()
    zeroed.eval()
    with torch.no_grad():
        out_full = tower(noisy, ctx, pad_id=0)
        out_zeroed = zeroed(noisy, ctx, pad_id=0)
    assert not torch.allclose(out_full, out_zeroed)


def test_arm_e_denoiser_arch_wired_through_twotower_config_and_roundtrips(
    tmp_path: Path,
) -> None:
    """Requirement #1: E constructs through the canonical factory
    (TwoTowerConfig.denoiser_arch), trains one step, and round-trips a
    checkpoint, same as every other built arm."""
    records = [ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32, n_heads=2, context_layers=1, denoiser_layers=2,
            denoiser_arch="stacked_matched_state",  # type: ignore[arg-type]
            grammar_constrained=False, gen_steps=2, seed=0,
        ),
        device="cpu",
    )
    assert isinstance(model.denoiser, StackedMatchedStateDenoiserTower)
    assert len(model.denoiser.layers) == 2
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()

    ckpt = tmp_path / "arm_e.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert loaded.config.denoiser_arch == "stacked_matched_state"
    assert isinstance(loaded.denoiser, StackedMatchedStateDenoiserTower)
    assert len(loaded.denoiser.layers) == 2


def test_recursive_control_initialization_includes_arm_e_with_disjoint_seed() -> None:
    """The fairness contract covers arm E too: common tensors (tok/pos/the
    unshared transition blocks) match across A/B/C/D/E, E's state/
    state_ctx_proj are its own architecture-specific tensors, and its
    reserved arch_specific:stacked_matched_state seed is disjoint from every
    other arm's."""
    base_seed = 0
    vocab, d_model, n_layers, max_len = 23, 16, 2, 32
    arm_ids = ("A", "B", "C", "D", "E")
    towers = {}
    for arm_id in arm_ids:
        torch.manual_seed(derive_seed(base_seed, "model_initialization"))
        towers[arm_id] = construct_arm_tower(
            arm_id, vocab_size=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=2, max_len=max_len, recursive_steps=2,
            recursive_transition_layers=n_layers,
        )
    report = build_recursive_control_initialization(
        base_seed=base_seed,
        arm_towers=towers,
        arm_denoiser_arch={arm_id: ARM_DENOISER_ARCH[arm_id] for arm_id in arm_ids},
    )
    assert report.common_tensor_hashes_match_across_arms is True
    assert "tok.weight" in report.common_tensor_names
    assert any(name.startswith("layers.") for name in report.common_tensor_names)
    # E's state/state_ctx_proj are architecture-specific to E alone.
    e_specific = report.architecture_specific_tensor_names_and_shapes["E"]
    assert "state" in e_specific
    assert "state_ctx_proj.weight" in e_specific
    assert "state_ctx_proj.bias" in e_specific
    assert "z_latent" not in e_specific
    assert "ctx_proj.weight" not in e_specific
    seeds = report.architecture_specific_seeds
    assert seeds["E"] not in {v for k, v in seeds.items() if k != "E"}
    assert len(set(seeds.values())) == len(seeds)


def test_deferred_arm_ids_now_only_contains_h() -> None:
    """SLM-241 (RSC-A05) follow-up: E is no longer deferred -- only H
    remains, so this is the honest fail-closed set construct_arm_tower
    raises NotImplementedError for."""
    assert DEFERRED_ARM_IDS == ("H",)
    assert "E" in BUILT_ARM_IDS
    assert set(BUILT_ARM_IDS) | set(DEFERRED_ARM_IDS) == set(ALL_ARM_IDS)
