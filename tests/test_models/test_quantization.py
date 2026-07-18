"""Tests for reference low-bit quantizers and physical-cost ledger (CAP3-01)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.quantization import (
    QuantFormat,
    binary_format,
    binary_plus_mask_format,
    bf16_format,
    build_model_ledger,
    compute_tensor_cost,
    convert_twotower,
    fake_quantize_weight,
    fp16_format,
    int4_format,
    int8_format,
    learned_four_level_zero_format,
    physical_cost_evidence,
    symmetric_four_level_format,
    ternary_format,
)
from slm_training.models.quantization.convert import QuantizationPolicy
from slm_training.models.quantization.formats import KERNEL_REGISTRY


def test_format_registry_has_required_formats() -> None:
    required = {
        "fp16",
        "bf16",
        "int8",
        "int4",
        "binary",
        "ternary",
        "symmetric_four_level",
        "learned_four_level_zero",
        "binary_plus_mask",
        "residual_binary_plane",
        "residual_ternary_plane",
    }
    assert required <= set(KERNEL_REGISTRY)


def test_int8_scale_and_reconstruction() -> None:
    w = torch.linspace(-1.5, 1.5, 5)
    q, scale, zp = fake_quantize_weight(w, int8_format())
    assert scale.item() == pytest.approx(1.5 / 127, abs=1e-6)
    assert zp is not None and torch.allclose(zp, torch.zeros_like(zp))
    assert q[0].item() == pytest.approx(-1.5, abs=1e-5)
    assert q[-1].item() == pytest.approx(1.5, abs=1e-5)
    assert (q - w).abs().max() <= scale.item() * 0.5


def test_int4_level_count_and_grid() -> None:
    fmt = int4_format()
    assert fmt.level_count == 15
    assert fmt.weight_levels[0] == -7.0
    assert fmt.weight_levels[-1] == 7.0
    w = torch.tensor([-7.0, -3.0, 0.0, 3.0, 7.0])
    q, scale, _ = fake_quantize_weight(w, fmt)
    assert scale.item() == pytest.approx(1.0, abs=1e-6)
    assert q.tolist() == pytest.approx(w.tolist(), abs=1e-5)


def test_binary_maps_to_plus_minus_one() -> None:
    w = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
    q, scale, _ = fake_quantize_weight(w, binary_format())
    assert scale.item() == pytest.approx(2.0, abs=1e-6)
    assert set(q.unique().tolist()) == {-2.0, 2.0}


def test_ternary_preserves_zero_threshold() -> None:
    fmt = ternary_format()
    assert fmt.physical_slot_bits == 2
    assert fmt.nominal_symbol_bits == pytest.approx(math.log2(3), abs=1e-9)
    w = torch.tensor([-1.0, -0.4, 0.0, 0.4, 1.0])
    q, scale, _ = fake_quantize_weight(w, fmt)
    assert scale.item() == pytest.approx(1.0, abs=1e-6)
    expected = [-1.0, 0.0, 0.0, 0.0, 1.0]
    assert q.tolist() == pytest.approx(expected, abs=1e-5)


def test_symmetric_four_level_grid() -> None:
    fmt = symmetric_four_level_format()
    assert fmt.level_count == 4
    assert fmt.physical_slot_bits == 2
    w = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    q, scale, _ = fake_quantize_weight(w, fmt)
    assert scale.item() == pytest.approx(1.0, abs=1e-6)
    assert q.tolist() == pytest.approx(w.tolist(), abs=1e-5)


def test_learned_four_level_contains_ternary_feasible_set() -> None:
    fmt = learned_four_level_zero_format(levels=(-1.0, 0.0, 1.0, 2.0))
    assert fmt.supports_exact_zero
    assert 0.0 in fmt.learned_levels
    ternary_set = {-1.0, 0.0, 1.0}
    assert ternary_set <= set(fmt.learned_levels)
    w = torch.tensor([-1.0, -0.1, 0.0, 0.9, 2.0])
    q, scale, _ = fake_quantize_weight(w, fmt)
    assert scale.item() == pytest.approx(1.0, abs=1e-6)
    assert set(q.unique().tolist()) <= set(fmt.learned_levels)


def test_learned_four_level_requires_zero() -> None:
    with pytest.raises(ValueError):
        learned_four_level_zero_format(levels=(-1.0, 0.5, 1.0, 2.0))


def test_binary_plus_mask_storage_is_two_bits() -> None:
    fmt = binary_plus_mask_format()
    assert fmt.physical_slot_bits == 2
    assert fmt.weight_levels == (-1.0, 0.0, 1.0)
    w = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
    q, scale, _ = fake_quantize_weight(w, fmt)
    assert scale.item() == pytest.approx(1.0, abs=1e-6)
    assert set(q.tolist()) == {-1.0, 0.0, 1.0}


def test_fp16_bf16_are_identity() -> None:
    w = torch.randn(8, 8)
    for fmt in [fp16_format(), bf16_format()]:
        q, scale, zp = fake_quantize_weight(w, fmt)
        assert torch.equal(q, w)
        assert scale.item() == pytest.approx(1.0, abs=1e-6)
        assert zp is None


def test_groupwise_scale_overhead() -> None:
    fmt = int8_format(group_size=128)
    w = torch.randn(64, 256)
    q, scale, _ = fake_quantize_weight(w, fmt, group_size=128)
    num_groups = math.ceil(w.numel() / 128)
    assert scale.numel() == num_groups
    # FP16 scale overhead: 2 bytes per group.
    assert num_groups * 2 == scale.numel() * 2


def test_groupwise_packing_padding_accounted() -> None:
    fmt = int4_format(group_size=128)
    w = torch.randn(1, 129)
    cost = compute_tensor_cost("w", w, fmt, group_size=128)
    groups = math.ceil(129 / 128)  # 2
    bytes_per_group = math.ceil(128 * 4 / 8)  # 64
    assert cost.physical_weight_bytes == groups * bytes_per_group


def test_groupwise_quantization_approximates_per_tensor_for_small_tensors() -> None:
    fmt = int8_format(group_size=128)
    w = torch.randn(2, 64)
    q_group, scale_group, _ = fake_quantize_weight(w, fmt, group_size=64)
    q_tensor, scale_tensor, _ = fake_quantize_weight(w, fmt, group_size=None)
    assert scale_group.numel() == 2
    assert scale_tensor.numel() == 1
    # Groupwise should not be dramatically worse than per-tensor on this data.
    assert (q_group - w).abs().max() <= 4 * (q_tensor - w).abs().max() + 1e-5


def test_tied_weight_conversion_fails_by_default() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(16, 16, bias=False)
            self.b = torch.nn.Linear(16, 16, bias=False)
            self.b.weight = self.a.weight  # tied

    model = Toy()
    policy = QuantizationPolicy(default_format=binary_format())
    with pytest.raises(ValueError, match="tied"):
        convert_twotower(model, policy, fail_on_tied=True)


def test_conversion_is_reversible_with_saved_state() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)

    model = Toy()
    original = {n: p.clone() for n, p in model.named_parameters()}
    policy = QuantizationPolicy(default_format=ternary_format())
    converted, records = convert_twotower(model, policy, fail_on_tied=False, in_place=True)
    assert converted is model
    assert any(r.format_id == "ternary" for r in records)
    assert not torch.equal(model.lin.weight, original["lin.weight"])
    from slm_training.models.quantization.convert import restore_original_weights

    restore_original_weights(model, original)
    assert torch.equal(model.lin.weight, original["lin.weight"])


def test_default_off_quant_format() -> None:
    from slm_training.models import TwoTowerConfig

    cfg = TwoTowerConfig()
    assert cfg.quant_format is None


def test_whole_model_ledger_includes_unquantized_tensors() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(10, 16)
            self.norm = torch.nn.LayerNorm(16)
            self.lin = torch.nn.Linear(16, 16)
            self.head = torch.nn.Linear(16, 10)

    model = Toy()
    ledger = build_model_ledger(model, {}, default_format=binary_format())
    # Embeddings / norms / heads should be counted as unquantized.
    assert ledger.unquantized_bytes > 0
    # At least one binary tensor should appear.
    assert "binary" in ledger.formats
    assert ledger.total() == sum(report.total_bytes for report in ledger.formats.values())
    assert ledger.checkpoint_bytes == ledger.total() + ledger.alignment_overhead_bytes
    # Resident memory includes activations + KV + scratch.
    assert ledger.resident_bytes > ledger.total()


def test_empirical_entropy_does_not_replace_physical_bytes() -> None:
    fmt = ternary_format()
    w = torch.tensor([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    cost = compute_tensor_cost("w", w, fmt)
    assert cost.empirical_entropy_bits is not None
    assert cost.empirical_entropy_bits <= cost.ideal_bits
    # Physical bytes include per-group packing padding, not entropy-coded size.
    groups = math.ceil(w.numel() / fmt.group_size)
    bytes_per_group = math.ceil(fmt.group_size * fmt.physical_slot_bits / 8)
    assert cost.physical_weight_bytes == groups * bytes_per_group


def test_missing_kernel_reported_explicitly() -> None:
    cap = KERNEL_REGISTRY["ternary"]
    assert cap.reference_pytorch
    assert not cap.cuda
    assert not cap.cpu_optimized
    assert "reference path" in cap.notes.lower()


def test_cap0_04_physical_cost_evidence() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)

    model = Toy()
    ledger = build_model_ledger(model, {}, default_format=int8_format())
    evidence = physical_cost_evidence(
        ledger,
        grammar_hash="abc",
        dataset_ids=("ds1",),
        checkpoint_ids=("ckpt1",),
    )
    assert evidence.evidence_kind.value == "estimated"
    assert evidence.sample_count >= 1
    assert "total_bytes" in evidence.coverage


def test_format_descriptor_rejects_invalid_bits() -> None:
    with pytest.raises(ValueError):
        QuantFormat(
            format_id="bad",
            weight_levels=(-1.0, 1.0),
            nominal_symbol_bits=-1.0,
            physical_slot_bits=1,
            group_size=128,
            scale_dtype="fp16",
            zero_point_dtype=None,
            bias_dtype="fp16",
            activation_dtype="fp16",
            accumulation_dtype="fp32",
            packing_layout="x",
            supports_exact_zero=False,
            entropy_coding=None,
            kernel_id=None,
        )
