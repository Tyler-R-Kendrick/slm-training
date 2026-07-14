"""Tests for accelerator detection and parallel unmask policies."""

from __future__ import annotations

import torch

from slm_training.runtime.accel import detect_device, maybe_compile
from slm_training.models.parallel_decode import select_unmask_indices


def test_detect_device_cpu_fallback() -> None:
    info = detect_device("cpu")
    assert info.device == "cpu"
    assert info.backend == "cpu"
    assert info.num_threads >= 1


def test_detect_device_auto() -> None:
    info = detect_device("auto")
    assert info.backend in {"cpu", "cuda", "npu", "dml"}
    assert info.device


def test_select_unmask_topk_and_adaptive() -> None:
    conf = torch.tensor([[0.9, 0.8, 0.1, 0.7, 0.6]])
    unknown = torch.tensor([[True, True, True, True, True]])
    topk = select_unmask_indices(conf, unknown, step=0, steps=4, mode="topk")
    assert len(topk) >= 1
    adaptive = select_unmask_indices(
        conf, unknown, step=0, steps=4, mode="adaptive", min_spacing=2
    )
    assert len(adaptive) >= 1
    # Spacing: chosen positions should not be adjacent.
    length = 5
    positions = sorted(i % length for i in adaptive)
    for a, b in zip(positions, positions[1:]):
        assert b - a >= 2


def test_maybe_compile_smoke() -> None:
    mod = torch.nn.Linear(8, 8)
    out = maybe_compile(mod, enabled=False)
    assert out is mod


def test_detect_device_explicit_directml(monkeypatch) -> None:
    import sys
    import types

    directml = types.SimpleNamespace(device=lambda: "privateuseone:0")
    monkeypatch.setitem(sys.modules, "torch_directml", directml)
    info = detect_device("directml")
    assert info.device == "privateuseone:0"
    assert info.backend == "dml"
    assert info.note == "torch-directml"


def test_detect_device_resolved_directml() -> None:
    info = detect_device("privateuseone:0")
    assert info.device == "privateuseone:0"
    assert info.backend == "dml"
    assert info.compile is False
