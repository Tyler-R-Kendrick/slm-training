"""Lossless BF16 codebook compression (brianbell-x candidate 0009)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from slm_training.compression import (
    LAYOUT_BYTESPLIT,
    LAYOUT_REGROUP,
    compress_state_dict,
    decode_bf16_u16_bytesplit,
    decode_bf16_u16_regroup,
    decompress_state_dict,
    encode_bf16_u16_bytesplit,
    encode_bf16_u16_regroup,
    summarize_stats,
    write_compressed_checkpoint,
)


def test_encode_decode_bitexact_random() -> None:
    rng = np.random.default_rng(0)
    words = rng.integers(0, 2**16, size=50_000, dtype=np.uint16)
    for encode, decode in (
        (encode_bf16_u16_regroup, decode_bf16_u16_regroup),
        (encode_bf16_u16_bytesplit, decode_bf16_u16_bytesplit),
    ):
        ct = encode(words, name="t", shape=(words.size,))
        assert np.array_equal(decode(ct), words)


def test_encode_decode_common_and_escapes() -> None:
    common = np.array(
        [0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000, 0x3F00],
        dtype=np.uint16,
    )
    rng = np.random.default_rng(1)
    words = np.concatenate(
        [np.tile(common, 1000), rng.integers(0, 2**16, size=2000, dtype=np.uint16)]
    )
    ct = encode_bf16_u16_regroup(words, name="mix", shape=(words.size,))
    assert np.array_equal(decode_bf16_u16_regroup(ct), words)


def test_state_dict_roundtrip_bf16_view() -> None:
    sd = {
        "linear.weight": torch.randn(32, 16, dtype=torch.float32),
        "linear.bias": torch.randn(32, dtype=torch.float32),
        "emb.weight": torch.randn(64, 8, dtype=torch.float32),
        "steps": torch.tensor(7, dtype=torch.int64),
    }
    payload, stats = compress_state_dict(sd, layout=LAYOUT_REGROUP, min_numel=16)
    assert stats
    summary = summarize_stats(stats)
    assert summary["reduction_pct"] >= 0.0
    restored = decompress_state_dict(payload)
    assert int(restored["steps"].item()) == 7
    for key, tensor in sd.items():
        if not tensor.is_floating_point():
            continue
        assert torch.equal(
            tensor.detach().cpu().to(torch.bfloat16),
            restored[key].detach().cpu().to(torch.bfloat16),
        ), key


def test_write_compressed_checkpoint(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "kind": "twotower",
            "config": {"model": {"kind": "test"}},
            "gen_len": 32,
            "state_dict": {
                "w": torch.randn(64, 64),
                "b": torch.zeros(64),
                "steps": torch.tensor(3, dtype=torch.int64),
            },
        },
        ckpt_path,
    )
    out = tmp_path / "model.pt.wc.json"
    summary = write_compressed_checkpoint(ckpt_path, out, layout=LAYOUT_REGROUP)
    assert out.is_file()
    assert summary["reduction_pct"] >= 0.0
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["weights"]["format"] == "slm-training-bf16-exponent-codebook-v1"
    assert loaded["weights"]["reference"].endswith("weight-compression/")
    restored = decompress_state_dict(loaded["weights"])
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert torch.equal(
        raw["state_dict"]["w"].to(torch.bfloat16),
        restored["w"].to(torch.bfloat16),
    )
    assert int(restored["steps"].item()) == 3


def test_bytesplit_layout_roundtrip() -> None:
    sd = {"w": torch.randn(48, 48)}
    payload, _ = compress_state_dict(sd, layout=LAYOUT_BYTESPLIT, min_numel=16)
    restored = decompress_state_dict(payload)
    assert torch.equal(sd["w"].to(torch.bfloat16), restored["w"].to(torch.bfloat16))
