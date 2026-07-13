"""Lossless BF16 fusible exponent-codebook compression.

Reference layout from https://brianbell-x.github.io/weight-compression/
(candidate 0009 “regroup”): per-tensor codebook of the top-K sign+exponent
symbols, fixed-width 4-bit index, raw 7-bit mantissa, in-order escape stream
for rare symbols. Bit-exact vs BF16; no quality tradeoff vs the BF16 view.

Kernel boundary: this module only encodes/decodes tensors for storage and
bundle export. Fused in-register matmul kernels stay outside this repo
(Cactus / NEON / Triton). The PyTorch path decompresses to dense float tensors.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"SLMWC001"
K_DEFAULT = 15  # +1 escape → 4-bit index
LAYOUT_REGROUP = "regroup_k15"
LAYOUT_BYTESPLIT = "bytesplit_k15"


@dataclass
class TensorCodecStats:
    name: str
    n_weights: int
    n_escape: int
    bf16_bytes: int
    compressed_bytes: int
    bits_per_weight: float
    reduction_pct: float
    layout: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompressedTensor:
    name: str
    shape: tuple[int, ...]
    layout: str
    codebook: np.ndarray  # uint16[K] sign+exp (regroup) or high-byte (bytesplit)
    index: np.ndarray  # uint8[n] values 0..K (K = escape)
    mantissa: np.ndarray  # uint8[n] 7-bit (regroup) or low-byte (bytesplit)
    escape: np.ndarray  # uint16[n_esc]
    source_dtype: str = "bfloat16"

    @property
    def n_weights(self) -> int:
        return int(self.index.size)


def _to_bf16_u16(array: np.ndarray) -> np.ndarray:
    """Convert float array to BF16 bit patterns as uint16 (torch RNE)."""
    import torch

    f32 = np.asarray(array, dtype=np.float32).reshape(-1)
    # Match torch.bfloat16 rounding (not float32 top-16 truncation).
    bf = torch.from_numpy(np.ascontiguousarray(f32)).to(torch.bfloat16)
    return bf.view(torch.uint16).cpu().numpy().astype(np.uint16, copy=False)


def _from_bf16_u16(u16: np.ndarray, *, to_float32: bool = True) -> np.ndarray:
    import torch

    bits = torch.from_numpy(np.ascontiguousarray(u16, dtype=np.uint16))
    bf = bits.view(torch.bfloat16)
    if to_float32:
        return bf.float().cpu().numpy()
    return bf.cpu().numpy()


def encode_bf16_u16_regroup(
    u16: np.ndarray,
    *,
    k: int = K_DEFAULT,
    name: str = "",
    shape: tuple[int, ...] | None = None,
) -> CompressedTensor:
    """Encode BF16 bits with regroup layout (headline ~11.3 b/w)."""
    u = np.asarray(u16, dtype=np.uint16).reshape(-1)
    sign = (u >> 15).astype(np.uint32)
    exp = ((u >> 7) & 0xFF).astype(np.uint32)
    mant = (u & 0x7F).astype(np.uint8)
    sym = ((sign << 8) | exp).astype(np.int64)
    hist = np.bincount(sym, minlength=512)
    top = np.argsort(hist)[::-1][:k].astype(np.uint16)
    code_map = np.full(512, k, dtype=np.uint8)
    code_map[top] = np.arange(k, dtype=np.uint8)
    idx = code_map[sym]
    esc = idx == k
    escape = sym[esc].astype(np.uint16)
    return CompressedTensor(
        name=name,
        shape=tuple(shape) if shape is not None else (int(u.size),),
        layout=LAYOUT_REGROUP,
        codebook=top,
        index=idx.astype(np.uint8),
        mantissa=mant,
        escape=escape,
        source_dtype="bfloat16",
    )


def decode_bf16_u16_regroup(ct: CompressedTensor) -> np.ndarray:
    k = int(ct.codebook.size)
    cb = np.zeros(k + 1, dtype=np.uint16)
    cb[:k] = ct.codebook
    sym = cb[ct.index].astype(np.uint32)
    esc = ct.index == k
    if int(esc.sum()) != int(ct.escape.size):
        raise ValueError(
            f"escape length mismatch: index escapes={int(esc.sum())} stream={ct.escape.size}"
        )
    sym[esc] = ct.escape.astype(np.uint32)
    sign = (sym >> 8).astype(np.uint16)
    exp = (sym & 0xFF).astype(np.uint16)
    mant = ct.mantissa.astype(np.uint16) & 0x7F
    return ((sign << 15) | (exp << 7) | mant).astype(np.uint16)


def encode_bf16_u16_bytesplit(
    u16: np.ndarray,
    *,
    k: int = K_DEFAULT,
    name: str = "",
    shape: tuple[int, ...] | None = None,
) -> CompressedTensor:
    """Encode BF16 bits with byte-split layout (GPU-validated variant)."""
    raw = np.asarray(u16, dtype=np.uint16).reshape(-1).view(np.uint8)
    # little-endian: low, high
    low, high = raw[0::2].copy(), raw[1::2].copy()
    hist = np.bincount(high, minlength=256)
    top = np.argsort(hist)[::-1][:k].astype(np.uint8)
    code_map = np.full(256, k, dtype=np.uint8)
    code_map[top] = np.arange(k, dtype=np.uint8)
    idx = code_map[high]
    esc = idx == k
    escape = high[esc].astype(np.uint16)
    return CompressedTensor(
        name=name,
        shape=tuple(shape) if shape is not None else (int(idx.size),),
        layout=LAYOUT_BYTESPLIT,
        codebook=top.astype(np.uint16),
        index=idx.astype(np.uint8),
        mantissa=low,  # full low byte
        escape=escape,
        source_dtype="bfloat16",
    )


def decode_bf16_u16_bytesplit(ct: CompressedTensor) -> np.ndarray:
    k = int(ct.codebook.size)
    cb = np.zeros(k + 1, dtype=np.uint8)
    cb[:k] = ct.codebook.astype(np.uint8)
    high = cb[ct.index]
    esc = ct.index == k
    if int(esc.sum()) != int(ct.escape.size):
        raise ValueError("escape length mismatch (bytesplit)")
    high[esc] = ct.escape.astype(np.uint8)
    out = np.empty(ct.index.size * 2, dtype=np.uint8)
    out[0::2] = ct.mantissa
    out[1::2] = high
    return out.view(np.uint16)


def encode_tensor(
    array: np.ndarray,
    *,
    name: str = "",
    layout: str = LAYOUT_REGROUP,
    k: int = K_DEFAULT,
) -> CompressedTensor:
    shape = tuple(int(x) for x in np.asarray(array).shape)
    u16 = _to_bf16_u16(array)
    if layout == LAYOUT_BYTESPLIT:
        return encode_bf16_u16_bytesplit(u16, k=k, name=name, shape=shape)
    if layout != LAYOUT_REGROUP:
        raise ValueError(f"unknown layout {layout!r}")
    return encode_bf16_u16_regroup(u16, k=k, name=name, shape=shape)


def decode_tensor(ct: CompressedTensor, *, to_float32: bool = True) -> np.ndarray:
    if ct.layout == LAYOUT_BYTESPLIT:
        u16 = decode_bf16_u16_bytesplit(ct)
    elif ct.layout == LAYOUT_REGROUP:
        u16 = decode_bf16_u16_regroup(ct)
    else:
        raise ValueError(f"unknown layout {ct.layout!r}")
    arr = _from_bf16_u16(u16, to_float32=to_float32)
    return arr.reshape(ct.shape)


def compressed_payload_bytes(ct: CompressedTensor) -> int:
    """Packed size estimate matching the fusible bit budget (+ byte alignment)."""
    n = ct.n_weights
    n_esc = int(ct.escape.size)
    k = int(ct.codebook.size)
    if ct.layout == LAYOUT_REGROUP:
        # 4-bit idx + 7-bit mant + 9-bit escapes + 9-bit codebook
        bits = n * 4 + n * 7 + n_esc * 9 + k * 9
    else:
        bits = n * 4 + n * 8 + n_esc * 8 + k * 8
    # store escape count + shape header overhead (~64 B)
    return int(math.ceil(bits / 8)) + 64


def tensor_stats(ct: CompressedTensor) -> TensorCodecStats:
    n = ct.n_weights
    bf16_bytes = n * 2
    compressed = compressed_payload_bytes(ct)
    bpw = (compressed * 8) / max(1, n)
    return TensorCodecStats(
        name=ct.name,
        n_weights=n,
        n_escape=int(ct.escape.size),
        bf16_bytes=bf16_bytes,
        compressed_bytes=compressed,
        bits_per_weight=round(bpw, 4),
        reduction_pct=round(100.0 * (1.0 - compressed / max(1, bf16_bytes)), 3),
        layout=ct.layout,
    )


def roundtrip_ok(array: np.ndarray, *, layout: str = LAYOUT_REGROUP) -> bool:
    """True if encode→decode matches the BF16 view of `array` bit-exactly."""
    u16 = _to_bf16_u16(array)
    ct = encode_tensor(array, layout=layout)
    if ct.layout == LAYOUT_REGROUP:
        rec = decode_bf16_u16_regroup(ct)
    else:
        rec = decode_bf16_u16_bytesplit(ct)
    return bool(np.array_equal(rec, u16))


def compress_state_dict(
    state_dict: dict[str, Any],
    *,
    layout: str = LAYOUT_REGROUP,
    min_numel: int = 64,
) -> tuple[dict[str, Any], list[TensorCodecStats]]:
    """
    Compress floating tensors in a torch state_dict.

    Non-float / tiny tensors are kept dense. Returns a JSON-serializable
    payload plus per-tensor stats.
    """
    import torch

    tensors: list[dict[str, Any]] = []
    dense: dict[str, Any] = {}
    stats: list[TensorCodecStats] = []

    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            dense[name] = value
            continue
        if not torch.is_floating_point(value) or value.numel() < min_numel:
            cpu = value.detach().cpu().contiguous()
            entry: dict[str, Any] = {
                "dtype": str(value.dtype).replace("torch.", ""),
                "shape": list(value.shape),
                "data": None,
                "raw_bytes": None,
            }
            if cpu.numel() <= 4096:
                # Preserve exact dtype (do not cast ints through float).
                entry["data"] = cpu.numpy().tolist()
            else:
                arr = cpu.numpy()
                entry["raw_bytes"] = arr.tobytes().hex()
                entry["numpy_dtype"] = str(arr.dtype)
            dense[name] = entry
            continue
        arr = value.detach().cpu().float().numpy()
        ct = encode_tensor(arr, name=name, layout=layout)
        if not roundtrip_ok(arr, layout=layout):
            raise RuntimeError(f"bit-exact round-trip failed for {name}")
        st = tensor_stats(ct)
        stats.append(st)
        tensors.append(
            {
                "name": name,
                "shape": list(ct.shape),
                "layout": ct.layout,
                "source_dtype": str(value.dtype).replace("torch.", ""),
                "codebook": ct.codebook.tolist(),
                "index": ct.index.tobytes().hex(),
                "mantissa": ct.mantissa.tobytes().hex(),
                "escape": ct.escape.tolist(),
                "stats": st.to_dict(),
            }
        )

    payload = {
        "format": "slm-training-bf16-exponent-codebook-v1",
        "layout": layout,
        "k": K_DEFAULT,
        "reference": "https://brianbell-x.github.io/weight-compression/",
        "kernel_separate": True,
        "note": (
            "Lossless vs BF16 view of weights. PyTorch loads decompress to float32. "
            "Fused kernels are not vendored here."
        ),
        "tensors": tensors,
        "dense": dense,
    }
    return payload, stats


def decompress_state_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Decode compressed payload back to a torch float32 state_dict."""
    import torch

    out: dict[str, Any] = {}
    for item in payload.get("tensors") or []:
        ct = CompressedTensor(
            name=item["name"],
            shape=tuple(item["shape"]),
            layout=item["layout"],
            codebook=np.asarray(item["codebook"], dtype=np.uint16),
            index=np.frombuffer(bytes.fromhex(item["index"]), dtype=np.uint8).copy(),
            mantissa=np.frombuffer(bytes.fromhex(item["mantissa"]), dtype=np.uint8).copy(),
            escape=np.asarray(item["escape"], dtype=np.uint16),
            source_dtype=item.get("source_dtype") or "bfloat16",
        )
        arr = decode_tensor(ct, to_float32=True)
        out[item["name"]] = torch.from_numpy(np.ascontiguousarray(arr))
    for name, meta in (payload.get("dense") or {}).items():
        if not isinstance(meta, dict):
            out[name] = meta
            continue
        if meta.get("data") is not None:
            t = torch.tensor(meta["data"])
            dtype_name = meta.get("dtype") or "float32"
            dtype = getattr(torch, dtype_name, torch.float32)
            out[name] = t.to(dtype)
        elif meta.get("raw_bytes"):
            dt = np.dtype(meta.get("numpy_dtype") or "float32")
            arr = np.frombuffer(bytes.fromhex(meta["raw_bytes"]), dtype=dt).reshape(
                meta["shape"]
            )
            out[name] = torch.from_numpy(np.ascontiguousarray(arr.copy()))
    return out


def summarize_stats(stats: list[TensorCodecStats]) -> dict[str, Any]:
    bf16 = sum(s.bf16_bytes for s in stats)
    comp = sum(s.compressed_bytes for s in stats)
    n = sum(s.n_weights for s in stats)
    esc = sum(s.n_escape for s in stats)
    return {
        "n_tensors": len(stats),
        "n_weights": n,
        "bf16_bytes": bf16,
        "compressed_bytes": comp,
        "bits_per_weight": round((comp * 8) / max(1, n), 4),
        "reduction_pct": round(100.0 * (1.0 - comp / max(1, bf16)), 3),
        "escape_pct": round(100.0 * esc / max(1, n), 4),
        "layout": stats[0].layout if stats else None,
    }


def write_compressed_checkpoint(
    checkpoint: Path,
    out_path: Path,
    *,
    layout: str = LAYOUT_REGROUP,
) -> dict[str, Any]:
    """Compress TwoTower (or generic) torch checkpoint weights to JSON payload."""
    import torch

    checkpoint = Path(checkpoint)
    out_path = Path(out_path)
    payload_in = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload_in.get("state_dict") or payload_in
    compressed, stats = compress_state_dict(state, layout=layout)
    summary = summarize_stats(stats)
    out = {
        "kind": payload_in.get("kind"),
        "config": payload_in.get("config"),
        "gen_len": payload_in.get("gen_len"),
        "compression": summary,
        "weights": compressed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    # Also write a compact sidecar summary.
    (out_path.with_suffix(".stats.json")).write_text(
        json.dumps({"checkpoint": str(checkpoint), **summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
