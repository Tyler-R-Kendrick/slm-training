"""Generic trainer/evaluator for LatentCodec-based bottleneck models.

This is the CAP2-02 counterpart to :func:`train_kary_bottleneck`.  It can train
any codec that exposes the :class:`LatentCodec` interface plus a decoder that
maps codec outputs to target logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.models.latent_codec import LatentCodec


class LatentCodecModel(nn.Module):
    """Wrap a LatentCodec and a decoder for end-to-end training.

    The decoder receives only the codec's decode_input output, preserving the
    no-bypass contract during hard evaluation.
    """

    def __init__(self, codec: LatentCodec, num_targets: int) -> None:
        super().__init__()
        self.codec = codec
        self.num_targets = num_targets
        # Decoder input size depends on codec family.
        self.decoder = nn.Linear(self._decoder_input_dim(), num_targets)

    def _decoder_input_dim(self) -> int:
        spec = self.codec.spec
        if spec.name == "learned_vq":
            # VQ decoder input is the latent_dim; spec.levels is (codebook_size,).
            return self.codec.config.latent_dim  # type: ignore[attr-defined]
        if spec.name == "continuous":
            return self.codec.config.latent_dim  # type: ignore[attr-defined]
        if spec.name == "binary_lfq":
            return self.codec.config.d  # type: ignore[attr-defined]
        # Uniform scalar and mixed-radix FSQ use concatenated one-hots.
        return sum(spec.levels)

    def forward(self, x: torch.Tensor, *, hard: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.codec.encode(x, hard=hard)
        decoder_input = self.codec.decode_input(encoding)
        return self.decoder(decoder_input), encoding.code_index


def train_latent_codec(
    model: LatentCodecModel,
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int = 800,
    lr: float = 2e-2,
    log_every: int = 0,
) -> dict[str, float]:
    """Train a latent-codec model with soft codes.

    The codec's aux_loss terms are added to the cross-entropy objective.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad()
        encoding = model.codec.encode(states, hard=False)
        decoder_input = model.codec.decode_input(encoding)
        out = model.decoder(decoder_input)
        loss = F.cross_entropy(out, targets)
        for aux in encoding.aux_loss.values():
            loss = loss + aux
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{steps} loss={loss.item():.4f}")
    return {"final_loss": losses[-1], "steps": steps, "lr": lr}


def evaluate_latent_codec(
    model: LatentCodecModel,
    states: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Evaluate with hard codes."""
    model.eval()
    with torch.no_grad():
        out, _ = model(states, hard=True)
        pred = out.argmax(dim=-1)
        exact_acc = (pred == targets).float().mean().item()
        # Collect diagnostics from encodings.
        encodings = [model.codec.encode(states[i : i + 1], hard=True) for i in range(states.shape[0])]
        diag = model.codec.diagnostics(encodings)
        return {
            "exact_reconstruction_rate": exact_acc,
            "occupied_codewords": diag.occupied_codes,
            "capacity": max(1, int(2 ** model.codec.nominal_bits())),
            "empirical_entropy_bits": diag.empirical_entropy_bits,
            "utilization": diag.utilization,
        }
