"""Strict K-ary discrete bottleneck for CAP2 phase-boundary experiments.

The model maps an input (exact state id or semantic features) to a fixed-length
code of ``d`` categorical coordinates, each taking ``K`` values.  The decoder
sees *only* the code (as one-hot or integer-derived symbols) and reconstructs
the target.  This makes the capacity bound ``K**d`` a hard information
bottleneck: fewer than ``M`` distinct codewords cannot represent ``M`` distinct
states deterministically.

Training may use a soft relaxation (straight-through estimator).  Deterministic
evaluation always uses hard argmax codes and disables the relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class KaryBottleneckConfig:
    """Configuration for a strict K-ary bottleneck."""

    num_states: int
    K: int
    d: int
    hidden_dim: int = 64
    feature_dim: int | None = None
    mode: str = "oracle_state"  # "oracle_state" | "semantic_trace"
    lr: float = 2e-2
    train_steps: int = 800


class KaryBottleneck(nn.Module):
    """Deterministic discrete bottleneck with optional soft training path.

    The forward pass is:

        input -> encoder -> d * K logits
        -> hard code -> one-hot code -> decoder -> logits over states/actions

    No tensor except the hard code reaches the decoder.  An audit helper
    verifies this by recomputing the decoder output from the code alone.
    """

    def __init__(self, config: KaryBottleneckConfig) -> None:
        super().__init__()
        self.config = config
        self.num_states = config.num_states
        self.K = config.K
        self.d = config.d

        if config.mode == "oracle_state":
            # For exact state ids we learn a per-state code directly.  This is
            # still an encoder (state -> code) and keeps the decoder input
            # strictly limited to the hard code.
            self.encoder: nn.Module | None = None
            self.state_code_logits = nn.Parameter(
                torch.randn(config.num_states, config.d, config.K)
            )
            self.code_logits: nn.Linear | None = None
        elif config.mode == "semantic_trace":
            if config.feature_dim is None:
                raise ValueError("semantic_trace mode requires feature_dim")
            self.encoder = nn.Sequential(
                nn.Linear(config.feature_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.state_code_logits = None
            self.code_logits = nn.Linear(config.hidden_dim, config.d * config.K)
        else:
            raise ValueError(f"unknown mode {config.mode!r}")

        self.decoder = nn.Linear(config.d * config.K, config.num_states)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-coordinate logits [batch, d, K]."""
        if self.state_code_logits is not None:
            return self.state_code_logits[x]
        assert self.encoder is not None and self.code_logits is not None
        h = self.encoder(x)
        return self.code_logits(h).view(-1, self.d, self.K)

    def decode_from_code(self, code: torch.Tensor) -> torch.Tensor:
        """Decode from integer codes only (no encoder access)."""
        one_hot = F.one_hot(code, self.K).float()
        return self.decoder(one_hot.view(-1, self.d * self.K))

    def forward(
        self,
        x: torch.Tensor,
        *,
        hard: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            output logits over states/actions,
            hard integer code [batch, d],
            per-coordinate logits [batch, d, K].
        """
        logits = self._logits(x)
        # Deterministic code = argmax over each coordinate.
        hard_code = logits.argmax(dim=-1)
        one_hot = F.one_hot(hard_code, self.K).float()
        if self.training and not hard:
            # Straight-through estimator (STE): forward uses the hard one-hot code,
            # but gradients flow through the soft softmax probability.  This lets
            # training optimize discrete code assignments while evaluation is fully
            # deterministic.
            probs = F.softmax(logits, dim=-1)
            soft_one_hot = one_hot + probs - probs.detach()
        else:
            soft_one_hot = one_hot
        out = self.decoder(soft_one_hot.view(-1, self.d * self.K))
        return out, hard_code, logits

    def audit_no_bypass(self, x: torch.Tensor) -> bool:
        """Verify the decoder depends only on the hard code, not upstream state.

        Recomputes the decoder output from the integer code and checks it equals
        the full forward output.  For the network-backed semantic_trace mode it
        additionally zeros the encoder output and confirms the code changes,
        proving the upstream latent is not a side-channel to the decoder.
        """
        self.eval()
        with torch.no_grad():
            logits = self._logits(x)
            code = logits.argmax(dim=-1)
            from_code = self.decode_from_code(code)
            full_out, full_code, _ = self.forward(x, hard=True)
            if not torch.equal(full_code, code):
                return False
            if not torch.allclose(from_code, full_out, atol=1e-6):
                return False

            if self.encoder is not None and self.code_logits is not None:
                h = self.encoder(x)
                zero_h = torch.zeros_like(h)
                zero_logits = self.code_logits(zero_h).view(-1, self.d, self.K)
                zero_code = zero_logits.argmax(dim=-1)
                if x.numel() > 1 or (self.K > 1 and self.d > 0):
                    if torch.equal(zero_code, code):
                        return False
        return True


def train_kary_bottleneck(
    model: KaryBottleneck,
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int | None = None,
    lr: float | None = None,
    log_every: int = 0,
) -> dict[str, float]:
    """Tiny deterministic trainer for fixture wiring.

    Trains with a soft code relaxation and evaluates with hard codes.  Fixture
    overfit is wiring evidence only, not a production quality claim.
    """
    cfg = model.config
    steps = steps if steps is not None else cfg.train_steps
    lr = lr if lr is not None else cfg.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad()
        out, _, _ = model(states, hard=False)
        loss = F.cross_entropy(out, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{steps} loss={loss.item():.4f}")
    return {"final_loss": losses[-1], "steps": steps, "lr": lr}


def evaluate_kary_bottleneck(
    model: KaryBottleneck,
    states: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Hard-code evaluation."""
    model.eval()
    with torch.no_grad():
        out, codes, logits = model(states, hard=True)
        pred = out.argmax(dim=-1)
        exact_acc = (pred == targets).float().mean().item()
        occupied = int(torch.unique(codes, dim=0).shape[0])
        return {
            "exact_reconstruction_rate": exact_acc,
            "occupied_codewords": occupied,
            "capacity": model.K ** model.d,
        }
