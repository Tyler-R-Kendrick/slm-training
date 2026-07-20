"""SLM-158 (SPV3-05): compare sequence mixers under a shared task.

Fixture/wiring-only harness. Defines a narrow ``ContextSequenceMixer`` protocol and
simplified reference implementations for Transformer, a Mamba-family selective SSM,
Gated DeltaNet, RWKV, xLSTM, and Hyena.  The shared head is a tiny sequence
classifier on a synthetic token-pattern task.  No production TwoTower wiring is
touched and no ship-gate claim is made.
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "MIXER_CAMPAIGN_ID",
    "MixerFamily",
    "MixerArm",
    "CommonConfig",
    "MixerOutput",
    "MixerReportRow",
    "MixerManifest",
    "MixerReport",
    "ContextSequenceMixer",
    "NoMixer",
    "TransformerMixer",
    "MambaReferenceMixer",
    "GatedDeltaNetMixer",
    "RWKVMixer",
    "xLSTMReferenceMixer",
    "HyenaReferenceMixer",
    "build_manifest",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
]

MATRIX_VERSION = "spv3-05-v1"
MATRIX_SET = "slm158_mixer_comparison"
MIXER_CAMPAIGN_ID = "slm158-mixer-comparison"


class MixerFamily(str, Enum):
    """Mixer family for SPV3-05."""

    NO_MIXER = "no_mixer"
    TRANSFORMER = "transformer"
    MAMBA_REFERENCE = "mamba_reference"
    GATED_DELTA_NET = "gated_delta_net"
    RWKV = "rwkv"
    XLSTM = "xlstm"
    HYENA = "hyena"


@dataclass(frozen=True)
class CommonConfig:
    """Frozen orthogonal controls shared by every arm."""

    seq_len: int = 32
    vocab_size: int = 64
    d_model: int = 32
    n_layers: int = 2
    n_classes: int = 8
    n_train: int = 128
    n_eval: int = 32
    seeds: tuple[int, ...] = (0, 1)
    lr: float = 1e-2
    epochs: int = 8
    batch_size: int = 16
    mixer_state_dim: int = 16
    metric_versions: dict[str, str] = field(default_factory=lambda: {"meaningful": "2.0.0"})

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["seeds"] = list(self.seeds)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommonConfig":
        return cls(
            seq_len=data.get("seq_len", 32),
            vocab_size=data.get("vocab_size", 64),
            d_model=data.get("d_model", 32),
            n_layers=data.get("n_layers", 2),
            n_classes=data.get("n_classes", 8),
            n_train=data.get("n_train", 256),
            n_eval=data.get("n_eval", 64),
            seeds=tuple(data.get("seeds", [0, 1, 2])),
            lr=data.get("lr", 1e-2),
            epochs=data.get("epochs", 10),
            batch_size=data.get("batch_size", 16),
            mixer_state_dim=data.get("mixer_state_dim", 16),
            metric_versions=data.get("metric_versions", {"meaningful": "2.0.0"}),
        )


@dataclass(frozen=True)
class MixerArm:
    """One arm in the mixer comparison."""

    arm_id: str
    family: MixerFamily
    name: str
    description: str
    promotable: bool = True
    reference: bool = True
    blocked: bool = False
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["family"] = self.family.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MixerArm":
        return cls(
            arm_id=data["arm_id"],
            family=MixerFamily(data.get("family", "no_mixer")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            promotable=data.get("promotable", True),
            reference=data.get("reference", True),
            blocked=data.get("blocked", False),
            blocker=data.get("blocker", ""),
        )


@dataclass(frozen=True)
class MixerOutput:
    """Output contract for every context mixer."""

    hidden: torch.Tensor
    pooled: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MixerReportRow:
    """Aggregated row for one arm/seed."""

    arm_id: str
    family: MixerFamily
    seed: int
    promotable: bool
    n_records: int
    mean_loss: float
    mean_accuracy: float
    mean_latency_ms: float
    param_count: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["family"] = self.family.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MixerReportRow":
        return cls(
            arm_id=data["arm_id"],
            family=MixerFamily(data.get("family", "no_mixer")),
            seed=data["seed"],
            promotable=data.get("promotable", True),
            n_records=data["n_records"],
            mean_loss=data["mean_loss"],
            mean_accuracy=data["mean_accuracy"],
            mean_latency_ms=data["mean_latency_ms"],
            param_count=data["param_count"],
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class MixerManifest:
    """Preregistered manifest for the SLM-158 campaign."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = MIXER_CAMPAIGN_ID
    hypothesis: str = (
        "A narrow sequence-mixer protocol with simplified reference implementations "
        "can expose whether non-Transformer mixers preserve task accuracy and improve "
        "latency/memory on a shared synthetic sequence task, before productionizing "
        "any one family."
    )
    falsifier: str = (
        "All mixers perform identically on the actual workload distribution, "
        "recurrent/SSM mixers require unrealistic lengths to show a cost win, "
        "or simplified references cannot be trained stably enough to separate "
        "mixer family effects from implementation noise."
    )
    common_config: CommonConfig = field(default_factory=CommonConfig)
    arms: tuple[MixerArm, ...] = ()
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["common_config"] = self.common_config.to_dict()
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MixerManifest":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", MIXER_CAMPAIGN_ID),
            hypothesis=data.get("hypothesis", ""),
            falsifier=data.get("falsifier", ""),
            common_config=CommonConfig.from_dict(data.get("common_config", {})),
            arms=tuple(MixerArm.from_dict(a) for a in data.get("arms", [])),
            claim_class=data.get("claim_class", "wiring"),
            status=data.get("status", "not_run"),
        )


@dataclass(frozen=True)
class MixerReport:
    """Full fixture report for SLM-158."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    manifest: MixerManifest
    rows: list[MixerReportRow]
    version_stamp: dict[str, Any] = field(default_factory=dict)
    claim_class: str = "wiring"

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "manifest": self.manifest.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MixerReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", MIXER_CAMPAIGN_ID),
            run_id=data.get("run_id", "slm158_fixture"),
            status=data.get("status", "fixture"),
            manifest=MixerManifest.from_dict(data.get("manifest", {})),
            rows=[MixerReportRow.from_dict(r) for r in data.get("rows", [])],
            version_stamp=data.get("version_stamp", {}),
            claim_class=data.get("claim_class", "wiring"),
        )


class ContextSequenceMixer(nn.Module, ABC):
    """Narrow mixer protocol: encode a padded token batch into a pooled vector."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def encode(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> MixerOutput:
        """x: [B, L, D]. pad_mask: [B, L] True where PAD."""
        ...

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(x, pad_mask).pooled


class NoMixer(ContextSequenceMixer):
    """No learned sequence mixer: mean pool token embeddings + tiny MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> MixerOutput:
        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = x.mean(dim=1)
        hidden = self.mlp(self.norm(pooled)).unsqueeze(1).expand(-1, x.size(1), -1)
        return MixerOutput(hidden=hidden, pooled=pooled, diagnostics={"family": "no_mixer"})


class TransformerMixer(ContextSequenceMixer):
    """Small Transformer encoder with mean pooling."""

    def __init__(self, cfg: CommonConfig) -> None:
        super().__init__(cfg.d_model)
        from slm_training.models.blocks import TokenEncoder

        self.encoder = TokenEncoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=max(1, cfg.d_model // 16),
            max_len=cfg.seq_len,
            dropout=0.0,
        )

    def encode(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> MixerOutput:
        # x is already embedded by the classifier; feed through the transformer body.
        hidden = self.encoder.layers[0](x, self_pad_mask=pad_mask)
        for layer in self.encoder.layers[1:]:
            hidden = layer(hidden, self_pad_mask=pad_mask)
        hidden = self.encoder.norm(hidden)
        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return MixerOutput(hidden=hidden, pooled=pooled, diagnostics={"family": "transformer"})


class MambaReferenceMixer(ContextSequenceMixer):
    """Simplified Mamba-family selective SSM reference (not a production kernel)."""

    def __init__(self, cfg: CommonConfig) -> None:
        super().__init__(cfg.d_model)
        d = cfg.d_model
        n = cfg.mixer_state_dim
        self.in_proj = nn.Linear(d, n * 3 + d + 1)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(n))
        self.out_proj = nn.Linear(n, d)

    def encode(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> MixerOutput:
        B, L, D = x.shape
        N = self.A_log.size(0)
        proj = self.in_proj(x)
        x_state, z, B_, C, dt = torch.split(
            proj, [N, D, N, N, 1], dim=-1
        )
        dt = F.softplus(dt.squeeze(-1)).clamp_min(1e-3)  # [B, L]
        A = -F.softplus(self.A_log)  # [N]
        h = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))  # [B, N]
            dB = (dt[:, t].unsqueeze(-1) * B_[:, t])  # [B, N]
            h = h * dA + dB * x_state[:, t]
            y = h * C[:, t] * self.D  # [B, N]
            ys.append(y)
        y = torch.stack(ys, dim=1)  # [B, L, N]
        hidden = self.out_proj(y) + z  # [B, L, D]
        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return MixerOutput(
            hidden=hidden,
            pooled=pooled,
            diagnostics={"family": "mamba_reference", "state_dim": N},
        )


class GatedDeltaNetMixer(ContextSequenceMixer):
    """Simplified gated DeltaNet-style linear attention reference."""

    def __init__(self, cfg: CommonConfig) -> None:
        super().__init__(cfg.d_model)
        d = cfg.d_model
        self.qkv = nn.Linear(d, d * 3)
        self.beta = nn.Linear(d, 1)
        self.gate = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def encode(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> MixerOutput:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        beta = torch.sigmoid(self.beta(x))
        g = torch.sigmoid(self.gate(x))
        S = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []
        for t in range(L):
            b = beta[:, t].unsqueeze(-1)
            kt = k[:, t].unsqueeze(-1)
            vt = v[:, t].unsqueeze(1)
            S = (1 - b) * S + b * torch.bmm(kt, vt)
            yt = torch.bmm(S, q[:, t].unsqueeze(-1)).squeeze(-1)
            ys.append(yt)
        y = torch.stack(ys, dim=1)
        hidden = self.out_proj(g * y + (1 - g) * x)
        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return MixerOutput(
            hidden=hidden,
            pooled=pooled,
            diagnostics={"family": "gated_delta_net"},
        )


class RWKVMixer(ContextSequenceMixer):
    """Simplified RWKV-style time-mixing reference."""

    def __init__(self, cfg: CommonConfig) -> None:
        super().__init__(cfg.d_model)
        d = cfg.d_model
        self.time_mix = nn.Linear(d, d * 5)
        self.out_proj = nn.Linear(d, d)

    def encode(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> MixerOutput:
        B, L, D = x.shape
        split = self.time_mix(x).chunk(5, dim=-1)
        r, k, v, g, w = split
        r = torch.sigmoid(r)
        k = -F.softplus(k)
        v = torch.tanh(v)
        g = torch.sigmoid(g)
        w = torch.exp(-F.softplus(w))
        state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []
        for t in range(L):
            e = torch.exp(k[:, t])
            state = state * w[:, t] + e * v[:, t]
            denom = state.abs().clamp_min(1e-6)
            y = r[:, t] * state / denom
            ys.append(y)
        y = torch.stack(ys, dim=1)
        hidden = self.out_proj(g * y + (1 - g) * x)
        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return MixerOutput(
            hidden=hidden,
            pooled=pooled,
            diagnostics={"family": "rwkv"},
        )


class xLSTMReferenceMixer(ContextSequenceMixer):
    """Simplified xLSTM mLSTM-style matrix-memory reference."""

    def __init__(self, cfg: CommonConfig) -> None:
        super().__init__(cfg.d_model)
        d = cfg.d_model
        self.qkv = nn.Linear(d, d * 3)
        self.f_gate = nn.Linear(d, 1)
        self.i_gate = nn.Linear(d, 1)
        self.out_proj = nn.Linear(d, d)

    def encode(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> MixerOutput:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        f = torch.sigmoid(self.f_gate(x))
        i = torch.exp(self.i_gate(x).clamp(max=5.0))
        C = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []
        for t in range(L):
            ft = f[:, t].unsqueeze(-1)
            it = i[:, t].unsqueeze(-1)
            kt = k[:, t].unsqueeze(1)
            vt = v[:, t].unsqueeze(-1)
            C = ft * C + it * torch.bmm(vt, kt)
            yt = (torch.bmm(C, q[:, t].unsqueeze(-1)) / it.clamp_min(1.0)).squeeze(-1)
            ys.append(yt)
        y = torch.stack(ys, dim=1)
        hidden = self.out_proj(y + x)
        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return MixerOutput(
            hidden=hidden,
            pooled=pooled,
            diagnostics={"family": "xlstm_reference"},
        )


class HyenaReferenceMixer(ContextSequenceMixer):
    """Simplified Hyena-style long-convolution reference."""

    def __init__(self, cfg: CommonConfig) -> None:
        super().__init__(cfg.d_model)
        d = cfg.d_model
        self.filter_order = min(16, cfg.seq_len)
        self.in_proj = nn.Linear(d, d * 2)
        self.filter_mlp = nn.Sequential(
            nn.Linear(self.filter_order, self.filter_order * 2),
            nn.GELU(),
            nn.Linear(self.filter_order * 2, self.filter_order),
        )
        self.out_proj = nn.Linear(d, d)

    def encode(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> MixerOutput:
        B, L, D = x.shape
        u, v = self.in_proj(x).chunk(2, dim=-1)
        positions = torch.linspace(0, 1, self.filter_order, device=x.device, dtype=x.dtype)
        kernel = self.filter_mlp(positions)  # [K]
        kernel = kernel.view(1, 1, self.filter_order).flip(-1)
        v_perm = v.permute(0, 2, 1)  # [B, D, L]
        conv = F.conv1d(
            F.pad(v_perm, (self.filter_order - 1, 0)),
            kernel.expand(D, -1, -1),
            groups=D,
        )  # [B, D, L]
        conv = conv.permute(0, 2, 1)  # [B, L, D]
        hidden = self.out_proj(u * conv + x)
        if pad_mask is not None:
            mask = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return MixerOutput(
            hidden=hidden,
            pooled=pooled,
            diagnostics={"family": "hyena_reference", "filter_order": self.filter_order},
        )


def _build_mixer(family: MixerFamily, cfg: CommonConfig) -> ContextSequenceMixer:
    if family is MixerFamily.NO_MIXER:
        return NoMixer(cfg.d_model)
    if family is MixerFamily.TRANSFORMER:
        return TransformerMixer(cfg)
    if family is MixerFamily.MAMBA_REFERENCE:
        return MambaReferenceMixer(cfg)
    if family is MixerFamily.GATED_DELTA_NET:
        return GatedDeltaNetMixer(cfg)
    if family is MixerFamily.RWKV:
        return RWKVMixer(cfg)
    if family is MixerFamily.XLSTM:
        return xLSTMReferenceMixer(cfg)
    if family is MixerFamily.HYENA:
        return HyenaReferenceMixer(cfg)
    raise ValueError(f"unsupported mixer family: {family}")


class _MixerClassifier(nn.Module):
    """Token embedding + mixer + classifier head."""

    def __init__(self, cfg: CommonConfig, family: MixerFamily) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.mixer = _build_mixer(family, cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok(ids)
        pad_mask = ids.eq(0)
        pooled = self.mixer.encode(x, pad_mask=pad_mask).pooled
        return self.head(self.norm(pooled))


def _make_dataset(
    cfg: CommonConfig, seed: int, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    torch_rng = torch.Generator().manual_seed(seed)
    ids = torch.randint(
        1, cfg.vocab_size, (n, cfg.seq_len), generator=torch_rng, dtype=torch.long
    )
    labels = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        # Synthetic rule: class = sum of first three token ids modulo n_classes.
        rule = int(ids[i, :3].sum().item()) % cfg.n_classes
        # Corrupt a fraction to make it nontrivial.
        if rng.random() < 0.1:
            rule = (rule + rng.randrange(1, cfg.n_classes)) % cfg.n_classes
        labels[i] = rule
    return ids, labels


def _evaluate(model: nn.Module, ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(ids)
        loss = F.cross_entropy(logits, labels).item()
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def _train_arm(
    arm: MixerArm, cfg: CommonConfig, seed: int, device: str = "cpu"
) -> MixerReportRow:
    torch.manual_seed(seed)
    train_ids, train_labels = _make_dataset(cfg, seed, cfg.n_train)
    eval_ids, eval_labels = _make_dataset(cfg, seed + 1000, cfg.n_eval)
    train_ids = train_ids.to(device)
    train_labels = train_labels.to(device)
    eval_ids = eval_ids.to(device)
    eval_labels = eval_labels.to(device)

    model = _MixerClassifier(cfg, arm.family).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    import time

    start = time.perf_counter()
    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(train_ids.size(0))
        for i in range(0, train_ids.size(0), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            optimizer.zero_grad()
            logits = model(train_ids[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            loss.backward()
            optimizer.step()
    train_wall = time.perf_counter() - start

    eval_loss, eval_acc = _evaluate(model, eval_ids, eval_labels)
    param_count = sum(p.numel() for p in model.parameters())
    mean_latency_ms = (train_wall * 1000) / max(1, cfg.epochs * (cfg.n_train // cfg.batch_size))

    notes = [
        f"family={arm.family.value}",
        "fixture-only: synthetic token-pattern classifier",
    ]
    if arm.reference:
        notes.append("reference implementation")

    return MixerReportRow(
        arm_id=arm.arm_id,
        family=arm.family,
        seed=seed,
        promotable=arm.promotable and not arm.blocked,
        n_records=cfg.n_eval,
        mean_loss=eval_loss,
        mean_accuracy=eval_acc,
        mean_latency_ms=mean_latency_ms,
        param_count=param_count,
        notes=notes,
    )


def build_manifest() -> MixerManifest:
    """Return the default SLM-158 fixture manifest."""
    arms = (
        MixerArm(
            arm_id="T0_no_mixer",
            family=MixerFamily.NO_MIXER,
            name="no_mixer_floor",
            description="Mean-pooled token embedding floor with no learned sequence mixer.",
        ),
        MixerArm(
            arm_id="T1_transformer",
            family=MixerFamily.TRANSFORMER,
            name="small_transformer",
            description="Small Transformer encoder (baseline).",
        ),
        MixerArm(
            arm_id="S1_mamba_reference",
            family=MixerFamily.MAMBA_REFERENCE,
            name="mamba_reference",
            description="Simplified Mamba-family selective SSM reference.",
        ),
        MixerArm(
            arm_id="L1_gated_delta_net",
            family=MixerFamily.GATED_DELTA_NET,
            name="gated_delta_net_reference",
            description="Simplified Gated DeltaNet-style linear-attention reference.",
        ),
        MixerArm(
            arm_id="R1_rwkv_reference",
            family=MixerFamily.RWKV,
            name="rwkv_reference",
            description="Simplified RWKV-style recurrent time-mixing reference.",
        ),
        MixerArm(
            arm_id="R2_xlstm_reference",
            family=MixerFamily.XLSTM,
            name="xlstm_reference",
            description="Simplified xLSTM mLSTM-style matrix-memory reference.",
        ),
        MixerArm(
            arm_id="C1_hyena_reference",
            family=MixerFamily.HYENA,
            name="hyena_reference",
            description="Simplified Hyena-style long-convolution reference.",
        ),
    )
    return MixerManifest(arms=arms)


def validate_manifest(manifest: MixerManifest) -> list[str]:
    """Validate manifest shape and honest constraints."""
    errors: list[str] = []
    if not manifest.arms:
        errors.append("arms must not be empty")
    seen: set[str] = set()
    for arm in manifest.arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
        if arm.blocked and arm.promotable:
            errors.append(f"{arm.arm_id}: blocked arm must be non-promotable")
    cfg = manifest.common_config
    if cfg.seq_len <= 0:
        errors.append("common_config.seq_len must be positive")
    if cfg.d_model <= 0:
        errors.append("common_config.d_model must be positive")
    if cfg.n_classes <= 0:
        errors.append("common_config.n_classes must be positive")
    return errors


def run_fixture_campaign(
    manifest: MixerManifest | None = None,
    *,
    run_id: str = "slm158_fixture",
    output_dir: Path | None = None,
    device: str = "cpu",
) -> MixerReport:
    """Run the SLM-158 mixer-comparison fixture campaign."""
    manifest = manifest or build_manifest()
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    cfg = manifest.common_config
    rows: list[MixerReportRow] = []
    for arm in manifest.arms:
        if arm.blocked:
            rows.append(
                MixerReportRow(
                    arm_id=arm.arm_id,
                    family=arm.family,
                    seed=-1,
                    promotable=False,
                    n_records=0,
                    mean_loss=float("nan"),
                    mean_accuracy=float("nan"),
                    mean_latency_ms=float("nan"),
                    param_count=0,
                    notes=["blocked", arm.blocker],
                )
            )
            continue
        for seed in cfg.seeds:
            rows.append(_train_arm(arm, cfg, seed, device=device))

    report = MixerReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=MIXER_CAMPAIGN_ID,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm158_mixer_comparison",
        ),
        claim_class="wiring",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm158_mixer_comparison_report.json")
    return report


def render_markdown(report: MixerReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-158 (SPV3-05): Sequence-mixer comparison fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no production "
        "TwoTower wiring was touched, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Arms",
        "",
        "| Arm | Family | Promotable | Reference | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.family.value} | {arm.promotable} | "
            f"{arm.reference} | {arm.description} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Arm | Seed | Records | Loss | Accuracy | Latency ms | Params |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        loss_s = f"{row.mean_loss:.3f}" if not math.isnan(row.mean_loss) else "N/A"
        acc_s = f"{row.mean_accuracy:.3f}" if not math.isnan(row.mean_accuracy) else "N/A"
        lat_s = f"{row.mean_latency_ms:.3f}" if not math.isnan(row.mean_latency_ms) else "N/A"
        lines.append(
            f"| {row.arm_id} | {row.seed} | {row.n_records} | {loss_s} | "
            f"{acc_s} | {lat_s} | {row.param_count} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "This is a fixture wiring run. The simplified reference mixers share a "
            "common input/output contract and are evaluated on a synthetic token-pattern "
            "task. Real claims require optimized kernels, the actual legal-action scorer "
            "and compiler state, measured wall-clock on target hardware, and held-out "
            "causal evaluation.",
            "",
        ]
    )
    return "\n".join(lines)
