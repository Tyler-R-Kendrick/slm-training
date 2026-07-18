"""LDI4-02 SAE decision-state diagnostic: torch-free schema (SLM-136).

The activation-capture contract, the SAE architecture/sparsity config, the matched
S0-S7 arm specs, and the train-only feature-selection boundary. Kept torch-free so a
CLI / campaign can resolve and fingerprint a diagnostic without importing torch; the
SAE ``nn.Module`` and the tensor math live in ``sae.py`` / ``interventions.py``.

Nothing here claims interpretability or steering utility: an SAE feature is a
*diagnostic candidate* until matched, multi-seed, held-group causal evidence clears the
baselines. Feature / layer / sign / threshold / dose selection uses train groups only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from slm_training.harnesses.preference.local_decisions import Split, split_for_group
from slm_training.lineage.records import content_sha

__all__ = [
    "SCHEMA_VERSION",
    "CaptureRole",
    "CaptureRow",
    "CaptureManifest",
    "SAEConfig",
    "SAEArm",
    "matched_sae_arms",
    "FeatureSelectionError",
    "select_features_train_only",
]

SCHEMA_VERSION = "ldi4-02-v1"

CaptureRole = Literal["target", "preservation", "unlabeled"]
_ROLES = ("target", "preservation", "unlabeled")
_NONLINEARITIES = ("relu", "jumprelu")
_SPARSITY = ("l1",)
_DECODER_NORM = ("unit", "none")


@dataclass(frozen=True)
class CaptureRow:
    """One activation-capture row, keyed to an exact DecisionEventV2 state. The tensor
    itself is stored out of band (sharded, content-addressed); this row records only its
    identity, shape, and content hash so a mismatched checkpoint/site/width fails closed."""

    state_id: str
    group_id: str
    split: Split
    architecture: str
    policy_checkpoint_sha: str
    tokenizer_sha: str
    decode_config_hash: str
    verifier_bundle_hash: str
    site: str  # e.g. "denoiser.block.3.residual"
    position: str  # "exact_decision" | "context_pooled"
    hidden_size: int
    dtype: str
    activation_content_hash: str
    role: CaptureRole = "unlabeled"
    generation_step: int | None = None
    decision_role: str = ""

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"bad capture role {self.role!r}")
        if self.split != split_for_group(self.group_id):
            raise ValueError(
                f"row split {self.split!r} disagrees with split_for_group({self.group_id!r})"
            )

    def identity(self) -> dict[str, Any]:
        """Capture identity (excludes labels/role) -- changing any field invalidates the
        cached activation."""
        return {
            "state_id": self.state_id,
            "architecture": self.architecture,
            "policy_checkpoint_sha": self.policy_checkpoint_sha,
            "tokenizer_sha": self.tokenizer_sha,
            "decode_config_hash": self.decode_config_hash,
            "verifier_bundle_hash": self.verifier_bundle_hash,
            "site": self.site,
            "position": self.position,
            "hidden_size": self.hidden_size,
            "generation_step": self.generation_step,
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class CaptureManifest:
    """A capture dataset over V2 states. Train and held groups must stay disjoint and
    every row must share one site/checkpoint identity (normalization stats are computed
    on train rows only -- enforced by the SAE trainer, recorded here by contract)."""

    site: str
    hidden_size: int
    rows: tuple[CaptureRow, ...]
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for r in self.rows:
            if r.site != self.site or r.hidden_size != self.hidden_size:
                raise ValueError("all capture rows must share the manifest site/width")
        train = {r.state_id for r in self.rows if r.split == "train"}
        held = {r.state_id for r in self.rows if r.split == "held_out"}
        if train & held:
            raise ValueError("train/held groups overlap on a state_id (leakage)")

    def train_rows(self) -> tuple[CaptureRow, ...]:
        return tuple(r for r in self.rows if r.split == "train")

    def held_rows(self) -> tuple[CaptureRow, ...]:
        return tuple(r for r in self.rows if r.split == "held_out")

    def identity_fingerprint(self) -> str:
        return content_sha(
            {"site": self.site, "hidden_size": self.hidden_size, "version": self.version,
             "rows": [r.identity() for r in self.rows]}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "hidden_size": self.hidden_size,
            "version": self.version,
            "identity_fingerprint": self.identity_fingerprint(),
            "row_count": len(self.rows),
            "train_count": len(self.train_rows()),
            "held_count": len(self.held_rows()),
            "rows": [r.to_dict() for r in self.rows],
        }


@dataclass(frozen=True)
class SAEConfig:
    """Transparent baseline SAE: ``z = act(encoder(h - bias_dec)); h_hat = decoder(z) +
    bias_dec; loss = recon + lambda * sparsity``. Declares the exact architecture and
    sparsity objective; dead features are resampled only by the deterministic policy."""

    d_in: int
    expansion_factor: int = 4
    nonlinearity: str = "relu"
    sparsity: str = "l1"
    lambda_sparse: float = 1e-3
    decoder_norm: str = "unit"
    dead_feature_window: int = 256
    dead_feature_threshold: float = 1e-6
    seed: int = 0
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.d_in <= 0 or self.expansion_factor <= 0:
            raise ValueError("d_in and expansion_factor must be positive")
        if self.nonlinearity not in _NONLINEARITIES:
            raise ValueError(f"nonlinearity must be one of {_NONLINEARITIES}")
        if self.sparsity not in _SPARSITY:
            raise ValueError(f"sparsity must be one of {_SPARSITY}")
        if self.decoder_norm not in _DECODER_NORM:
            raise ValueError(f"decoder_norm must be one of {_DECODER_NORM}")
        if self.lambda_sparse < 0:
            raise ValueError("lambda_sparse must be non-negative")

    @property
    def dict_width(self) -> int:
        return self.d_in * self.expansion_factor

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SAEConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown SAE config field(s): {sorted(unknown)}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    def fingerprint(self) -> str:
        return content_sha(self.to_dict())[:16]


# --------------------------------------------------------------------------- #
# Matched S0-S7 arm specs (the SAE arms must not out-parameter the controls).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SAEArm:
    arm_id: str  # S0..S7
    method: str
    purpose: str
    selection_data: str  # "none" | "train_only"
    dose_grid: tuple[float, ...] = ()
    trainable_params: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def matched_sae_arms(
    *, site: str, dose_grid: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
) -> tuple[SAEArm, ...]:
    """The S0-S7 matched matrix at one pinned parent/site/corpus. Every steering arm uses
    train-only selection and the identical symmetric dose grid; the SAE arms are compared
    against a generic-perturbation, a simple-representation, a supervised-dense, a learned
    (ReFT), and a weight-space control before any steering conclusion."""
    return (
        SAEArm("S0", "no_intervention", "parent control", "none", (), 0),
        SAEArm("S1", "random_normalized_direction", "generic perturbation control", "none", dose_grid, 0),
        SAEArm("S2", "raw_diffmean", "simple representation control", "train_only", dose_grid, 0),
        SAEArm("S3", "linear_probe_direction", "supervised dense control", "train_only", dose_grid),
        SAEArm("S4", "reft_r1", "learned representation control (SLM-134)", "train_only", dose_grid),
        SAEArm("S5", "direct_weight_adapter", "weight-space control (SLM-126)", "train_only", dose_grid),
        SAEArm("S6", "top_sae_feature", "sparse single-feature test", "train_only", dose_grid),
        SAEArm("S7", "sparse_sae_feature_set", "capacity control", "train_only", dose_grid),
    )


class FeatureSelectionError(ValueError):
    """Raised when feature/sign/threshold/dose selection would touch held-out groups."""


def select_features_train_only(
    scores_by_split: Mapping[Split, Sequence[float]], *, top_k: int
) -> tuple[int, ...]:
    """Select the ``top_k`` feature indices by train-group score. Fails closed if a
    held-out score vector is supplied for *selection* (held groups are for replication
    only -- never for feature, sign, threshold, layer, or dose choice)."""
    if "held_out" in scores_by_split:
        raise FeatureSelectionError(
            "held-out scores must not drive selection; use train groups only"
        )
    train = list(scores_by_split.get("train", ()))
    if not train:
        raise FeatureSelectionError("no train-group scores supplied")
    order = sorted(range(len(train)), key=lambda i: train[i], reverse=True)
    return tuple(order[:top_k])
