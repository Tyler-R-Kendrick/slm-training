"""PEFT actuator factory for the LDI1-02 causal adapter path.

Resolves a validated, fingerprinted :class:`AdapterSpec` (method + rank/alpha/
dropout + target modules + identities) independently of whether ``peft`` is
installed, so specs can be constructed, hashed, and unit-tested with no training
extra present. The actual ``peft`` config object is built lazily in
:func:`build_peft_config`, which fails *visibly* when a requested lever
(DoRA / PiSSA / AdaLoRA) is not exposed by the installed ``peft`` — an
unsupported option is never silently dropped to plain LoRA.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal

from slm_training.lineage.records import content_sha
from slm_training.lineage.tracks import CAUSAL_LORA_RECIPE

__all__ = [
    "AdapterMethod",
    "AdapterSpec",
    "SUPPORTED_METHODS",
    "EXPERIMENTAL_METHODS",
    "build_peft_config",
    "dependency_versions",
]

AdapterMethod = Literal["lora", "dora", "pissa", "adalora"]
SUPPORTED_METHODS: tuple[str, ...] = ("lora", "dora", "pissa", "adalora")
# AdaLoRA is a distinct experimental method, never an implicit fallback.
EXPERIMENTAL_METHODS: tuple[str, ...] = ("adalora",)

_DEFAULT_TARGETS: tuple[str, ...] = tuple(CAUSAL_LORA_RECIPE["target_modules"])


def dependency_versions() -> dict[str, str]:
    """Installed transformers/peft versions, or ``"absent"`` when not installed."""
    versions: dict[str, str] = {}
    for name in ("transformers", "peft"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


@dataclass(frozen=True)
class AdapterSpec:
    """A validated, fingerprintable causal PEFT adapter specification."""

    base_model_id: str
    base_model_revision: str
    tokenizer_sha: str
    method: AdapterMethod = "lora"
    rank: int = int(CAUSAL_LORA_RECIPE["rank"])
    alpha: int = int(CAUSAL_LORA_RECIPE["alpha"])
    dropout: float = float(CAUSAL_LORA_RECIPE["dropout"])
    target_modules: tuple[str, ...] = _DEFAULT_TARGETS
    include_lm_head: bool = False
    layer_pattern: str | None = None
    allow_experimental: bool = False
    objective_fingerprint: str = ""
    corpus_fingerprint: str = ""
    materializer_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"unknown adapter method {self.method!r}; "
                f"expected one of {list(SUPPORTED_METHODS)}"
            )
        if self.method in EXPERIMENTAL_METHODS and not self.allow_experimental:
            raise ValueError(
                f"{self.method!r} is experimental; pass allow_experimental=True to opt in"
            )
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if not self.target_modules:
            raise ValueError("target_modules must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "tokenizer_sha": self.tokenizer_sha,
            "method": self.method,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "include_lm_head": self.include_lm_head,
            "layer_pattern": self.layer_pattern,
            "objective_fingerprint": self.objective_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "materializer_fingerprint": self.materializer_fingerprint,
        }

    def fingerprint(self) -> str:
        """Deterministic content hash folding the spec and dependency versions."""
        return content_sha({"spec": self.to_dict(), "versions": dependency_versions()})


def _require_peft() -> Any:
    try:
        import peft  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional training extra
        raise RuntimeError(
            "install slm-training[hf] to build a causal PEFT adapter"
        ) from exc
    return peft


def _supports_kwarg(func: Any, name: str) -> bool:  # pragma: no cover - needs peft
    import inspect

    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def build_peft_config(spec: AdapterSpec) -> Any:  # pragma: no cover - needs peft extra
    """Build the concrete ``peft`` config for ``spec``, failing visibly.

    A requested DoRA/PiSSA/AdaLoRA lever that the installed ``peft`` does not
    expose raises rather than silently degrading to plain LoRA.
    """
    peft = _require_peft()
    modules_to_save = ["lm_head"] if spec.include_lm_head else None

    if spec.method == "adalora":
        adalora = getattr(peft, "AdaLoraConfig", None)
        if adalora is None:
            raise RuntimeError("installed peft does not expose AdaLoraConfig")
        return adalora(
            init_r=spec.rank,
            lora_alpha=spec.alpha,
            lora_dropout=spec.dropout,
            target_modules=list(spec.target_modules),
            modules_to_save=modules_to_save,
            task_type="CAUSAL_LM",
        )

    lora_config = peft.LoraConfig
    kwargs: dict[str, Any] = dict(
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
    )
    if spec.layer_pattern is not None:
        kwargs["layers_pattern"] = spec.layer_pattern
    if spec.method == "dora":
        if not _supports_kwarg(lora_config, "use_dora"):
            raise RuntimeError("installed peft does not support DoRA (use_dora)")
        kwargs["use_dora"] = True
    elif spec.method == "pissa":
        if not _supports_kwarg(lora_config, "init_lora_weights"):
            raise RuntimeError("installed peft does not support PiSSA initialization")
        kwargs["init_lora_weights"] = "pissa"
    return lora_config(**kwargs)
