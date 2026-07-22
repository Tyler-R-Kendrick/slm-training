"""Typed decode-path registry for the decode-invariance audit (EFS0-02).

A ``DecodePathSpec`` is a *declarative* bundle of decode levers plus
compatibility predicates. It names one semantically-distinct way to decode a
frozen checkpoint so an eval-only factorial (checkpoint × decode-path) can
attribute historical quality variance to the decoder rather than the weights
(the E288 lesson: byte-identical weights went parse 0 → 1.0 after a decoder
correction).

The registry is the single source of truth for the three required paths so the
matrix does not scatter booleans:

* ``checkpoint_declared`` — decode exactly as the checkpoint's own config
  declared (the historical control; no runtime overrides);
* ``current_native`` — current production model-native decode with current bug
  fixes, preserving the checkpoint's own output representation;
* ``current_exact_or_compiler`` — the strongest current exact choice-completion
  (choice codec) or compiler-tree greedy (surface/lexer), **preserving** the
  checkpoint's target representation — never coercing a surface checkpoint into
  a choice codec.

Levers map onto ``TwoTowerConfig``/``ModelBuildConfig`` fields consumed by
``apply_runtime_overrides`` (see ``factory.py``) and gated by an
``Experiment.runtime_override_fields`` whitelist. The module is Torch-free; it
only declares configuration and compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from slm_training.harnesses.model_build.eval_policy import STRICT_EVALUATION_POLICY
from slm_training.lineage.records import content_sha

__all__ = [
    "DECODE_PATH_SCHEMA",
    "KNOWN_OUTPUT_CODECS",
    "CompletionKind",
    "DecodePathSpec",
    "REQUIRED_DECODE_PATH_IDS",
    "get_decode_path",
    "all_decode_paths",
    "compatible_decode_paths",
]

DECODE_PATH_SCHEMA = "decode_path/v1"

# Output-codec identities (the checkpoint's own ``output_tokenizer``). "surface"
# is the semantic-bits stream name for the compositional codec.
KNOWN_OUTPUT_CODECS: tuple[str, ...] = ("compositional", "lexer", "choice")

CompletionKind = Literal["declared", "exact", "greedy", "beam", "stochastic"]

REQUIRED_DECODE_PATH_IDS: tuple[str, ...] = (
    "checkpoint_declared",
    "current_native",
    "current_exact_or_compiler",
)


@dataclass(frozen=True)
class DecodePathSpec:
    """One semantically-distinct decode path for eval-only comparison."""

    path_id: str
    description: str
    generation_entry: str
    completion_kind: CompletionKind
    grammar_policy: str
    seed_policy: str
    expected_fallback: str
    impl_version: str
    supported_model_families: tuple[str, ...] = ()
    supported_output_codecs: tuple[str, ...] = ()
    min_output_contract_version: int | None = None
    sampling: tuple[tuple[str, Any], ...] = ()
    lever_overrides: tuple[tuple[str, Any], ...] = ()
    # Per-codec extra overrides so one path can preserve each checkpoint's own
    # representation (e.g. compiler-tree for surface, exact pushdown for choice).
    codec_lever_overrides: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    schema_version: str = DECODE_PATH_SCHEMA

    def __post_init__(self) -> None:
        if self.completion_kind not in {
            "declared",
            "exact",
            "greedy",
            "beam",
            "stochastic",
        }:
            raise ValueError(f"unknown completion_kind {self.completion_kind!r}")
        codecs = {codec for codec, _ in self.codec_lever_overrides}
        unknown = codecs - set(self.supported_output_codecs)
        if unknown:
            raise ValueError(
                f"{self.path_id}: codec overrides for unsupported codecs {sorted(unknown)}"
            )

    def runtime_override_fields(self) -> tuple[str, ...]:
        """All config fields this path may mutate (base + every codec branch)."""
        fields: set[str] = {name for name, _ in self.lever_overrides}
        for _codec, overrides in self.codec_lever_overrides:
            fields.update(name for name, _ in overrides)
        return tuple(sorted(fields))

    def resolve_config_overrides(self, output_codec: str) -> dict[str, Any]:
        """Concrete config overrides to apply for a checkpoint of ``output_codec``."""
        overrides: dict[str, Any] = dict(self.lever_overrides)
        for codec, codec_overrides in self.codec_lever_overrides:
            if codec == output_codec:
                overrides.update(dict(codec_overrides))
        return overrides

    def is_compatible(
        self,
        *,
        model_family: str,
        output_codec: str,
        output_contract_version: int | None = None,
    ) -> tuple[bool, str | None]:
        """Whether this path can run on a checkpoint of the given identity.

        Returns ``(ok, reason)``; ``reason`` is a stable string when incompatible
        so the matrix can emit an explicit incompatible cell instead of coercing
        semantics.
        """
        if self.supported_model_families and model_family not in self.supported_model_families:
            return False, (
                f"path {self.path_id!r} supports model families "
                f"{list(self.supported_model_families)}, not {model_family!r}"
            )
        if self.supported_output_codecs and output_codec not in self.supported_output_codecs:
            return False, (
                f"path {self.path_id!r} supports output codecs "
                f"{list(self.supported_output_codecs)}, not {output_codec!r}; "
                "refusing to coerce the checkpoint's target representation"
            )
        if self.min_output_contract_version is not None:
            if output_contract_version is None or (
                output_contract_version < self.min_output_contract_version
            ):
                return False, (
                    f"path {self.path_id!r} requires output_contract_version >= "
                    f"{self.min_output_contract_version}, got {output_contract_version}"
                )
        return True, None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Stable implementation/version fingerprint of this path spec."""
        return content_sha(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecodePathSpec":
        return cls(
            path_id=str(data["path_id"]),
            description=str(data["description"]),
            generation_entry=str(data["generation_entry"]),
            completion_kind=str(data["completion_kind"]),  # type: ignore[arg-type]
            grammar_policy=str(data["grammar_policy"]),
            seed_policy=str(data["seed_policy"]),
            expected_fallback=str(data["expected_fallback"]),
            impl_version=str(data["impl_version"]),
            supported_model_families=tuple(data.get("supported_model_families", ())),
            supported_output_codecs=tuple(data.get("supported_output_codecs", ())),
            min_output_contract_version=data.get("min_output_contract_version"),
            sampling=tuple((str(k), v) for k, v in data.get("sampling", ())),
            lever_overrides=tuple((str(k), v) for k, v in data.get("lever_overrides", ())),
            codec_lever_overrides=tuple(
                (str(codec), tuple((str(k), v) for k, v in overrides))
                for codec, overrides in data.get("codec_lever_overrides", ())
            ),
            schema_version=str(data.get("schema_version", DECODE_PATH_SCHEMA)),
        )


# --- the three required paths --------------------------------------------------

_CHECKPOINT_DECLARED = DecodePathSpec(
    path_id="checkpoint_declared",
    description=(
        "Historical control: decode exactly as the checkpoint's own recorded "
        "config/decoder declared. No runtime overrides are applied."
    ),
    generation_entry="as_declared",
    completion_kind="declared",
    grammar_policy="checkpoint_declared",
    seed_policy="checkpoint_declared_seed",
    expected_fallback="as recorded with the checkpoint",
    impl_version="decode_path.checkpoint_declared/v1",
    supported_model_families=(),
    supported_output_codecs=(),
    sampling=(("deterministic", True),),
    lever_overrides=(),
    codec_lever_overrides=(),
)

_CURRENT_NATIVE = DecodePathSpec(
    path_id="current_native",
    description=(
        "Current production model-native decode with current bug fixes, "
        "preserving the checkpoint's own output representation."
    ),
    generation_entry="model_native_greedy_ltr",
    completion_kind="greedy",
    grammar_policy="current_native_grammar",
    seed_policy="fixed_seed_0",
    expected_fallback="unconstrained fallback permitted (current default)",
    impl_version="decode_path.current_native/v1",
    supported_model_families=("twotower",),
    supported_output_codecs=(),
    sampling=(("deterministic", True), ("grammar_sample_decode", False)),
    lever_overrides=(
        ("grammar_constrained", True),
        ("grammar_ltr_primary", True),
        ("compiler_decode_mode", "off"),
        ("grammar_sample_decode", False),
        ("allow_unconstrained_fallback", True),
    ),
    codec_lever_overrides=(),
)

_CURRENT_EXACT_OR_COMPILER = DecodePathSpec(
    path_id="current_exact_or_compiler",
    description=(
        "Strongest current exact choice-completion (choice codec) or "
        "compiler-tree greedy (surface/lexer), preserving the checkpoint's "
        "target representation without coercion."
    ),
    generation_entry="exact_or_compiler_tree",
    completion_kind="exact",
    grammar_policy="strict_exact_or_compiler",
    seed_policy="fixed_seed_0",
    expected_fallback="no unconstrained fallback (fail-closed)",
    impl_version="decode_path.current_exact_or_compiler/v1",
    supported_model_families=("twotower",),
    supported_output_codecs=("choice", "lexer", "compositional"),
    sampling=(("deterministic", True),),
    lever_overrides=tuple(STRICT_EVALUATION_POLICY.items()),
    codec_lever_overrides=(
        # Choice codec routes to the exact ChoiceDecodeState pushdown automatically
        # when grammar is on; compiler-tree is not applicable.
        ("choice", (("compiler_decode_mode", "off"),)),
        # Surface/lexer checkpoints use compiler-tree greedy.
        ("lexer", (("compiler_decode_mode", "tree"), ("compiler_search_mode", "greedy"))),
        (
            "compositional",
            (("compiler_decode_mode", "tree"), ("compiler_search_mode", "greedy")),
        ),
    ),
)

_REGISTRY: dict[str, DecodePathSpec] = {
    spec.path_id: spec
    for spec in (_CHECKPOINT_DECLARED, _CURRENT_NATIVE, _CURRENT_EXACT_OR_COMPILER)
}


def get_decode_path(path_id: str) -> DecodePathSpec:
    try:
        return _REGISTRY[path_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown decode path {path_id!r}; known: {sorted(_REGISTRY)}"
        ) from exc


def all_decode_paths() -> tuple[DecodePathSpec, ...]:
    """Every registered path, in stable required order then extras."""
    ordered = [_REGISTRY[pid] for pid in REQUIRED_DECODE_PATH_IDS if pid in _REGISTRY]
    extras = [s for pid, s in sorted(_REGISTRY.items()) if pid not in REQUIRED_DECODE_PATH_IDS]
    return tuple(ordered + extras)


def compatible_decode_paths(
    *,
    model_family: str,
    output_codec: str,
    output_contract_version: int | None = None,
) -> list[tuple[DecodePathSpec, bool, str | None]]:
    """Every path with its ``(spec, compatible, reason)`` for one checkpoint."""
    results: list[tuple[DecodePathSpec, bool, str | None]] = []
    for spec in all_decode_paths():
        ok, reason = spec.is_compatible(
            model_family=model_family,
            output_codec=output_codec,
            output_contract_version=output_contract_version,
        )
        results.append((spec, ok, reason))
    return results
