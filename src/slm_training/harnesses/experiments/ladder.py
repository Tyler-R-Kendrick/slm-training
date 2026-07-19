"""Scaling ladder definitions (P1c)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from slm_training.harnesses.model_build.config import ModelBuildConfig

# CAP3-05: imports for synthetic-model byte estimation.
import torch
import torch.nn as nn

from slm_training.models.quantization.cost import build_model_ledger
from slm_training.models.quantization.formats import (
    binary_format,
    fp16_format,
    int4_format,
    int8_format,
    learned_four_level_zero_format,
    symmetric_four_level_format,
    ternary_format,
)

# Output representations the trainer supports today. "choice" (the B1
# choice-sequence codec, SLM-42) is a valid ladder axis for planning and
# corpus-bit accounting, but training it requires twotower support for a
# production-token output head — model_build_config_for_point fails closed
# until that lands.
TRAINABLE_REPRESENTATIONS: tuple[str, ...] = ("compositional", "lexer", "choice")


@dataclass(frozen=True)
class LadderPoint:
    d_model: int
    n_heads: int
    context_layers: int
    denoiser_layers: int
    target_token_budget: int
    horizon_multiplier: float = 1.0
    # Output representation (B3 axis): compositional | lexer | choice.
    representation: str = "compositional"
    # CAP3-05 equal-byte / precision metadata (all optional; default keeps legacy
    # ladder points unchanged).
    byte_budget: int | None = None
    precision_format: str | None = None
    actual_bytes: int | None = None
    budget_delta: float | None = None
    status: str = "feasible"

    @property
    def point_id(self) -> str:
        base = (
            f"d{self.d_model}_h{self.n_heads}_c{self.context_layers}_"
            f"dn{self.denoiser_layers}_t{self.target_token_budget}_x{self.horizon_multiplier:g}"
        )
        # Default keeps legacy ids stable; axes show up only when used.
        if self.representation != "compositional":
            base = f"{base}_r{self.representation}"
        if self.byte_budget is not None:
            base = f"{base}_b{self.byte_budget}"
        if self.precision_format:
            base = f"{base}_p{self.precision_format}"
        return base


@dataclass(frozen=True)
class ScalingLadder:
    ladder_id: str
    track: Literal["scratch", "hf"]
    points: tuple[LadderPoint, ...]
    token_horizons: tuple[float, ...] = (0.5, 1.0, 2.0)
    decode_frozen: Mapping[str, Any] | None = None
    # Output representation for every point on this ladder (B3 capacity arm):
    # "compositional" | "lexer" | "choice". The capacity ladder pairs two
    # otherwise-identical ladders differing only in this field.
    output_tokenizer: str = "compositional"


def proportional_depths(d_model: int) -> tuple[int, int, int]:
    """Return (n_heads, context_layers, denoiser_layers) scaled with width."""
    n_heads = max(2, d_model // 32)
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1
    context_layers = max(1, d_model // 64)
    denoiser_layers = max(2, d_model // 32)
    return n_heads, context_layers, denoiser_layers


def scratch_ladder_default(
    *,
    base_token_budget: int = 50_000,
    widths: tuple[int, ...] = (64, 96, 128, 192),
    horizons: tuple[float, ...] = (0.5, 1.0, 2.0),
    representation: str = "compositional",
    ladder_id: str = "scratch_v1",
) -> ScalingLadder:
    points: list[LadderPoint] = []
    # Constant tokens-per-trainable-param proxy: budget ∝ d_model².
    ref = 128
    for d in widths:
        n_heads, ctx, den = proportional_depths(d)
        scale = (d / ref) ** 2
        budget = max(1_000, int(base_token_budget * scale))
        for h in horizons:
            points.append(
                LadderPoint(
                    d_model=d,
                    n_heads=n_heads,
                    context_layers=ctx,
                    denoiser_layers=den,
                    target_token_budget=max(1_000, int(budget * h)),
                    horizon_multiplier=h,
                    representation=representation,
                )
            )
    return ScalingLadder(
        ladder_id=ladder_id,
        track="scratch",
        points=tuple(points),
        token_horizons=horizons,
        decode_frozen={
            "gen_steps": 8,
            "best_of_n": 1,
            "grammar_constrained": True,
            "parallel_unmask": "adaptive",
        },
    )


def capacity_ladder_pair(
    *,
    base_token_budget: int = 50_000,
    widths: tuple[int, ...] = (64, 96, 128, 192),
    horizons: tuple[float, ...] = (1.0,),
    representations: tuple[str, ...] = ("lexer", "choice"),
) -> tuple[ScalingLadder, ...]:
    """B3 (SLM-23): matched ladders differing only in output representation.

    Same widths, budgets, and frozen decode per arm, so quality-vs-d_model
    curves are comparable across representations and `params_per_bit` (E1,
    `evals/semantic_bits.py`) can be reported at matched quality. Arms whose
    representation is not in TRAINABLE_REPRESENTATIONS are constructible for
    planning/bit accounting but fail closed at config creation.
    """
    return tuple(
        scratch_ladder_default(
            base_token_budget=base_token_budget,
            widths=widths,
            horizons=horizons,
            representation=representation,
            ladder_id=f"capacity_{representation}_v1",
        )
        for representation in representations
    )


def hf_ladder_default(
    *,
    base_token_budget: int = 50_000,
    widths: tuple[int, ...] = (64, 96, 128, 192),
    horizons: tuple[float, ...] = (0.5, 1.0, 2.0),
    representation: str = "compositional",
) -> ScalingLadder:
    points: list[LadderPoint] = []
    for d in widths:
        n_heads, _, den = proportional_depths(d)
        # Frozen HF: scale only the denoiser; context_layers unused for hf tower.
        scale = (d / 128) ** 2
        budget = max(1_000, int(base_token_budget * scale))
        for h in horizons:
            points.append(
                LadderPoint(
                    d_model=d,
                    n_heads=n_heads,
                    context_layers=2,
                    denoiser_layers=den,
                    target_token_budget=max(1_000, int(budget * h)),
                    horizon_multiplier=h,
                    representation=representation,
                )
            )
    return ScalingLadder(
        ladder_id="hf_v1",
        track="hf",
        points=tuple(points),
        token_horizons=horizons,
        decode_frozen={
            "gen_steps": 8,
            "best_of_n": 1,
            "grammar_constrained": True,
            "freeze_context": True,
            "hf_model_name": "HuggingFaceTB/SmolLM2-135M",
        },
    )


#: Default d_model rungs for the B3 capacity ladder (three from-scratch widths).
CAPACITY_WIDTHS: tuple[int, ...] = (64, 128, 192)
#: Output-representation arms compared by the B3 capacity ladder. ``lexer`` is
#: the surface-token control (matched to quality-matrix E255); ``choice`` is the
#: B1 choice-sequence codec (matched to E262). Everything else is held fixed.
CAPACITY_ARMS: tuple[str, ...] = ("lexer", "choice")


def capacity_ladder(
    output_tokenizer: str,
    *,
    base_token_budget: int = 50_000,
    widths: tuple[int, ...] = CAPACITY_WIDTHS,
    horizons: tuple[float, ...] = (1.0,),
    mask_pattern: str = "diffusion",
) -> ScalingLadder:
    """One arm of the B3 capacity ladder (SLM-23).

    Reuses the from-scratch ``scratch_ladder_default`` point construction
    (constant tokens-per-trainable-param proxy, ``budget ∝ d_model²``) so the
    two arms share identical widths/depths/budgets and differ *only* in
    ``output_tokenizer``. The recipe (diffusion masking, non-LTR MaskGIT
    decode, grammar-constrained) matches the quality-matrix representation
    controls E255 (lexer) / E262 (choice) it is the capacity-swept form of.
    """
    base = scratch_ladder_default(
        base_token_budget=base_token_budget, widths=widths, horizons=horizons
    )
    decode = dict(base.decode_frozen or {})
    decode["mask_pattern"] = mask_pattern
    decode["grammar_ltr_primary"] = False
    return ScalingLadder(
        ladder_id=f"capacity_{output_tokenizer}_v1",
        track="scratch",
        points=base.points,
        token_horizons=base.token_horizons,
        decode_frozen=decode,
        output_tokenizer=output_tokenizer,
    )


def capacity_ladder_arms(
    *,
    base_token_budget: int = 50_000,
    widths: tuple[int, ...] = CAPACITY_WIDTHS,
    horizons: tuple[float, ...] = (1.0,),
    arms: tuple[str, ...] = CAPACITY_ARMS,
    mask_pattern: str = "diffusion",
) -> dict[str, ScalingLadder]:
    """The full B3 capacity ladder: one matched ladder per output-tokenizer arm.

    All arms share the same ``LadderPoint`` set (same widths/depths/budgets/
    horizons), so a row is fully identified by (arm, point_id). Only
    ``output_tokenizer`` varies across arms — the matched-recipe contract the
    B3 regression pins.
    """
    return {
        arm: capacity_ladder(
            arm,
            base_token_budget=base_token_budget,
            widths=widths,
            horizons=horizons,
            mask_pattern=mask_pattern,
        )
        for arm in arms
    }


def ladder_run_id(ladder_id: str, point: LadderPoint, seed: int) -> str:
    return f"{ladder_id}__{point.point_id}__s{seed}"


def model_build_config_for_point(
    point: LadderPoint,
    ladder: ScalingLadder,
    *,
    train_dir: Path,
    test_dir: Path | None,
    run_root: Path,
    seed: int,
    steps: int = 10_000,
    batch_size: int = 4,
    lr: float = 3e-4,
    extra: Mapping[str, Any] | None = None,
) -> ModelBuildConfig:
    if point.representation not in TRAINABLE_REPRESENTATIONS:
        raise ValueError(
            f"representation {point.representation!r} is not trainable yet: the "
            f"trainer supports {TRAINABLE_REPRESENTATIONS}. The 'choice' arm "
            "(B1 choice-sequence codec, SLM-42) needs a production-token output "
            "head in the twotower before its ladder rows can run."
        )
    decode = dict(ladder.decode_frozen or {})
    kwargs: dict[str, Any] = {
        "train_dir": Path(train_dir),
        "test_dir": Path(test_dir) if test_dir else None,
        "run_root": Path(run_root),
        "run_id": ladder_run_id(ladder.ladder_id, point, seed),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "d_model": point.d_model,
        "n_heads": point.n_heads,
        "context_layers": point.context_layers,
        "denoiser_layers": point.denoiser_layers,
        "target_token_budget": point.target_token_budget,
        "context_backend": "scratch" if ladder.track == "scratch" else "hf",
        # B3 axis reconciliation: main's capacity_ladder encodes the arm in
        # ladder.output_tokenizer (points stay "compositional"); #277's
        # capacity_ladder_pair encodes it in point.representation (ladder stays
        # "compositional"). Prefer the non-default of the two.
        "output_tokenizer": (
            ladder.output_tokenizer
            if ladder.output_tokenizer != "compositional"
            else point.representation
        ),
        "mask_pattern": str(decode.get("mask_pattern", "random")),
        "grammar_ltr_primary": bool(decode.get("grammar_ltr_primary", False)),
        "freeze_context": bool(decode.get("freeze_context", ladder.track == "hf")),
        "hf_model_name": str(
            decode.get("hf_model_name") or "HuggingFaceTB/SmolLM2-135M"
        ),
        "gen_steps": int(decode.get("gen_steps", 8)),
        "best_of_n": int(decode.get("best_of_n", 1)),
        "grammar_constrained": bool(decode.get("grammar_constrained", True)),
        "parallel_unmask": str(decode.get("parallel_unmask", "adaptive")),
        "loss_eval_every": max(1, steps // 10),
    }
    if point.precision_format is not None:
        kwargs["quant_format"] = point.precision_format
        kwargs["use_dynamic_quant"] = True
    if point.byte_budget is not None:
        kwargs["byte_budget"] = point.byte_budget
    if extra:
        kwargs.update(dict(extra))
    return ModelBuildConfig(**kwargs)


#: CLI-friendly aliases -> canonical format factories used by CAP3-05.
_FORMAT_FACTORIES: dict[str, Any] = {
    "fp16": fp16_format,
    "int8": int8_format,
    "int4": int4_format,
    "binary": binary_format,
    "ternary": ternary_format,
    "learned4zero": learned_four_level_zero_format,
    "learned_four_level_zero": learned_four_level_zero_format,
    "symmetric4": symmetric_four_level_format,
    "symmetric_four_level": symmetric_four_level_format,
}


def _format_factory(format_id: str, group_size: int = 128) -> Any:
    """Return a ``QuantFormat`` for a CLI alias."""
    factory = _FORMAT_FACTORIES.get(format_id)
    if factory is None:
        raise ValueError(
            f"unknown precision format {format_id!r}; "
            f"supported: {sorted(_FORMAT_FACTORIES)}"
        )
    return factory(group_size=group_size)


class _SyntheticBlock(nn.Module):
    """Minimal transformer-like block for byte estimation."""

    def __init__(self, d_model: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.attn = nn.Linear(d_model, d_model, bias=True)
        self.mlp = nn.Linear(d_model, d_model * ffn_mult, bias=True)
        self.mlp_out = nn.Linear(d_model * ffn_mult, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)


class _SyntheticTwoTower(nn.Module):
    """Deterministic synthetic model whose named parameters match CAP3-04 groups.

    The names intentionally contain the substrings used by the default grouping
    policy and the CAP3-01 ledger's auto-exclusion heuristics (``embed``,
    ``norm``, ``bias``).  It is *not* a real TwoTower; it is a byte-estimation
    surrogate used only for planning.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_layers: int,
        denoiser_layers: int,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.semantic_input = nn.Linear(d_model, d_model, bias=True)
        self.input_projection = nn.Linear(d_model, d_model, bias=True)
        self.context_encoder = nn.ModuleList(
            [_SyntheticBlock(d_model, ffn_mult=ffn_mult) for _ in range(context_layers)]
        )
        self.denoiser = nn.ModuleList(
            [_SyntheticBlock(d_model, ffn_mult=ffn_mult) for _ in range(denoiser_layers)]
        )
        self.latent_projection = nn.Linear(d_model, d_model, bias=True)
        self.local_head = nn.ModuleDict({"scorer": nn.Linear(d_model, 8, bias=True)})
        # Embeddings are auto-excluded from quantization by the ledger.
        self.action_embeddings = nn.Parameter(torch.randn(8, d_model))


def estimate_bytes(
    d_model: int,
    n_heads: int,
    context_layers: int,
    denoiser_layers: int,
    format_id: str,
    *,
    group_size: int = 128,
    ffn_mult: int = 4,
) -> int:
    """Return modeled whole-model bytes for a config/format using the CAP3-01 ledger."""
    model = _SyntheticTwoTower(
        d_model=d_model,
        n_heads=n_heads,
        context_layers=context_layers,
        denoiser_layers=denoiser_layers,
        ffn_mult=ffn_mult,
    )
    fmt = _format_factory(format_id, group_size=group_size)
    ledger = build_model_ledger(
        model,
        format_map={},
        default_format=fmt,
        d_model=d_model,
    )
    return ledger.total()


def plan_equal_byte_ladder(
    byte_budgets: tuple[int, ...],
    formats: tuple[str, ...],
    *,
    widths: tuple[int, ...] = (32, 64, 96, 128, 192),
    horizons: tuple[float, ...] = (1.0,),
    depth_policy: str = "proportional",
    head_policy: str = "proportional",
    base_token_budget: int = 50_000,
    group_size: int = 128,
    tolerance: float = 0.03,
    representation: str = "compositional",
    ffn_mult: int = 4,
) -> tuple[ScalingLadder, ...]:
    """Plan a width × precision ladder where every point targets the same byte budget.

    For each ``(byte_budget, format)`` pair the planner searches ``widths`` and
    ``horizons`` and selects the candidate whose modeled whole-model bytes are
    closest to ``byte_budget`` within ``tolerance``.  Formats that cannot fit the
    budget are marked ``infeasible`` but still emitted so the manifest records
    the failure explicitly.

    Bytes are *modeled* via ``estimate_bytes`` (a synthetic TwoTower-like module
    + ``build_model_ledger``), not measured on a deployed device.
    """
    ladders: list[ScalingLadder] = []

    for budget in byte_budgets:
        for fmt_id in formats:
            points: list[LadderPoint] = []
            ref = 128
            candidates: list[tuple[int, int, int, int, float, int, float]] = []
            for d in widths:
                if head_policy == "proportional":
                    n_heads, ctx, den = proportional_depths(d)
                else:
                    n_heads, ctx, den = max(2, d // 32), 2, 4
                if depth_policy == "fixed":
                    ctx = max(1, widths[0] // 64) if widths else 1
                    den = max(2, widths[0] // 32) if widths else 2
                scale = (d / ref) ** 2
                token_budget = max(1_000, int(base_token_budget * scale))
                for h in horizons:
                    actual = estimate_bytes(
                        d,
                        n_heads,
                        ctx,
                        den,
                        fmt_id,
                        group_size=group_size,
                        ffn_mult=ffn_mult,
                    )
                    delta = (actual - budget) / budget if budget else 0.0
                    candidates.append((d, n_heads, ctx, den, h, actual, delta))

            feasible = [c for c in candidates if abs(c[6]) <= tolerance]
            if feasible:
                chosen = min(feasible, key=lambda c: abs(c[6]))
                status = "feasible"
            else:
                under = [c for c in candidates if c[5] <= budget]
                if under:
                    chosen = max(under, key=lambda c: c[5])
                    status = "infeasible"
                else:
                    # No configuration fits; record the smallest width as infeasible
                    # so the failure is visible in the manifest.
                    chosen = candidates[0]
                    status = "infeasible"

            d, n_heads, ctx, den, h, actual, delta = chosen
            points.append(
                LadderPoint(
                    d_model=d,
                    n_heads=n_heads,
                    context_layers=ctx,
                    denoiser_layers=den,
                    target_token_budget=max(1_000, int(token_budget * h)),
                    horizon_multiplier=h,
                    representation=representation,
                    byte_budget=budget,
                    precision_format=fmt_id,
                    actual_bytes=actual,
                    budget_delta=delta,
                    status=status,
                )
            )

            ladders.append(
                ScalingLadder(
                    ladder_id=f"equal_byte_{budget}_{fmt_id}_v1",
                    track="scratch",
                    points=tuple(points),
                    token_horizons=horizons,
                    decode_frozen={
                        "gen_steps": 8,
                        "best_of_n": 1,
                        "grammar_constrained": True,
                        "parallel_unmask": "adaptive",
                    },
                )
            )

    return tuple(ladders)
