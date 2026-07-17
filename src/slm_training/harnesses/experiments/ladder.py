"""Scaling ladder definitions (P1c)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from slm_training.harnesses.model_build.config import ModelBuildConfig


@dataclass(frozen=True)
class LadderPoint:
    d_model: int
    n_heads: int
    context_layers: int
    denoiser_layers: int
    target_token_budget: int
    horizon_multiplier: float = 1.0

    @property
    def point_id(self) -> str:
        return (
            f"d{self.d_model}_h{self.n_heads}_c{self.context_layers}_"
            f"dn{self.denoiser_layers}_t{self.target_token_budget}_x{self.horizon_multiplier:g}"
        )


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
                )
            )
    return ScalingLadder(
        ladder_id="scratch_v1",
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


def hf_ladder_default(
    *,
    base_token_budget: int = 50_000,
    widths: tuple[int, ...] = (64, 96, 128, 192),
    horizons: tuple[float, ...] = (0.5, 1.0, 2.0),
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
        "output_tokenizer": ladder.output_tokenizer,
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
    if extra:
        kwargs.update(dict(extra))
    return ModelBuildConfig(**kwargs)
