"""CAP3-03: equal-physical-storage ternary falsification matrix.

Fixture/wiring harness that compares low-bit weight representations at matched
physical storage on a local action scorer.  Real checkpoints/GPU are not required
for the wiring path; ship-grade claims need the full eval suite and --ship-gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.nn.functional as F

from slm_training.harnesses.quantization.calibration import (
    CalibrationCorpusManifest,
    CalibrationSample,
    calibrate_scales_ptq,
    qat_reconstruct_local_scorer,
)
from slm_training.models.quantization import (
    binary_format,
    binary_plus_mask_format,
    build_model_ledger,
    int4_format,
    int8_format,
    learned_four_level_zero_format,
    symmetric_four_level_format,
    ternary_format,
)
from slm_training.models.quantization.formats import QuantFormat

if TYPE_CHECKING:
    from slm_training.models.local_action_head import LocalFlatHead


CAP3_03_VERSION = "cap3-03-v1"


@dataclass(frozen=True)
class ArmConfig:
    """One matrix arm."""

    arm_id: str
    format_id: str
    group_size: int
    seed: int
    checkpoint_id: str
    calibration_manifest_sha: str
    qat_steps: int = 0
    qat_lr: float = 1e-2


@dataclass
class ArmResult:
    """Results for one (format, seed) arm."""

    arm_id: str
    format_id: str
    group_size: int
    seed: int
    checkpoint_id: str
    sample_count: int
    top1_accuracy: float
    teacher_top1_accuracy: float
    action_flip_rate: float
    kl_to_teacher: float
    margin_preservation: float
    mean_regret: float
    cvar90_regret: float
    zero_rate: float
    support_rate: float
    symbol_entropy_bits: float
    physical_weight_bytes: int
    total_bytes: int
    ledger_sha256: str
    status: str = "ok"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "format_id": self.format_id,
            "group_size": self.group_size,
            "seed": self.seed,
            "checkpoint_id": self.checkpoint_id,
            "sample_count": self.sample_count,
            "top1_accuracy": self.top1_accuracy,
            "teacher_top1_accuracy": self.teacher_top1_accuracy,
            "action_flip_rate": self.action_flip_rate,
            "kl_to_teacher": self.kl_to_teacher,
            "margin_preservation": self.margin_preservation,
            "mean_regret": self.mean_regret,
            "cvar90_regret": self.cvar90_regret,
            "zero_rate": self.zero_rate,
            "support_rate": self.support_rate,
            "symbol_entropy_bits": self.symbol_entropy_bits,
            "physical_weight_bytes": self.physical_weight_bytes,
            "total_bytes": self.total_bytes,
            "ledger_sha256": self.ledger_sha256,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class MatrixReport:
    """Versioned CAP3-03 matrix report."""

    version: str
    run_id: str
    timestamp: str
    checkpoint_id: str
    formats: tuple[str, ...]
    group_size: int
    seeds: tuple[int, ...]
    sample_count: int
    sampling_strategy: str
    calibration_manifest_sha: str
    arms: list[ArmResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "checkpoint_id": self.checkpoint_id,
            "formats": list(self.formats),
            "group_size": self.group_size,
            "seeds": list(self.seeds),
            "sample_count": self.sample_count,
            "sampling_strategy": self.sampling_strategy,
            "calibration_manifest_sha": self.calibration_manifest_sha,
            "arms": [a.as_dict() for a in self.arms],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)


@dataclass(frozen=True)
class MatchedConditions:
    """Fields that must be identical across arms; used to assert fair comparison."""

    checkpoint_id: str
    group_size: int
    physical_slot_bits: int
    calibration_manifest_sha: str
    sample_count: int
    sampling_strategy: str
    activation_dtype: str
    accumulation_dtype: str
    qat_steps: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_matches(self, other: MatchedConditions) -> None:
        diffs = [
            k
            for k in self.as_dict()
            if self.as_dict()[k] != other.as_dict()[k]
        ]
        if diffs:
            raise ValueError(f"matched conditions differ: {diffs}")


FORMAT_FACTORIES: dict[str, Any] = {
    "fp16": lambda group_size=128: _raise_unsupported("fp16"),
    "int8": int8_format,
    "int4": int4_format,
    "binary": binary_format,
    "ternary": ternary_format,
    "symmetric4": symmetric_four_level_format,
    "symmetric_four_level": symmetric_four_level_format,
    "learned4zero": lambda group_size=128: learned_four_level_zero_format(
        levels=(-1.0, 0.0, 1.0, 2.0), group_size=group_size
    ),
    "learned_four_level_zero": lambda group_size=128: learned_four_level_zero_format(
        levels=(-1.0, 0.0, 1.0, 2.0), group_size=group_size
    ),
    "binary_plus_mask": binary_plus_mask_format,
}


def _raise_unsupported(name: str) -> QuantFormat:
    raise ValueError(
        f"{name!r} is a control format; include it via the baseline arm, not this factory"
    )


def make_format(format_id: str, group_size: int) -> QuantFormat:
    """Build a QuantFormat from the matrix arm id."""
    factory = FORMAT_FACTORIES.get(format_id)
    if factory is None:
        raise ValueError(f"unknown format_id: {format_id!r}")
    return factory(group_size=group_size)


def _format_fingerprint(fmt: QuantFormat) -> str:
    """Stable string for matched-condition hashing."""
    levels = fmt.learned_levels if fmt.is_learned else fmt.weight_levels
    return json.dumps(
        {
            "format_id": fmt.format_id,
            "physical_slot_bits": fmt.physical_slot_bits,
            "group_size": fmt.group_size,
            "levels": levels,
            "activation_dtype": fmt.activation_dtype,
            "accumulation_dtype": fmt.accumulation_dtype,
        },
        sort_keys=True,
        default=str,
    )


def _sha256_json(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hidden_for_sample(sample: CalibrationSample, hidden_dim: int, device: torch.device) -> torch.Tensor:
    """Deterministic hidden vector for a calibration sample."""
    seed = int(_sha256_json(sample.trace_id)[:16], 16)
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, hidden_dim, generator=generator, device=device)


def _warm_embeddings(head: "LocalFlatHead", samples: list[CalibrationSample]) -> None:
    """Ensure every legal action in the corpus has a lazy embedding parameter."""
    from slm_training.models.local_action_head import StateContext

    ctx = StateContext(state_family_id="cap3-03")
    actions: set[str] = set()
    for s in samples:
        actions.update(s.legal_action_ids)
    # Lazy action_embeddings is created on the first score() call.
    if not hasattr(head, "action_embeddings"):
        h = torch.randn(1, head.hidden_dim)
        head.score(h, ctx, ["__warmup__"])
    for action in sorted(actions):
        if action not in head.action_embeddings:
            # Trigger creation via a tiny forward.
            h = torch.randn(1, head.hidden_dim)
            head.score(h, ctx, [action])


def _quantize_action_embeddings(
    head: "LocalFlatHead",
    fmt: QuantFormat,
    samples: list[CalibrationSample],
) -> dict[str, torch.Tensor]:
    """Return PTQ-quantized action-embedding tensors for all corpus actions."""
    _warm_embeddings(head, samples)
    actions = sorted({a for s in samples for a in s.legal_action_ids})
    quantized: dict[str, torch.Tensor] = {}
    for action in actions:
        param = head.action_embeddings.get(action)
        if param is None:
            continue
        q, _, _ = calibrate_scales_ptq(param.data, fmt, group_size=fmt.group_size)
        quantized[action] = q.detach().clone()
    return quantized


def _student_logits(
    hidden: torch.Tensor,
    legal_actions: list[str],
    quantized_embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute logits from quantized action embeddings."""
    embeddings = [quantized_embeddings[a] for a in legal_actions]
    stacked = torch.stack(embeddings, dim=0)
    return hidden @ stacked.T


def _symbol_entropy(quantized_embeddings: dict[str, torch.Tensor], fmt: QuantFormat) -> float:
    """Empirical Shannon entropy of the quantized symbol distribution."""
    levels = fmt.learned_levels if fmt.is_learned else fmt.weight_levels
    if not levels:
        return 0.0
    flat = torch.cat([v.flatten() for v in quantized_embeddings.values()])
    level_t = torch.tensor(levels, dtype=flat.dtype, device=flat.device)
    idx = flat.sub(level_t.view(-1, 1)).abs().argmin(dim=0)
    counts = torch.bincount(idx, minlength=len(levels)).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    if probs.numel() == 0:
        return 0.0
    return float(-(probs * torch.log2(probs)).sum().item())


def _zero_and_support_rates(
    quantized_embeddings: dict[str, torch.Tensor],
    fmt: QuantFormat,
) -> tuple[float, float]:
    """Return (zero_rate, support_rate) across quantized embeddings."""
    flat = torch.cat([v.flatten() for v in quantized_embeddings.values()])
    if flat.numel() == 0:
        return 0.0, 0.0
    zero_mask = flat == 0.0
    zero_rate = float(zero_mask.float().mean().item())
    support_rate = 1.0 - zero_rate
    return zero_rate, support_rate


def _cvar(values: list[float], alpha: float = 0.9) -> float:
    """Average of the worst (1-alpha) fraction of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values, reverse=True)
    k = max(1, math.ceil((1 - alpha) * len(sorted_vals)))
    return sum(sorted_vals[:k]) / k


def evaluate_arm(
    arm: ArmConfig,
    model: Any,
    manifest: CalibrationCorpusManifest,
    samples: list[CalibrationSample],
    *,
    hidden_dim: int = 64,
    device: torch.device | None = None,
) -> ArmResult:
    """Evaluate one matrix arm and return a versioned result row."""
    from slm_training.models.local_action_head import LocalFlatHead, StateContext

    if device is None:
        device = torch.device("cpu")

    fmt = make_format(arm.format_id, arm.group_size)

    # Find the local flat head target.
    head: LocalFlatHead | None = None
    for name, module in model.named_modules():
        if isinstance(module, LocalFlatHead):
            head = module
            break
    if head is None:
        return ArmResult(
            arm_id=arm.arm_id,
            format_id=arm.format_id,
            group_size=arm.group_size,
            seed=arm.seed,
            checkpoint_id=arm.checkpoint_id,
            sample_count=0,
            top1_accuracy=0.0,
            teacher_top1_accuracy=0.0,
            action_flip_rate=0.0,
            kl_to_teacher=0.0,
            margin_preservation=0.0,
            mean_regret=0.0,
            cvar90_regret=0.0,
            zero_rate=0.0,
            support_rate=0.0,
            symbol_entropy_bits=0.0,
            physical_weight_bytes=0,
            total_bytes=0,
            ledger_sha256="",
            status="error",
            notes=["no LocalFlatHead found in model"],
        )

    _warm_embeddings(head, samples)

    # Precompute teacher logits on unquantized head with deterministic hidden vectors.
    teacher_batches: list[tuple[torch.Tensor, list[str], torch.Tensor]] = []
    ctx = StateContext(state_family_id="cap3-03")
    for s in samples:
        hidden = _hidden_for_sample(s, hidden_dim, device)
        legal = list(s.legal_action_ids)
        with torch.no_grad():
            out = head.score(hidden, ctx, legal)
        teacher_logits = out.logits if out.logits is not None else torch.zeros(1, len(legal))
        teacher_batches.append((hidden, legal, teacher_logits))

    # Optional short QAT before final PTQ.
    if arm.qat_steps > 0:
        qat_reconstruct_local_scorer(head, fmt, teacher_batches, steps=arm.qat_steps, lr=arm.qat_lr)

    quantized_embeddings = _quantize_action_embeddings(head, fmt, samples)

    # Evaluate.
    top1_correct = 0
    teacher_top1_correct = 0
    flips = 0
    kls: list[float] = []
    margin_deltas: list[float] = []
    regrets: list[float] = []

    for s, (hidden, legal, teacher_logits) in zip(samples, teacher_batches):
        if not legal:
            continue
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_top1_idx = int(teacher_probs.argmax(dim=-1).item())
        teacher_top1_action = legal[teacher_top1_idx]

        student_logits = _student_logits(hidden, legal, quantized_embeddings)
        student_probs = F.softmax(student_logits, dim=-1)
        student_idx = int(student_probs.argmax(dim=-1).item())
        student_action = legal[student_idx]

        if s.selected_action_id is not None and student_action == s.selected_action_id:
            top1_correct += 1
        if student_action == teacher_top1_action:
            teacher_top1_correct += 1
        else:
            flips += 1

        kl = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction="batchmean",
        )
        kls.append(float(kl.item()))

        # Margin preservation for selected action (if known).
        if s.selected_action_id is not None and s.selected_action_id in legal:
            sel_idx = legal.index(s.selected_action_id)
            teacher_margin = (
                teacher_logits[0, sel_idx].item() - teacher_logits[0].max().item()
            )
            student_margin = (
                student_logits[0, sel_idx].item() - student_logits[0].max().item()
            )
            margin_deltas.append(student_margin - teacher_margin)

        regrets.append(0.0 if student_action == teacher_top1_action else 1.0)

    n = len(samples)
    zero_rate, support_rate = _zero_and_support_rates(quantized_embeddings, fmt)
    symbol_entropy = _symbol_entropy(quantized_embeddings, fmt)

    # Physical-cost ledger.
    ledger = build_model_ledger(model, {}, default_format=fmt)
    ledger_dict = ledger.as_dict()
    ledger_sha256 = _sha256_json(ledger_dict)
    fmt_report = ledger.formats.get(fmt.format_id)
    physical_weight_bytes = fmt_report.physical_weight_bytes if fmt_report else 0
    total_bytes = ledger.total()

    notes: list[str] = []
    if arm.qat_steps > 0:
        notes.append(f"QAT reconstruction: {arm.qat_steps} steps")

    return ArmResult(
        arm_id=arm.arm_id,
        format_id=arm.format_id,
        group_size=arm.group_size,
        seed=arm.seed,
        checkpoint_id=arm.checkpoint_id,
        sample_count=n,
        top1_accuracy=top1_correct / n if n else 0.0,
        teacher_top1_accuracy=teacher_top1_correct / n if n else 0.0,
        action_flip_rate=flips / n if n else 0.0,
        kl_to_teacher=sum(kls) / len(kls) if kls else 0.0,
        margin_preservation=sum(margin_deltas) / len(margin_deltas) if margin_deltas else 0.0,
        mean_regret=sum(regrets) / len(regrets) if regrets else 0.0,
        cvar90_regret=_cvar(regrets, alpha=0.9),
        zero_rate=zero_rate,
        support_rate=support_rate,
        symbol_entropy_bits=symbol_entropy,
        physical_weight_bytes=physical_weight_bytes,
        total_bytes=total_bytes,
        ledger_sha256=ledger_sha256,
        status="ok",
        notes=notes,
    )


def build_arms(
    checkpoint_id: str,
    formats: Sequence[str],
    group_size: int,
    seeds: Sequence[int],
    calibration_manifest_sha: str,
    qat_steps: int = 0,
    qat_lr: float = 1e-2,
) -> list[ArmConfig]:
    """Return one ArmConfig per (format, seed)."""
    arms: list[ArmConfig] = []
    for fmt in formats:
        for seed in seeds:
            arm_id = f"{fmt}_gs{group_size}_s{seed}"
            arms.append(
                ArmConfig(
                    arm_id=arm_id,
                    format_id=fmt,
                    group_size=group_size,
                    seed=seed,
                    checkpoint_id=checkpoint_id,
                    calibration_manifest_sha=calibration_manifest_sha,
                    qat_steps=qat_steps,
                    qat_lr=qat_lr,
                )
            )
    return arms


def run_matrix(
    model: Any,
    manifest: CalibrationCorpusManifest,
    samples: list[CalibrationSample],
    formats: Sequence[str],
    *,
    group_size: int = 128,
    seeds: Sequence[int] = (0,),
    qat_steps: int = 0,
    qat_lr: float = 1e-2,
    hidden_dim: int = 64,
    run_id: str | None = None,
) -> MatrixReport:
    """Run the full CAP3-03 falsification matrix."""
    if run_id is None:
        run_id = f"cap3-03-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    manifest_sha = _sha256_json(manifest.as_dict())
    arms_cfg = build_arms(
        manifest.checkpoint_id,
        formats,
        group_size,
        seeds,
        manifest_sha,
        qat_steps=qat_steps,
        qat_lr=qat_lr,
    )

    # Build a shared baseline conditions object for cross-arm parity checks.
    baseline_fmt = make_format(formats[0], group_size)
    baseline = MatchedConditions(
        checkpoint_id=manifest.checkpoint_id,
        group_size=group_size,
        physical_slot_bits=baseline_fmt.physical_slot_bits,
        calibration_manifest_sha=manifest_sha,
        sample_count=manifest.sample_count,
        sampling_strategy=manifest.sampling_strategy,
        activation_dtype=baseline_fmt.activation_dtype,
        accumulation_dtype=baseline_fmt.accumulation_dtype,
        qat_steps=qat_steps,
    )

    results: list[ArmResult] = []
    for arm in arms_cfg:
        fmt = make_format(arm.format_id, arm.group_size)
        current = MatchedConditions(
            checkpoint_id=arm.checkpoint_id,
            group_size=arm.group_size,
            physical_slot_bits=fmt.physical_slot_bits,
            calibration_manifest_sha=arm.calibration_manifest_sha,
            sample_count=manifest.sample_count,
            sampling_strategy=manifest.sampling_strategy,
            activation_dtype=fmt.activation_dtype,
            accumulation_dtype=fmt.accumulation_dtype,
            qat_steps=arm.qat_steps,
        )
        try:
            baseline.assert_matches(current)
        except ValueError as exc:
            results.append(
                ArmResult(
                    arm_id=arm.arm_id,
                    format_id=arm.format_id,
                    group_size=arm.group_size,
                    seed=arm.seed,
                    checkpoint_id=arm.checkpoint_id,
                    sample_count=0,
                    top1_accuracy=0.0,
                    teacher_top1_accuracy=0.0,
                    action_flip_rate=0.0,
                    kl_to_teacher=0.0,
                    margin_preservation=0.0,
                    mean_regret=0.0,
                    cvar90_regret=0.0,
                    zero_rate=0.0,
                    support_rate=0.0,
                    symbol_entropy_bits=0.0,
                    physical_weight_bytes=0,
                    total_bytes=0,
                    ledger_sha256="",
                    status="error",
                    notes=[str(exc)],
                )
            )
            continue

        # Set seed for reproducible model initialization / QAT.
        torch.manual_seed(arm.seed)
        random.seed(arm.seed)
        result = evaluate_arm(arm, model, manifest, samples, hidden_dim=hidden_dim)
        results.append(result)

    return MatrixReport(
        version=CAP3_03_VERSION,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checkpoint_id=manifest.checkpoint_id,
        formats=tuple(formats),
        group_size=group_size,
        seeds=tuple(seeds),
        sample_count=manifest.sample_count,
        sampling_strategy=manifest.sampling_strategy,
        calibration_manifest_sha=manifest_sha,
        arms=results,
    )
