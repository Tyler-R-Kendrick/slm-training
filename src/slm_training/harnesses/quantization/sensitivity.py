"""CAP3-04: grammar-conditioned quantization sensitivity profiling.

Fixture/wiring harness that measures direct task perturbation when a single
parameter group is quantized, then restores the baseline.  Ship-grade profiling
needs GPU + real checkpoints + full --ship-gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F

from slm_training.harnesses.quantization.calibration import (
    CalibrationCorpusManifest,
    CalibrationSample,
    calibrate_scales_ptq,
)
from slm_training.models.local_action_head import LocalFlatHead, StateContext

if TYPE_CHECKING:
    from slm_training.models.quantization.formats import QuantFormat


CAP3_04_VERSION = "cap3-04-v1"


@dataclass(frozen=True)
class ParameterGroup:
    """One quantizable parameter group in a model."""

    group_id: str
    path_patterns: tuple[str, ...] = ()
    type_patterns: tuple[str, ...] = ()
    param_name_patterns: tuple[str, ...] = ()
    quantize_kind: Literal["linear_weight", "embedding_dict", "none"] = "linear_weight"
    exclusion_reason: str | None = None

    def matches(self, module_path: str, module: torch.nn.Module, param_name: str) -> bool:
        if self.exclusion_reason:
            return False
        if any(re.search(p, module_path) for p in self.path_patterns):
            return True
        type_name = type(module).__name__
        if any(t.lower() in type_name.lower() for t in self.type_patterns):
            return True
        full_name = f"{module_path}.{param_name}" if module_path else param_name
        if any(re.search(p, full_name) for p in self.param_name_patterns):
            return True
        return False


@dataclass(frozen=True)
class GroupingPolicy:
    """Versioned collection of parameter groups."""

    version: str
    groups: tuple[ParameterGroup, ...]
    default_exclusion: str = "unmatched by grouping policy"


@dataclass
class GroupFormatPoint:
    """One direct-perturbation measurement: quantize ``group_id`` with ``format_id``."""

    group_id: str
    format_id: str
    group_size: int
    packed_bytes: int
    total_bytes: int
    sample_count: int
    top1_accuracy: float
    teacher_top1_accuracy: float
    action_flip_rate: float
    kl_to_teacher: float
    margin_preservation: float
    mean_regret: float
    cvar90_regret: float
    gradient_proxy: float | None = None
    status: str = "ok"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "format_id": self.format_id,
            "group_size": self.group_size,
            "packed_bytes": self.packed_bytes,
            "total_bytes": self.total_bytes,
            "sample_count": self.sample_count,
            "top1_accuracy": self.top1_accuracy,
            "teacher_top1_accuracy": self.teacher_top1_accuracy,
            "action_flip_rate": self.action_flip_rate,
            "kl_to_teacher": self.kl_to_teacher,
            "margin_preservation": self.margin_preservation,
            "mean_regret": self.mean_regret,
            "cvar90_regret": self.cvar90_regret,
            "gradient_proxy": self.gradient_proxy,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class SensitivityReport:
    """Versioned envelope for a sensitivity profile."""

    version: str
    run_id: str
    timestamp: str
    checkpoint_id: str
    calibration_manifest_sha: str
    grouping_policy_version: str
    formats: tuple[str, ...]
    sample_count: int
    gradient_proxies: dict[str, float]
    points: list[GroupFormatPoint]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "checkpoint_id": self.checkpoint_id,
            "calibration_manifest_sha": self.calibration_manifest_sha,
            "grouping_policy_version": self.grouping_policy_version,
            "formats": list(self.formats),
            "sample_count": self.sample_count,
            "gradient_proxies": self.gradient_proxies,
            "points": [p.as_dict() for p in self.points],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)


def default_grouping_policy() -> GroupingPolicy:
    """A conservative default policy for TwoTower + LocalFlatHead models."""
    return GroupingPolicy(
        version="cap3-04-default-v1",
        groups=(
            ParameterGroup(
                group_id="semantic_input",
                path_patterns=(r"^semantic_input", r"^input_projection"),
                quantize_kind="linear_weight",
            ),
            ParameterGroup(
                group_id="context_encoder",
                path_patterns=(r"\.context_encoder\.", r"\.context\."),
                quantize_kind="linear_weight",
            ),
            ParameterGroup(
                group_id="denoiser",
                path_patterns=(r"\.denoiser\.", r"\.decoder\."),
                quantize_kind="linear_weight",
            ),
            ParameterGroup(
                group_id="attention",
                path_patterns=(r"\.attn\.", r"\.attention\."),
                quantize_kind="linear_weight",
            ),
            ParameterGroup(
                group_id="mlp",
                path_patterns=(r"\.mlp\.", r"\.ffn\."),
                quantize_kind="linear_weight",
            ),
            ParameterGroup(
                group_id="latent_projection",
                path_patterns=(r"latent", r"\.z_proj\."),
                quantize_kind="linear_weight",
            ),
            ParameterGroup(
                group_id="local_head/scorer",
                path_patterns=(r"local_head\.scorer", r"action_scorer"),
                quantize_kind="linear_weight",
            ),
            ParameterGroup(
                group_id="local_head/embeddings",
                param_name_patterns=(r"local_head\.action_embeddings\.",),
                quantize_kind="embedding_dict",
            ),
            ParameterGroup(
                group_id="norms_and_biases",
                param_name_patterns=(r"\.norm\.", r"\.bias", r"embed", r"lm_head"),
                quantize_kind="none",
                exclusion_reason="excluded by policy: norm/bias/embed/head",
            ),
        ),
    )


def _sha256_json(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hidden_for_sample(sample: CalibrationSample, hidden_dim: int, device: torch.device) -> torch.Tensor:
    """Deterministic hidden vector for a calibration sample."""
    seed = int(_sha256_json(sample.trace_id)[:16], 16)
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, hidden_dim, generator=generator, device=device)


def _find_local_head(model: torch.nn.Module) -> LocalFlatHead | None:
    for module in model.modules():
        if isinstance(module, LocalFlatHead):
            return module
    return None


def _warm_local_head(head: LocalFlatHead, legal_action_ids: set[str]) -> None:
    """Ensure every legal action has a lazy embedding parameter."""
    ctx = StateContext(state_family_id="cap3-04")
    if not hasattr(head, "action_embeddings"):
        h = torch.randn(1, head.hidden_dim)
        head.score(h, ctx, ["__warmup__"])
    for action in sorted(legal_action_ids):
        if action not in head.action_embeddings:
            h = torch.randn(1, head.hidden_dim)
            head.score(h, ctx, [action])


def _iter_group_params(
    model: torch.nn.Module,
    group: ParameterGroup,
) -> list[tuple[str, torch.nn.Parameter, torch.nn.Module | None]]:
    """Return (full_param_name, param, parent_module) tuples matched by ``group``."""
    matched: list[tuple[str, torch.nn.Parameter, torch.nn.Module | None]] = []
    seen_names: set[str] = set()
    for module_path, module in model.named_modules():
        # LocalFlatHead lazy action embeddings are stored in a dict and not
        # returned by named_parameters(recurse=False); surface them explicitly.
        if isinstance(module, LocalFlatHead) and hasattr(module, "action_embeddings"):
            for action, param in module.action_embeddings.items():
                param_name = f"action_embeddings.{action}"
                full_name = f"{module_path}.{param_name}" if module_path else param_name
                if full_name in seen_names:
                    continue
                if group.matches(module_path, module, param_name):
                    matched.append((full_name, param, module))
                    seen_names.add(full_name)
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_path}.{param_name}" if module_path else param_name
            if full_name in seen_names:
                continue
            if group.matches(module_path, module, param_name):
                matched.append((full_name, param, module))
                seen_names.add(full_name)
    # Also catch parameters that live directly on the root module.
    for param_name, param in model.named_parameters(recurse=False):
        if param_name in seen_names:
            continue
        if group.matches("", model, param_name):
            matched.append((param_name, param, None))
            seen_names.add(param_name)
    return matched


def _snapshot_group(
    model: torch.nn.Module,
    group: ParameterGroup,
) -> dict[str, torch.Tensor]:
    """Clone all parameter tensors belonging to ``group``."""
    snapshots: dict[str, torch.Tensor] = {}
    for full_name, param, _ in _iter_group_params(model, group):
        snapshots[full_name] = param.data.clone()
    return snapshots


def _restore_group(snapshots: dict[str, torch.Tensor]) -> None:
    for full_name, original in snapshots.items():
        # Caller must keep param references; we store by object in practice.
        pass


def _quantize_linear_weight(
    module: torch.nn.Module,
    fmt: QuantFormat,
) -> None:
    """In-place fake-quantize a Linear module's weight."""
    from slm_training.models.quantization.fake_quant import fake_quantize_weight

    if not isinstance(module, torch.nn.Linear):
        return
    q, _, _ = fake_quantize_weight(module.weight.data, fmt, group_size=fmt.group_size)
    module.weight.data = q.to(module.weight.dtype)


def _quantize_embedding_dict(
    head: LocalFlatHead,
    fmt: QuantFormat,
) -> None:
    """In-place PTQ-quantize every lazy action embedding on ``head``."""
    for action, param in head.action_embeddings.items():
        q, _, _ = calibrate_scales_ptq(param.data, fmt, group_size=fmt.group_size)
        param.data = q.to(param.dtype)


def _quantize_group(
    model: torch.nn.Module,
    group: ParameterGroup,
    fmt: QuantFormat,
) -> None:
    """Apply quantization to one group in-place."""
    if group.quantize_kind == "none":
        return
    for full_name, param, module in _iter_group_params(model, group):
        if group.quantize_kind == "linear_weight" and isinstance(module, torch.nn.Linear):
            _quantize_linear_weight(module, fmt)
        elif group.quantize_kind == "embedding_dict" and isinstance(module, LocalFlatHead):
            _quantize_embedding_dict(module, fmt)


def _format_map_for_group(
    model: torch.nn.Module,
    group: ParameterGroup,
    fmt: QuantFormat,
) -> dict[str, QuantFormat]:
    """Return {param_name: fmt} for group members; others will default to FP16."""
    mapping: dict[str, QuantFormat] = {}
    for full_name, _, _ in _iter_group_params(model, group):
        mapping[full_name] = fmt
    return mapping


def _group_bytes_from_ledger(
    ledger: Any,
    group_param_names: set[str],
    fmt_id: str,
) -> tuple[int, int]:
    """Return (packed_weight_bytes, total_bytes) for group members under ``fmt_id``."""
    from slm_training.models.quantization.cost import PhysicalCostLedger

    if not isinstance(ledger, PhysicalCostLedger):
        return 0, 0
    packed = 0
    total = 0
    seen = set()
    for fmt_report in ledger.formats.values():
        for tensor in fmt_report.tensors:
            if tensor.name in group_param_names and tensor.format_id == fmt_id:
                packed += tensor.physical_weight_bytes
                total += tensor.total_bytes
                seen.add(tensor.name)
    return packed, total, seen


def _compute_group_bytes(
    model: torch.nn.Module,
    group: ParameterGroup,
    fmt: QuantFormat,
    hidden_dim: int,
) -> tuple[int, int]:
    """Physical bytes for group members, with a fallback for lazy dict parameters."""
    from slm_training.models.quantization.cost import compute_tensor_cost

    packed = 0
    total = 0
    activation_shape = (1, hidden_dim)
    for full_name, param, _ in _iter_group_params(model, group):
        cost = compute_tensor_cost(
            full_name,
            param,
            fmt,
            group_size=fmt.group_size,
            bias=None,
            activation_shape=activation_shape,
            exclusion_reason=None,
        )
        packed += cost.physical_weight_bytes
        total += cost.total_bytes
    return packed, total


def _cvar(values: list[float], alpha: float = 0.9) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values, reverse=True)
    k = max(1, math.ceil((1 - alpha) * len(sorted_vals)))
    return sum(sorted_vals[:k]) / k


def _score_local_head(
    head: LocalFlatHead,
    samples: list[CalibrationSample],
    hidden_dim: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, list[str], torch.Tensor]]:
    """Return deterministic (hidden, legal_actions, logits) batches for a local head."""
    ctx = StateContext(state_family_id="cap3-04")
    batches: list[tuple[torch.Tensor, list[str], torch.Tensor]] = []
    all_actions: set[str] = set()
    for s in samples:
        all_actions.update(s.legal_action_ids)
    _warm_local_head(head, all_actions)

    with torch.no_grad():
        for s in samples:
            hidden = _hidden_for_sample(s, hidden_dim, device)
            legal = list(s.legal_action_ids)
            if not legal:
                legal = ["__noop__"]
            out = head.score(hidden, ctx, legal)
            logits = out.logits if out.logits is not None else torch.zeros(1, len(legal), device=device)
            batches.append((hidden, legal, logits))
    return batches


def _evaluate_perturbation(
    head: LocalFlatHead,
    teacher_batches: list[tuple[torch.Tensor, list[str], torch.Tensor]],
    samples: list[CalibrationSample],
) -> dict[str, float]:
    """Compare the current (possibly quantized) head against ``teacher_batches``."""
    ctx = StateContext(state_family_id="cap3-04")
    top1_correct = 0
    teacher_top1_correct = 0
    flips = 0
    kls: list[float] = []
    margin_deltas: list[float] = []
    regrets: list[float] = []

    with torch.no_grad():
        for s, (hidden, legal, teacher_logits) in zip(samples, teacher_batches):
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            teacher_top1_idx = int(teacher_probs.argmax(dim=-1).item())
            teacher_top1_action = legal[teacher_top1_idx]

            out = head.score(hidden, ctx, legal)
            student_logits = out.logits if out.logits is not None else torch.zeros(1, len(legal), device=hidden.device)
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

            if s.selected_action_id is not None and s.selected_action_id in legal:
                sel_idx = legal.index(s.selected_action_id)
                teacher_margin = teacher_logits[0, sel_idx].item() - teacher_logits[0].max().item()
                student_margin = student_logits[0, sel_idx].item() - student_logits[0].max().item()
                margin_deltas.append(student_margin - teacher_margin)

            regrets.append(0.0 if student_action == teacher_top1_action else 1.0)

    n = len(samples)
    return {
        "top1_accuracy": top1_correct / n if n else 0.0,
        "teacher_top1_accuracy": teacher_top1_correct / n if n else 0.0,
        "action_flip_rate": flips / n if n else 0.0,
        "kl_to_teacher": sum(kls) / len(kls) if kls else 0.0,
        "margin_preservation": sum(margin_deltas) / len(margin_deltas) if margin_deltas else 0.0,
        "mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
        "cvar90_regret": _cvar(regrets, alpha=0.9),
    }


def compute_gradient_proxy(
    model: torch.nn.Module,
    samples: list[CalibrationSample],
    grouping_policy: GroupingPolicy,
    *,
    hidden_dim: int = 64,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Diagnostic squared-gradient proxy per group (eval-mode, no state mutation)."""
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.train()
    head = _find_local_head(model)
    if head is None:
        return {g.group_id: 0.0 for g in grouping_policy.groups}

    teacher_batches = _score_local_head(head, samples, hidden_dim, device)
    group_params = {
        g.group_id: [p for _, p, _ in _iter_group_params(model, g)]
        for g in grouping_policy.groups
        if g.quantize_kind != "none"
    }
    sums: dict[str, float] = {gid: 0.0 for gid in group_params}
    ctx = StateContext(state_family_id="cap3-04")

    for s, (hidden, legal, teacher_logits) in zip(samples, teacher_batches):
        model.zero_grad()
        out = head.score(hidden, ctx, legal)
        student_logits = out.logits if out.logits is not None else torch.zeros(1, len(legal), device=device)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        loss = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction="batchmean")
        loss.backward()
        for gid, params in group_params.items():
            sq = sum(float((p.grad ** 2).sum().item()) for p in params if p.grad is not None)
            sums[gid] += sq
        model.zero_grad()

    model.zero_grad()
    model.eval()
    return {gid: math.sqrt(sq / max(len(samples), 1)) for gid, sq in sums.items()}


def profile_group_sensitivity(
    model: torch.nn.Module,
    manifest: CalibrationCorpusManifest,
    samples: list[CalibrationSample],
    grouping_policy: GroupingPolicy,
    formats: tuple[str, ...],
    *,
    group_size: int = 128,
    hidden_dim: int = 64,
    device: torch.device | None = None,
    run_id: str | None = None,
) -> SensitivityReport:
    """Run the direct-perturbation sensitivity matrix and return a versioned report."""
    from slm_training.models.quantization import build_model_ledger
    from slm_training.models.quantization.formats import (
        binary_format,
        binary_plus_mask_format,
        fp16_format,
        int4_format,
        int8_format,
        learned_four_level_zero_format,
        symmetric_four_level_format,
        ternary_format,
    )

    _FORMAT_FACTORIES = {
        "fp16": fp16_format,
        "int8": int8_format,
        "int4": int4_format,
        "binary": binary_format,
        "ternary": ternary_format,
        "symmetric4": symmetric_four_level_format,
        "symmetric_four_level": symmetric_four_level_format,
        "learned4zero": learned_four_level_zero_format,
        "learned_four_level_zero": learned_four_level_zero_format,
        "binary_plus_mask": binary_plus_mask_format,
    }

    def _make_format(format_id: str, group_size: int) -> QuantFormat:
        factory = _FORMAT_FACTORIES.get(format_id)
        if factory is None:
            raise ValueError(f"unknown format_id: {format_id!r}")
        return factory(group_size=group_size)

    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    if run_id is None:
        run_id = f"cap3-04-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    head = _find_local_head(model)
    if head is None:
        return SensitivityReport(
            version=CAP3_04_VERSION,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checkpoint_id=manifest.checkpoint_id,
            calibration_manifest_sha=_sha256_json(manifest.as_dict()),
            grouping_policy_version=grouping_policy.version,
            formats=formats,
            sample_count=len(samples),
            gradient_proxies={},
            points=[],
        )

    # Baseline teacher logits.
    teacher_batches = _score_local_head(head, samples, hidden_dim, device)

    # Gradient proxy (diagnostic).
    gradient_proxies = compute_gradient_proxy(
        model, samples, grouping_policy, hidden_dim=hidden_dim, device=device
    )

    manifest_sha = _sha256_json(manifest.as_dict())
    fp16 = fp16_format(group_size=group_size)
    points: list[GroupFormatPoint] = []

    for group in grouping_policy.groups:
        group_params = _iter_group_params(model, group)
        group_param_names = {name for name, _, _ in group_params}

        if group.exclusion_reason or not group_param_names:
            points.append(
                GroupFormatPoint(
                    group_id=group.group_id,
                    format_id="excluded",
                    group_size=group_size,
                    packed_bytes=0,
                    total_bytes=0,
                    sample_count=0,
                    top1_accuracy=0.0,
                    teacher_top1_accuracy=0.0,
                    action_flip_rate=0.0,
                    kl_to_teacher=0.0,
                    margin_preservation=0.0,
                    mean_regret=0.0,
                    cvar90_regret=0.0,
                    gradient_proxy=gradient_proxies.get(group.group_id),
                    status="excluded",
                    notes=[group.exclusion_reason or "no matched parameters"],
                )
            )
            continue

        snapshots = {name: param.data.clone() for name, param, _ in group_params}

        for format_id in formats:
            # Restore baseline before each format probe.
            for name, param, _ in group_params:
                param.data = snapshots[name]

            fmt = _make_format(format_id, group_size=group_size)
            _quantize_group(model, group, fmt)

            metrics = _evaluate_perturbation(head, teacher_batches, samples)

            format_map = _format_map_for_group(model, group, fmt)
            ledger = build_model_ledger(model, format_map, default_format=fp16, d_model=hidden_dim)
            ledger_dict = ledger.as_dict()
            ledger_sha256 = _sha256_json(ledger_dict)
            packed_bytes, total_bytes, seen = _group_bytes_from_ledger(
                ledger, group_param_names, fmt.format_id
            )
            if not seen:
                # Lazy dict parameters (e.g., LocalFlatHead action embeddings) are
                # not surfaced by named_parameters; compute their cost directly.
                packed_bytes, total_bytes = _compute_group_bytes(
                    model, group, fmt, hidden_dim
                )

            points.append(
                GroupFormatPoint(
                    group_id=group.group_id,
                    format_id=fmt.format_id,
                    group_size=group_size,
                    packed_bytes=packed_bytes,
                    total_bytes=total_bytes,
                    sample_count=len(samples),
                    top1_accuracy=metrics["top1_accuracy"],
                    teacher_top1_accuracy=metrics["teacher_top1_accuracy"],
                    action_flip_rate=metrics["action_flip_rate"],
                    kl_to_teacher=metrics["kl_to_teacher"],
                    margin_preservation=metrics["margin_preservation"],
                    mean_regret=metrics["mean_regret"],
                    cvar90_regret=metrics["cvar90_regret"],
                    gradient_proxy=gradient_proxies.get(group.group_id),
                    status="ok",
                    notes=[f"ledger_sha256={ledger_sha256}"],
                )
            )

        # Restore baseline after all formats for this group.
        for name, param, _ in group_params:
            param.data = snapshots[name]

    return SensitivityReport(
        version=CAP3_04_VERSION,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checkpoint_id=manifest.checkpoint_id,
        calibration_manifest_sha=manifest_sha,
        grouping_policy_version=grouping_policy.version,
        formats=formats,
        sample_count=len(samples),
        gradient_proxies=gradient_proxies,
        points=points,
    )
