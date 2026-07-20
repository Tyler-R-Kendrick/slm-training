"""SLM-164 (SDE1-02): confusion-targeted legal-sibling contrast margin fixture.

Deterministic, CPU-only harness that exercises targeted margin losses over
synthetic legal-sibling contrast rows.  It trains no GPU model and makes no
ship-gate claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ContrastFamily",
    "LegalContrastRow",
    "LegalContrastManifest",
    "TargetedMarginArm",
    "TargetedMarginMetrics",
    "TargetedMarginReport",
    "build_default_families",
    "build_synthetic_manifest",
    "compute_targeted_margin_loss",
    "run_fixture_campaign",
    "render_markdown",
    "validate_manifest",
]

MATRIX_VERSION = "sde1-02-v1"
MATRIX_SET = "slm164_targeted_margin"
EXPERIMENT_ID = "slm164-targeted-margin"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_MARGIN = 1.0

_ACTION_VOCABULARY = [
    "+Stack",
    "+Card",
    "+Button",
    "+Input",
    "+Tabs",
    "+SwitchItem",
    "+Slider",
    "+Form",
    "close",
    "slot0",
    "slot1",
    "arity_1",
    "arity_2",
]


@dataclass(frozen=True)
class ContrastFamily:
    """Built-in legal-sibling contrast families for SLM-164."""

    empty_vs_child: str = "empty_vs_child"
    stack_vs_card: str = "stack_vs_card"
    rare_component_substitution: str = "rare_component_substitution"
    binder_arity: str = "binder_arity"
    slot_pointer: str = "slot_pointer"
    same_type_different_role: str = "same_type_different_role"

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContrastFamily":
        return cls(
            empty_vs_child=str(data.get("empty_vs_child", "empty_vs_child")),
            stack_vs_card=str(data.get("stack_vs_card", "stack_vs_card")),
            rare_component_substitution=str(
                data.get("rare_component_substitution", "rare_component_substitution")
            ),
            binder_arity=str(data.get("binder_arity", "binder_arity")),
            slot_pointer=str(data.get("slot_pointer", "slot_pointer")),
            same_type_different_role=str(
                data.get("same_type_different_role", "same_type_different_role")
            ),
        )


@dataclass(frozen=True)
class LegalContrastRow:
    """One legal-sibling contrast row."""

    state_signature: str
    expected_action: str
    contrast_actions: tuple[str, ...]
    family: str
    weight: float
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_signature": self.state_signature,
            "expected_action": self.expected_action,
            "contrast_actions": tuple(self.contrast_actions),
            "family": self.family,
            "weight": self.weight,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegalContrastRow":
        return cls(
            state_signature=str(data["state_signature"]),
            expected_action=str(data["expected_action"]),
            contrast_actions=tuple(str(a) for a in data.get("contrast_actions", [])),
            family=str(data["family"]),
            weight=float(data.get("weight", 1.0)),
            provenance=str(data.get("provenance", "")),
        )


@dataclass(frozen=True)
class LegalContrastManifest:
    """Synthetic contrast-row manifest for the SLM-164 fixture."""

    manifest_id: str
    families: tuple[str, ...]
    rows: tuple[LegalContrastRow, ...]
    source_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "families": tuple(self.families),
            "rows": [row.to_dict() for row in self.rows],
            "source_hashes": tuple(self.source_hashes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegalContrastManifest":
        return cls(
            manifest_id=str(data.get("manifest_id", "slm164_default")),
            families=tuple(str(f) for f in data.get("families", [])),
            rows=tuple(
                LegalContrastRow.from_dict(r) for r in data.get("rows", [])
            ),
            source_hashes=tuple(str(h) for h in data.get("source_hashes", [])),
        )


@dataclass(frozen=True)
class TargetedMarginArm:
    """One margin-source arm in the SLM-164 campaign."""

    arm_id: str
    source: str
    name: str
    description: str
    promotable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetedMarginArm":
        return cls(
            arm_id=str(data["arm_id"]),
            source=str(data["source"]),
            name=str(data["name"]),
            description=str(data["description"]),
            promotable=bool(data.get("promotable", False)),
        )


@dataclass(frozen=True)
class TargetedMarginMetrics:
    """Per-arm, per-seed targeted-margin metrics."""

    arm_id: str
    source: str
    seed: int
    active_contrasts: int
    violation_rate: float
    mean_margin_loss: float
    family_coverage: float
    family_violation_rates: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "source": self.source,
            "seed": self.seed,
            "active_contrasts": self.active_contrasts,
            "violation_rate": self.violation_rate,
            "mean_margin_loss": self.mean_margin_loss,
            "family_coverage": self.family_coverage,
            "family_violation_rates": dict(self.family_violation_rates),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetedMarginMetrics":
        return cls(
            arm_id=str(data["arm_id"]),
            source=str(data["source"]),
            seed=int(data["seed"]),
            active_contrasts=int(data.get("active_contrasts", 0)),
            violation_rate=float(data.get("violation_rate", 0.0)),
            mean_margin_loss=float(data.get("mean_margin_loss", 0.0)),
            family_coverage=float(data.get("family_coverage", 0.0)),
            family_violation_rates=dict(
                data.get("family_violation_rates", {}) or {}
            ),
        )


@dataclass(frozen=True)
class TargetedMarginReport:
    """Full fixture report for SLM-164."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    arms: tuple[TargetedMarginArm, ...]
    manifest: LegalContrastManifest
    rows: list[TargetedMarginMetrics]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "arms": [arm.to_dict() for arm in self.arms],
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
    def from_dict(cls, data: dict[str, Any]) -> "TargetedMarginReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm164_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            arms=tuple(TargetedMarginArm.from_dict(a) for a in data.get("arms", [])),
            manifest=LegalContrastManifest.from_dict(
                data.get("manifest", {"manifest_id": "slm164_default"})
            ),
            rows=[TargetedMarginMetrics.from_dict(r) for r in data.get("rows", [])],
            version_stamp=data.get("version_stamp", {}),
        )


def build_default_families() -> list[str]:
    """Return the built-in SLM-164 contrast family identifiers."""
    return list(ContrastFamily().to_dict().values())


def _action_score(action: str, seed: int) -> float:
    """Deterministic dummy score for ``action`` under ``seed``."""
    digest = hashlib.sha256(
        f"{action}:{seed}:{MATRIX_VERSION}".encode("utf-8")
    ).hexdigest()
    # Map to a finite float in (-2, 2).
    return (int(digest[:16], 16) % 10000) / 2500.0 - 2.0


def _make_state_signature(idx: int, family: str, seed: int) -> str:
    return hashlib.sha256(
        f"state:{idx}:{family}:{seed}:{MATRIX_VERSION}".encode("utf-8")
    ).hexdigest()[:16]


def build_synthetic_manifest(seed: int = 0) -> LegalContrastManifest:
    """Generate a deterministic ``LegalContrastManifest`` with ≥20 rows.

    The manifest covers every built-in family using a small action vocabulary.
    It requires no real model or DSL state.
    """
    rng = random.Random(seed)
    families = build_default_families()
    vocab = list(_ACTION_VOCABULARY)

    rows: list[LegalContrastRow] = []
    base = 4  # rows per family; 6 * 4 = 24 ≥ 20.
    for family in families:
        for i in range(base):
            rng.shuffle(vocab)
            expected = vocab[0]
            n_contrasts = rng.choice((1, 2, 3))
            contrasts = tuple(sorted(set(vocab[1 : 1 + n_contrasts])))
            rows.append(
                LegalContrastRow(
                    state_signature=_make_state_signature(
                        len(rows), family, seed
                    ),
                    expected_action=expected,
                    contrast_actions=contrasts,
                    family=family,
                    weight=1.0,
                    provenance=f"synthetic:{family}:{i}",
                )
            )

    source_hashes = [
        hashlib.sha256(
            json.dumps(rows, default=str, sort_keys=True).encode("utf-8")
        ).hexdigest()
    ]
    return LegalContrastManifest(
        manifest_id=f"slm164_synthetic_seed{seed}",
        families=tuple(families),
        rows=tuple(rows),
        source_hashes=tuple(source_hashes),
    )


def compute_targeted_margin_loss(
    scores: dict[str, float],
    expected: str,
    contrasts: tuple[str, ...],
    margin: float,
    mode: str,
) -> tuple[float, bool]:
    """Return ``(loss, violation)`` for one contrast row.

    Modes:
      * ``none`` — always zero loss, no violation.
      * ``uniform`` — E228-style hardest-contrast margin.
      * ``targeted_hardest`` — same formula restricted to the contrast set.
      * ``targeted_weighted`` — weighted log-sum-exp over contrasts.
      * ``shuffled`` — same as ``targeted_weighted`` (family labels are shuffled
        at aggregation time as a control).
    """
    if mode == "none":
        return 0.0, False

    s_expected = scores.get(expected, 0.0)
    contrast_scores = [scores.get(a, 0.0) for a in contrasts]

    if mode in {"uniform", "targeted_hardest"}:
        if not contrast_scores:
            return 0.0, False
        worst = max(contrast_scores)
        loss = max(0.0, margin - s_expected + worst)
        return loss, loss > 0.0

    if mode in {"targeted_weighted", "shuffled"}:
        if not contrast_scores:
            return 0.0, False
        total = sum(
            math.exp(margin - s_expected + s_contrast)
            for s_contrast in contrast_scores
        )
        loss = math.log1p(total)
        # A row violates when the hardest contrast exceeds the margin, matching
        # the uniform/hardest criterion so the fixture can show non-uniform
        # family violation rates.
        violates = any(margin - s_expected + s_contrast > 0.0 for s_contrast in contrast_scores)
        return loss, violates

    raise ValueError(f"unknown legal margin mode: {mode!r}")


def build_manifest() -> tuple[TargetedMarginArm, ...]:
    """Return the default SLM-164 fixture arms."""
    return (
        TargetedMarginArm(
            arm_id="A_none",
            source="none",
            name="none",
            description="No margin loss; baseline zero metrics.",
        ),
        TargetedMarginArm(
            arm_id="B_uniform",
            source="uniform",
            name="uniform",
            description="E228-style hardest-contrast margin over the contrast set.",
        ),
        TargetedMarginArm(
            arm_id="C_targeted_hardest",
            source="targeted_hardest",
            name="targeted_hardest",
            description="Hardest-contrast margin focused on the targeted siblings.",
        ),
        TargetedMarginArm(
            arm_id="D_targeted_weighted",
            source="targeted_weighted",
            name="targeted_weighted",
            description="Weighted log-sum-exp margin over all targeted contrasts.",
        ),
        TargetedMarginArm(
            arm_id="E_shuffled",
            source="shuffled",
            name="shuffled",
            description="Control arm: same rows with family labels randomly reassigned.",
        ),
    )


def validate_manifest(arms: tuple[TargetedMarginArm, ...]) -> list[str]:
    """Validate manifest shape."""
    errors: list[str] = []
    if not arms:
        errors.append("arms must not be empty")
    seen: set[str] = set()
    for arm in arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
        if arm.promotable:
            errors.append(f"{arm.arm_id}: fixture arms must be non-promotable")
    return errors


def _aggregate_metrics(
    arm: TargetedMarginArm,
    seed: int,
    manifest: LegalContrastManifest,
    margin: float,
) -> TargetedMarginMetrics:
    """Compute per-arm, per-seed metrics from deterministic dummy scores."""
    rng = random.Random(seed)

    # For the shuffled control, permute family labels deterministically.
    family_labels = list(manifest.families)
    shuffled_labels = list(family_labels)
    if arm.source == "shuffled":
        rng.shuffle(shuffled_labels)
    label_for = dict(zip(family_labels, shuffled_labels))

    losses: list[float] = []
    violations: list[bool] = []
    family_violation: dict[str, list[bool]] = {
        family: [] for family in manifest.families
    }

    for row in manifest.rows:
        # Build a deterministic score table for this row + seed.
        actions = {row.expected_action, *row.contrast_actions}
        scores = {a: _action_score(a, seed) for a in actions}
        # Perturb the expected-action score by row signature so the same action
        # does not always win/lose across rows, creating non-uniform family
        # violation rates that let the shuffled control differ.
        row_perturb = (
            int(hashlib.sha256(row.state_signature.encode("utf-8")).hexdigest()[:8], 16)
            % 10000
        ) / 2500.0 - 2.0
        scores[row.expected_action] += row_perturb

        loss, violation = compute_targeted_margin_loss(
            scores,
            row.expected_action,
            row.contrast_actions,
            margin,
            arm.source,
        )
        losses.append(loss)
        violations.append(violation)
        family = label_for.get(row.family, row.family)
        family_violation[family].append(violation)

    active = len(losses)
    violation_rate = sum(violations) / active if active else 0.0
    mean_loss = sum(losses) / active if active else 0.0

    covered = {family for family, bits in family_violation.items() if bits}
    family_coverage = len(covered) / len(manifest.families) if manifest.families else 0.0
    family_violation_rates = {
        family: (sum(bits) / len(bits) if bits else 0.0)
        for family, bits in family_violation.items()
    }

    return TargetedMarginMetrics(
        arm_id=arm.arm_id,
        source=arm.source,
        seed=seed,
        active_contrasts=active,
        violation_rate=violation_rate,
        mean_margin_loss=mean_loss,
        family_coverage=family_coverage,
        family_violation_rates=family_violation_rates,
    )


def run_fixture_campaign(
    arms: tuple[TargetedMarginArm, ...] | None = None,
    *,
    run_id: str = "slm164-targeted-margin",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    margin: float = _DEFAULT_MARGIN,
) -> TargetedMarginReport:
    """Run the SLM-164 confusion-targeted legal-sibling margin fixture campaign."""
    arms = arms or build_manifest()
    errors = validate_manifest(arms)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    manifest = build_synthetic_manifest(seed=seeds[0] if seeds else 0)
    rows: list[TargetedMarginMetrics] = []
    for arm in arms:
        for seed in seeds:
            rows.append(_aggregate_metrics(arm, seed, manifest, margin))

    report = TargetedMarginReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        arms=arms,
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm164_targeted_margin",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm164_targeted_margin_report.json")
    return report


def render_markdown(report: TargetedMarginReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-164 (SDE1-02): Confusion-targeted legal-sibling contrast margin fixture ({report.run_id})",
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
        "Targeting margin loss at confusion-prone legal siblings (e.g. Stack vs Card, "
        "same-type/different-role, rare-component substitutions) improves the "
        "separation between the expected action and its legal contrast set compared "
        "to a uniform or no-margin baseline.",
        "",
        "## Falsifier",
        "",
        "Targeted margin arms do not reduce violation rate or mean margin loss over "
        "the shuffled control arm, or the none baseline performs as well as the "
        "targeted arms.",
        "",
        "## Arms",
        "",
        "| Arm | Source | Promotable | Description |",
        "| --- | --- | --- | --- |",
    ]
    for arm in report.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.source} | {arm.promotable} | {arm.description} |"
        )

    lines.extend(
        [
            "",
            "## Contrast manifest",
            "",
            f"Manifest id: `{report.manifest.manifest_id}`  ",
            f"Families: {', '.join(report.manifest.families)}  ",
            f"Rows: {len(report.manifest.rows)}",
            "",
            "## Results",
            "",
            "| Arm | Seed | Active contrasts | Violation rate | Mean margin loss | Family coverage |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.seed} | {row.active_contrasts} | "
            f"{row.violation_rate:.3f} | {row.mean_margin_loss:.3f} | "
            f"{row.family_coverage:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Family violation rates",
            "",
            "| Arm | Seed | Family | Violation rate |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        for family, rate in sorted(row.family_violation_rates.items()):
            lines.append(f"| {row.arm_id} | {row.seed} | {family} | {rate:.3f} |")

    lines.extend(
        [
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** Every arm is explicitly non-promotable. The "
            "harness proves the wiring and metrics plumbing over deterministic "
            "synthetic contrast rows, but it does not train or evaluate a real "
            "model. The mechanism remains ``retain_diagnostic`` / "
            "``blocked_pending_real_model`` until a trained scorer and AgentV "
            "evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- Scores are deterministic hash-based dummy values, not a trained model.",
            "- Contrast rows are synthetic and cover a hand-picked action vocabulary.",
            "- Family weights are uniform in this fixture; real runs may load a "
            "  manifest with per-family weights.",
            "- No Pareto or ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm164_targeted_margin_fixture --mode plan-only",
            "python -m scripts.run_slm164_targeted_margin_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
