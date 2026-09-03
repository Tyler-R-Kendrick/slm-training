"""Human-readable rendering of a :class:`~slm_training.quality.gate.QualityReport`.

Formatting only -- no measurement and no policy, so the numbers a reader sees
are exactly the numbers the ratchet gated on.
"""

from __future__ import annotations

from slm_training.quality.gate import MAX_COMPONENT_LINES, QualityReport
from slm_training.quality.complexity import counts_by_rule, worst_functions
from slm_training.quality.sources import MAX_MODULE_LINES

RULE_LIMIT = 10


def _heading(text: str) -> list[str]:
    return ["", text, "-" * len(text)]


def summary(report: QualityReport) -> list[str]:
    """Top-level counts for every dimension."""

    total = sum(item.lines for item in report.files)
    lines = _heading("Summary")
    lines.append(f"  repo-owned source files : {len(report.files):,} ({total:,} lines)")
    lines.append(
        f"  modules over {MAX_MODULE_LINES} lines  : {len(report.oversized):,} "
        f"({len(report.oversized) / max(len(report.files), 1):.1%} of files)"
    )
    if report.complexity is not None:
        lines.append(f"  complexity findings     : {len(report.complexity):,}")
    lines.append(f"  components              : {len(report.components):,}")
    lines.append(
        f"  components over {MAX_COMPONENT_LINES:,} :"
        f" {len(report.oversized_components):,}"
    )
    return lines


def oversized_modules(report: QualityReport, *, limit: int = RULE_LIMIT) -> list[str]:
    if not report.oversized:
        return []
    lines = _heading(f"Largest modules (budget {MAX_MODULE_LINES} lines)")
    for item in report.oversized[:limit]:
        over = item.lines / MAX_MODULE_LINES
        lines.append(f"  {item.lines:>7,} lines  ({over:>5.1f}x budget)  {item.path}")
    remaining = len(report.oversized) - limit
    if remaining > 0:
        lines.append(f"  ... and {remaining:,} more over budget")
    return lines


def complexity_section(report: QualityReport, *, limit: int = RULE_LIMIT) -> list[str]:
    if report.complexity is None:
        return [*_heading("Complexity"), "  skipped (ruff unavailable)"]
    lines = _heading("Complexity (ruff)")
    for rule, count in counts_by_rule(report.complexity).items():
        lines.append(f"  {rule:<9} {count:>6,}")
    lines.append("  most-burdened functions:")
    worst = worst_functions(report.complexity, limit=limit)
    lines.extend(f"    {entry}" for entry in worst)
    return lines


def package_section(report: QualityReport, *, limit: int = RULE_LIMIT) -> list[str]:
    """Martin metrics, largest components first."""

    lines = _heading("Component metrics (Ca / Ce / I / A / D)")
    lines.append(
        "  "
        f"{'lines':>8} {'mod':>4} {'Ca':>4} {'Ce':>4}"
        f" {'I':>5} {'A':>5} {'D':>5}  component"
    )
    ranked = sorted(report.components, key=lambda item: -item.lines)
    for item in ranked[:limit]:
        lines.append(
            f"  {item.lines:>8,} {item.modules:>4}"
            f" {item.afferent:>4} {item.efferent:>4} "
            f"{item.instability:>5.2f} {item.abstractness:>5.2f} {item.distance:>5.2f}"
            f"  {item.name}"
        )
    return lines


def principles_section(report: QualityReport, *, limit: int = 5) -> list[str]:
    """ADP / SDP / SAP findings."""

    lines = _heading("Package principles")
    lines.append(
        f"  ADP  {len(report.cycles):>4} dependency cycle(s) covering "
        f"{report.components_in_cycles} component(s)"
    )
    for group in report.cycles[:limit]:
        shown = ", ".join(group[:4])
        suffix = f", ... (+{len(group) - 4})" if len(group) > 4 else ""
        lines.append(f"       cycle of {len(group):>3}: {shown}{suffix}")
    lines.append(f"  SDP  {len(report.sdp):>4} stable-dependency violation(s)")
    lines.extend(f"       {entry}" for entry in report.sdp[:limit])
    lines.append(f"  SAP  {len(report.sap):>4} stable-abstraction violation(s)")
    lines.extend(f"       {entry}" for entry in report.sap[:limit])
    return lines


def render(report: QualityReport) -> str:
    """The complete report."""

    lines = ["Code quality report", "==================="]
    for section in (
        summary,
        oversized_modules,
        complexity_section,
        package_section,
        principles_section,
    ):
        lines.extend(section(report))
    return "\n".join(lines)
