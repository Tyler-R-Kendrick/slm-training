"""Collect every quality dimension and ratchet it against the baseline.

One pass over the repository produces the four ratchet dimensions:

``module_lines``       oversized modules, by physical line count
``complexity``         ruff complexity findings, counted per file
``package_lines``      oversized components, by physical line count
``package_principles`` aggregate ADP / SDP / SAP violation counts

The first three are keyed by path so debt is attributable; the fourth is
aggregate because a cycle's *membership* churns with every import while its
*size* is the thing that has to fall.

See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from slm_training.quality import baseline as baseline_module
from slm_training.quality import complexity as complexity_module
from slm_training.quality import martin, sources

#: Physical-line budget for one component. Martin's Common Closure Principle
#: asks that a component change for one reason; past this size that is not
#: credible and the component wants splitting.
MAX_COMPONENT_LINES = 20_000

DIMENSION_UNITS = {
    "module_lines": "lines",
    "complexity": "findings",
    "package_lines": "lines",
    "package_principles": "violations",
}


@dataclass(frozen=True)
class QualityReport:
    """Everything one analysis pass measured."""

    files: list[sources.SourceFile]
    oversized: list[sources.SourceFile]
    complexity: list[complexity_module.Violation] | None
    components: list[martin.ComponentMetrics]
    edges: dict[str, set[str]]
    cycles: list[list[str]]
    sdp: list[str]
    sap: list[str]
    ruff_version: str | None = None

    @property
    def oversized_components(self) -> list[martin.ComponentMetrics]:
        return sorted(
            (item for item in self.components if item.lines > MAX_COMPONENT_LINES),
            key=lambda item: -item.lines,
        )

    @property
    def components_in_cycles(self) -> int:
        return sum(len(group) for group in self.cycles)

    def dimensions(self) -> dict[str, dict[str, int]]:
        """The ratchet payload for this pass."""

        payload: dict[str, dict[str, int]] = {
            "module_lines": {item.path: item.lines for item in self.oversized},
            "package_lines": {
                item.name: item.lines for item in self.oversized_components
            },
            "package_principles": {
                "adp_components_in_cycles": self.components_in_cycles,
                "adp_cycles": len(self.cycles),
                "sdp_violations": len(self.sdp),
                "sap_violations": len(self.sap),
            },
        }
        if self.complexity is not None:
            payload["complexity"] = complexity_module.counts_by_file(self.complexity)
        return payload


def analyse(*, root: Path, skip_complexity: bool = False) -> QualityReport:
    """Measure every dimension in one pass."""

    files = sources.discover(root=root)
    components, edges = martin.build(root=root)
    findings: list[complexity_module.Violation] | None = None
    version: str | None = None
    if not skip_complexity:
        findings = complexity_module.run_ruff(root=root)
        version = complexity_module.ruff_version()
    return QualityReport(
        files=files,
        oversized=sources.over_budget(files),
        complexity=findings,
        components=components,
        edges=edges,
        cycles=martin.cycles(edges),
        sdp=martin.stable_dependency_violations(components, edges),
        sap=martin.stable_abstraction_violations(components),
        ruff_version=version,
    )


def evaluate(
    report: QualityReport, recorded: dict[str, dict]
) -> list[baseline_module.RatchetOutcome]:
    """Ratchet every measured dimension against its recorded ceilings."""

    outcomes: list[baseline_module.RatchetOutcome] = []
    for name, current in sorted(report.dimensions().items()):
        outcomes.append(
            baseline_module.compare(
                current,
                baseline_module.dimension(recorded, name),
                dimension=name,
                unit=DIMENSION_UNITS[name],
            )
        )
    return outcomes


def tooling_drift(report: QualityReport, recorded: dict[str, dict]) -> str | None:
    """A note when complexity counts were taken under a different ruff.

    Returned only when it is actually relevant, so an ordinary regression is
    not muddied by an irrelevant version line.
    """

    was = baseline_module.tooling(recorded).get("ruff")
    now = report.ruff_version
    if not was or not now or was == now:
        return None
    return (
        f"complexity counts were recorded under {was!r} but this run used "
        f"{now!r}; a ruff upgrade moves many counts at once. Re-freeze with "
        "--update in a dedicated commit rather than treating these as code "
        "regressions."
    )


def merged_dimensions(
    report: QualityReport, recorded: dict[str, dict]
) -> dict[str, dict[str, int]]:
    """Fresh measurements, preserving dimensions this pass did not measure.

    ``--update`` with complexity skipped must not silently erase the recorded
    complexity ceilings.
    """

    payload = report.dimensions()
    for name in DIMENSION_UNITS:
        if name not in payload and name in recorded:
            payload[name] = dict(recorded[name])
    return payload
