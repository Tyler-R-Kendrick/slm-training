"""The ratchet: a frozen debt baseline that may shrink but never grow.

A 400-line budget declared against a codebase with 531 oversized modules is a
budget nobody can honour, so it gets waived and stops meaning anything. The
ratchet keeps the budget real: every existing violation is recorded at its
current magnitude, any increase fails, and any decrease must be written back so
the ceiling drops permanently. Debt is thereby monotonically non-increasing --
the Boy Scout rule, mechanised.

See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "code_quality_baseline/v1"

BASELINE_PATH = "src/slm_training/resources/code_quality_baseline.json"

README = [
    "Frozen code-quality debt. Regenerate with"
    " `python -m scripts.verify_code_quality --update`.",
    "Contract: docs/design/code-quality-contract.md.",
    "Every number is a ceiling that may fall and must never rise. Do not hand-raise an",
    "entry to land a change: split the module or simplify the function instead.",
]


@dataclass(frozen=True)
class RatchetOutcome:
    """What a single ratchet dimension found against its baseline."""

    dimension: str
    regressions: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        """Findings that fail the gate: debt grew, or new debt appeared."""

        return [*self.regressions, *self.unlisted]

    @property
    def drift(self) -> list[str]:
        """Findings that mean the baseline is looser than reality."""

        return [*self.improvements, *self.stale]

    @property
    def clean(self) -> bool:
        return not self.failures and not self.drift


def compare(
    current: Mapping[str, int],
    recorded: Mapping[str, int],
    *,
    dimension: str,
    unit: str,
) -> RatchetOutcome:
    """Ratchet ``current`` measurements against their ``recorded`` ceilings.

    ``current`` holds only present-day violations, so a key absent from it has
    fallen back inside budget and its baseline entry is stale.
    """

    regressions: list[str] = []
    unlisted: list[str] = []
    improvements: list[str] = []

    for key, value in sorted(current.items()):
        ceiling = recorded.get(key)
        if ceiling is None:
            unlisted.append(f"{key}: new {dimension} violation ({value} {unit})")
        elif value > ceiling:
            delta = value - ceiling
            regressions.append(
                f"{key}: {dimension} grew {ceiling} -> {value} {unit} (+{delta})"
            )
        elif value < ceiling:
            improvements.append(
                f"{key}: improved {ceiling} -> {value} {unit}; lower the baseline"
            )

    stale = [
        f"{key}: no longer violates {dimension}; drop the baseline entry"
        for key in sorted(set(recorded) - set(current))
    ]
    return RatchetOutcome(
        dimension=dimension,
        regressions=regressions,
        unlisted=unlisted,
        improvements=improvements,
        stale=stale,
    )


def load(*, root: Path, path: str = BASELINE_PATH) -> dict[str, dict]:
    """Read the baseline document, tolerating a first-ever run."""

    target = root / path
    if not target.exists():
        return {}
    document = json.loads(target.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: expected schema {SCHEMA!r}, found {document.get('schema')!r}"
        )
    return document


def dimension(document: Mapping[str, dict], name: str) -> dict[str, int]:
    """One ratchet dimension out of a loaded baseline document."""

    return dict(document.get(name, {}))


def save(
    dimensions: Mapping[str, Mapping[str, int]],
    *,
    root: Path,
    path: str = BASELINE_PATH,
) -> Path:
    """Write the baseline document with deterministic key ordering."""

    document = {
        "_readme": README,
        "schema": SCHEMA,
        **{
            name: dict(sorted(values.items()))
            for name, values in sorted(dimensions.items())
        },
    }
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return target
