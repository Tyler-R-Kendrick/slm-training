"""Deterministic, no-write census of evidence bundles and checkpoint references.

Scans committed result artifacts for ``evidence_bundle/v1`` payloads and the
model-card checkpoint roster for checkpoint locations, then verifies each
against the fail-closed SLM-291 bar:

* every ``EvidenceBundleV1`` must pass :func:`verify_evidence_bundle`;
* a roster row that positively claims a durable class (promoted / champion /
  deployed / frontier / ship_candidate) must reference a durable, digest-
  pinned location — ``outputs/``, ``/tmp``, and pytest-temporary paths fail
  closed with an explicit reason.

Historical rows that honestly disclaim ship status are reported, never
rewritten. The census mutates nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.harness_core.evidence_bundle import (
    SCHEMA_VERSION,
    VERIFIER_VERSION,
    bundle_from_dict,
    verify_evidence_bundle,
)

CENSUS_SCHEMA = "evidence_bundle_census/v1"

_REPO_ROOT = Path(__file__).resolve().parents[4]

_BACKTICK_SPAN = re.compile(r"`([^`]+)`")
# Positive durable claims. Negated/honest rows ("not promotable or ship",
# "rejected", "never sync", "not a ship claim") must not match.
_DURABLE_CLAIM = re.compile(
    r"\b(promoted|champion|deployed|frontier|ship_candidate)\b", re.IGNORECASE
)
_NEGATION = re.compile(
    r"\b(not|never|rejected|non|no|invalidated|diagnostic|demo|scratch|fixture)\b",
    re.IGNORECASE,
)


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _iter_bundles(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, dict):
        if value.get("schema") == SCHEMA_VERSION:
            yield value
        for child in value.values():
            yield from _iter_bundles(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_bundles(child)


def census_design_dir(
    design_dir: Path | str,
    *,
    artifact_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Verify every committed evidence bundle under a design directory."""
    root = Path(design_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for index, raw in enumerate(_iter_bundles(payload)):
            rel = path.relative_to(root).as_posix()
            try:
                bundle = bundle_from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                rows.append(
                    {
                        "source": f"{rel}#{index}",
                        "run_id": raw.get("run_id"),
                        "verdict": "invalid",
                        "failures": [f"malformed bundle: {exc}"],
                    }
                )
                continue
            failures = verify_evidence_bundle(
                bundle, artifact_root=artifact_root, store=None
            )
            rows.append(
                {
                    "source": f"{rel}#{index}",
                    "run_id": bundle.run_id,
                    "claim_class": bundle.claim_class,
                    "bundle_sha256": bundle.sha,
                    "verdict": "pass" if not failures else "fail",
                    "failures": failures,
                }
            )
    rows.sort(key=lambda row: row["source"])
    return rows


def _roster_rows(model_card: str) -> Iterable[tuple[int, list[str]]]:
    in_roster = False
    for lineno, line in enumerate(model_card.splitlines(), start=1):
        if line.startswith("## "):
            in_roster = line.strip().lower().startswith("## current checkpoint roster")
            continue
        if not in_roster or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] in {"Role", "---", ":---"} or set(cells[0]) <= {"-", ":"}:
            continue
        yield lineno, cells


def _classify_location(location: str) -> tuple[str, str]:
    """Return (path_or_uri, durability) for a roster Location cell."""
    match = _BACKTICK_SPAN.search(location)
    target = match.group(1) if match else location
    if target.startswith("hf://"):
        return target, "durable_remote"
    if target.startswith("/tmp/") or target.startswith("/var/"):
        return target, "ephemeral_local"
    if target.startswith(("outputs/", "src/")) or target == "pytest temporary checkpoint":
        return target, "repo_local"
    return target, "unparsed"


def census_model_card(model_card_path: Path | str) -> list[dict[str, Any]]:
    """Classify roster checkpoint references; fail closed on durable claims."""
    text = Path(model_card_path).read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for lineno, cells in _roster_rows(text):
        role, run_id, _kind, location, status = cells[:5]
        target, durability = _classify_location(location)
        claims_durable = bool(_DURABLE_CLAIM.search(status)) and not _NEGATION.search(
            status
        )
        failures: list[str] = []
        if claims_durable and durability != "durable_remote":
            failures.append(
                f"non-fixture claim references a non-durable location "
                f"({durability}, model-card line {lineno}); promoted/champion/"
                "frontier/ship evidence must be a digest-pinned durable URI"
            )
        rows.append(
            {
                "line": lineno,
                "role": role,
                "run_id": run_id.strip("`"),
                # Ephemeral paths stay only in the model card (line-referenced
                # here) so census output carries no machine-absolute paths.
                "location": (
                    target if durability != "ephemeral_local" else "<ephemeral-local>"
                ),
                "durability": durability,
                "claims_durable": claims_durable,
                "verdict": "fail" if failures else "pass",
                "failures": failures,
            }
        )
    return rows


def build_census(
    *,
    model_card: Path | str | None = None,
    design_dir: Path | str | None = None,
    artifact_root: Path | str | None = None,
) -> dict[str, Any]:
    """Full deterministic census payload (schema ``evidence_bundle_census/v1``)."""
    model_card = Path(model_card) if model_card else _REPO_ROOT / "docs/MODEL_CARD.md"
    design_dir = (
        Path(design_dir) if design_dir else _REPO_ROOT / "docs" / "design"
    )
    artifact_root = Path(artifact_root) if artifact_root else _REPO_ROOT
    bundle_rows = census_design_dir(design_dir, artifact_root=artifact_root)
    roster_rows = census_model_card(model_card)
    failures = [row for row in (*bundle_rows, *roster_rows) if row["failures"]]

    def _display(path: Path) -> str:
        try:
            return path.resolve().relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            return path.name

    payload = {
        "schema": CENSUS_SCHEMA,
        "verifier_version": VERIFIER_VERSION,
        "model_card": _display(model_card),
        "design_dir": _display(design_dir),
        "summary": {
            "evidence_bundles": len(bundle_rows),
            "roster_rows": len(roster_rows),
            "roster_durable_remote": sum(
                1 for row in roster_rows if row["durability"] == "durable_remote"
            ),
            "roster_repo_local": sum(
                1 for row in roster_rows if row["durability"] == "repo_local"
            ),
            "roster_ephemeral_local": sum(
                1 for row in roster_rows if row["durability"] == "ephemeral_local"
            ),
            "roster_claims_durable": sum(
                1 for row in roster_rows if row["claims_durable"]
            ),
            "failures": len(failures),
        },
        "evidence_bundles": bundle_rows,
        "roster": roster_rows,
    }
    payload["census_sha256"] = _canonical_sha(
        {key: value for key, value in payload.items() if key != "census_sha256"}
    )
    return payload


def render_markdown(census: Mapping[str, Any]) -> str:
    summary = census["summary"]
    lines = [
        "# Evidence bundle census",
        "",
        f"Schema: `{census['schema']}` · verifier `{census['verifier_version']}`",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "| --- | --- |",
        f"| Evidence bundles verified | {summary['evidence_bundles']} |",
        f"| Roster rows | {summary['roster_rows']} |",
        f"| Roster durable remote | {summary['roster_durable_remote']} |",
        f"| Roster repo-local | {summary['roster_repo_local']} |",
        f"| Roster ephemeral local | {summary['roster_ephemeral_local']} |",
        f"| Roster rows claiming durable | {summary['roster_claims_durable']} |",
        f"| Failures | {summary['failures']} |",
        "",
        "## Failures",
        "",
    ]
    failures = [
        row
        for row in (*census["evidence_bundles"], *census["roster"])
        if row["failures"]
    ]
    if not failures:
        lines.append("None.")
    for row in failures:
        label = row.get("source") or f"line {row['line']} ({row['run_id']})"
        for reason in row["failures"]:
            lines.append(f"- `{label}`: {reason}")
    lines.append("")
    return "\n".join(lines)
