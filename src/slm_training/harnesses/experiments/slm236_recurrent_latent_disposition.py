"""Fail-closed RSC4 disposition assembled from immutable RSC gate evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp, git_commit

REPORT_SCHEMA = "RecurrentLatentDispositionV1"
DEFAULT_JSON = "docs/design/iter-slm236-recurrent-latent-disposition-20260724.json"
DEFAULT_MARKDOWN = "docs/design/recurrent-semantic-computation-looped-latent-disposition.md"
EVIDENCE = {
    "differentiation": ("docs/design/iter-slm229-looped-latent-differentiation-20260721.json", "blocked_by_recurrence"),
    "observability": ("docs/design/iter-slm230-recurrence-observability-20260724.json", "stagnant"),
    "dynamics": ("docs/design/iter-slm231-recurrence-dynamics-20260724.json", "expansive_unstable"),
    "z_state": ("docs/design/iter-slm232-latent-state-use-20260724.json", "unstable"),
    "recursive_core": ("docs/design/iter-slm233-recursive-campaign-20260724.json", "architecture_not_identifiable"),
    "minimal_probe": ("docs/design/iter-slm234-compiler-latent-closeout-20260724.json", "not_authorized"),
}


@dataclass(frozen=True)
class EvidenceRefV1:
    mechanism_id: str
    path: str
    verdict: str
    sha256: str


@dataclass(frozen=True)
class RecurrentLatentDispositionV1:
    evidence_cutoff_commit: str
    evidence: tuple[EvidenceRefV1, ...]
    disposition: str
    claim_class: str
    rationale: str
    reopen_conditions: tuple[str, ...]
    schema: str = REPORT_SCHEMA
    matrix_set: str = "slm236_recurrent_latent_disposition"
    matrix_version: str = "rsc4-02-v1"
    version_stamp: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [asdict(row) for row in self.evidence]
        payload["reopen_conditions"] = list(self.reopen_conditions)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RecurrentLatentDispositionV1:
        return cls(
            **{**value, "evidence": tuple(EvidenceRefV1(**row) for row in value["evidence"]), "reopen_conditions": tuple(value["reopen_conditions"])},
        )


def _verdict(payload: dict[str, Any]) -> str:
    for key in ("gate", "recursive_core_gate"):
        if isinstance(payload.get(key), dict) and payload[key].get("verdict"):
            return str(payload[key]["verdict"])
    return str(payload.get("decision") or payload.get("verdict") or payload.get("status"))


def build_report(repo_root: Path) -> RecurrentLatentDispositionV1:
    rows = []
    for mechanism_id, (path, expected) in EVIDENCE.items():
        raw = (repo_root / path).read_bytes()
        payload = json.loads(raw)
        verdict = _verdict(payload)
        if verdict != expected:
            raise ValueError(f"{mechanism_id} verdict {verdict!r}, expected {expected!r}")
        rows.append(EvidenceRefV1(mechanism_id, path, verdict, hashlib.sha256(raw).hexdigest()))
    return RecurrentLatentDispositionV1(
        evidence_cutoff_commit=git_commit(),
        evidence=tuple(rows), disposition="blocked", claim_class="governance_disposition",
        rationale="No RSC mechanism is authorized for optional or primary use: the differentiation contract is absent, the floor is inconclusive, recurrence is stagnant or unstable, the recursive core is unidentifiable, and the minimal probe is not authorized.",
        reopen_conditions=(
            "An authorized MinimalCompilerLatentContractV1 with an exact hash.",
            "A SemanticFloorGateV1 floor_escaped verdict.",
            "A matched recursive-core verdict with a durable checkpoint and bounded stable dynamics.",
        ), version_stamp=build_version_stamp("harness.experiments.slm236_recurrent_latent_disposition"),
    )


def validate_report(report: RecurrentLatentDispositionV1, repo_root: Path) -> list[str]:
    errors = []
    if report.schema != REPORT_SCHEMA or report.disposition != "blocked":
        errors.append("RSC4 must remain a blocked disposition")
    if report.claim_class != "governance_disposition":
        errors.append("RSC4 is not a performance or ship claim")
    try:
        expected = build_report(repo_root)
    except ValueError as exc:
        return [str(exc)]
    if [(row.mechanism_id, row.verdict, row.sha256) for row in report.evidence] != [(row.mechanism_id, row.verdict, row.sha256) for row in expected.evidence]:
        errors.append("evidence references are stale or incomplete")
    return errors


def render_markdown(report: RecurrentLatentDispositionV1) -> str:
    rows = "\n".join(f"| {row.mechanism_id} | `{row.verdict}` | [{row.path}]({row.path.removeprefix('docs/design/')}) |" for row in report.evidence)
    conditions = "\n".join(f"{index}. {item}" for index, item in enumerate(report.reopen_conditions, 1))
    return f"""# RSC4 recurrent semantic computation and looped-latent disposition\n\n**Schema:** `{report.schema}`\n\n**Disposition:** **blocked** — not a performance, production, or ship claim.\n\n## Executive finding\n\n{report.rationale}\n\nNo latent/plan configuration may become a default, champion, optional production path, or promotion candidate from this evidence. No checkpoint, model card, or README model-card update is warranted.\n\n## Evidence\n\n| Mechanism | Verdict | Artifact |\n| --- | --- | --- |\n{rows}\n\n## Reopening conditions\n\n{conditions}\n\n## Relationship to SLM-160\n\nThis is an RSC addendum only. It preserves the historical [SLM-160 disposition](semantic-planning-valid-state-disposition.md) and does not revise its mechanism verdicts.\n\n## Reproduction\n\n```bash\npython -m scripts.publish_recurrent_latent_disposition --check\n```\n"""
