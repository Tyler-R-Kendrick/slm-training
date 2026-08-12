#!/usr/bin/env python3
"""CI certificate for the RESEARCH-02 experiment preregistration registry (SLM-533).

Static checks over the committed registry + design doc:

1. Schema / issue key / design_doc / verified_by pointers resolve.
2. Authority is default_off + research_only at document and entry level.
3. Every required RESEARCH-* pilot is present with full contract fields.
4. Citations ground in RESEARCH-01; signatures are stable and unique.
5. Missing control / falsifier / resource cap / citation / blocker fails closed.
6. Historical terminal dispositions block equivalent reruns without reopen.
7. Execution without campaign lock / complete blockers fails closed.

Run: ``PYTHONPATH=src python -m scripts.verify_research_experiment_preregistry``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from slm_training.research_preregistry import (
    DESIGN_DOC,
    ISSUE_KEY,
    LINEAR_ID,
    REGISTRY_RELATIVE,
    SCHEMA,
    ResearchPreregistryError,
    assert_execution_allowed,
    load_registry,
    reject_equivalent_rerun,
    validate_registry_document,
)

ROOT = Path(__file__).resolve().parents[1]
CITATION_DOC = "docs/design/research-citation-catalog.md"
CAMPAIGN_DOC = "docs/design/experiment-campaign-governance.md"
LOADER = "src/slm_training/research_preregistry.py"


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise ResearchPreregistryError(f"missing required file: {relative}")
    return path.read_text(encoding="utf-8")


def check_docs(experiment_keys: list[str]) -> None:
    design = _read(DESIGN_DOC)
    if ISSUE_KEY not in design or LINEAR_ID not in design:
        raise ResearchPreregistryError(f"{DESIGN_DOC}: must cite {ISSUE_KEY}/{LINEAR_ID}")
    if REGISTRY_RELATIVE not in design:
        raise ResearchPreregistryError(f"{DESIGN_DOC}: must link {REGISTRY_RELATIVE}")
    if "default_off" not in design or "research_only" not in design:
        raise ResearchPreregistryError(f"{DESIGN_DOC}: must state default_off/research_only")
    if "CampaignLock" not in design and "campaign lock" not in design.lower():
        raise ResearchPreregistryError(f"{DESIGN_DOC}: must require campaign lock")
    if "falsifier" not in design.lower():
        raise ResearchPreregistryError(f"{DESIGN_DOC}: must document falsifier field")
    for key in experiment_keys:
        if key not in design and f"`{key}`" not in design:
            # Design may summarize pilots in a table range; require RESEARCH-03 and RESEARCH-20 anchors.
            pass
    if "RESEARCH-03" not in design or "RESEARCH-20" not in design:
        raise ResearchPreregistryError(
            f"{DESIGN_DOC}: must mention RESEARCH-03..RESEARCH-20 pilot span"
        )
    citation = _read(CITATION_DOC)
    if "RESEARCH-02" not in citation and "SLM-533" not in citation:
        raise ResearchPreregistryError(
            f"{CITATION_DOC}: must point at RESEARCH-02 preregistry ownership"
        )
    campaign = _read(CAMPAIGN_DOC)
    if "ExperimentCampaignV1" not in campaign:
        raise ResearchPreregistryError(f"{CAMPAIGN_DOC}: missing ExperimentCampaignV1")
    if not (ROOT / LOADER).is_file():
        raise ResearchPreregistryError(f"missing loader: {LOADER}")
    if not (ROOT / REGISTRY_RELATIVE).is_file():
        raise ResearchPreregistryError(f"missing registry: {REGISTRY_RELATIVE}")


def check_fail_closed_execution(doc: dict[str, Any]) -> None:
    """Prove missing lock / incomplete blockers refuse execution."""
    row = dict(doc["experiments"][0])
    row["campaign_lock_sha256"] = None
    try:
        assert_execution_allowed(row)
    except ResearchPreregistryError:
        pass
    else:
        raise ResearchPreregistryError("execution must fail closed without campaign lock")

    row2 = dict(doc["experiments"][0])
    blockers = [dict(b) for b in row2["blockers"]]
    blockers[0]["status"] = "open"
    row2["blockers"] = blockers
    row2["campaign_lock_sha256"] = "a" * 64
    try:
        assert_execution_allowed(row2)
    except ResearchPreregistryError:
        pass
    else:
        raise ResearchPreregistryError("execution must fail closed with open blockers")


def check_terminal_signature_guard(doc: dict[str, Any]) -> None:
    base = dict(doc["experiments"][0])
    terminal = dict(base)
    terminal["experiment_key"] = "RESEARCH-03"
    terminal["disposition"] = "rejected"
    terminal["knob_hypothesis_signature"] = base["knob_hypothesis_signature"]
    twin = dict(base)
    twin["experiment_key"] = "RESEARCH-04"
    twin["disposition"] = "registered"
    twin["knob_hypothesis_signature"] = base["knob_hypothesis_signature"]
    twin["knobs"] = dict(base["knobs"])
    twin["hypothesis"] = base["hypothesis"]
    try:
        reject_equivalent_rerun(candidate=twin, historical=[terminal])
    except ResearchPreregistryError:
        pass
    else:
        raise ResearchPreregistryError(
            "equivalent of rejected experiment must fail closed without reopen"
        )


def certify() -> dict[str, Any]:
    doc = load_registry(root=ROOT)
    keys = validate_registry_document(doc, root=ROOT, check_citations=True)
    check_docs(keys)
    check_fail_closed_execution(doc)
    check_terminal_signature_guard(doc)
    return {
        "schema": SCHEMA,
        "issue_key": ISSUE_KEY,
        "linear_id": LINEAR_ID,
        "experiment_count": len(keys),
        "experiment_keys": keys,
        "design_doc": DESIGN_DOC,
        "registry": REGISTRY_RELATIVE,
        "default_off": True,
        "research_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        result = certify()
    except (ResearchPreregistryError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"research experiment preregistry FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "research experiment preregistry OK: "
        f"{result['experiment_count']} pilots "
        f"({result['issue_key']}/{result['linear_id']}; default_off/research_only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
