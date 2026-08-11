#!/usr/bin/env python3
"""Regenerate or verify JSON Schema generated from goal-support contracts."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    from slm_training.bridge_utils import repo_root
    from slm_training.dsl.solver.goal_support import (
        DomainObstructionCoreV1,
        GoalActionEvidenceV1,
        GoalDomainAdequacyReportV1,
        GoalTerminalEvidenceV1,
        GoalVerifierProfileV1,
    )

    resources = repo_root() / "src" / "slm_training" / "resources"
    schemas = {
        "goal_verifier_profile.schema.json": GoalVerifierProfileV1.model_json_schema(),
        "goal_terminal_evidence.schema.json": GoalTerminalEvidenceV1.model_json_schema(),
        "domain_obstruction_core.schema.json": DomainObstructionCoreV1.model_json_schema(),
        "goal_action_evidence.schema.json": GoalActionEvidenceV1.model_json_schema(),
        "goal_domain_adequacy_report.schema.json": GoalDomainAdequacyReportV1.model_json_schema(),
    }
    for filename, schema in schemas.items():
        path = resources / filename
        rendered = json.dumps(schema, indent=2) + "\n"
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(f"Goal support schema snapshot is stale: {path}")
            print(f"Goal support schema snapshot is current: {path}")
            continue
        path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
