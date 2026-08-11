#!/usr/bin/env python3
"""Regenerate or verify JSON Schema generated from CompiledGoalConstraintSetV1."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    from slm_training.bridge_utils import repo_root
    from slm_training.data.progspec.goal_constraints import CompiledGoalConstraintSetV1

    schema_path = (
        repo_root()
        / "src"
        / "slm_training"
        / "resources"
        / "compiled_goal_constraint_set.schema.json"
    )
    rendered = json.dumps(CompiledGoalConstraintSetV1.model_json_schema(), indent=2) + "\n"
    if args.check:
        if not schema_path.is_file() or schema_path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"CompiledGoalConstraintSetV1 schema snapshot is stale: {schema_path}"
            )
        print(f"CompiledGoalConstraintSetV1 schema snapshot is current: {schema_path}")
        return 0
    schema_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {schema_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
