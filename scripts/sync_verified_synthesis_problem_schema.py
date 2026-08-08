#!/usr/bin/env python3
"""Regenerate or verify the JSON Schema generated from VerifiedSynthesisProblemV1.

SGS-007 (SLM-444) requires the schema to be generated from the typed owner,
never hand-maintained; this mirrors sync_openui_schema.py's --check shape.
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    from slm_training.bridge_utils import repo_root
    from slm_training.data.progspec.synthesis_problem import (
        VerifiedSynthesisProblemV1,
    )

    schema_path = (
        repo_root()
        / "src"
        / "slm_training"
        / "resources"
        / "verified_synthesis_problem.schema.json"
    )
    rendered = json.dumps(VerifiedSynthesisProblemV1.model_json_schema(), indent=2) + "\n"
    if args.check:
        if not schema_path.is_file() or schema_path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"VerifiedSynthesisProblemV1 schema snapshot is stale: {schema_path}"
            )
        print(f"VerifiedSynthesisProblemV1 schema snapshot is current: {schema_path}")
        return 0
    schema_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {schema_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
