#!/usr/bin/env python3
"""Regenerate or verify the committed schema from the pinned OpenUI library."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
    from slm_training.dsl.lang_core import library_schema

    schema_path = GRAMMARS_DIR / "openui_schema.json"
    order_path = GRAMMARS_DIR / "openui_prop_order.json"
    schema = library_schema(refresh=True, allow_snapshot=False)
    expected_order = {
        name: list((definition.get("properties") or {}).keys())
        for name, definition in (schema.get("$defs") or {}).items()
    }
    actual_order = json.loads(order_path.read_text(encoding="utf-8"))
    if expected_order != actual_order:
        raise RuntimeError(
            "official schema property order differs from openui_prop_order.json"
        )
    # JSON object insertion order is the positional OpenUI argument contract.
    rendered = json.dumps(schema, indent=2) + "\n"
    if args.check:
        if not schema_path.is_file() or schema_path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"OpenUI schema snapshot is stale: {schema_path}")
        print(f"OpenUI schema snapshot is current: {schema_path}")
        return 0
    schema_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {schema_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
