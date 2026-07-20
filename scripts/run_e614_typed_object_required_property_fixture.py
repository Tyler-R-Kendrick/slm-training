#!/usr/bin/env python3
"""E614 grammar-only diagnostic: typed-object required-property closure.

E613 (``docs/design/iter-e613-schema-derived-typed-item-20260720.md``) floored
the schema-derived object opener for authored typed-array items (e.g.
``ImageGallery.images``) but observed the resulting object frame did not carry
the item schema, so the model filled arbitrary keys/nested components out to
the 160-token canvas. E613's own next-step note: "propagate typed object
property schemas into the choice state, require known keys, and make required
property closure explicit before testing another decode margin."

This script exercises that harness change (``ChoiceDecodeState`` in
``slm_training.models.choice_tokenizer``, default-off
``require_object_schema_properties`` flag / model-level
``semantic_plan_typed_object_required_property_closure`` config knob) directly
against the grammar, with no trained checkpoint: no checkpoint was available
in this environment to reproduce E608-E613's matched OOD replay, so this is
wiring/fixture evidence only, not a quality-metric replay. It shows, for the
real ``ImageGallery`` component contract:

- default off: the schema-derived object item may close with zero keys
  (reproducing E613's arbitrary-key failure mode's precondition), and
- opt-in on: the pushdown grammar refuses to close the object until its
  schema's required properties (``src``) are filled, and ``minimal_completion_length``
  stays finite (the new constraint does not create a decode dead end).

Usage:
  python -m scripts.run_e614_typed_object_required_property_fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.models.choice_tokenizer import ChoiceDecodeState, ChoiceTokenizer

_DESIGN_JSON = "docs/design/iter-e614-typed-object-required-property-20260720.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _probe_gallery_object_close(
    tok: ChoiceTokenizer, *, require_object_schema_properties: bool
) -> dict[str, Any]:
    """Open ``+ImageGallery([{`` and report whether ``}`` is legal immediately."""
    state = ChoiceDecodeState(
        tok,
        slot_count=1,
        require_object_schema_properties=require_object_schema_properties,
    )
    for tok_str in ("+ImageGallery", "[", "{"):
        assert state.advance_id(tok.token_to_id[tok_str]), tok_str
    allowed = state.allowed_ids(16)
    close_id = tok.token_to_id["}"]
    src_id = tok.token_to_id["n:src"]
    minimal_len = state.clone().minimal_completion_length()
    return {
        "require_object_schema_properties": require_object_schema_properties,
        "close_legal_with_zero_keys": close_id in allowed,
        "required_property_name_legal": src_id in allowed,
        "minimal_completion_length": minimal_len,
        "minimal_completion_feasible": minimal_len < 1025,
    }


def _probe_full_completion_with_required_property(tok: ChoiceTokenizer) -> dict[str, Any]:
    """Confirm a full legal completion exists once the required key is filled."""
    from slm_training.dsl.production_codec import LIT_PREFIX

    state = ChoiceDecodeState(
        tok, slot_count=1, require_object_schema_properties=True
    )
    tokens = (
        "+ImageGallery",
        "[",
        "{",
        "n:src",
        f'{LIT_PREFIX}""',
        "}",
        "]",
        "-",
    )
    for tok_str in tokens:
        if not state.advance_id(tok.token_to_id[tok_str]):
            return {"reached_eos": False, "failed_at": tok_str}
    return {"reached_eos": bool(state.advance_id(tok.eos_id))}


def run_fixture() -> dict[str, Any]:
    tok = ChoiceTokenizer.build()
    off = _probe_gallery_object_close(tok, require_object_schema_properties=False)
    on = _probe_gallery_object_close(tok, require_object_schema_properties=True)
    completion = _probe_full_completion_with_required_property(tok)
    checks = {
        "default_off_preserves_bare_close": off["close_legal_with_zero_keys"] is True,
        "opt_in_blocks_bare_close": on["close_legal_with_zero_keys"] is False,
        "opt_in_offers_required_property_name": on["required_property_name_legal"]
        is True,
        "opt_in_completion_remains_feasible": on["minimal_completion_feasible"]
        is True,
        "filled_required_property_permits_full_completion": completion.get(
            "reached_eos"
        )
        is True,
    }
    return {
        "schema_version": 1,
        "experiment_id": "E614",
        "run_id": "e614-typed-object-required-property-fixture-r1",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "status": "completed",
        "run_class": "fixture",
        "hypothesis": (
            "Propagating typed-array item object schemas into the choice-codec "
            "pushdown state and gating object closure on required properties "
            "will make ImageGallery's schema-derived object frame refuse to "
            "close before its required `src` property is filled, without "
            "creating a decode dead end."
        ),
        "recipe": {
            "component": "ImageGallery",
            "field": "images[].src (required)",
            "mode": "grammar-only fixture; no trained checkpoint available in "
            "this environment",
            "max_wall_minutes": 3.0,
        },
        "checkpoint": {
            "created": False,
            "note": (
                "No prior checkpoint (e.g. e569-e561-matched-cont48-r1-48s used "
                "by E608-E613) was present under outputs/ in this environment; "
                "a matched OOD quality-metric replay is deferred to the next "
                "iteration that has checkpoint access."
            ),
        },
        "probes": {"default_off": off, "opt_in": on, "full_completion": completion},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "version_stamp": build_version_stamp(
            "model.twotower", "harness.model_build.eval"
        ),
        "generated_at": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=_DESIGN_JSON,
        help="Path to write the JSON report (default: docs/design mirror).",
    )
    args = parser.parse_args(argv)

    report = run_fixture()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
