#!/usr/bin/env python3
"""E615 grammar-only diagnostic: broadened coverage of E614's required-property closure.

E614 (``docs/design/iter-e614-typed-object-required-property-20260720.md``)
propagated a typed-array item's object schema (``ImageGallery.images[].src``)
into ``ChoiceDecodeState``'s pushdown stack and grammar-rejected closing the
object before its schema's required properties were filled. That iteration's
own fixture (``run_e614_typed_object_required_property_fixture.py``) and unit
tests only drove the ``OBJ_OPEN`` handler's ``parent.kind == "variadic"``
branch (a typed-array item) with a single required key (``src``). No prior
evidence exercised:

1. The sibling ``parent.kind == "component"`` branch (a directly-typed,
   non-array object *argument*, e.g. what ``Input.rules`` would be if it
   named required keys) in ``ChoiceDecodeState._accept_expression_token``'s
   ``OBJ_OPEN`` handler.
2. A regression check that objects whose schema names *no* required
   properties (true of every shipped directly-typed object argument today,
   e.g. ``Input.rules``) are left unaffected by the opt-in flag.
3. Behaviour with *more than one* required property: whether closure stays
   blocked until every key is filled, whether ``minimal_completion_length``
   keeps targeting the next missing key (not just the first), and whether
   fill order matters.

This script (and the mirrored pytest cases in
``tests/test_models/test_choice_tokenizer.py``) closes those three gaps,
still with no trained checkpoint and no schema change: (1)/(2) push a
synthetic ``_ChoiceFrame`` directly rather than routing through
``_component_contracts()`` (no shipped schema has a required, directly-typed
object argument to exercise this branch against); (3) synthetically widens
the real ``ImageGallery`` item schema's ``object_required`` tuple from
``("src",)`` to ``("src", "alt")`` after opening (no shipped schema currently
requires two keys either). Both are grammar/state-machine invariant checks,
not claims about the real component library's current contracts.

No prior checkpoint (e.g. ``e569-e561-matched-cont48-r1-48s``, used by
E608-E613) was present under ``outputs/`` in this environment, matching
E614's finding; a matched OOD quality-metric replay remains deferred to the
next iteration that has checkpoint access.

Usage:
  python -m scripts.run_e615_typed_object_multi_key_closure_fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.production_codec import CLOSE, LIT_PREFIX
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.models.choice_tokenizer import (
    ChoiceDecodeState,
    ChoiceTokenizer,
    _ChoiceFrame,
)

_DESIGN_JSON = "docs/design/iter-e615-typed-object-multi-key-closure-20260720.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _probe_component_argument_path(tok: ChoiceTokenizer) -> dict[str, Any]:
    """Drive the untested ``parent.kind == "component"`` OBJ_OPEN branch."""
    state = ChoiceDecodeState(tok, slot_count=1, require_object_schema_properties=True)
    state.frames.append(
        _ChoiceFrame(
            "component",
            "element:Synthetic",
            close=CLOSE,
            schemas=({"type": "object", "required": ["src", "alt"]},),
            required_args=1,
            arg_index=0,
        )
    )
    opened = state.advance_id(tok.token_to_id["{"])
    object_required = state.frames[-1].object_required if opened else ()
    close_id = tok.token_to_id["}"]
    src_id = tok.token_to_id["n:src"]
    allowed = state.allowed_ids(16)
    return {
        "object_opened": opened,
        "object_required_derived": list(object_required),
        "close_legal_with_zero_keys": close_id in allowed,
        "required_property_name_legal": src_id in allowed,
    }


def _probe_component_argument_no_required(tok: ChoiceTokenizer) -> dict[str, Any]:
    """Regression: an object argument with no required keys stays unaffected."""
    state = ChoiceDecodeState(tok, slot_count=1, require_object_schema_properties=True)
    state.frames.append(
        _ChoiceFrame(
            "component",
            "element:Synthetic",
            close=CLOSE,
            schemas=({"type": "object"},),
            required_args=1,
            arg_index=0,
        )
    )
    opened = state.advance_id(tok.token_to_id["{"])
    object_required = state.frames[-1].object_required if opened else ("<unopened>",)
    closes = state.advance_id(tok.token_to_id["}"]) if opened else False
    return {
        "object_opened": opened,
        "object_required_derived": list(object_required),
        "bare_close_still_legal": closes,
    }


def _open_gallery_widened(tok: ChoiceTokenizer) -> ChoiceDecodeState:
    """Open ``+ImageGallery([{`` then synthetically widen to two required keys."""
    state = ChoiceDecodeState(tok, slot_count=1, require_object_schema_properties=True)
    for tok_str in ("+ImageGallery", "[", "{"):
        assert state.advance_id(tok.token_to_id[tok_str]), tok_str
    state.frames[-1].object_required = ("src", "alt")
    return state


def _probe_multi_key_order(tok: ChoiceTokenizer, *, order: tuple[str, str]) -> dict[str, Any]:
    state = _open_gallery_widened(tok)
    steps: list[dict[str, Any]] = []
    for name in order:
        name_id = tok.token_to_id[f"n:{name}"]
        close_id = tok.token_to_id["}"]
        pre_close_blocked = not state.clone().advance_id(close_id)
        completion_target = state._completion_id()
        assert state.advance_id(name_id)
        assert state.advance_id(tok.token_to_id[f'{LIT_PREFIX}""'])
        steps.append(
            {
                "filled": name,
                "pre_fill_close_blocked": pre_close_blocked,
                "pre_fill_completion_targeted": tok.id_to_token.get(completion_target),
            }
        )
    final_close = state.advance_id(tok.token_to_id["}"])
    return {"order": list(order), "steps": steps, "final_close_legal": final_close}


def run_fixture() -> dict[str, Any]:
    tok = ChoiceTokenizer.build()
    component_arg = _probe_component_argument_path(tok)
    component_arg_no_required = _probe_component_argument_no_required(tok)
    forward = _probe_multi_key_order(tok, order=("src", "alt"))
    reverse = _probe_multi_key_order(tok, order=("alt", "src"))

    checks = {
        "component_argument_branch_derives_required": component_arg[
            "object_required_derived"
        ]
        == ["src", "alt"],
        "component_argument_branch_blocks_bare_close": component_arg[
            "close_legal_with_zero_keys"
        ]
        is False,
        "component_argument_branch_offers_required_key": component_arg[
            "required_property_name_legal"
        ]
        is True,
        "no_required_object_argument_unaffected": component_arg_no_required[
            "bare_close_still_legal"
        ]
        is True,
        "forward_order_blocks_until_both_filled": all(
            step["pre_fill_close_blocked"] for step in forward["steps"]
        ),
        "forward_order_targets_next_missing_key": forward["steps"][0][
            "pre_fill_completion_targeted"
        ]
        == "n:src"
        and forward["steps"][1]["pre_fill_completion_targeted"] == "n:alt",
        "forward_order_reaches_close": forward["final_close_legal"] is True,
        "reverse_order_blocks_until_both_filled": all(
            step["pre_fill_close_blocked"] for step in reverse["steps"]
        ),
        # Honest finding, not a bug: `_completion_id` targets the first still-
        # missing key in the schema's *declared* `required` order, not the
        # key the caller is actually about to fill next. That is correct for
        # its purpose (`minimal_completion_length` only needs *a* legal,
        # feasible completion to exist) but it means both reverse-order steps
        # report "n:src" here even though "alt" is filled first in reality.
        "reverse_order_completion_target_is_declaration_order": reverse[
            "steps"
        ][0]["pre_fill_completion_targeted"]
        == "n:src"
        and reverse["steps"][1]["pre_fill_completion_targeted"] == "n:src",
        "reverse_order_reaches_close": reverse["final_close_legal"] is True,
    }
    return {
        "schema_version": 1,
        "experiment_id": "E615",
        "run_id": "e615-typed-object-multi-key-closure-fixture-r1",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "status": "completed",
        "run_class": "fixture",
        "hypothesis": (
            "E614's required-property closure gate (ChoiceDecodeState."
            "require_object_schema_properties) generalizes beyond the single "
            "typed-array-item / single-required-key case it shipped with: it "
            "also gates the sibling directly-typed-component-argument OBJ_OPEN "
            "branch, leaves no-required-key objects unaffected, and correctly "
            "sequences and order-independently tracks two required keys "
            "without creating a decode dead end."
        ),
        "recipe": {
            "component": "synthetic component-argument frame + widened "
            "ImageGallery.images[] item schema",
            "field": "object_required tuple widened to (src, alt); no shipped "
            "schema currently requires two keys or names required keys on a "
            "directly-typed (non-array) object argument",
            "mode": "grammar-only fixture; no trained checkpoint available in "
            "this environment (same finding as E614)",
            "max_wall_minutes": 3.0,
        },
        "checkpoint": {
            "created": False,
            "note": (
                "No prior checkpoint (e.g. e569-e561-matched-cont48-r1-48s used "
                "by E608-E613) was present under outputs/ in this environment; "
                "a matched OOD quality-metric replay remains deferred to the "
                "next iteration that has checkpoint access."
            ),
        },
        "probes": {
            "component_argument_branch": component_arg,
            "component_argument_branch_no_required": component_arg_no_required,
            "multi_key_forward_order": forward,
            "multi_key_reverse_order": reverse,
        },
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
