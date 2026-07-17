#!/usr/bin/env python3
"""G3 (SLM-47): synthesize a DSL pack from a task spec (task → grammar → pack).

Deterministic, no LLM, no training run. Reads a :class:`LatentTaskSpec` (inline
JSON or a fixture file), calls ``synthesize_pack``, writes the synthesized
``.lark`` to an output dir, and prints the pack's filled slots plus a parse →
serialize → re-parse round-trip demo.

Examples
--------
    python -m scripts.synthesize_pack --spec tests/test_dsl/fixtures/latent_task.json
    python -m scripts.synthesize_pack --spec-json '{"task_id":"kv","components":[{"name":"row","props":["a","b"]}]}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.latent import LatentTaskSpec, synthesize_pack


def _demo_program(spec: LatentTaskSpec) -> str:
    """A trivial program in the synthesized DSL for the round-trip demo."""
    if spec.components:
        comp = spec.components[0]
        args = ", ".join(f'":demo.{p}"' for p in comp.props)
        return f"{spec.root_name} = {comp.name}({args})"
    return f'{spec.root_name} = text(":demo.hello")'


def _load_spec(args: argparse.Namespace) -> LatentTaskSpec:
    if args.spec_json:
        data = json.loads(args.spec_json)
    else:
        data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    return LatentTaskSpec.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", type=Path, help="path to a task-spec JSON fixture")
    source.add_argument("--spec-json", type=str, help="inline task-spec JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dsl/latent"),
        help="directory to write the synthesized .lark into",
    )
    parser.add_argument(
        "--program",
        type=str,
        default=None,
        help="override the round-trip demo program",
    )
    args = parser.parse_args(argv)

    spec = _load_spec(args)
    args.output.mkdir(parents=True, exist_ok=True)
    pack = synthesize_pack(spec, grammars_dir=args.output)

    grammar_path = args.output / f"{spec.dsl_id}.lark"
    print(f"task {spec.task_id!r} -> pack {pack.pack_id!r}")
    print(f"grammar: {grammar_path}")
    print(f"filled slots: {list(pack.filled_slots())}")
    print(f"reward_label: {pack.reward_label}")
    print(f"prop_order: {pack.prop_order()}")

    program = args.program or _demo_program(spec)
    parsed = pack.backend.parse(program)
    serialized = pack.backend.serialize(parsed)
    reparsed = pack.backend.parse(serialized)
    ok = reparsed.root == parsed.root
    print(f"demo program:  {program!r}")
    print(f"serialized:    {serialized!r}")
    print(f"round-trip root-stable: {ok}")
    scopes = sorted({s.scope for s in pack.scope_extractor(program)})
    print(f"scopes: {scopes}")
    print(f"slot_contract: {pack.placeholder_policy.slot_contract(program)}")
    engine = pack.incremental_engine()
    head = spec.components[0].name if spec.components else "text"
    open_call = f"{spec.root_name} = {head}("
    print(
        f"incremental engine can_complete_with_holes({open_call!r}): "
        f"{engine.can_complete_with_holes(open_call)}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
