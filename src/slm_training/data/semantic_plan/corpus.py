"""Deterministic fixture corpus for SLM-144 SPV1-01 plan-predictor wiring."""

from __future__ import annotations

import random
from typing import Any

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.dsl.language_contract import contract_id as current_contract_id
from slm_training.dsl.pack import get_pack

__all__ = ["build_fixture_plan_corpus"]

_ROOT_CONTAINERS = ["Stack", "Card", "List", "Grid"]
_DIRECTIONS = ["column", "row"]
_LEAF_COMPONENTS = ["TextContent", "Button", "Input", "CardHeader"]
_CONTENT_PROP = {
    "TextContent": "text",
    "Button": "label",
    "Input": "placeholder",
    "CardHeader": "title",
}


def _make_leaf(counter: int, family: str) -> dict[str, Any]:
    prop = _CONTENT_PROP[family]
    return {
        "typeName": family,
        "props": {prop: f":slm144_{counter:04d}.{prop}"},
    }


def _make_program(index: int, rng: random.Random) -> dict[str, Any]:
    root_type = rng.choice(_ROOT_CONTAINERS)
    n_children = rng.randint(1, 3)
    children: list[dict[str, Any]] = []
    base = index * 10
    for offset in range(n_children):
        family = rng.choice(_LEAF_COMPONENTS)
        children.append(_make_leaf(base + offset, family))

    props: dict[str, Any] = {"children": children}
    if root_type == "Stack":
        props["direction"] = rng.choice(_DIRECTIONS)

    return {
        "typeName": root_type,
        "props": props,
    }


def build_fixture_plan_corpus(
    count: int = 64,
    seed: int = 0,
) -> dict[str, list[tuple[ProgramSpec, SemanticPlanV1]]]:
    """Generate a deterministic fixture corpus of ProgramSpec + gold plans.

    The corpus is split 80/20 train/val. All records use the ``openui`` pack and
    the ``OpenUISemanticPlanExtractor`` so the predictor wiring reuses the same
    semantic-plan schema, extractor, and fingerprints as production code.
    """
    rng = random.Random(seed)
    pack = get_pack("openui")
    extractor = OpenUISemanticPlanExtractor()
    contract_id = current_contract_id()

    records: list[tuple[ProgramSpec, SemanticPlanV1]] = []
    for i in range(count):
        ast = {"root": _make_program(i, rng)}
        spec = ProgramSpec(
            id=f"slm144_{i:04d}",
            ast=ast,
            canonical_openui="fixture",
            facts={},
            contract_id=contract_id,
            program_family_id="slm144_fixture",
            lineage_id="slm144",
            split_group_id=f"sg_{i % 8}",
            split="train",
        )
        plan = extractor.extract(spec, pack)
        records.append((spec, plan))

    split = int(count * 0.8)
    return {"train": records[:split], "val": records[split:]}
