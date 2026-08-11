"""HARN-06 / SLM-546: constructivization generation + validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harnesses.reasoning.revmath.constructivization import (
    assert_not_masquerading,
    evaluate_constructivization,
    generate_constructivization_variant,
    parse_constructivization_meta,
)
from slm_training.harnesses.reasoning.revmath.plugins import (
    ConstructivizationPlugin,
    default_plugin_registry,
    resolve_plugin,
)
from slm_training.harnesses.reasoning.revmath.runner import run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskV1,
    parse_revmath_task,
)

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "src" / "slm_training" / "resources" / "revmath" / "fixtures"


def _load_pair(stem: str) -> tuple[RevmathTaskV1, object]:
    task = parse_revmath_task(
        json.loads((FIXTURES / f"{stem}.task.json").read_text(encoding="utf-8"))
    )
    meta = parse_constructivization_meta(
        json.loads((FIXTURES / f"{stem}.meta.json").read_text(encoding="utf-8"))
    )
    return task, meta


def _run(task: RevmathTaskV1, meta: object):
    def run_variant(baseline: RevmathTaskV1, _variant):
        plugin = ConstructivizationPlugin(meta_override=meta)  # type: ignore[arg-type]
        registry = default_plugin_registry()
        registry["constructivization"] = plugin
        return run_revmath_task(baseline, hermetic=True, plugin_registry=registry)

    return evaluate_constructivization(task, meta, run_variant=run_variant)  # type: ignore[arg-type]


def test_plugin_registered() -> None:
    task, _meta = _load_pair("constructivization_bounded")
    plugin = resolve_plugin(task)
    assert plugin is not None
    assert plugin.task_kind == "constructivization"


def test_bounded_finite_analogue_states_weakening() -> None:
    task, meta = _load_pair("constructivization_bounded")
    assert meta.mode == "bounded_finite_analogue"  # type: ignore[union-attr]
    assert meta.bound_ast_id == "bound.finite_N_8"  # type: ignore[union-attr]
    assert meta.constructivized_statement_sha256 != meta.original_statement_sha256  # type: ignore[union-attr]
    report = _run(task, meta)
    assert report.outcome == "witnessed"
    assert report.to_dict()["masquerades_as_original"] is False
    assert report.weakening_description
    assert report.original_statement_sha256 != report.constructivized_statement_sha256
    assert report.variant.bound_ast_id == "bound.finite_N_8"


def test_witness_oracle_remainder_modes() -> None:
    for stem, mode in (
        ("constructivization_witness", "explicit_witness_producing"),
        ("constructivization_oracle", "oracle_relative"),
        ("constructivization_remainder", "documented_nonconstructive_remainder"),
    ):
        task, meta = _load_pair(stem)
        assert meta.mode == mode  # type: ignore[union-attr]
        report = _run(task, meta)
        assert report.outcome == "witnessed"
        assert report.constructivized_statement_sha256 == task.proposition.statement_sha256


def test_timeout_is_unknown_not_impossibility() -> None:
    task, meta = _load_pair("constructivization_timeout")
    report = _run(task, meta)
    assert report.outcome == "unknown"
    assert report.judgment["outcome"] == JudgmentOutcome.UNKNOWN.value


def test_masquerade_rejected() -> None:
    _task, meta = _load_pair("constructivization_bounded")
    raw = meta.to_dict()  # type: ignore[union-attr]
    raw["constructivized_statement"] = raw["original_statement"]
    raw["constructivized_statement_sha256"] = raw["original_statement_sha256"]
    with pytest.raises(RevmathSchemaError, match="masquerade|equals original"):
        broken = parse_constructivization_meta(raw)
        assert_not_masquerading(broken)


def test_task_must_carry_constructivized_not_original() -> None:
    _task, meta = _load_pair("constructivization_bounded")
    payload = json.loads(
        (FIXTURES / "constructivization_bounded.task.json").read_text(encoding="utf-8")
    )
    payload["proposition"]["statement"] = meta.original_statement  # type: ignore[union-attr]
    payload["proposition"]["statement_sha256"] = meta.original_statement_sha256  # type: ignore[union-attr]
    broken = parse_revmath_task(payload)
    with pytest.raises(RevmathSchemaError, match="constructivized"):
        generate_constructivization_variant(broken, meta)  # type: ignore[arg-type]


def test_variant_generation_deterministic() -> None:
    task, meta = _load_pair("constructivization_bounded")
    first = generate_constructivization_variant(task, meta)  # type: ignore[arg-type]
    second = generate_constructivization_variant(task, meta)  # type: ignore[arg-type]
    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["masquerades_as_original"] is False
