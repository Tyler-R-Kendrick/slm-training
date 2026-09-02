"""Fixture-scale wall-clock guard for strict compiler-tree decode (P5).

The screening eval decodes every smoke record under
``STRICT_COMPILER_TREE_POLICY``; its per-record cost is dominated by the
grammar authority (completion-forest builds inside the terminal-witness
search), not by the denoiser.  This test decodes three smoke prompts with a
deterministic, untrained fixture model through the same
``generate_batch_requests`` seam the eval harness uses and asserts the median
per-record wall time stays under ``PER_RECORD_BUDGET_S``.

Calibration is explicit, never silent: a fixed pure-Python workload is timed
first and compared against ``CALIBRATION_REFERENCE_S`` (measured on the
4-CPU, no-GPU box that set the budget).  A machine slower than
``MAX_CALIBRATION_SLOWDOWN`` times the reference skips with the measured
ratio in the skip reason, so a skip is visible in the report and never mistaken
for a pass.  Fixture-scale performance only; no quality or ship claim
(docs/design/compiler-decode-cost-20260902.md).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest
import torch

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.eval_policy import (
    STRICT_COMPILER_TREE_POLICY_ID,
    apply_strict_compiler_tree_policy,
)
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

# Median per-record wall budget (seconds) for the three-prompt workload below.
PER_RECORD_BUDGET_S = 5.0
# Wall seconds the calibration workload took on the reference box
# (4 CPU, no GPU, quiet; 0.10-0.12 s over five runs).  Re-measure when the box
# changes.
CALIBRATION_REFERENCE_S = 0.11
# Skip (loudly) when this machine is more than this many times slower.
MAX_CALIBRATION_SLOWDOWN = 2.5
SMOKE_RECORDS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "data"
    / "eval"
    / "e938_role_safe_all_targets_smoke24_v1"
    / "suites"
    / "smoke"
    / "records.jsonl"
)
N_PROMPTS = 3
MIN_CONTENT_COMPONENTS = 3


def _calibration_seconds() -> float:
    """Time a fixed pure-Python workload (dict/tuple/sort churn, like the kernel)."""
    started = time.perf_counter()
    table: dict[tuple[int, int], int] = {}
    for index in range(120_000):
        key = (index % 977, index % 331)
        table[key] = table.get(key, 0) + index
    _ = sorted(table.items())[:10]
    return time.perf_counter() - started


def _smoke_records(limit: int) -> list[ExampleRecord]:
    rows = [
        json.loads(line)
        for line in SMOKE_RECORDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        ExampleRecord(
            id=row["id"],
            prompt=row["prompt"],
            openui=row["openui"],
            placeholders=list(row.get("placeholders", [])),
            split="smoke",
            source=row.get("source", "fixture"),
            meta=row.get("meta", {}),
        )
        for row in rows[:limit]
    ]


def _request_for(record: ExampleRecord) -> GenerationRequest:
    # Same request shape as harnesses/model_build/eval_runner._request_for.
    request = GenerationRequest.from_record(record, schema=None)
    data = request.to_dict()
    data["runtime_symbols"] = [
        RuntimeSymbol(surface=slot, role="external_entity").to_dict()
        for slot in request.slot_contract
    ]
    return GenerationRequest.from_dict(data)


def _fixture_model(records: list[ExampleRecord]) -> TwoTowerModel:
    torch.manual_seed(0)
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer="lexer",
        d_model=32,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        gen_steps=2,
        grammar_constrained=True,
        grammar_ltr_primary=True,
        grammar_ltr_max_tokens=192,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=config, device="cpu")
    model.eval()
    apply_strict_compiler_tree_policy(model.config)
    # A4 minimum-content floor: an untrained model closes the layout after one
    # component; requiring three keeps the workload at the multi-binder size
    # the screening records actually decode (~30 tokens, ~20 domain queries).
    model.config.decode_min_content = MIN_CONTENT_COMPONENTS
    assert model.config.evaluation_policy == STRICT_COMPILER_TREE_POLICY_ID
    assert model.config.compiler_decode_mode == "tree"
    return model


def test_strict_compiler_tree_decode_per_record_wall_budget() -> None:
    # Min of three so a single scheduler hiccup does not fake a slow machine.
    calibration = min(_calibration_seconds() for _ in range(3))
    slowdown = calibration / CALIBRATION_REFERENCE_S
    if slowdown > MAX_CALIBRATION_SLOWDOWN:
        pytest.skip(
            f"machine is {slowdown:.1f}x slower than the calibration reference "
            f"({calibration:.3f}s vs {CALIBRATION_REFERENCE_S:.3f}s); the "
            f"{PER_RECORD_BUDGET_S:.0f}s per-record budget is not meaningful here"
        )

    records = _smoke_records(N_PROMPTS)
    assert len(records) == N_PROMPTS
    model = _fixture_model(records)
    cap = int(model.config.grammar_ltr_max_tokens)

    walls: list[float] = []
    for record in records:
        request = _request_for(record)
        started = time.perf_counter()
        with collect_decode_stats() as stats:
            texts = model.generate_batch_requests([request], max_len=cap)
        walls.append(time.perf_counter() - started)
        assert len(texts) == 1 and texts[0]
        # The grammar authority is the cost here, and it must stay attributed.
        assert stats.compiler_ms > 0.0
        assert stats.tokens_emitted > 0

    median_wall = statistics.median(walls)
    assert median_wall <= PER_RECORD_BUDGET_S, (
        f"median per-record strict compiler-tree decode wall {median_wall:.2f}s "
        f"exceeds {PER_RECORD_BUDGET_S:.1f}s budget (walls={[round(w, 2) for w in walls]}, "
        f"calibration slowdown {slowdown:.2f}x)"
    )
