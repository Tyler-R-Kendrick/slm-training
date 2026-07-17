"""G3 (SLM-47): latent-DSL generator — task in, oracle-valid pack out."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.pack import get_pack
from slm_training.harnesses.experiments.latent_dsl import (
    TaskSpec,
    generate_corpus,
    instantiate_pack,
    run_fixture,
    synthesize_grammar,
)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="checkout-form",
        description="Render a checkout confirmation",
        components={
            "panel": ("children",),
            "label": ("text",),
            "amount": ("value",),
        },
        content_slots=(":order.total", ":order.status"),
    )


def test_task_spec_is_strictly_typed() -> None:
    with pytest.raises(ValueError, match="slug"):
        TaskSpec(task_id="Bad Slug!", description="x", components={"a": ()})
    with pytest.raises(ValueError, match="at least one component"):
        TaskSpec(task_id="empty", description="x")
    with pytest.raises(ValueError, match="start with ':'"):
        TaskSpec(
            task_id="slots",
            description="x",
            components={"a": ()},
            content_slots=("nope",),
        )


def test_instantiated_pack_registers_backend_and_oracle(tmp_path: Path) -> None:
    task = _task()
    grammar = synthesize_grammar(task)
    assert "start: statement*" in grammar and task.task_id in grammar
    pack = instantiate_pack(task, tmp_path)
    # The pack resolves through the shared registries (F1 contract).
    assert get_pack(pack.id) is pack
    assert pack.backend().available()
    # Oracle decides: legal program passes, garbage raises.
    program = pack.validity_oracle('root = panel([x])\nx = label(":order.status")')
    assert program is not None
    with pytest.raises(Exception):
        pack.validity_oracle("root = ((((")
    # Canonicalizer is idempotent with a stable fingerprint.
    source = 'root = panel([x])\nx = amount(":order.total")'
    canonical = pack.canonicalize(source)
    assert pack.canonicalize(canonical) == canonical
    assert pack.canonical_fingerprint(source) == pack.canonical_fingerprint(canonical)
    # Contract id is derived from the synthesized grammar content.
    assert pack.contract_id().startswith("latent-checkout-form-")


def test_generated_corpus_passes_the_packs_own_oracle(tmp_path: Path) -> None:
    task = _task()
    pack = instantiate_pack(task, tmp_path)
    records = generate_corpus(task)
    assert len(records) == 3 * 4
    for record in records:
        pack.validity_oracle(record.openui)
        assert record.placeholders and all(
            pack.placeholder_policy.is_placeholder(p) for p in record.placeholders
        )
    # Every component is exercised.
    assert {r.meta["component"] for r in records} == set(task.components)


def test_end_to_end_fixture_task_to_trained_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    summary = run_fixture(_task(), tmp_path, train_steps=2)
    assert summary["pack_id"] == "latent-checkout-form"
    assert summary["records"] == 12
    assert summary["backend_available"] is True
    assert len(summary["train_losses"]) == 2
    assert all(loss == loss for loss in summary["train_losses"])  # no NaN
    # The honest contract: the summary REPORTS whether the tiny scratch
    # decode passed the oracle; at 2 fixture steps it may legitimately fail.
    assert isinstance(summary["decoded_oracle_valid"], bool)
