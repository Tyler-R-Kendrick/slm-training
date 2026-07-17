"""F1 (SLM-34): DSL pack contract — registry, invariants, and fixture loop."""

from __future__ import annotations

import pytest

from slm_training.dsl.packs import available_packs, get_pack
from slm_training.dsl.stream_types import StreamStatus


def test_builtin_packs_registered() -> None:
    ids = available_packs()
    assert "openui" in ids
    assert "toy-layout" in ids
    assert get_pack("default").id == "openui"
    with pytest.raises(KeyError):
        get_pack("no-such-dsl")


@pytest.mark.parametrize("pack_id", ["openui", "toy-layout"])
def test_pack_contract_invariants(pack_id: str) -> None:
    """Every pack: generated corpus is valid by its own oracle, scope-clean,
    and canonicalization is an idempotent normal form."""
    pack = get_pack(pack_id)
    assert pack.backend().info.id == pack.grammar
    assert pack.corpus_generator is not None
    records = pack.corpus_generator(3, 0)
    assert len(records) == 3
    # Determinism: same (count, seed) -> same sources.
    again = pack.corpus_generator(3, 0)
    assert [r.openui for r in records] == [r.openui for r in again]
    for record in records:
        source = record.openui
        pack.validity_oracle(source, "document")
        status = pack.scope_check(source)
        assert isinstance(status, StreamStatus)
        assert status.ok, f"{pack_id} generated scope-invalid source"
        canonical = pack.canonicalize(source)
        assert pack.canonicalize(canonical) == canonical
        assert pack.canonical_equal(source, canonical)


def test_openui_pack_placeholder_policy() -> None:
    pack = get_pack("openui")
    assert pack.placeholders.is_placeholder(":hero.title")
    assert not pack.placeholders.is_placeholder("plain text")
    assert pack.placeholders.content_props
    records = pack.corpus_generator(2, 1)
    extracted = pack.placeholders.extract(records[0].openui)
    assert all(pack.placeholders.is_placeholder(p) for p in extracted)


def test_engine_seam_resolves_registered_backends() -> None:
    """The fastpath engine now finds any registered backend's grammar by id
    instead of a hard-coded path table."""
    from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl

    assert engine_for_dsl("toy-layout") is not None
    assert engine_for_dsl("lark-openui") is not None
    assert engine_for_dsl("definitely-unknown") is None


def test_pack_fixture_loop_generate_train_eval() -> None:
    """End-to-end through the pack interface: generate -> scratch-train ->
    decode -> gold re-validated by the pack oracle. Fixture-scale by design."""
    torch = pytest.importorskip("torch")

    from slm_training.harnesses.model_build.plugin import GenerationRequest
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    pack = get_pack("openui")
    records = pack.corpus_generator(3, 7)
    config = TwoTowerConfig(
        output_tokenizer="lexer",
        context_backend="scratch",
        grammar_constrained=False,
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=48,
        max_target_len=160,
        seed=7,
    )
    model = TwoTowerModel.from_records(records, config=config, device="cpu")
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    outputs = model.generate_batch_requests(
        [
            GenerationRequest(
                prompt=records[0].prompt,
                slot_contract=tuple(records[0].placeholders or ()),
            )
        ]
    )
    assert len(outputs) == 1 and isinstance(outputs[0], str)
    for record in records:
        pack.validity_oracle(record.openui, "document")
