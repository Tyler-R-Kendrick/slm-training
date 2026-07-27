"""C2 (SLM-26): dynamic pseudo-embeddings for symbol tokens."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.binding_consistency import binding_consistency_probe
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero_title, hero_body], "column")\n'
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")'
)


def _model(mode: str) -> TwoTowerModel:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
        )
    ]
    return TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            seed=0,
            gen_steps=2,
            output_tokenizer="lexer",
            runtime_symbol_features=mode,
        ),
        device="cpu",
    )


def test_replace_mode_uses_opaque_ordinal_features() -> None:
    from slm_training.models.dsl_tokenizer import SymbolTable

    model = _model("replace")
    table = SymbolTable.from_placeholders(
        [":slot_0", ":slot_1"], max_slots=model.tokenizer.sym_slots
    )
    features = model._runtime_feature_tensor([table])
    assert features is not None
    weight = model.denoiser.tok.weight
    for slot, _surface in enumerate(table.placeholders):
        token_id = model.tokenizer.sym_id(slot)
        delta = features[0, token_id]
        assert torch.any(delta != 0), "replace mode must write a delta"
        effective = weight[token_id] + delta
        byte_ids = model.tokenizer._encode_bytes(
            f"content:{slot} role:external_entity"
        )
        composed = weight.index_select(0, torch.tensor(byte_ids)).mean(0)
        assert torch.allclose(effective, composed, atol=1e-6)
    model.denoiser.set_runtime_symbol_features(None)


def test_marker_embedding_depends_on_ordinal() -> None:
    from slm_training.models.dsl_tokenizer import SymbolTable

    model = _model("replace")
    weight = model.denoiser.tok.weight
    table = SymbolTable.from_placeholders(
        [":slot_0", ":slot_1"], max_slots=model.tokenizer.sym_slots
    )
    features = model._runtime_feature_tensor([table])
    assert features is not None
    first = weight[model.tokenizer.sym_id(0)] + features[0, model.tokenizer.sym_id(0)]
    second = weight[model.tokenizer.sym_id(1)] + features[0, model.tokenizer.sym_id(1)]
    assert not torch.allclose(first, second, atol=1e-6)
    model.denoiser.set_runtime_symbol_features(None)


def test_slot_component_texts_use_opaque_ordinals() -> None:
    model = _model("none")
    model._opaque_slot_projection_active = True
    try:
        assert model._slot_component_texts([":slot_0", ":slot_1"]) == [
            "content:0",
            "content:1",
        ]
        assert model._slot_role_token(":slot_0") == "content"
    finally:
        model._opaque_slot_projection_active = False


def test_context_rejects_named_marker_contracts() -> None:
    model = _model("none")
    model.config.slot_contract_in_context = True
    with pytest.raises(ValueError, match="opaque :slot_<ordinal>"):
        model._context_prompts(
            ["Use :hero.title"], slot_contracts=[[":hero.title"]]
        )
    canonical = model._context_prompts(
        ["Use :slot_0 then :slot_1"],
        slot_contracts=[[":slot_0", ":slot_1"]],
    )
    assert canonical[0].count(":slot_") >= 2


def test_opaque_replace_mode_uses_canonical_inventory() -> None:
    from slm_training.models.dsl_tokenizer import SymbolTable

    model = _model("replace")
    table = SymbolTable.from_placeholders(
        [":slot_0", ":slot_1"],
        max_slots=model.tokenizer.sym_slots,
    )
    model._opaque_slot_projection_active = True
    try:
        features = model._runtime_feature_tensor([table])
        assert features is not None
    finally:
        model._opaque_slot_projection_active = False
        model.denoiser.set_runtime_symbol_features(None)


def test_replace_mode_trains_and_decodes() -> None:
    records = [
        ExampleRecord(
            id=f"r{i}",
            prompt=f"Hero {i}",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
        )
        for i in range(4)
    ]
    model = _model("replace")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model.training_loss(records)
        loss.backward()
        optimizer.step()
        assert float(loss.item()) == float(loss.item())  # finite
    out = model.generate("Hero 0", grammar_constrained=False)
    assert isinstance(out, str)


def test_binding_consistency_probe_reports_margin() -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
        ),
        ExampleRecord(
            id="b",
            prompt="Hero again",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
        ),
    ]
    model = _model("replace")
    report = binding_consistency_probe(model, records)
    assert report["mode"] == "replace"
    assert report["surfaces"] == 2
    assert report["same_surface_pairs"] > 0
    assert report["cross_surface_pairs"] > 0
    assert report["binding_margin"] is not None
    # Unknown modes are rejected loudly (config validation).
    model.config.runtime_symbol_features = "bogus"
    from slm_training.models.dsl_tokenizer import SymbolTable

    with pytest.raises(ValueError, match="unknown runtime_symbol_features"):
        model._runtime_feature_tensor(
            [SymbolTable.from_placeholders([":a.b"], max_slots=4)]
        )


def test_v13_registers_matched_c2_row() -> None:
    from scripts.run_quality_matrix import _v11_experiments, _v15_experiments

    rows = _v15_experiments(Path("outputs/data/train/v1"))
    assert [row.eid for row in rows] == ["E278"]
    (c2_row,) = rows
    assert c2_row.runtime_symbol_features == "replace"
    (control,) = [r for r in _v11_experiments(c2_row.train_dir) if r.eid == "E255"]
    a, b = asdict(control), asdict(c2_row)
    assert {k for k in a if a[k] != b[k]} == {
        "eid",
        "run_id",
        "description",
        "runtime_symbol_features",
    }
