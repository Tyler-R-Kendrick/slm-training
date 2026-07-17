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
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")'
)


def _model(mode: str) -> TwoTowerModel:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            placeholders=[":hero.title", ":hero.body"],
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


def test_replace_mode_substitutes_the_learned_pool_row_exactly() -> None:
    from slm_training.models.dsl_tokenizer import SymbolTable

    model = _model("replace")
    table = SymbolTable.from_placeholders(
        [":hero.title", ":hero.body"], max_slots=model.tokenizer.sym_slots
    )
    features = model._runtime_feature_tensor([table])
    assert features is not None
    weight = model.denoiser.tok.weight
    for slot, surface in enumerate(table.placeholders):
        token_id = model.tokenizer.sym_id(slot)
        delta = features[0, token_id]
        assert torch.any(delta != 0), "replace mode must write a delta"
        effective = weight[token_id] + delta
        # The effective row is the deterministic byte-compositional vector —
        # the learned pool row cancels entirely.
        byte_ids = model.tokenizer._encode_bytes(surface)
        composed = weight.index_select(0, torch.tensor(byte_ids)).mean(0)
        assert torch.allclose(effective, composed, atol=1e-6)
    model.denoiser.set_runtime_symbol_features(None)


def test_same_surface_means_identical_embedding_across_slots() -> None:
    from slm_training.models.dsl_tokenizer import SymbolTable

    model = _model("replace")
    weight = model.denoiser.tok.weight
    # The same surface bound to DIFFERENT slots (different examples) yields
    # the identical effective embedding — referent identity, not slot identity.
    table_a = SymbolTable.from_placeholders(
        [":hero.title", ":hero.body"], max_slots=model.tokenizer.sym_slots
    )
    table_b = SymbolTable.from_placeholders(
        [":hero.body", ":hero.title"], max_slots=model.tokenizer.sym_slots
    )
    fa = model._runtime_feature_tensor([table_a])
    fb = model._runtime_feature_tensor([table_b])
    assert fa is not None and fb is not None
    eff_a = weight[model.tokenizer.sym_id(0)] + fa[0, model.tokenizer.sym_id(0)]
    eff_b = weight[model.tokenizer.sym_id(1)] + fb[0, model.tokenizer.sym_id(1)]
    assert torch.allclose(eff_a, eff_b, atol=1e-6)  # both are :hero.title
    model.denoiser.set_runtime_symbol_features(None)


def test_replace_mode_trains_and_decodes() -> None:
    records = [
        ExampleRecord(
            id=f"r{i}",
            prompt=f"Hero {i}",
            openui=HERO,
            placeholders=[":hero.title", ":hero.body"],
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
            placeholders=[":hero.title", ":hero.body"],
        ),
        ExampleRecord(
            id="b",
            prompt="Hero again",
            openui=HERO,
            placeholders=[":hero.title", ":hero.body"],
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
    assert [row.eid for row in rows] == ["E264"]
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
