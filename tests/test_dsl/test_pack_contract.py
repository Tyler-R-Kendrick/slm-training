"""F1 (SLM-34): DSL pack contract + OpenUI as the first pack instance."""

from __future__ import annotations

import pytest

from slm_training.dsl.pack import available_packs, get_pack

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)


def test_openui_pack_registers_and_resolves() -> None:
    assert "openui" in available_packs()
    pack = get_pack("openui")
    assert pack.id == "openui"
    # Default + env-fallback resolution mirror the grammar-backend switch.
    assert get_pack(None).id == "openui"
    assert get_pack("default").id == "openui"
    with pytest.raises(KeyError):
        get_pack("no-such-dsl")


def test_openui_pack_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SLM_DSL_PACK", "openui")
    assert get_pack(None).id == "openui"
    monkeypatch.delenv("SLM_DSL_PACK")
    monkeypatch.setenv("SLM_GRAMMAR_DSL", "openui")
    assert get_pack(None).id == "openui"


def test_openui_pack_members_are_the_existing_owners() -> None:
    pack = get_pack("openui")
    # Grammar backend resolves through the existing registry.
    assert pack.backend().info.id == "openui"
    # Placeholder policy is the routing defense (C4 context).
    assert pack.placeholder_policy.is_placeholder(":hero.title")
    assert not pack.placeholder_policy.is_placeholder("hello")
    assert pack.placeholder_policy.extract(HERO) == [":hero.title", ":hero.body"]
    # Scope rules expose the codec's reference representations.
    assert set(pack.scope_rules.bind_encodings) == {"absolute", "relative"}
    assert "verifier" in pack.scope_rules.reference_legality or "ParseError" in (
        pack.scope_rules.reference_legality
    )
    families = pack.scope_rules.scope_families()
    assert families and all(isinstance(f, str) for f in families)
    # Language contract id is stable and content-derived.
    cid = pack.contract_id()
    assert isinstance(cid, str) and cid == pack.contract_id()


def test_openui_pack_canonicalizer_and_oracle() -> None:
    pack = get_pack("openui")
    program = pack.validity_oracle(HERO)
    assert getattr(program, "serialized", None)
    canonical = pack.canonicalize(HERO)
    # Idempotent normal form with a stable fingerprint.
    assert pack.canonicalize(canonical) == canonical
    assert pack.canonical_fingerprint(HERO) == pack.canonical_fingerprint(canonical)
    with pytest.raises(Exception):
        pack.validity_oracle("root = NotAComponent((((")


def test_openui_pack_end_to_end_generate_train_eval() -> None:
    """The F1 verify gate: generate → train scratch → eval via the pack."""
    torch = pytest.importorskip("torch")  # noqa: F841

    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    pack = get_pack("openui")

    # Generate: typed-AST scope corpus from one canonical root program.
    build = pack.corpus_generator()
    records, pairs = build(
        root_id="f1-fixture",
        openui=HERO,
        split="train",
        split_group_id="f1-fixture-group",
        program_family_id="f1-fixture-family",
        lineage_id="f1-fixture-lineage",
    )
    assert records, "pack corpus generator produced no records"
    assert all(r.openui for r in records)
    # Every generated program passes the pack's own validity oracle
    # (fragment kinds may be non-document outputs; check document records).
    documents = [r for r in records if "root =" in r.openui]
    assert documents
    for record in documents[:4]:
        pack.validity_oracle(record.openui)

    # Train scratch: tiny CPU model from the generated records, with real
    # gradient steps so the fixture exercises the training path end-to-end.
    model = TwoTowerModel.from_records(
        documents[:8],
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            seed=0,
            gen_steps=2,
        ),
        device="cpu",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = model.training_loss(documents[:4])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    assert all(loss == loss for loss in losses), "training produced NaN loss"

    # Eval: decode and certify through the pack's oracle/canonicalizer path.
    out = model.generate(documents[0].prompt, grammar_constrained=True)
    assert isinstance(out, str)
    if out.strip():
        try:
            program = pack.validity_oracle(out)
        except Exception:
            # Fixture-tiny model may emit unparseable text; the contract
            # only promises the oracle *decides*, not that the model passes.
            program = None
        if program is not None and getattr(program, "serialized", None):
            canonical = pack.canonicalize(program.serialized)
            assert pack.canonicalize(canonical) == canonical
