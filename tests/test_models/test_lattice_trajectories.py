from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _model() -> TwoTowerModel:
    record = ExampleRecord(
        id="trajectory",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")\n',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="lexer",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=32,
            max_target_len=32,
            grammar_ltr_max_tokens=32,
            gen_steps=1,
            seed=0,
            compiler_search_mode="ptrm",
            compiler_search_trigger="always",
            compiler_search_width=4,
            compiler_search_noise=1.0,
        ),
        device="cpu",
    )
    model.eval()
    return model


def test_ptrm_runs_full_validated_continuations_without_invalid_selection() -> None:
    model = _model()
    ctx, ctx_pad = model._encode_context(["card"])

    with collect_decode_stats() as stats:
        model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )

    assert stats.compiler_lattice_verifier_calls == 4
    assert stats.compiler_lattice_trajectories >= 4
    assert stats.compiler_lattice_invalid_selected_over_valid == 0
