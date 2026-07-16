from pathlib import Path
from types import SimpleNamespace

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.factory import apply_runtime_overrides


def test_none_decode_overrides_preserve_checkpoint_settings() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(grammar_ltr_primary=True, grammar_ltr_repair=True)
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        grammar_ltr_primary=None,
        grammar_ltr_repair=None,
    )

    apply_runtime_overrides(model, config)

    assert model.config.grammar_ltr_primary is True
    assert model.config.grammar_ltr_repair is True


def test_compiler_decode_mode_activates_primary_ltr_path() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            compiler_decode_mode="off",
            grammar_ltr_primary=False,
        )
    )
    config = ModelBuildConfig(
        train_dir=Path("."),
        compiler_decode_mode="tree",
        grammar_ltr_primary=None,
    )

    apply_runtime_overrides(model, config)

    assert model.config.compiler_decode_mode == "tree"
    assert model.config.grammar_ltr_primary is True
