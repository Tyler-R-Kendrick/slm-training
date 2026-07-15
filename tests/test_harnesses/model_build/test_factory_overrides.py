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
