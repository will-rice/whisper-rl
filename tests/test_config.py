"""Tests for the configuration model."""

from whisper_rl.config import Config


def test_default_config_instantiates() -> None:
    """The default config should be valid and PoC-sized."""
    config = Config()
    assert config.base_model == "openai/whisper-tiny"
    assert config.num_generations > 1
    assert config.kl_beta >= 0.0


def test_config_overrides_apply() -> None:
    """Overrides should be honored by the pydantic model."""
    config = Config(num_generations=4, learning_rate=5e-6)
    assert config.num_generations == 4
    assert config.learning_rate == 5e-6
