"""Tests for Whisper prompt reconstruction against the installed transformers."""

import pytest
import torch
from transformers import WhisperForConditionalGeneration

from whisper_rl.config import Config
from whisper_rl.modeling import build_policy, decoder_prompt


@pytest.fixture(scope="module")
def policy() -> WhisperForConditionalGeneration:
    """The real (tiny) Whisper policy; pins the generate() output contract."""
    return build_policy(Config())


def test_generate_strips_decoder_prompt(
    policy: WhisperForConditionalGeneration,
) -> None:
    """Transformers >= 5 returns only sampled tokens, with no forced prompt.

    The GRPO loss slices completions relative to the prompt, so this contract
    must hold; if a transformers upgrade changes it, this test fails first.
    """
    features = torch.zeros(2, 80, 3000)
    sequences = policy.generate(  # ty: ignore[missing-argument]
        input_features=features, task="transcribe", max_new_tokens=4
    )
    start = policy.generation_config.decoder_start_token_id
    assert (sequences != start).all()  # ty: ignore[unresolved-attribute]


def test_decoder_prompt_pins_known_locale(
    policy: WhisperForConditionalGeneration,
) -> None:
    """The prompt's language token comes from the known locale, not detection.

    Whisper mis-detects the language of lower-resource clips (e.g. Urdu as
    Hindi), so when Common Voice gives us the locale we must pin it. The clip's
    features must not change the language token here.
    """
    config = policy.generation_config
    features = torch.zeros(2, 80, 3000)
    prompt = decoder_prompt(policy, features, "transcribe", ["ur", "ka"])

    assert prompt.shape == (2, 4)
    assert (prompt[:, 0] == config.decoder_start_token_id).all()
    assert prompt[0, 1].item() == config.lang_to_id["<|ur|>"]  # ty: ignore[unresolved-attribute]
    assert prompt[1, 1].item() == config.lang_to_id["<|ka|>"]  # ty: ignore[unresolved-attribute]
    assert (prompt[:, 2] == config.task_to_id["transcribe"]).all()  # ty: ignore[unresolved-attribute]
    assert (prompt[:, 3] == config.no_timestamps_token_id).all()  # ty: ignore[unresolved-attribute]


def test_decoder_prompt_strips_locale_region(
    policy: WhisperForConditionalGeneration,
) -> None:
    """Common Voice region-coded locales map to the base Whisper language."""
    config = policy.generation_config
    prompt = decoder_prompt(policy, torch.zeros(1, 80, 3000), "transcribe", ["sv-SE"])
    assert prompt[0, 1].item() == config.lang_to_id["<|sv|>"]  # ty: ignore[unresolved-attribute]


def test_decoder_prompt_falls_back_for_unsupported_locale(
    policy: WhisperForConditionalGeneration,
) -> None:
    """Locales Whisper has no language token for fall back to detection."""
    config = policy.generation_config
    prompt = decoder_prompt(policy, torch.zeros(1, 80, 3000), "transcribe", ["ast"])
    lang_ids = set(config.lang_to_id.values())  # ty: ignore[unresolved-attribute]
    assert int(prompt[0, 1].item()) in lang_ids
