"""Tests for Whisper prompt reconstruction against the installed transformers."""

import pytest
import torch

from whisper_rl.config import Config
from whisper_rl.modeling import build_policy, decoder_prompt


@pytest.fixture(scope="module")
def policy():
    """The real (tiny) Whisper policy; pins the generate() output contract."""
    return build_policy(Config())


def test_generate_strips_decoder_prompt(policy) -> None:
    """Transformers >= 5 returns only sampled tokens, with no forced prompt.

    The GRPO loss slices completions relative to the prompt, so this contract
    must hold; if a transformers upgrade changes it, this test fails first.
    """
    features = torch.zeros(2, 80, 3000)
    sequences = policy.generate(
        input_features=features, task="transcribe", max_new_tokens=4
    )
    start = policy.generation_config.decoder_start_token_id
    assert (sequences != start).all()


def test_decoder_prompt_matches_whisper_forced_tokens(policy) -> None:
    """The reconstructed prompt is the 4-token prefix generate() conditions on."""
    config = policy.generation_config
    features = torch.zeros(2, 80, 3000)
    prompt = decoder_prompt(policy, features, task="transcribe")

    assert prompt.shape == (2, 4)
    assert (prompt[:, 0] == config.decoder_start_token_id).all()
    lang_ids = set(config.lang_to_id.values())
    assert all(int(token) in lang_ids for token in prompt[:, 1])
    assert (prompt[:, 2] == config.task_to_id["transcribe"]).all()
    assert (prompt[:, 3] == config.no_timestamps_token_id).all()
