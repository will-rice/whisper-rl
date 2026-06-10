"""Whisper model construction for GRPO."""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_rl.config import Config

# Whisper forces a 4-token decoder prompt before the transcription:
# ``[<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]``.
# The language token varies per clip but the length is constant, which is all
# we need to separate the prompt from sampled completion tokens.
PROMPT_LEN = 4


def build_processor(config: Config) -> WhisperProcessor:
    """Load the (multilingual) Whisper processor for ``config.base_model``.

    No language is fixed here: Whisper auto-detects each clip's language at
    generation time so a single model can be trained across many languages.

    Args:
        config: Project configuration.

    Returns:
        The loaded :class:`~transformers.WhisperProcessor`.
    """
    return WhisperProcessor.from_pretrained(config.base_model)


def build_policy(config: Config) -> WhisperForConditionalGeneration:
    """Load a trainable Whisper policy model.

    Any checkpoint-baked ``forced_decoder_ids`` are cleared so Whisper's
    per-clip language auto-detection is what conditions decoding.

    Args:
        config: Project configuration.

    Returns:
        A trainable Whisper model.
    """
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    model.generation_config.forced_decoder_ids = None  # ty: ignore[unresolved-attribute]
    model.config.forced_decoder_ids = None
    return model


def build_reference(config: Config) -> WhisperForConditionalGeneration:
    """Load a frozen reference Whisper model for the KL penalty.

    Args:
        config: Project configuration.

    Returns:
        A Whisper model in eval mode with gradients disabled.
    """
    model = build_policy(config)
    model.eval()
    model.requires_grad_(False)
    return model


def repeat_features(input_features: torch.Tensor, num_generations: int) -> torch.Tensor:
    """Repeat each audio clip's features ``num_generations`` times.

    The expansion is interleaved so the ``num_generations`` completions of a
    clip stay contiguous, matching :func:`whisper_rl.grpo.group_advantages`.

    Args:
        input_features: Features of shape ``(batch, n_mels, frames)``.
        num_generations: Completions per clip.

    Returns:
        Features of shape ``(batch * num_generations, n_mels, frames)``.
    """
    return input_features.repeat_interleave(num_generations, dim=0)
