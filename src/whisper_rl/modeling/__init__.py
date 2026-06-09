"""Whisper model construction for GRPO."""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_rl.config import Config


def build_processor(config: Config) -> WhisperProcessor:
    """Load the Whisper processor for ``config.base_model``.

    Args:
        config: Project configuration.

    Returns:
        The loaded :class:`~transformers.WhisperProcessor`.
    """
    return WhisperProcessor.from_pretrained(
        config.base_model, language=config.language, task=config.task
    )


def build_policy(config: Config) -> WhisperForConditionalGeneration:
    """Load a trainable Whisper policy model.

    The generation config is set up for the configured language/task and
    caching is disabled for gradient compatibility during the loss pass.

    Args:
        config: Project configuration.

    Returns:
        A trainable Whisper model.
    """
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    # ``language``/``task`` are dynamic GenerationConfig attributes; forced
    # decoder ids are recomputed by ``generate`` from language/task.
    gen = model.generation_config
    gen.language = config.language  # ty: ignore[unresolved-attribute]
    gen.task = config.task  # ty: ignore[unresolved-attribute]
    gen.forced_decoder_ids = None  # ty: ignore[unresolved-attribute]
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


def decoder_prompt_ids(
    processor: WhisperProcessor, model: WhisperForConditionalGeneration
) -> list[int]:
    """Return the forced decoder prompt that precedes generated tokens.

    The prompt is ``[decoder_start, <|lang|>, <|task|>, <|notimestamps|>]`` and
    its length is needed to separate the prompt from sampled completion tokens.

    Args:
        processor: The Whisper processor.
        model: The Whisper model (for its decoder start token id).

    Returns:
        The list of forced decoder prompt token ids.
    """
    start = int(model.config.decoder_start_token_id)
    forced = processor.get_decoder_prompt_ids(no_timestamps=True)
    return [start] + [int(token_id) for _, token_id in forced]


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
