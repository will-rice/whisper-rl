"""Whisper model construction for GRPO."""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_rl.config import Config


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


def decoder_prompt(
    model: WhisperForConditionalGeneration,
    input_features: torch.Tensor,
    task: str,
) -> torch.Tensor:
    """Reconstruct the decoder prompt that ``generate`` conditioned on.

    Whisper forces ``[<|startoftranscript|>, <|lang|>, <|task|>,
    <|notimestamps|>]`` before the transcription, but transformers >= 5 strips
    it from the returned sequences. Language detection is a deterministic
    argmax over the language logits, so re-running it reproduces exactly the
    prompt used during generation.

    Args:
        model: The Whisper model that generated (or will generate) with
            language auto-detection.
        input_features: Log-mel features of shape ``(batch, n_mels, frames)``.
        task: The Whisper task, e.g. ``"transcribe"``.

    Returns:
        Prompt token ids of shape ``(batch, 4)``.
    """
    generation_config = model.generation_config
    with torch.no_grad():
        lang_ids = model.detect_language(input_features=input_features)  # ty: ignore[invalid-argument-type]
    lang = lang_ids.unsqueeze(1)
    start = torch.full_like(lang, generation_config.decoder_start_token_id)
    task_id = torch.full_like(lang, generation_config.task_to_id[task])  # ty: ignore[unresolved-attribute]
    no_timestamps = torch.full_like(
        lang,
        generation_config.no_timestamps_token_id,  # ty: ignore[unresolved-attribute]
    )
    return torch.cat([start, lang, task_id, no_timestamps], dim=1)


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
