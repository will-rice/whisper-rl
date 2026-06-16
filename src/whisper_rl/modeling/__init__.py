"""Whisper model construction for GRPO."""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_rl.config import Config


def build_processor(config: Config) -> WhisperProcessor:
    """Load the (multilingual) Whisper processor for ``config.base_model``.

    The decoder language is pinned per clip from its Common Voice locale at
    generation time (see :func:`decoder_prompt`), so one model trains across
    many languages without relying on Whisper's language auto-detection.

    Args:
        config: Project configuration.

    Returns:
        The loaded :class:`~transformers.WhisperProcessor`.
    """
    return WhisperProcessor.from_pretrained(config.base_model)


def build_policy(config: Config) -> WhisperForConditionalGeneration:
    """Load a trainable Whisper policy model.

    Any checkpoint-baked ``forced_decoder_ids`` are cleared so the decoder
    prompt we build (see :func:`decoder_prompt`) is what conditions decoding.

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
    locales: list[str],
) -> torch.Tensor:
    """Build the Whisper decoder prompt, pinning each clip's known language.

    Whisper conditions on ``[<|sot|>, <|lang|>, <|task|>, <|notimestamps|>]``.
    Auto-detecting ``<|lang|>`` mislabels lower-resource clips (e.g. Urdu as
    Hindi), so the clip transcribes in the wrong language and WER explodes.
    Common Voice gives us the locale, so the language token is taken from it
    (region stripped, e.g. ``sv-SE`` -> ``sv``); locales Whisper has no token
    for fall back to its detected language. The same prompt is fed to
    ``generate`` (as ``decoder_input_ids``) and to the log-prob forward, so the
    two never disagree.

    Args:
        model: The Whisper model.
        input_features: Log-mel features of shape ``(batch, n_mels, frames)``.
        task: The Whisper task, e.g. ``"transcribe"``.
        locales: Common Voice locale per clip, used to pin the language.

    Returns:
        Prompt token ids of shape ``(batch, 4)``.
    """
    generation_config = model.generation_config
    lang_to_id = generation_config.lang_to_id  # ty: ignore[unresolved-attribute]
    tokens = [lang_to_id.get(f"<|{loc.split('-')[0]}|>") for loc in locales]

    if any(token is None for token in tokens):
        with torch.no_grad():
            detected = model.detect_language(input_features=input_features)  # ty: ignore[invalid-argument-type]
    else:
        detected = torch.zeros(len(tokens), dtype=torch.long)
    lang_ids = [
        int(detected[i]) if token is None else token for i, token in enumerate(tokens)
    ]
    lang = torch.tensor(lang_ids, device=input_features.device).unsqueeze(1)
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
