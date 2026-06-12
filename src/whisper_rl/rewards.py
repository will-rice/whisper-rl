"""Reward functions for Whisper GRPO.

The reward signal is derived from the word error rate (WER) between a sampled
transcription and the ground-truth reference. Lower WER is better, so the
reward is the negated WER, optionally floored so that catastrophic
hypotheses do not dominate the per-group advantage normalization.
"""

import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

_normalizer = BasicTextNormalizer()


def normalize(text: str) -> str:
    """Normalize text for fair WER comparison.

    Applies Whisper's basic text normalizer (lowercasing, punctuation and
    symbol removal, whitespace collapsing) so scoring is not dominated by
    casing or punctuation differences.

    Args:
        text: Raw text to normalize.

    Returns:
        The normalized text.
    """
    return _normalizer(text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute normalized word error rate between a reference and hypothesis.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model-produced transcription.

    Returns:
        The WER as a float. A reference that normalizes to the empty string
        returns ``0.0`` for an empty hypothesis and ``1.0`` otherwise.
    """
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def wer_reward(reference: str, hypothesis: str, floor: float = -1.0) -> float:
    """Reward a hypothesis by its negated word error rate.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model-produced transcription.
        floor: Lower bound applied to the reward so a single very bad
            hypothesis cannot dominate group-relative normalization.

    Returns:
        ``max(floor, -WER)``: ``0.0`` for a perfect transcription and more
        negative as errors increase.
    """
    return max(floor, -word_error_rate(reference, hypothesis))
