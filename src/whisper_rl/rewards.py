"""Reward functions for Whisper GRPO.

The reward signal is derived from the error rate between a sampled
transcription and the ground-truth reference. Lower error is better, so the
reward is the negated error rate, optionally floored so that catastrophic
hypotheses do not dominate the per-group advantage normalization. Word error
rate (WER) and character error rate (CER) are both supported; CER is finer
grained (partial credit per character), which keeps the reward off the floor
on hard clips and is the sensible choice for languages without word spaces.
"""

import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

_normalizer = BasicTextNormalizer()


def normalize(text: str) -> str:
    """Normalize text for fair error-rate comparison.

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


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute normalized character error rate between reference and hypothesis.

    CER is well defined for languages without word spaces (e.g. Japanese,
    Chinese, Thai) where word-level WER is meaningless, and is finer grained
    than WER everywhere.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model-produced transcription.

    Returns:
        The CER as a float, with the same empty-reference convention as
        :func:`word_error_rate`.
    """
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


ERROR_RATES = {"wer": word_error_rate, "cer": character_error_rate}


def error_reward(
    reference: str, hypothesis: str, metric: str, floor: float = -1.0
) -> float:
    """Reward a hypothesis by its negated error rate.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model-produced transcription.
        metric: Which error rate to use, ``"wer"`` or ``"cer"``.
        floor: Lower bound applied to the reward so a single very bad
            hypothesis cannot dominate group-relative normalization.

    Returns:
        ``max(floor, -error_rate)``: ``0.0`` for a perfect transcription and
        more negative as errors increase.
    """
    return max(floor, -ERROR_RATES[metric](reference, hypothesis))
