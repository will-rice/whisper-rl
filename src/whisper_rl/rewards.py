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


def length_penalty(reference: str, hypothesis: str) -> float:
    """Penalize a hypothesis that is longer than the reference.

    Counters the rambling / runaway-insertion drift that inflates WER: the
    excess character length relative to the reference, saturated at ``1.0`` so
    a single runaway completion cannot dominate the blended reward. A
    hypothesis no longer than the reference is not penalized.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model-produced transcription.

    Returns:
        A penalty in ``[0, 1]``.
    """
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0
    return min(1.0, max(0.0, (len(hyp) - len(ref)) / len(ref)))


def repetition_penalty(reference: str, hypothesis: str) -> float:
    """Penalize repeated word bigrams (Whisper's hallucination loops).

    Args:
        reference: Unused; kept for a uniform component signature.
        hypothesis: Model-produced transcription.

    Returns:
        The fraction of non-unique word bigrams in ``[0, 1]``; ``0.0`` when the
        hypothesis has fewer than two words.
    """
    del reference
    words = normalize(hypothesis).split()
    bigrams = list(zip(words, words[1:], strict=False))
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


REWARD_COMPONENTS = {
    "wer": word_error_rate,
    "cer": character_error_rate,
    "length": length_penalty,
    "repetition": repetition_penalty,
}


def combined_reward(
    reference: str,
    hypothesis: str,
    weights: dict[str, float],
    floor: float = -1.0,
) -> float:
    """Reward a hypothesis by the negated, weight-averaged penalty blend.

    Each enabled component (see :data:`REWARD_COMPONENTS`) returns a penalty
    where lower is better; the reward is the negated weighted mean, floored.
    Group-relative advantages normalize the overall scale, so the weights set
    only the relative emphasis between components.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model-produced transcription.
        weights: Per-component weights; only listed components contribute.
        floor: Lower bound so a single very bad hypothesis cannot dominate
            group-relative normalization.

    Returns:
        ``max(floor, -weighted_mean(penalties))``.
    """
    total = sum(weights.values())
    penalty = sum(
        weight * REWARD_COMPONENTS[name](reference, hypothesis)
        for name, weight in weights.items()
    )
    return max(floor, -penalty / total)
