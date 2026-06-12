"""Tests for the WER-based reward functions."""

from whisper_rl.rewards import normalize, wer_reward, word_error_rate


def test_normalize_lowercases_and_strips_punctuation() -> None:
    """Normalization removes casing and punctuation differences."""
    assert normalize("Hello, World!") == normalize("hello world")


def test_word_error_rate_perfect_match_is_zero() -> None:
    """A perfect (post-normalization) transcription scores zero WER."""
    assert word_error_rate("The quick brown fox", "the quick brown fox!") == 0.0


def test_word_error_rate_single_substitution() -> None:
    """One wrong word out of four is a WER of 0.25."""
    assert word_error_rate("the quick brown fox", "the quick brown dog") == 0.25


def test_word_error_rate_empty_reference() -> None:
    """An empty reference scores zero for empty and one otherwise."""
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "something") == 1.0


def test_wer_reward_is_negated_wer_and_floored() -> None:
    """Reward is the negated WER, floored to avoid dominating outliers."""
    assert wer_reward("the quick brown fox", "the quick brown fox") == 0.0
    assert wer_reward("the quick brown fox", "the quick brown dog") == -0.25
    # A wildly wrong hypothesis is clamped at the floor.
    assert wer_reward("a", "x y z w v", floor=-1.0) == -1.0
