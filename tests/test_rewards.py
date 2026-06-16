"""Tests for the error-rate reward functions."""

import pytest

from whisper_rl.rewards import (
    character_error_rate,
    error_reward,
    normalize,
    word_error_rate,
)


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


def test_character_error_rate_single_substitution() -> None:
    """One wrong character out of three is a CER of 1/3."""
    assert character_error_rate("abc", "abx") == pytest.approx(1 / 3)


def test_character_error_rate_empty_reference() -> None:
    """An empty reference scores zero for empty and one otherwise."""
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "x") == 1.0


def test_error_reward_negates_and_floors() -> None:
    """Reward is the negated error rate, floored to avoid dominating outliers."""
    assert error_reward("the quick brown fox", "the quick brown fox", "wer") == 0.0
    assert error_reward("the quick brown fox", "the quick brown dog", "wer") == -0.25
    assert error_reward("a", "x y z w v", "wer", floor=-1.0) == -1.0


def test_cer_reward_gives_partial_credit_where_wer_floors() -> None:
    """CER yields graded reward where word-level WER would pin at the floor."""
    # Every word wrong -> WER reward floors at -1, but half the characters
    # are right, so the CER reward carries gradient-bearing signal.
    assert error_reward("abcd", "abxy", "wer", floor=-1.0) == -1.0
    assert error_reward("abcd", "abxy", "cer", floor=-1.0) == pytest.approx(-0.5)
