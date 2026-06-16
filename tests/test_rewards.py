"""Tests for the error-rate and shaping reward functions."""

import pytest

from whisper_rl.rewards import (
    character_error_rate,
    combined_reward,
    length_penalty,
    normalize,
    repetition_penalty,
    word_error_rate,
)


def test_normalize_lowercases_and_strips_punctuation() -> None:
    """Normalization removes casing and punctuation differences."""
    assert normalize("Hello, World!") == normalize("hello world")


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


def test_length_penalty_only_punishes_over_length() -> None:
    """Excess length is penalized; a shorter hypothesis is not."""
    assert length_penalty("abc", "abcabc") == pytest.approx(1.0)
    assert length_penalty("abc", "ab") == 0.0
    # Runaway length saturates at 1.0 so it cannot dominate the blend.
    assert length_penalty("abc", "abcabcabcabc") == 1.0


def test_repetition_penalty_flags_repeated_ngrams() -> None:
    """Repeated word bigrams score high; varied text scores zero."""
    assert repetition_penalty("ignored", "a b a b a b") == pytest.approx(0.6)
    assert repetition_penalty("ignored", "a b c d") == 0.0
    assert repetition_penalty("ignored", "hello") == 0.0


def test_combined_reward_perfect_transcription_is_zero() -> None:
    """A perfect, non-repetitive transcription incurs no penalty."""
    weights = {"wer": 1.0, "cer": 1.0, "length": 0.5, "repetition": 0.5}
    assert combined_reward("the quick brown fox", "the quick brown fox", weights) == 0.0


def test_combined_reward_blends_and_floors() -> None:
    """The reward is the negated weight-averaged penalty, floored."""
    # Only WER weighted -> reproduces the negated WER.
    assert combined_reward(
        "the quick brown fox", "the quick brown dog", {"wer": 1.0}
    ) == pytest.approx(-0.25)
    # A catastrophic hypothesis is clamped at the floor.
    assert combined_reward("a", "x y z w v", {"wer": 1.0}, floor=-1.0) == -1.0
