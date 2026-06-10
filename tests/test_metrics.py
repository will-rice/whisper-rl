"""Tests for per-language word error rate aggregation."""

from whisper_rl.metrics import LanguageWER


def test_per_language_and_overall_corpus_wer() -> None:
    """WER is reported per language and as a combined corpus figure."""
    metric = LanguageWER()
    # English: perfect transcription -> 0 errors / 3 words.
    metric.update("en", "the cat sat", "the cat sat")
    # German: one substitution -> 1 error / 2 words = 0.5.
    metric.update("de", "der hund", "der katze")

    results = metric.compute()
    assert results["en"] == 0.0
    assert results["de"] == 0.5
    # Corpus overall: 1 edit over 5 reference words = 0.2.
    assert abs(results["overall"] - 0.2) < 1e-9


def test_reset_clears_state() -> None:
    """Reset empties the accumulator."""
    metric = LanguageWER()
    metric.update("en", "hello world", "hello world")
    metric.reset()
    assert metric.compute() == {}


def test_empty_reference_is_skipped() -> None:
    """References that normalize to empty do not break aggregation."""
    metric = LanguageWER()
    metric.update("en", "...", "anything")
    metric.update("en", "real words here", "real words here")
    results = metric.compute()
    assert results["en"] == 0.0
