"""Tests for per-language error rate aggregation."""

from whisper_rl.metrics import LanguageErrorRate


def test_per_language_and_overall_corpus_rates() -> None:
    """WER and CER are reported per language and as a combined corpus figure."""
    metric = LanguageErrorRate()
    # English: perfect transcription -> 0 errors.
    metric.update("en", "the cat sat", "the cat sat")
    # German: one word substitution -> 1 error / 2 words = 0.5 WER.
    metric.update("de", "der hund", "der katze")

    results = metric.compute()
    assert results["wer"]["en"] == 0.0
    assert results["wer"]["de"] == 0.5
    # Corpus overall WER: 1 edit over 5 reference words = 0.2.
    assert abs(results["wer"]["overall"] - 0.2) < 1e-9
    # CER is reported alongside WER: zero for a perfect match, positive when
    # characters differ, and aggregated overall.
    assert results["cer"]["en"] == 0.0
    assert results["cer"]["de"] > 0.0
    assert "overall" in results["cer"]


def test_reset_clears_state() -> None:
    """Reset empties the accumulator."""
    metric = LanguageErrorRate()
    metric.update("en", "hello world", "hello world")
    metric.reset()
    assert metric.compute() == {}


def test_empty_reference_is_skipped() -> None:
    """References that normalize to empty do not break aggregation."""
    metric = LanguageErrorRate()
    metric.update("en", "...", "anything")
    metric.update("en", "real words here", "real words here")
    results = metric.compute()
    assert results["wer"]["en"] == 0.0
    assert results["cer"]["en"] == 0.0
