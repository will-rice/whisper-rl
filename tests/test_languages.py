"""Tests for Common Voice locale to Whisper language mapping."""

from whisper_rl.languages import (
    WHISPER_LANGUAGES,
    locale_to_whisper,
    supported_locales,
)


def test_plain_code_maps_to_itself() -> None:
    """A supported plain locale maps to the same Whisper code."""
    assert locale_to_whisper("en") == "en"
    assert locale_to_whisper("de") == "de"


def test_region_suffix_is_stripped() -> None:
    """Region-suffixed locales drop the region before mapping."""
    assert locale_to_whisper("zh-CN") == "zh"
    assert locale_to_whisper("sv-SE") == "sv"


def test_case_insensitive() -> None:
    """Mapping is case-insensitive."""
    assert locale_to_whisper("EN") == "en"


def test_unsupported_locale_returns_none() -> None:
    """Locales Whisper cannot transcribe return None."""
    assert locale_to_whisper("zz") is None
    assert locale_to_whisper("xx-YY") is None


def test_supported_locales_filters_and_preserves_order() -> None:
    """supported_locales keeps supported codes in order."""
    assert supported_locales(["en", "zz", "de"]) == ["en", "de"]


def test_whisper_languages_is_populated() -> None:
    """The Whisper language set is non-empty and includes English."""
    assert "en" in WHISPER_LANGUAGES
    assert len(WHISPER_LANGUAGES) > 50
