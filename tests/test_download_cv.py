"""Tests for the Common Voice download catalog parser."""

from whisper_rl.scripts.download_cv import parse_catalog

_CARD = (
    '<a class="block h-full" href="/datasets/{id}"><div data-slot="card">'
    "<span>Common Voice</span> Common Voice Scripted Speech 26.0 - {name}"
    '<span class="truncate">A collection of read speech recordings in {name}.</span>'
    "<span>License</span>: CC0-1.0 <span>Locale</span>: {locale} "
    "<span>Task</span>: ASR <span>Format</span>: MP3 <span>Size</span>: {size}</div></a>"
)


def _page(*cards: str) -> str:
    body = "".join(cards)
    return f"<html><body><nav></nav>{body}</body></html>"


def test_parse_catalog_extracts_id_name_locale() -> None:
    """Each dataset card yields its id, language name, and locale."""
    html = _page(
        _CARD.format(
            id="cmtest00000000000000en", name="English", locale="en", size="88.1 GB"
        ),
        _CARD.format(
            id="cmtest00000000000ast", name="Asturian", locale="ast", size="10.2 MB"
        ),
    )
    cards = parse_catalog(html, "Scripted Speech 26.0")
    by_locale = {c["locale"]: c for c in cards}
    assert by_locale["en"]["id"] == "cmtest00000000000000en"
    assert by_locale["en"]["name"] == "Scripted Speech 26.0 - English"
    assert set(by_locale) == {"en", "ast"}


def test_parse_catalog_keeps_region_codes() -> None:
    """Region-coded locales like sv-SE are captured in full."""
    html = _page(
        _CARD.format(
            id="cmtest0000000000000sv", name="Swedish", locale="sv-SE", size="1 GB"
        )
    )
    assert parse_catalog(html, "Scripted Speech 26.0")[0]["locale"] == "sv-SE"


def test_parse_catalog_ignores_other_releases() -> None:
    """Cards from a different release are not returned."""
    html = _page(
        _CARD.format(
            id="cmtest0000000000000xx", name="Bodo", locale="brx", size="1 MB"
        ).replace("Scripted Speech 26.0", "Spontaneous Speech 4.0")
    )
    assert parse_catalog(html, "Scripted Speech 26.0") == []
