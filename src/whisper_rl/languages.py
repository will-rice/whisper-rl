"""Whisper language codes and Common Voice locale mapping.

Common Voice ships one config per locale (e.g. ``en``, ``de``, ``zh-CN``).
Whisper identifies languages by short codes (e.g. ``en``, ``de``, ``zh``). This
module maps the former to the latter and filters out locales the loaded
Whisper checkpoint cannot transcribe.
"""

from collections.abc import Iterable

from transformers.models.whisper.tokenization_whisper import LANGUAGES

# Language codes the (multilingual) Whisper tokenizer knows about.
WHISPER_LANGUAGES: frozenset[str] = frozenset(LANGUAGES)

# Common Voice locale prefixes whose base code differs from Whisper's.
_LOCALE_ALIASES: dict[str, str] = {"nb": "no", "iw": "he", "jw": "jv"}


def locale_to_whisper(locale: str) -> str | None:
    """Map a Common Voice locale to a Whisper language code.

    Region suffixes are dropped (``zh-CN`` -> ``zh``, ``sv-SE`` -> ``sv``) and a
    few aliases are normalized. Locales Whisper does not support return ``None``.

    Args:
        locale: A Common Voice locale string.

    Returns:
        The Whisper language code, or ``None`` if unsupported.
    """
    base = locale.strip().lower().split("-")[0]
    base = _LOCALE_ALIASES.get(base, base)
    return base if base in WHISPER_LANGUAGES else None


def supported_locales(locales: Iterable[str]) -> list[str]:
    """Filter locales to those the loaded Whisper checkpoint supports.

    Order is preserved and duplicates that map to the same Whisper code are
    kept (they are distinct Common Voice configs).

    Args:
        locales: Candidate Common Voice locale strings.

    Returns:
        The subset of ``locales`` that Whisper can transcribe.
    """
    return [locale for locale in locales if locale_to_whisper(locale) is not None]
