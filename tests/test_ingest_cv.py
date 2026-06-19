"""Tests for the Common Voice ingest helpers."""

from pathlib import Path

from whisper_rl.scripts.ingest_cv import build_records, whisper_supported


def _write_archive(root: Path) -> Path:
    """Create a minimal Common Voice locale layout and return its directory."""
    locale_dir = root / "en"
    clips = locale_dir / "clips"
    clips.mkdir(parents=True)
    (clips / "a.mp3").write_bytes(b"audio-a")
    (clips / "b.mp3").write_bytes(b"audio-b")
    # path<TAB>sentence; one good row, one empty sentence, one missing clip.
    (locale_dir / "train.tsv").write_text(
        "path\tsentence\na.mp3\tHello world\nb.mp3\t   \nmissing.mp3\tUnused\n"
    )
    return locale_dir


def test_build_records_keeps_usable_rows(tmp_path: Path) -> None:
    """Rows with a transcript and an existing clip become records."""
    locale_dir = _write_archive(tmp_path)
    records = build_records(locale_dir, "train.tsv", "en")

    assert len(records) == 1
    record = records[0]
    assert record["sentence"] == "Hello world"
    assert record["locale"] == "en"
    assert record["audio"].endswith("clips/a.mp3")
    assert Path(record["audio"]).is_absolute()


def test_build_records_missing_tsv_is_empty(tmp_path: Path) -> None:
    """A split with no TSV yields no records rather than erroring."""
    locale_dir = _write_archive(tmp_path)
    assert build_records(locale_dir, "dev.tsv", "en") == []


def test_whisper_supported_includes_major_languages_and_excludes_non_whisper() -> None:
    """Region codes are stripped; non-Whisper Common Voice locales are excluded."""
    supported = whisper_supported()
    assert "en" in supported
    assert "ja" in supported
    assert "sv" in supported  # base of sv-SE
    assert "ast" not in supported  # Asturian is not a Whisper language
