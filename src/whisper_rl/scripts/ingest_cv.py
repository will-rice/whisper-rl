"""Ingest Mozilla Common Voice archives into a streamable parquet index.

Common Voice from the Mozilla Data Collective ships as one ``.tar.gz`` per
locale that extracts to ``<locale>/clips/*.mp3`` plus ``<split>.tsv`` metadata
(``path``, ``sentence``, …). Rather than re-encode the audio, this builds a
small parquet index of ``(audio, sentence, locale)`` per split where ``audio``
is the absolute path to the on-disk clip; the training stream casts that column
to ``datasets.Audio`` and decodes the mp3s on the fly. Only locales Whisper has
a language token for are kept (others can never be transcribed correctly).

Point ``Config.dataset_name`` at the output directory to train on it.
"""

import argparse
import csv
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers.models.whisper.tokenization_whisper import LANGUAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common Voice split file -> the split name we expose (dev is our validation).
SPLITS = {"train": "train.tsv", "validation": "dev.tsv"}


def main() -> None:
    """Entry point for the ``ingest-cv`` console script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory of extracted Common Voice locales (<locale>/clips, tsv).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Where to write the <split>/<locale>.parquet index.",
    )
    args = parser.parse_args()

    supported = whisper_supported()
    locale_dirs = sorted(p for p in args.source_dir.iterdir() if p.is_dir())
    for locale_dir in tqdm(locale_dirs, desc="locales"):
        locale = locale_dir.name
        if locale.split("-")[0] not in supported:
            logger.info("Skipping non-Whisper locale %s", locale)
            continue
        for split, tsv_name in SPLITS.items():
            records = build_records(locale_dir, tsv_name, locale)
            if not records:
                continue
            out_path = args.output_dir / split / f"{locale}.parquet"
            write_parquet(records, out_path)
            logger.info("%s/%s: %d clips", locale, split, len(records))


def whisper_supported() -> set[str]:
    """Return the set of Whisper language codes (Common Voice locale bases)."""
    return set(LANGUAGES)


def build_records(locale_dir: Path, tsv_name: str, locale: str) -> list[dict]:
    """Build ``(audio, sentence, locale)`` records from a split's TSV.

    Args:
        locale_dir: The extracted ``<locale>`` directory.
        tsv_name: The split TSV file name, e.g. ``"train.tsv"``.
        locale: The Common Voice locale to stamp on every record.

    Returns:
        One record per row with a non-empty transcript and an existing clip.
    """
    tsv_path = locale_dir / tsv_name
    if not tsv_path.exists():
        return []
    clips = locale_dir / "clips"
    records: list[dict] = []
    with tsv_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sentence = (row.get("sentence") or "").strip()
            clip = clips / (row.get("path") or "")
            if not sentence or not clip.is_file():
                continue
            records.append(
                {"audio": str(clip.resolve()), "sentence": sentence, "locale": locale}
            )
    return records


def write_parquet(records: list[dict], out_path: Path) -> None:
    """Write records to a parquet file, creating parent directories.

    Args:
        records: ``(audio, sentence, locale)`` records.
        out_path: Destination parquet path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path)


if __name__ == "__main__":
    main()
