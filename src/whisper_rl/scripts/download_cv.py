"""Download curated Common Voice locales from the Mozilla Data Collective.

MDC distributes Common Voice as one dataset per locale, gated behind an API key
(``Authorization: Bearer``) and per-dataset terms accepted through the website.
This enumerates the catalog for a release, keeps the locales Whisper supports,
and downloads each archive to ``output_dir`` via the documented presigned-URL
endpoint (resumable). Terms must already be accepted for a dataset, else the API
returns 403; downloads are rate limited to 30 per day per organization.

Set ``MDC_API_KEY`` in the environment or ``.env``. Feed the extracted archives
to ``ingest-cv`` next.
"""

import argparse
import logging
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from whisper_rl.scripts.ingest_cv import whisper_supported

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://mozilladatacollective.com"
ID_PATTERN = re.compile(r"^[a-z0-9]{20,}$")
# 30 downloads/day per organization; default a run to that ceiling.
DAILY_LIMIT = 30


class RateLimitError(Exception):
    """Raised when the API's daily download limit is reached."""


def main() -> None:
    """Entry point for the ``download-cv`` console script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory (e.g. /data/common_voice_26).",
    )
    parser.add_argument("--release", default="Scripted Speech 26.0", type=str)
    parser.add_argument(
        "--max", default=DAILY_LIMIT, type=int, help="Max downloads this run."
    )
    parser.add_argument(
        "--locales",
        default="",
        type=str,
        help="Comma-separated locales to restrict to (default: all supported).",
    )
    parser.add_argument("--list_only", action="store_true", help="Only print the plan.")
    args = parser.parse_args()
    load_dotenv()

    supported = whisper_supported()
    only = {loc for loc in args.locales.split(",") if loc}
    datasets = [
        card
        for card in enumerate_release(args.release)
        if card["locale"].split("-")[0] in supported
        and (not only or card["locale"] in only)
    ]
    logger.info("%d Whisper-supported locales for %s", len(datasets), args.release)
    if args.list_only:
        for card in datasets:
            logger.info("%s  %s  %s", card["locale"], card["id"], card["name"])
        return

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {os.environ['MDC_API_KEY']}"
    archives = args.output_dir / "archives"
    downloaded = 0
    for card in datasets:
        if downloaded >= args.max:
            logger.info("Reached --max=%d for this run; rerun to continue.", args.max)
            break
        try:
            info = request_download(session, card["id"])
        except RateLimitError:
            logger.error("Daily download limit hit; rerun after the UTC reset.")
            break
        if info is None:
            continue
        if stream_download(info["downloadUrl"], archives / info["filename"], info):
            downloaded += 1


def enumerate_release(release: str) -> list[dict]:
    """Scrape the catalog for every dataset card of ``release``.

    Args:
        release: The release name, e.g. ``"Scripted Speech 26.0"``.

    Returns:
        ``{"id", "name", "locale"}`` for each matching dataset.
    """
    found: list[dict] = []
    seen: set[str] = set()
    for page in range(1, 60):
        html = requests.get(
            f"{BASE_URL}/datasets",
            params={"q": f"Common Voice {release}", "page": page},
            timeout=60,
        ).text
        fresh = [c for c in parse_catalog(html, release) if c["id"] not in seen]
        if not fresh:
            break
        for card in fresh:
            seen.add(card["id"])
            found.append(card)
    return found


def parse_catalog(html: str, release: str) -> list[dict]:
    """Parse dataset cards out of a catalog search page.

    Args:
        html: Raw catalog HTML.
        release: Release name a card's title must contain.

    Returns:
        ``{"id", "name", "locale"}`` for each card of ``release``.
    """
    cards: list[dict] = []
    for chunk in html.split('"/datasets/')[1:]:
        dataset_id = chunk[: chunk.find('"')]
        if not ID_PATTERN.match(dataset_id):
            continue
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", chunk))
        name = re.search(
            rf"({re.escape(release)} - .+?) (?:A collection|License)", text
        )
        locale = re.search(r"Locale[:\s]+([A-Za-z][A-Za-z0-9-]*)", text)
        if not name or not locale:
            continue
        cards.append(
            {"id": dataset_id, "name": name.group(1).strip(), "locale": locale.group(1)}
        )
    return cards


def request_download(session: requests.Session, dataset_id: str) -> dict | None:
    """Ask the API for a presigned download URL.

    Args:
        session: Authenticated session.
        dataset_id: The dataset to download.

    Returns:
        The JSON payload, or ``None`` if the dataset's terms have not been
        accepted (403).

    Raises:
        RateLimitError: If the API returns 429 (daily limit reached).
    """
    response = session.post(
        f"{BASE_URL}/api/datasets/{dataset_id}/download", timeout=60
    )
    if response.status_code == 403:
        logger.warning("Terms not accepted for %s — accept on the website.", dataset_id)
        return None
    if response.status_code == 429:
        raise RateLimitError
    response.raise_for_status()
    return response.json()


def stream_download(url: str, dest: Path, info: dict) -> bool:
    """Stream a presigned URL to ``dest``, resuming a partial file.

    Args:
        url: The presigned download URL (no auth needed).
        dest: Destination path.
        info: The download payload (for the expected size).

    Returns:
        ``True`` if the file is complete after this call.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = int(info.get("sizeBytes") or 0)
    have = dest.stat().st_size if dest.exists() else 0
    if expected and have >= expected:
        logger.info("Already complete: %s", dest.name)
        return True
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        with (
            dest.open("ab" if have else "wb") as handle,
            tqdm(
                total=expected or None,
                initial=have,
                unit="B",
                unit_scale=True,
                desc=dest.name[:30],
            ) as bar,
        ):
            for block in response.iter_content(chunk_size=1 << 20):
                handle.write(block)
                bar.update(len(block))
    return not expected or dest.stat().st_size >= expected


if __name__ == "__main__":
    main()
