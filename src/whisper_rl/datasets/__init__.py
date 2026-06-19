"""Datasets and data loading for Whisper GRPO."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import torch
from datasets import (
    Audio,
    get_dataset_config_names,
    interleave_datasets,
    load_dataset,
)
from datasets import (
    IterableDataset as HFIterableDataset,
)
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import WhisperProcessor

from whisper_rl.config import Config

logger = logging.getLogger(__name__)

# Examples buffered for approximate shuffling of the training stream.
SHUFFLE_BUFFER_SIZE = 1000

# (input_features, reference, common_voice_locale)
Example = tuple[torch.Tensor, str, str]


class Batch(NamedTuple):
    """A batch of audio features, references, and languages.

    Attributes:
        input_features: Log-mel features of shape ``(batch, n_mels, frames)``.
        references: Ground-truth transcriptions, one per audio clip.
        languages: Common Voice locale for each clip; pins the decoder
            language and buckets per-language metrics.
    """

    input_features: torch.Tensor
    references: list[str]
    languages: list[str]


def load_stream(config: Config, split: str) -> HFIterableDataset:
    """Stream and interleave every configured locale for ``split``.

    Args:
        config: Project configuration.
        split: Dataset split to stream, e.g. ``"train"``.

    Returns:
        A Hugging Face streaming dataset with audio cast to
        ``config.sample_rate``.
    """
    if Path(config.dataset_name).is_dir():
        return _load_local_stream(config, split)

    token = os.environ.get("HUGGINGFACE_TOKEN")
    locales = config.languages
    if locales is None:
        locales = get_dataset_config_names(config.dataset_name, token=token)
    streams = []
    for locale in locales:
        try:
            stream = load_dataset(
                config.dataset_name,
                locale,
                split=split,
                streaming=True,
                token=token,
            )
        except Exception as error:  # pragma: no cover - network/config issues
            logger.warning("Skipping locale %s: %s", locale, error)
            continue
        stream = stream.cast_column(
            config.audio_column,
            Audio(sampling_rate=config.sample_rate),
        )
        streams.append(stream)

    if not streams:
        raise RuntimeError(f"No usable locales for {config.dataset_name!r} ({split}).")
    if len(streams) == 1:
        return streams[0]
    return interleave_datasets(streams, stopping_strategy="all_exhausted")


def _load_local_stream(config: Config, split: str) -> HFIterableDataset:
    """Stream a local ``ingest-cv`` parquet index for ``split``.

    The index rows hold the clip's on-disk path in the audio column; casting to
    :class:`~datasets.Audio` decodes the mp3s on the fly, and the per-clip
    ``locale`` column drives per-language metrics. ``config.languages`` selects
    which per-locale parquet files to stream (all of them when ``None``).

    Args:
        config: Project configuration; ``dataset_name`` is the index directory.
        split: Dataset split to stream.

    Returns:
        A streaming dataset with audio cast to ``config.sample_rate``.
    """
    base = Path(config.dataset_name) / split
    if config.languages:
        data_files: str | list[str] = [
            str(base / f"{locale}.parquet") for locale in config.languages
        ]
    else:
        data_files = str(base / "*.parquet")
    stream = load_dataset(
        "parquet", data_files=data_files, split="train", streaming=True
    )
    return stream.cast_column(
        config.audio_column, Audio(sampling_rate=config.sample_rate)
    )


def prepare_example(
    row: dict, config: Config, processor: WhisperProcessor
) -> Example | None:
    """Turn a streamed row into an :data:`Example`, or ``None`` if unusable.

    Args:
        row: A streamed dataset row.
        config: Project configuration.
        processor: Whisper processor used for feature extraction.

    Returns:
        The ``(input_features, reference, locale)`` example, or ``None`` when
        the reference transcript is empty.
    """
    reference = str(row[config.text_column]).strip()
    if not reference:
        return None
    locale = str(row[config.locale_column])
    # ``Audio`` columns decode to torchcodec ``AudioDecoder`` objects;
    # ``cast_column`` already resamples to ``config.sample_rate``.
    samples = row[config.audio_column].get_all_samples()
    features = processor.feature_extractor(
        samples.data[0],
        sampling_rate=config.sample_rate,
        return_tensors="pt",
    )
    return (features.input_features[0], reference, locale)


class SpeechDataset(Dataset):
    """In-memory speech dataset materialized from a stream.

    Used for evaluation: the slice is pulled once and its log-mel features
    pre-computed, so repeated validation epochs cost no network traffic and
    score the exact same clips.
    """

    def __init__(
        self,
        config: Config,
        processor: WhisperProcessor,
        split: str,
        max_samples: int | None,
    ) -> None:
        dataset = load_stream(config, split)
        if max_samples is not None:
            dataset = dataset.take(max_samples)
        self.examples: list[Example] = [
            example
            for row in dataset
            if (example := prepare_example(row, config, processor)) is not None
        ]

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        """Return the ``(input_features, reference, language)`` example."""
        return self.examples[index]


class StreamingSpeechDataset(IterableDataset):
    """Speech dataset that decodes and featurizes on the fly while training.

    Wraps the Hugging Face stream (which shards itself across DataLoader
    workers) with approximate shuffling, so arbitrarily large splits train
    with flat memory and no startup materialization.
    """

    def __init__(
        self,
        config: Config,
        processor: WhisperProcessor,
        split: str,
        max_samples: int | None,
    ) -> None:
        self.config = config
        self.processor = processor
        stream = load_stream(config, split)
        stream = stream.shuffle(seed=config.seed, buffer_size=SHUFFLE_BUFFER_SIZE)
        if max_samples is not None:
            stream = stream.take(max_samples)
        self.stream = stream

    def __iter__(self) -> Iterator[Example]:
        """Yield prepared examples from the shuffled stream."""
        for row in self.stream:
            example = prepare_example(row, self.config, self.processor)
            if example is not None:
                yield example


def collate(examples: list[Example]) -> Batch:
    """Collate examples into a :class:`Batch`.

    Args:
        examples: ``(input_features, reference, locale)`` items.

    Returns:
        A batched :class:`Batch`.
    """
    features, references, locales = zip(*examples, strict=True)
    return Batch(
        input_features=torch.stack(features),
        references=list(references),
        languages=list(locales),
    )


class SpeechDataModule(LightningDataModule):
    """Lightning data module wrapping :class:`SpeechDataset`."""

    def __init__(self, config: Config, processor: WhisperProcessor) -> None:
        super().__init__()
        self.config = config
        self.processor = processor
        self.train_dataset: StreamingSpeechDataset | None = None
        self.eval_dataset: SpeechDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create the streamed train dataset and materialize the eval set."""
        self.train_dataset = StreamingSpeechDataset(
            self.config,
            self.processor,
            self.config.train_split,
            self.config.max_train_samples,
        )
        self.eval_dataset = SpeechDataset(
            self.config,
            self.processor,
            self.config.eval_split,
            self.config.max_eval_samples,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader (shuffled by the stream itself)."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        assert self.eval_dataset is not None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate,
        )
