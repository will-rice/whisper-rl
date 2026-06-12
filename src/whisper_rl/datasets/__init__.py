"""Datasets and data loading for Whisper GRPO."""

import logging
import os
from typing import NamedTuple

import torch
from datasets import (
    Audio,
    get_dataset_config_names,
    interleave_datasets,
    load_dataset,
)
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor

from whisper_rl.config import Config

logger = logging.getLogger(__name__)

# (input_features, reference, common_voice_locale)
Example = tuple[torch.Tensor, str, str]


class Batch(NamedTuple):
    """A batch of audio features, references, and languages.

    Attributes:
        input_features: Log-mel features of shape ``(batch, n_mels, frames)``.
        references: Ground-truth transcriptions, one per audio clip.
        languages: Common Voice locale for each clip, used to bucket
            per-language metrics (the decoder language is auto-detected).
    """

    input_features: torch.Tensor
    references: list[str]
    languages: list[str]


class SpeechDataset(Dataset):
    """In-memory multilingual speech dataset materialized from a stream.

    Several Common Voice locale configs are streamed and interleaved; only the
    requested number of examples are pulled and their log-mel features
    pre-computed. Streaming avoids downloading whole corpora, keeping
    proof-of-concept runs light while remaining a map-style dataset.
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
        self.examples: list[Example] = []

        dataset = self._load_stream(split)
        if max_samples is not None:
            dataset = dataset.take(max_samples)

        for row in dataset:
            reference = str(row[config.text_column]).strip()
            if not reference:
                continue
            locale = str(row[config.locale_column])
            # ``Audio`` columns decode to torchcodec ``AudioDecoder`` objects;
            # ``cast_column`` already resamples to ``config.sample_rate``.
            samples = row[config.audio_column].get_all_samples()
            features = processor.feature_extractor(
                samples.data[0],
                sampling_rate=config.sample_rate,
                return_tensors="pt",
            )
            self.examples.append((features.input_features[0], reference, locale))

    def _resolve_languages(self, token: str | None) -> list[str]:
        """Return the locale configs to stream for this dataset."""
        if self.config.languages is not None:
            return self.config.languages
        return get_dataset_config_names(self.config.dataset_name, token=token)

    def _load_stream(self, split: str):  # noqa: ANN202
        """Stream and interleave every configured locale for ``split``."""
        token = os.environ.get("HUGGINGFACE_TOKEN")
        locales = self._resolve_languages(token)
        streams = []
        for locale in locales:
            try:
                stream = load_dataset(
                    self.config.dataset_name,
                    locale,
                    split=split,
                    streaming=True,
                    token=token,
                )
            except Exception as error:  # pragma: no cover - network/config issues
                logger.warning("Skipping locale %s: %s", locale, error)
                continue
            stream = stream.cast_column(
                self.config.audio_column,
                Audio(sampling_rate=self.config.sample_rate),
            )
            streams.append(stream)

        if not streams:
            raise RuntimeError(
                f"No usable locales for {self.config.dataset_name!r} ({split})."
            )
        if len(streams) == 1:
            return streams[0]
        return interleave_datasets(streams, stopping_strategy="all_exhausted")

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        """Return the ``(input_features, reference, language)`` example."""
        return self.examples[index]


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
        self.train_dataset: SpeechDataset | None = None
        self.eval_dataset: SpeechDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Materialize the train and eval datasets."""
        self.train_dataset = SpeechDataset(
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
        """Return the training dataloader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
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
