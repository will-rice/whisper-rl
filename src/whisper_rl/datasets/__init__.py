"""Datasets and data loading for Whisper GRPO."""

import os
from typing import NamedTuple

import torch
from datasets import Audio, load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor

from whisper_rl.config import Config


class Batch(NamedTuple):
    """A batch of audio features and their reference transcriptions.

    Attributes:
        input_features: Log-mel features of shape ``(batch, n_mels, frames)``.
        references: Ground-truth transcriptions, one per audio clip.
    """

    input_features: torch.Tensor
    references: list[str]


class SpeechDataset(Dataset):
    """In-memory speech dataset materialized from a (streamed) HF dataset.

    Streaming avoids downloading an entire corpus; only the requested number
    of examples are pulled and their log-mel features pre-computed. This keeps
    proof-of-concept runs light while remaining a standard map-style dataset.
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
        token = os.environ.get("HUGGINGFACE_TOKEN")
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=split,
            streaming=True,
            trust_remote_code=True,
            token=token,
        )
        dataset = dataset.cast_column(
            config.audio_column, Audio(sampling_rate=config.sample_rate)
        )
        if max_samples is not None:
            dataset = dataset.take(max_samples)

        self.examples: list[tuple[torch.Tensor, str]] = []
        for row in dataset:
            reference = str(row[config.text_column]).strip()
            if not reference:
                continue
            audio = row[config.audio_column]
            features = processor.feature_extractor(
                audio["array"],
                sampling_rate=config.sample_rate,
                return_tensors="pt",
            )
            self.examples.append((features.input_features[0], reference))

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        """Return the ``(input_features, reference)`` example at ``index``."""
        return self.examples[index]


def collate(examples: list[tuple[torch.Tensor, str]]) -> Batch:
    """Collate examples into a :class:`Batch`.

    Args:
        examples: ``(input_features, reference)`` items from a dataset.

    Returns:
        A batched :class:`Batch`.
    """
    input_features = torch.stack([features for features, _ in examples])
    references = [reference for _, reference in examples]
    return Batch(input_features=input_features, references=references)


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
