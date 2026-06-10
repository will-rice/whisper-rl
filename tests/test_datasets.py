"""Tests for dataset collation."""

import torch

from whisper_rl.datasets import Batch, collate


def test_collate_stacks_features_and_keeps_alignment() -> None:
    """Collation stacks features and preserves reference/language order."""
    examples = [
        (torch.zeros(80, 3000), "hello", "en"),
        (torch.ones(80, 3000), "hallo", "de"),
    ]
    batch = collate(examples)
    assert isinstance(batch, Batch)
    assert batch.input_features.shape == (2, 80, 3000)
    assert batch.references == ["hello", "hallo"]
    assert batch.languages == ["en", "de"]
    # Row order is preserved: second clip is the all-ones tensor.
    assert torch.equal(batch.input_features[1], torch.ones(80, 3000))
