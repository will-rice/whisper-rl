"""Tests for training-script helpers."""

from huggingface_hub.utils import validate_repo_id

from whisper_rl.scripts.train import hub_repo_name


def test_hub_repo_name_passes_short_names_through() -> None:
    """A name already within the limit is unchanged."""
    name = "whisper-small-grpo-1fada6f-continue-100k"
    assert hub_repo_name(name) == name


def test_hub_repo_name_truncates_compounded_warm_start_names() -> None:
    """Warm-starting from a Hub model can overflow HF's 96-char repo limit."""
    name = (
        "whisper-small-grpo-1fada6f-small-all50-blendreward-t0.7-lr3e-6-bs4"
        "-grpo-49579fe-small-all50-blend-continue-100k"
    )
    result = hub_repo_name(name)
    assert len(result) <= 96
    validate_repo_id(result)  # raises if invalid
