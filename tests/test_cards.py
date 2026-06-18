"""Tests for model-card metric selection and per-language formatting."""

from whisper_rl.cards import model_index, select_best


def test_select_best_prefers_max_reward() -> None:
    """When val/reward is present, the highest-reward row is chosen."""
    rows = [
        {"val/reward": -0.4, "val/wer": 0.30},
        {"val/reward": -0.2, "val/wer": 0.35},  # best reward, worse WER
        {"val/reward": -0.5, "val/wer": 0.25},
    ]
    assert select_best(rows) is rows[1]


def test_select_best_falls_back_to_min_wer() -> None:
    """Without val/reward (older runs), the lowest-WER row is chosen."""
    rows = [{"val/wer": 0.30}, {"val/wer": 0.22}, {"val/wer": 0.41}]
    assert select_best(rows) is rows[1]


def test_select_best_none_when_no_validation() -> None:
    """No validation rows yields no selection."""
    assert select_best([]) is None
    assert select_best([{"train/loss": 0.1}]) is None


def test_model_index_has_per_language_metrics() -> None:
    """Each language and an overall entry become model-index results."""
    row = {
        "val/wer": 0.4,
        "val/cer": 0.155,
        "val/wer_en": 0.1,
        "val/cer_en": 0.05,
        "val/wer_de": 0.2,
        "val/cer_de": 0.08,
    }
    block = model_index("model", row, "fixie-ai/common_voice_17_0")
    assert block.startswith("model-index:\n- name: model\n")
    # One result per language plus an overall "all" entry.
    assert "config: en" in block
    assert "config: de" in block
    assert "config: all" in block
    # Both WER and CER for each (3 configs x 2 metrics).
    assert block.count("type: wer") == 3
    assert block.count("type: cer") == 3
    assert "value: 0.1000" in block


def test_model_index_empty_without_validation() -> None:
    """No validation row yields no model-index block."""
    assert model_index("model", {}, "ds") == ""
    assert model_index("model", {"train/loss": 0.1}, "ds") == ""
