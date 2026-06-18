"""Tests for model-card metric selection and per-language formatting."""

from whisper_rl.cards import language_table, select_best


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


def test_language_table_lists_per_language_rates() -> None:
    """The table renders one sorted row per language with WER and CER."""
    row = {
        "val/wer": 0.4,
        "val/wer_en": 0.1,
        "val/cer_en": 0.05,
        "val/wer_de": 0.2,
        "val/cer_de": 0.08,
    }
    table = language_table(row)
    lines = table.splitlines()
    assert lines[0] == "| Language | WER | CER |"
    # Languages are sorted; German before English.
    assert lines[2] == "| `de` | 0.200 | 0.080 |"
    assert lines[3] == "| `en` | 0.100 | 0.050 |"


def test_language_table_empty_without_per_language_keys() -> None:
    """A row with no per-language keys produces no table."""
    assert language_table({"val/wer": 0.4}) == ""
