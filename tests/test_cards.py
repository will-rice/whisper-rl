"""Tests for model-card metric selection and per-language formatting."""

from whisper_rl.cards import fetch_validation_rows, model_index, select_best


class _StubRun:
    """Minimal stand-in for a W&B run exposing summary keys and history."""

    def __init__(self, summary_keys: list[str]) -> None:
        self.summary = dict.fromkeys(summary_keys, 0.0)
        self.requested: list[str] | None = None

    def scan_history(self, keys: list[str]) -> list[dict]:
        self.requested = keys
        return [dict.fromkeys(keys, 0.1)]


def test_fetch_validation_rows_skips_unlogged_optional_metrics() -> None:
    """Old runs without val/cer or val/reward are not excluded by the scan."""
    run = _StubRun(["val/wer", "val/wer_en", "train/loss"])
    fetch_validation_rows(run)  # ty: ignore[invalid-argument-type]
    assert run.requested is not None
    assert "val/cer" not in run.requested
    assert "val/reward" not in run.requested
    assert "val/wer" in run.requested
    assert "val/wer_en" in run.requested


def test_fetch_validation_rows_includes_logged_optional_metrics() -> None:
    """Newer runs request val/cer and val/reward when present."""
    run = _StubRun(["val/wer", "val/cer", "val/reward", "val/wer_de", "val/cer_de"])
    fetch_validation_rows(run)  # ty: ignore[invalid-argument-type]
    assert run.requested is not None
    assert {"val/cer", "val/reward", "val/wer_de", "val/cer_de"} <= set(run.requested)


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
