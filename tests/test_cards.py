"""Tests for model-card metric selection and per-language formatting."""

from whisper_rl.cards import (
    dataset_label,
    fetch_validation_rows,
    language_table,
    model_index,
    qualify_repo_id,
    select_best,
)


class _StubApi:
    """Minimal HfApi stand-in exposing the authenticated user's name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def whoami(self) -> dict[str, str]:
        return {"name": self.name}


def test_qualify_repo_id_adds_namespace_to_bare_name() -> None:
    """A bare experiment name is qualified with the authenticated user."""
    got = qualify_repo_id("whisper-tiny-grpo-abc", _StubApi("wrice"))  # ty: ignore[invalid-argument-type]
    assert got == "wrice/whisper-tiny-grpo-abc"


def test_qualify_repo_id_leaves_namespaced_id_untouched() -> None:
    """An already-namespaced id passes through without a whoami lookup."""
    got = qualify_repo_id("wrice/model", _StubApi("other"))  # ty: ignore[invalid-argument-type]
    assert got == "wrice/model"


def test_dataset_label_strips_local_index_path() -> None:
    """A local index directory shows a clean dataset name."""
    assert dataset_label("/data/common_voice_26/index") == "common_voice_26"
    assert dataset_label("/data/cv25") == "cv25"


def test_dataset_label_passes_through_hub_ids() -> None:
    """Hub dataset ids are left untouched."""
    assert dataset_label("fixie-ai/common_voice_17_0") == "fixie-ai/common_voice_17_0"


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


def test_language_table_lists_sorted_per_language_rates() -> None:
    """The markdown table renders one sorted row per language with WER and CER."""
    row = {"val/wer_en": 0.1, "val/cer_en": 0.05, "val/wer_de": 0.2, "val/cer_de": 0.08}
    lines = language_table(row).splitlines()
    assert lines[0] == "| Language | WER | CER |"
    assert lines[2] == "| `de` | 0.200 | 0.080 |"
    assert lines[3] == "| `en` | 0.100 | 0.050 |"


def test_language_table_empty_without_per_language_keys() -> None:
    """A row with no per-language keys produces no table."""
    assert language_table({"val/wer": 0.4}) == ""
