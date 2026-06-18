"""Build and upload Hugging Face model cards with W&B training curves.

Pulls a run's logged hyperparameters and metric history from Weights & Biases,
renders the training curves to a static PNG (Hub model cards do not render live
W&B panels), and uploads both the rendered card and the image to the model repo.
Used both by the ``upload-card`` CLI and by the training loop, which regenerates
the card whenever it pushes a new best checkpoint.
"""

import io
import logging

import matplotlib
import wandb
from huggingface_hub import HfApi

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP_KEY = "trainer/global_step"

# (metric key, axis label) for each curve to plot, in display order.
CURVES = [
    ("val/reward", "Validation reward (higher is better)"),
    ("val/wer", "Validation WER (lower is better)"),
    ("val/cer", "Validation CER (lower is better)"),
    ("train/reward", "Train reward (blended, on rollouts)"),
    ("train/kl", "Train KL to reference"),
]

# (config key, display name) for the hyperparameter table, in display order.
HYPERPARAMS = [
    ("base_model", "Base model"),
    ("dataset_name", "Dataset"),
    ("learning_rate", "Learning rate"),
    ("temperature", "Sampling temperature"),
    ("num_generations", "Group size (generations/clip)"),
    ("reward_weights", "Reward weights"),
    ("kl_beta", "KL penalty (β)"),
    ("batch_size", "Batch size (clips/step)"),
    ("max_steps", "Max optimizer steps"),
    ("warmup_steps", "Warmup steps"),
]


def write_card(
    repo_id: str, run: wandb.apis.public.Run, license_id: str = "mit"
) -> None:
    """Render and upload the card and training-curve image for ``run``.

    Args:
        repo_id: Target HF model repo id.
        run: The W&B run to read config and metric history from.
        license_id: SPDX license identifier for the card metadata.
    """
    best = select_best(fetch_validation_rows(run))
    api = HfApi()
    api.upload_file(
        path_or_fileobj=render_curves(fetch_series(run)),
        path_in_repo="training_curves.png",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=build_card(repo_id, run, best, license_id).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    best_wer = best.get("val/wer") if best else None
    logging.info("Uploaded card to %s (overall val/wer=%s)", repo_id, best_wer)


def fetch_validation_rows(run: wandb.apis.public.Run) -> list[dict]:
    """Return each validation row with its overall and per-language rates.

    Args:
        run: The W&B run to read.

    Returns:
        A list of history rows that carry ``val/wer`` (i.e. validation epochs),
        each including ``val/reward`` when logged and every per-language
        ``val/wer_<lang>`` / ``val/cer_<lang>``.
    """
    # scan_history inner-joins on the requested keys, so only request metrics
    # the run actually logged — older runs predate val/cer and val/reward.
    summary = set(run.summary.keys())
    keys = [STEP_KEY, "val/wer"]
    keys += [opt for opt in ("val/cer", "val/reward") if opt in summary]
    keys += sorted(key for key in summary if key.startswith(("val/wer_", "val/cer_")))
    return list(run.scan_history(keys=keys))


def select_best(rows: list[dict]) -> dict | None:
    """Pick the validation row matching the kept checkpoint.

    The checkpoint is selected on the maximum ``val/reward``; for older runs
    that never logged it, fall back to the minimum ``val/wer``.

    Args:
        rows: Validation history rows.

    Returns:
        The chosen row, or ``None`` if there are no validation rows.
    """
    val_rows = [row for row in rows if row.get("val/wer") is not None]
    if not val_rows:
        return None
    with_reward = [row for row in val_rows if row.get("val/reward") is not None]
    if with_reward:
        return max(with_reward, key=lambda row: row["val/reward"])
    return min(val_rows, key=lambda row: row["val/wer"])


def model_index(repo_name: str, row: dict, dataset: str) -> str:
    """Build a ``model-index`` YAML block with per-language metrics.

    Each language (and an ``all`` overall entry) becomes a result keyed by a
    Common Voice config, with WER and CER, so the Hub renders structured
    per-language evaluation results.

    Args:
        repo_name: The model repo name (model-index ``name``).
        row: The best validation row with overall and per-language rates.
        dataset: Hub dataset id the metrics were computed on.

    Returns:
        The ``model-index:`` YAML block, or ``""`` if there are no metrics.
    """
    if not row or row.get("val/wer") is None:
        return ""
    langs = sorted(
        key.removeprefix("val/wer_") for key in row if key.startswith("val/wer_")
    )
    configs = [(lang, f"val/wer_{lang}", f"val/cer_{lang}") for lang in langs]
    configs.append(("all", "val/wer", "val/cer"))

    results = ""
    for config, wer_key, cer_key in configs:
        wer = row.get(wer_key)
        cer = row.get(cer_key)
        metrics = ""
        if wer is not None:
            metrics += (
                f"    - type: wer\n      value: {wer:.4f}\n      name: WER ({config})\n"
            )
        if cer is not None:
            metrics += (
                f"    - type: cer\n      value: {cer:.4f}\n      name: CER ({config})\n"
            )
        if not metrics:
            continue
        results += (
            "  - task:\n"
            "      type: automatic-speech-recognition\n"
            "      name: Automatic Speech Recognition\n"
            "    dataset:\n"
            f"      type: {dataset}\n"
            "      name: Common Voice 17.0\n"
            f"      config: {config}\n"
            "      split: validation\n"
            "    metrics:\n" + metrics
        )
    if not results:
        return ""
    return f"model-index:\n- name: {repo_name}\n  results:\n{results}"


def language_table(row: dict) -> str:
    """Render a human-readable per-language WER/CER markdown table.

    Args:
        row: A validation row with ``val/wer_<lang>`` / ``val/cer_<lang>`` keys.

    Returns:
        A markdown table sorted by language, or ``""`` when the row has no
        per-language keys.
    """
    langs = sorted(
        key.removeprefix("val/wer_") for key in row if key.startswith("val/wer_")
    )
    if not langs:
        return ""
    lines = ["| Language | WER | CER |", "| --- | --- | --- |"]
    for lang in langs:
        wer = row.get(f"val/wer_{lang}")
        cer = row.get(f"val/cer_{lang}")
        wer_s = f"{wer:.3f}" if wer is not None else "—"
        cer_s = f"{cer:.3f}" if cer is not None else "—"
        lines.append(f"| `{lang}` | {wer_s} | {cer_s} |")
    return "\n".join(lines)


def fetch_series(run: wandb.apis.public.Run) -> dict[str, tuple[list, list]]:
    """Return ``{metric: (steps, values)}`` from the run's full history.

    Args:
        run: The W&B run to read.

    Returns:
        Mapping from metric key to parallel lists of optimizer steps and values,
        for every metric that has at least one logged point.
    """
    series: dict[str, tuple[list, list]] = {}
    for key, _ in CURVES:
        steps, values = [], []
        for row in run.scan_history(keys=[STEP_KEY, key]):
            if row.get(key) is not None and row.get(STEP_KEY) is not None:
                steps.append(row[STEP_KEY])
                values.append(row[key])
        if values:
            series[key] = (steps, values)
    return series


def render_curves(series: dict[str, tuple[list, list]]) -> bytes:
    """Render the available curves to a PNG and return its bytes.

    Args:
        series: Mapping from metric key to ``(steps, values)``.

    Returns:
        PNG image bytes of the training curves.
    """
    labels = dict(CURVES)
    keys = [key for key, _ in CURVES if key in series]
    n_rows = (len(keys) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, key in zip(axes.flat, keys, strict=False):
        ax.set_visible(True)
        steps, values = series[key]
        ax.plot(steps, values, linewidth=1.5)
        ax.set_title(labels[key])
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def build_card(
    repo_id: str,
    run: wandb.apis.public.Run,
    best: dict | None,
    license_id: str,
) -> str:
    """Compose the model card markdown (YAML metadata + body).

    Args:
        repo_id: Target HF model repo id.
        run: The W&B run (source of config and links).
        best: The best validation row (overall + per-language rates), or
            ``None`` if the run never validated.
        license_id: SPDX license identifier for the metadata.

    Returns:
        The full README.md contents.
    """
    config = run.config
    languages = config.get("languages") or ["en"]
    dataset = config.get("dataset_name", "fixie-ai/common_voice_17_0")
    base_model = config.get("base_model", "openai/whisper-tiny")
    base_url = f"https://huggingface.co/{base_model}"

    best = best or {}
    overall_wer = best.get("val/wer")
    overall_cer = best.get("val/cer")
    n_langs = sum(1 for key in best if key.startswith("val/wer_"))
    scope = f"overall across {n_langs} languages" if n_langs > 1 else "overall"

    configured = config.get("languages")
    if configured:
        lang_scope = ", ".join(f"`{code}`" for code in configured)
    elif n_langs:
        lang_scope = f"all {n_langs} Common Voice locales"
    else:
        lang_scope = "all available Common Voice locales"
    sample_cap = config.get("max_train_samples")
    sample_scope = (
        f"a {sample_cap}-clip slice" if sample_cap else "the full training split"
    )

    yaml = (
        "---\n"
        "library_name: transformers\n"
        f"license: {license_id}\n"
        "pipeline_tag: automatic-speech-recognition\n"
        f"base_model: {base_model}\n"
        "datasets:\n"
        f"- {dataset}\n"
        "language:\n"
        + "".join(f"- {code}\n" for code in languages)
        + "tags:\n- whisper\n- grpo\n- reinforcement-learning\n- asr\n"
        "metrics:\n- wer\n- cer\n"
        + model_index(repo_id.split("/")[-1], best, dataset)
        + "---\n"
    )

    hp_rows = "\n".join(
        f"| {name} | `{config[key]}` |" for key, name in HYPERPARAMS if key in config
    )
    if overall_wer is not None:
        cer_part = f", CER {overall_cer:.3f}" if overall_cer is not None else ""
        result_line = (
            f"**Best validation ({scope}): WER {overall_wer:.3f}{cer_part}**\n"
        )
    else:
        result_line = ""
    table = language_table(best)
    performance_section = (
        f"\n## Performance per language\n\nValidation WER and CER at the best "
        f"checkpoint, per Common Voice locale (also in the Evaluation Results "
        f"metadata above):\n\n{table}\n"
        if table
        else ""
    )

    return f"""{yaml}
# {repo_id.split("/")[-1]}

A [Whisper]({base_url}) model fine-tuned with **GRPO** (Group Relative Policy
Optimization) using a **blended error-rate reward**. Trained with
[whisper-rl](https://github.com/will-rice/whisper-rl).

{result_line}
## Training data

Fine-tuned on [{dataset}](https://huggingface.co/datasets/{dataset}) —
{lang_scope}, streamed and decoded on the fly from {sample_scope}. Each clip's
language is pinned from its Common Voice locale during training.
{performance_section}
## How it was trained

Instead of cross-entropy against a single reference, for each audio clip the
policy samples a group of `num_generations` transcriptions, scores each by a
negated blend of word error rate, character error rate, and length / repetition
penalties, and is nudged toward the better candidates with a clipped
policy-gradient objective regularized by a per-token KL penalty to the frozen
base model. Advantages are the group-relative, standardized rewards
(`A = (r - mean) / (std + eps)`), so no value network is needed. The clip's
language is pinned from its Common Voice locale, and the policy's own greedy
transcriptions are scored as validation WER and CER.

## Hyperparameters

| Field | Value |
| --- | --- |
{hp_rows}

## Training curves

Pulled from the [Weights & Biases run]({run.url}) (static snapshot):

![training curves](training_curves.png)

## Usage

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="{repo_id}")
print(asr("audio.wav")["text"])
```

## Limitations

A proof-of-concept GRPO recipe, not a tuned production system. WER and CER are
reported on a held-out Common Voice validation slice after text normalization;
real-world performance varies by domain, accent, language, and audio quality.
"""
