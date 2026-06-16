"""Generate a model card with W&B training curves and upload it to the Hub.

Pulls a run's logged hyperparameters and metric history from Weights & Biases,
renders the training curves to a static PNG (Hub model cards do not render live
W&B panels), and uploads both the rendered card and the image to the model repo.
"""

import argparse
import io
import logging

import matplotlib
import wandb
from huggingface_hub import HfApi

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO)

ENTITY = "will-rice"
PROJECT = "whisper-rl"
STEP_KEY = "trainer/global_step"

# (metric key, axis label) for each curve to plot, in display order.
CURVES = [
    ("val/wer", "Validation WER (lower is better)"),
    ("train/reward", "Train reward (-WER on rollouts)"),
    ("train/kl", "Train KL to reference"),
    ("train/completion_len", "Mean completion length"),
]

# (config key, display name) for the hyperparameter table, in display order.
HYPERPARAMS = [
    ("base_model", "Base model"),
    ("dataset_name", "Dataset"),
    ("learning_rate", "Learning rate"),
    ("temperature", "Sampling temperature"),
    ("num_generations", "Group size (generations/clip)"),
    ("kl_beta", "KL penalty (β)"),
    ("clip_eps", "PPO clip ε"),
    ("batch_size", "Batch size (clips/step)"),
    ("max_steps", "Max optimizer steps"),
    ("warmup_steps", "Warmup steps"),
    ("max_new_tokens", "Max new tokens"),
]


def main() -> None:
    """Entry point for the ``upload-card`` console script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", type=str, help="Target HF model repo id.")
    parser.add_argument("run_id", type=str, help="8-character W&B run id.")
    parser.add_argument("--entity", default=ENTITY, type=str)
    parser.add_argument("--project", default=PROJECT, type=str)
    parser.add_argument("--license", default="mit", type=str)
    args = parser.parse_args()

    run = wandb.Api().run(f"{args.entity}/{args.project}/{args.run_id}")
    series = fetch_series(run)
    best_wer = min(series["val/wer"][1]) if "val/wer" in series else None

    api = HfApi()
    api.upload_file(
        path_or_fileobj=render_curves(series),
        path_in_repo="training_curves.png",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=build_card(args.repo_id, run, best_wer, args.license).encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    logging.info("Uploaded card to %s (best val/wer=%s)", args.repo_id, best_wer)


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
    best_wer: float | None,
    license_id: str,
) -> str:
    """Compose the model card markdown (YAML metadata + body).

    Args:
        repo_id: Target HF model repo id.
        run: The W&B run (source of config and links).
        best_wer: Best validation WER, or ``None`` if never logged.
        license_id: SPDX license identifier for the metadata.

    Returns:
        The full README.md contents.
    """
    config = run.config
    languages = config.get("languages") or ["en"]
    dataset = config.get("dataset_name", "fixie-ai/common_voice_17_0")
    base_model = config.get("base_model", "openai/whisper-tiny")
    base_url = f"https://huggingface.co/{base_model}"

    metric_yaml = ""
    if best_wer is not None:
        metric_yaml = (
            "model-index:\n"
            f"- name: {repo_id.split('/')[-1]}\n"
            "  results:\n"
            "  - task:\n"
            "      type: automatic-speech-recognition\n"
            "      name: Automatic Speech Recognition\n"
            "    dataset:\n"
            f"      type: {dataset}\n"
            "      name: Common Voice 17.0\n"
            "    metrics:\n"
            "    - type: wer\n"
            f"      value: {best_wer:.4f}\n"
            "      name: Validation WER\n"
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
        "metrics:\n- wer\n" + metric_yaml + "---\n"
    )

    hp_rows = "\n".join(
        f"| {name} | `{config[key]}` |" for key, name in HYPERPARAMS if key in config
    )
    result_line = (
        f"**Best validation WER: {best_wer:.3f}**\n" if best_wer is not None else ""
    )

    return f"""{yaml}
# {repo_id.split("/")[-1]}

A [Whisper]({base_url}) model fine-tuned with **GRPO** (Group Relative Policy
Optimization) using **word error rate (WER)** as the reward. Trained with
[whisper-rl](https://github.com/will-rice/whisper-rl).

{result_line}
## How it was trained

Instead of cross-entropy against a single reference, for each audio clip the
policy samples a group of `num_generations` transcriptions, scores each by its
negated WER against the ground truth, and is nudged toward the lower-error
candidates with a clipped policy-gradient objective regularized by a per-token
KL penalty to the frozen base model. Advantages are the group-relative,
standardized rewards (`A = (r - mean) / (std + eps)`), so no value network is
needed. Audio is streamed from Common Voice and the policy's own greedy
transcriptions are evaluated as validation WER.

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

A proof-of-concept GRPO recipe on `whisper-tiny`-scale models, not a tuned
production system. WER is reported on a held-out Common Voice validation slice
after text normalization; real-world performance varies by domain, accent, and
audio quality.
"""


if __name__ == "__main__":
    main()
