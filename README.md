# whisper-rl

A proof-of-concept for finetuning [Whisper](https://huggingface.co/openai/whisper-tiny)
with **GRPO** (Group Relative Policy Optimization) using **word error rate (WER)**
as the reward, **across many languages**.

Instead of cross-entropy against a single reference, the model samples a group
of candidate transcriptions per audio clip, scores each one by its WER against
the ground truth, and is nudged toward the lower-error candidates with a
policy-gradient objective regularized by a KL penalty to the original model.

## How it works

For each audio clip in a batch:

1. **Sample a group.** Draw `num_generations` transcriptions from the current
   policy with temperature sampling. Whisper auto-detects each clip's language
   and prepends its fixed 4-token decoder prompt
   (`<|sot|><|lang|><|transcribe|><|notimestamps|>`); GRPO scores only the
   transcription tokens that follow.
2. **Score.** Reward each completion with `-WER` (post text-normalization)
   against the reference transcript. Lower error → higher reward.
3. **Group-relative advantages.** Center and scale the rewards within each
   group: `A = (r - mean) / (std + eps)`. No value network is needed — the
   group mean is the baseline.
4. **Policy update.** Optimize the clipped GRPO objective on the per-token
   log-probabilities of the sampled completions, with a per-token KL penalty
   toward a frozen reference copy of the initial model.

The reward, advantage, log-prob, and loss math live in
[`grpo.py`](src/whisper_rl/grpo.py) and [`rewards.py`](src/whisper_rl/rewards.py)
and are covered by unit tests.

## Project structure

```
src/whisper_rl/
├── config.py              # Pydantic hyperparameter config
├── rewards.py             # WER normalization + reward
├── grpo.py                # Advantages, token log-probs, GRPO loss (pure torch)
├── metrics.py             # Per-language corpus-WER accumulator
├── modeling/__init__.py   # Whisper policy / frozen reference builders
├── datasets/__init__.py   # Streamed multilingual Common Voice + DataModule
├── lightning_module.py    # WhisperGRPOModule (the training loop)
└── scripts/train.py       # CLI entry point (the `train` command)
tests/                     # Unit tests for grpo, rewards, metrics, datasets
```

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure credentials

```bash
cp .env.example .env
# Add your WANDB_API_KEY (and HUGGINGFACE_TOKEN if you hit rate limits)
```

The default dataset is [`fixie-ai/common_voice_17_0`](https://huggingface.co/datasets/fixie-ai/common_voice_17_0),
an unofficial parquet Common Voice 17 mirror. It replaces the official
`mozilla-foundation/common_voice_*` repos, which were removed from the Hub in
October 2025 (script-based mirrors like `fsicoli/common_voice_21_0` no longer
load either: `datasets` >= 4 dropped script loaders). Column names follow the
Common Voice convention (`sentence`, `audio`, `locale`); override them in
`config.py` if your dataset differs.

Audio decoding uses [`torchcodec`](https://github.com/pytorch/torchcodec),
which needs the FFmpeg shared libraries (FFmpeg 4–7) installed on the host.

## Usage

By default `train` streams **every Common Voice 17 locale** (auto-discovered
and interleaved) and finetunes `openai/whisper-tiny`. Training audio is
decoded and featurized on the fly in the dataloader workers, so the full
splits train with flat memory; the eval slice is materialized once so every
validation scores the exact same clips. The language of each clip is
auto-detected by Whisper:

```bash
uv run train
```

Useful flags:

- `--num_devices`: number of GPUs (default `1`)
- `--no_wandb`: disable Weights & Biases logging
- `--fast_dev_run`: run a single train/val batch to smoke-test the loop
- `--checkpoint_path`: resume from a Lightning checkpoint
- `--log_root`: directory for checkpoints and logs (default `logs`)

A quick end-to-end smoke test without external logging:

```bash
uv run train --fast_dev_run --no_wandb
```

### Configuration

All hyperparameters live in [`src/whisper_rl/config.py`](src/whisper_rl/config.py).
Notable knobs:

| Field                                    | Default                      | Meaning                                          |
| ---------------------------------------- | ---------------------------- | ------------------------------------------------ |
| `base_model`                             | `openai/whisper-tiny`        | Multilingual Whisper checkpoint to finetune      |
| `dataset_name`                           | `fixie-ai/common_voice_17_0` | Parquet Common Voice 17 mirror                   |
| `languages`                              | `None`                       | Locales to stream; `None` = all dataset locales  |
| `num_generations`                        | `8`                          | Completions sampled per clip (group size)        |
| `temperature`                            | `1.0`                        | Sampling temperature for rollouts                |
| `kl_beta`                                | `0.04`                       | Weight of the KL penalty to the reference        |
| `clip_eps`                               | `0.2`                        | PPO-style ratio clipping                         |
| `learning_rate`                          | `1e-6`                       | AdamW learning rate (RL finetuning is sensitive) |
| `max_train_samples` / `max_eval_samples` | `None` / `256`               | Streamed slice caps (`None` = full split)        |

To train on a specific set of languages, set
`languages=["en", "de", "fr", "zh-CN"]` (Common Voice locale codes).

### Metrics

Logged during training: `train/loss`, `train/reward`, `train/kl`,
`train/completion_len`. Validation reports corpus WER overall (`val/wer`) **and
per language** (`val/wer_en`, `val/wer_de`, …), so you can see which languages
improve or regress.

## Development

```bash
uv run pytest                    # tests
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run ty check                  # type check
uv run pre-commit run -a         # everything (matches CI)
```

## Notes & caveats

- This is a **proof of concept**, not a tuned recipe. RL finetuning of ASR is
  sensitive to learning rate, group size, and KL weight — expect to sweep.
- Whisper-tiny is chosen for fast iteration; bump `base_model` to `whisper-base`
  or `whisper-small` once the loop is validated.
- Streaming ~50 language configs and interleaving them is convenient but not
  perfectly balanced; for serious per-language evaluation, raise
  `max_eval_samples` (or set it to `None`) so every language is well represented.
- One gradient update is taken per rollout (the importance ratio is 1), which
  keeps the implementation on-policy and simple. Add inner epochs over cached
  rollouts if you want the full off-policy PPO-style reuse.
