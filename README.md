# whisper-rl

A proof-of-concept for finetuning [Whisper](https://huggingface.co/openai/whisper-tiny)
with **GRPO** (Group Relative Policy Optimization) using **word error rate (WER)**
as the reward.

Instead of cross-entropy against a single reference, the model samples a group
of candidate transcriptions per audio clip, scores each one by its WER against
the ground truth, and is nudged toward the lower-error candidates with a
policy-gradient objective regularized by a KL penalty to the original model.

## How it works

For each audio clip in a batch:

1. **Sample a group.** Draw `num_generations` transcriptions from the current
   policy with temperature sampling.
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
├── modeling/__init__.py   # Whisper policy / frozen reference builders
├── datasets/__init__.py   # Streamed Common Voice dataset + DataModule
├── lightning_module.py    # WhisperGRPOModule (the training loop)
└── scripts/train.py       # CLI entry point (the `train` command)
tests/                     # Unit tests for grpo, rewards, config
```

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure credentials

```bash
cp .env.example .env
# Add your HUGGINGFACE_TOKEN and WANDB_API_KEY
```

Common Voice is a **gated** dataset: log in to Hugging Face and accept the
terms for `mozilla-foundation/common_voice_17_0` (or point `dataset_name` at
another corpus). The `HUGGINGFACE_TOKEN` from `.env` is used to authenticate
the streamed download.

## Usage

Train (defaults: `openai/whisper-tiny`, English Common Voice, a small streamed
slice so a PoC run stays light):

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

| Field                                    | Default                                       | Meaning                                          |
| ---------------------------------------- | --------------------------------------------- | ------------------------------------------------ |
| `base_model`                             | `openai/whisper-tiny`                         | Whisper checkpoint to finetune                   |
| `dataset_name` / `dataset_config`        | `mozilla-foundation/common_voice_17_0` / `en` | Dataset and locale                               |
| `num_generations`                        | `8`                                           | Completions sampled per clip (group size)        |
| `temperature`                            | `1.0`                                         | Sampling temperature for rollouts                |
| `kl_beta`                                | `0.04`                                        | Weight of the KL penalty to the reference        |
| `clip_eps`                               | `0.2`                                         | PPO-style ratio clipping                         |
| `learning_rate`                          | `1e-6`                                        | AdamW learning rate (RL finetuning is sensitive) |
| `max_train_samples` / `max_eval_samples` | `512` / `64`                                  | Streamed slice sizes (`None` = full split)       |

Metrics logged during training: `train/loss`, `train/reward`, `train/kl`,
`train/completion_len`, and `val/wer` (greedy-decode WER on the eval split).

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
- One gradient update is taken per rollout (the importance ratio is 1), which
  keeps the implementation on-policy and simple. Add inner epochs over cached
  rollouts if you want the full off-policy PPO-style reuse.
