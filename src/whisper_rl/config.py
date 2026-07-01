"""Configuration for Whisper GRPO finetuning."""

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Hyperparameters for Whisper GRPO finetuning.

    The defaults are tuned for a quick proof-of-concept run on a single GPU
    using ``openai/whisper-tiny`` and a small slice of Common Voice.
    """

    # Reproducibility.
    seed: int = 42

    # Model.
    base_model: str = "openai/whisper-tiny"
    task: str = "transcribe"

    # Data.
    # Parquet Common Voice 17 mirror (the official mozilla-foundation repos
    # were removed from the Hub in Oct 2025, and ``datasets`` >= 4 dropped
    # script-based loaders entirely).
    dataset_name: str = "fixie-ai/common_voice_17_0"
    # Common Voice locale configs to stream and interleave. ``None`` means
    # "every locale in the dataset" (auto-discovered). The locale pins each
    # clip's decoder language at generation time and buckets per-language
    # metrics.
    languages: list[str] | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    audio_column: str = "audio"
    text_column: str = "sentence"
    locale_column: str = "locale"
    sample_rate: int = 16000
    # Cap the number of streamed examples. ``None`` uses the full split: the
    # train side streams and featurizes on the fly, so any size is fine; the
    # eval side is materialized in memory and should stay capped.
    max_train_samples: int | None = None
    max_eval_samples: int | None = 256
    batch_size: int = 8
    num_workers: int = 8

    # GRPO.
    # Reward is the negated, weight-averaged blend of these penalty components
    # (see ``whisper_rl.rewards.REWARD_COMPONENTS``): ``wer`` and ``cer`` are
    # the error rates (CER is finer grained and works for languages without
    # word spaces), ``length`` penalizes runaway-long completions, and
    # ``repetition`` penalizes hallucination loops. A weight of 0 (or omission)
    # drops a component. WER and CER are reported at validation regardless.
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "wer": 1.0,
            "cer": 1.0,
            "length": 0.5,
            "repetition": 0.5,
        }
    )
    # Number of completions sampled per audio clip (the "group").
    num_generations: int = 8
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    # KL penalty weight against the frozen reference policy.
    kl_beta: float = 0.04
    # PPO-style ratio clipping epsilon.
    clip_eps: float = 0.2
    # Small constant added to the per-group std when normalizing advantages.
    advantage_eps: float = 1e-4
    # Weight of the supervised cross-entropy term added to the GRPO loss. This
    # is a direct teaching signal toward the reference, so it can move languages
    # the policy never samples correctly (where GRPO alone has no gradient).
    # ``0`` disables it (pure GRPO). Heavy SFT bootstraps low-resource languages
    # early but over-corrects a strong base model late (on CV22, the languages
    # Whisper is strongest at — en/de/pt — degrade past ~8k). So hold
    # ``sft_weight`` through ``sft_anneal_start`` (full teaching while the
    # floored languages are still learning), decay linearly to
    # ``sft_weight_final`` by ``sft_anneal_end``, then hold the floor.
    sft_weight: float = 1.0
    sft_weight_final: float = 0.1
    sft_anneal_start: int = 6000
    sft_anneal_end: int = 12000

    # Training. The streamed train dataset has no length, so the run is bound
    # solely by ``max_steps`` (epochs are disabled in the trainer); validation
    # cadence is likewise in optimizer steps.
    max_steps: int = 500
    learning_rate: float = 1e-6
    min_learning_rate: float = 1e-7
    weight_decay: float = 0.0
    warmup_steps: int = 20
    grad_clip: float = 1.0
    val_check_interval: int = 250
