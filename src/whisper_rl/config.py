"""Configuration for Whisper GRPO finetuning."""

from pydantic import BaseModel


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
    # Unofficial Common Voice 21 mirror (the official mozilla-foundation repos
    # were removed from the Hub in Oct 2025). It is a script-based loader, so
    # ``trust_remote_code=True`` is required.
    dataset_name: str = "fsicoli/common_voice_21_0"
    # Common Voice locale configs to stream and interleave. ``None`` means
    # "every locale in the dataset" (auto-discovered). The clip's language is
    # auto-detected by Whisper at generation time; the locale is used only to
    # bucket per-language metrics.
    languages: list[str] | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    audio_column: str = "audio"
    text_column: str = "sentence"
    locale_column: str = "locale"
    sample_rate: int = 16000
    # Cap the number of streamed examples so a PoC run stays light. ``None``
    # uses the full splits.
    max_train_samples: int | None = 1024
    max_eval_samples: int | None = 256
    batch_size: int = 4
    num_workers: int = 4

    # GRPO.
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

    # Training.
    max_epochs: int = 1
    max_steps: int = 500
    learning_rate: float = 1e-6
    min_learning_rate: float = 1e-7
    weight_decay: float = 0.0
    warmup_steps: int = 20
    grad_clip: float = 1.0
    val_check_interval: float = 0.5
